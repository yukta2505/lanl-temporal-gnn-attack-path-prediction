"""Train XGBoost with disk-backed feature arrays and chronological splits."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from xgboost import XGBClassifier

FEATURES = ["source_out_events", "target_in_events", "pair_events",
            "pair_recency_seconds", "pair_success_rate"]


def _load_memmaps(path: Path, temp_dir: Path):
    """Two-pass JSONL reader; avoid retaining millions of Python dicts."""
    with path.open(encoding="utf-8") as stream:
        row_count = sum(1 for line in stream if line.strip())
    if not row_count:
        raise ValueError("No examples found")
    x = np.memmap(temp_dir / "features.bin", mode="w+", dtype=np.float32,
                  shape=(row_count, len(FEATURES)))
    y = np.memmap(temp_dir / "labels.bin", mode="w+", dtype=np.uint8, shape=(row_count,))
    times = np.memmap(temp_dir / "times.bin", mode="w+", dtype=np.int64, shape=(row_count,))
    groups = np.memmap(temp_dir / "groups.bin", mode="w+", dtype=np.int32, shape=(row_count,))
    group_ids: dict[tuple[int, str], int] = {}
    i = 0
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                values = row["features"]
                if len(values) != len(FEATURES):
                    raise ValueError("unexpected feature count")
                timestamp = int(row["window_start"])
                source = str(row["source_host"])
                group_key = (timestamp, source)
                groups[i] = group_ids.setdefault(group_key, len(group_ids))
                x[i] = values
                y[i] = int(row["label"])
                times[i] = timestamp
                i += 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid example at line {line_no}: {exc}") from exc
    if i != row_count:
        raise ValueError(f"Example file changed during reading ({row_count} rows counted, {i} read)")
    x.flush(); y.flush(); times.flush(); groups.flush()
    return x, y, times, groups, group_ids


def _time_masks(times: np.ndarray):
    windows = np.unique(times)
    if len(windows) < 3:
        raise ValueError("Need at least 3 distinct prediction windows for train/validation/test")
    train_cut = min(len(windows) - 2, max(1, int(len(windows) * 0.70)))
    validation_cut = min(len(windows) - 1, max(train_cut + 1, int(len(windows) * 0.85)))
    train_end, validation_end = windows[train_cut], windows[validation_cut]
    return times < train_end, (times >= train_end) & (times < validation_end), times >= validation_end


def _precision_at_10(labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray) -> float | None:
    if not len(labels):
        return None
    order = np.lexsort((-probabilities, groups))
    sorted_groups, sorted_labels = groups[order], labels[order]
    _, starts, counts = np.unique(sorted_groups, return_index=True, return_counts=True)
    scores = [float(sorted_labels[start:start + min(10, count)].sum() / min(10, count))
              for start, count in zip(starts, counts)]
    return float(np.mean(scores)) if scores else None


def _evaluate(model: XGBClassifier, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    if len(y) == 0:
        return {"examples": 0, "positives": 0, "roc_auc": None, "pr_auc": None,
                "precision": None, "recall": None, "f1": None, "precision_at_10": None}
    probabilities = model.predict_proba(np.asarray(x))[:, 1]
    predictions = probabilities >= 0.5
    both_classes = np.unique(y).size == 2
    has_positives = bool(np.sum(y))
    return {
        "examples": int(len(y)), "positives": int(np.sum(y)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if both_classes else None,
        "pr_auc": float(average_precision_score(y, probabilities)) if has_positives else None,
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "precision_at_10": _precision_at_10(y, probabilities, groups),
    }


def train(input_path: Path, output_dir: Path, seed: int = 42) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xgb_arrays_", dir=output_dir) as temp:
        x, y, times, groups, _ = _load_memmaps(input_path, Path(temp))
        train_mask, validation_mask, test_mask = _time_masks(times)
        train_y = np.asarray(y[train_mask])
        if np.unique(train_y).size < 2:
            raise ValueError("Training split must include positive and negative examples")
        positive = int(train_y.sum())
        negative = len(train_y) - positive
        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=negative / max(positive, 1),
            objective="binary:logistic", eval_metric="aucpr",
            tree_method="hist", random_state=seed, n_jobs=2,
        )
        model.fit(np.asarray(x[train_mask]), train_y)
        joblib.dump({"model": model, "features": FEATURES}, output_dir / "xgboost_next_hop.joblib")
        metrics = {
            "task": "next-window red-team hop prediction conditioned on previously known compromised hosts",
            "split": {
                "train_windows": int(np.unique(times[train_mask]).size),
                "validation_windows": int(np.unique(times[validation_mask]).size),
                "test_windows": int(np.unique(times[test_mask]).size),
                "train_examples": int(train_mask.sum()),
                "validation_examples": int(validation_mask.sum()),
                "test_examples": int(test_mask.sum()),
            },
            "validation": _evaluate(model, x[validation_mask], y[validation_mask], groups[validation_mask]),
            "test": _evaluate(model, x[test_mask], y[test_mask], groups[test_mask]),
            "top_feature_importance": sorted(
                zip(FEATURES, map(float, model.feature_importances_)),
                key=lambda item: item[1], reverse=True,
            ),
        }
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(metrics, stream, indent=2)
        # Windows cannot remove the temporary directory while its memmaps are
        # still open. Close each backing file before TemporaryDirectory cleanup.
        for array in (x, y, times, groups):
            array._mmap.close()
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost next-hop baseline")
    parser.add_argument("--input", type=Path, default=Path("data/processed/next_hop_examples_15d.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/xgboost"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
