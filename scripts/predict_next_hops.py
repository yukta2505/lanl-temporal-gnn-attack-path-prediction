"""Score next-hop examples and export local feature explanations as CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import DMatrix


def predict(model_path: Path, examples_path: Path, output_path: Path, batch_size: int = 50000) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["window_start", "source_host", "target_host", "probability", "explanation"]
    count = 0
    with examples_path.open(encoding="utf-8") as stream, output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        batch = []
        def write_batch(rows: list[dict]) -> None:
            nonlocal count
            if not rows:
                return
            frame = pd.DataFrame(rows)
            matrix = DMatrix(frame[features], feature_names=features)
            frame["probability"] = model.predict_proba(frame[features])[:, 1]
            contributions = model.get_booster().predict(matrix, pred_contribs=True)
            for row, contribution in zip(frame.to_dict("records"), contributions):
                order = sorted(range(len(features)), key=lambda idx: abs(float(contribution[idx])), reverse=True)[:3]
                explanation = "; ".join(
                    f"{features[idx]}={'raises' if contribution[idx] > 0 else 'lowers'} score by {abs(float(contribution[idx])):.3f}"
                    for idx in order
                )
                writer.writerow({key: row[key] for key in columns[:-1]} | {"explanation": explanation})
                count += 1
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                values = record["features"]
                if len(values) != len(features):
                    raise ValueError("unexpected feature count")
                batch.append({
                    "window_start": int(record["window_start"]),
                    "source_host": record["source_host"],
                    "target_host": record["target_host"],
                    **dict(zip(features, values)),
                })
                if len(batch) >= batch_size:
                    write_batch(batch)
                    batch.clear()
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid example at line {line_no}: {exc}") from exc
        write_batch(batch)
    if count == 0:
        raise ValueError("No candidate examples to score")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Score next-hop candidates with local XGBoost explanations")
    parser.add_argument("--model", type=Path, default=Path("outputs/xgboost/xgboost_next_hop.joblib"))
    parser.add_argument("--input", type=Path, default=Path("data/processed/next_hop_examples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=50000)
    args = parser.parse_args()
    print(json.dumps({"scored_candidates": predict(args.model, args.input, args.output, args.batch_size),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
