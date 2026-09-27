"""Disk-backed LANL next-hop example generation with resumable checkpoints."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sqlite3
import zlib
from collections import defaultdict
from itertools import groupby
from pathlib import Path

from .lanl_dataset import iter_auth_events, iter_flow_events, load_redteam

FEATURES = ["source_out_events", "target_in_events", "pair_events",
            "pair_recency_seconds", "pair_success_rate"]
RESULT_KEYS = ("examples", "positives", "negatives", "windows", "events_seen")


def _input_signature(auth: Path, redteam: Path, flows: Path | None,
                     days: int, window_minutes: int, negatives: int, seed: int) -> str:
    sources = []
    for path in (auth, redteam, flows):
        if path is None:
            sources.append(None)
            continue
        resolved = path.resolve()
        stat = resolved.stat()
        sources.append({"path": str(resolved), "size": stat.st_size,
                        "modified_ns": stat.st_mtime_ns})
    return json.dumps({"sources": sources, "days": days,
                       "window_minutes": window_minutes,
                       "negatives": negatives, "seed": seed}, sort_keys=True)


def generate(auth: Path, redteam: Path, output: Path, flows: Path | None = None,
             days: int = 3, window_minutes: int = 30, negatives: int = 20,
             seed: int = 42, state_dir: Path | None = None,
             resume: bool = False) -> dict:
    """Stream events, persisting aggregate state and an output offset per window.

    Gzip inputs are reread from the beginning on resume because gzip streams
    cannot seek efficiently. Completed windows are skipped using the SQLite
    checkpoint, and the output is truncated to its last committed byte offset.
    """
    if days <= 0 or window_minutes <= 0 or negatives <= 0:
        raise ValueError("days, window_minutes and negatives must be positive")
    signature = _input_signature(auth, redteam, flows, days, window_minutes,
                                 negatives, seed)
    width = window_minutes * 60
    labels = load_redteam(redteam)
    labels_by_window: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for timestamp, _user, source, target in labels:
        labels_by_window[timestamp // width].add((source, target))
    streams = [iter_auth_events(auth, labels)]
    if flows:
        streams.append(iter_flow_events(flows))
    merged = heapq.merge(*streams, key=lambda event: event["timestamp"])
    first_time = None

    def limited():
        nonlocal first_time
        for event in merged:
            if first_time is None:
                first_time = event["timestamp"]
            if event["timestamp"] - first_time >= days * 86400:
                break
            yield event

    output.parent.mkdir(parents=True, exist_ok=True)
    if state_dir is None:
        state_dir = output.with_name(output.name + ".state")
    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / "history.sqlite"
    if resume:
        if not db_path.exists() or not output.exists():
            raise FileNotFoundError("Cannot resume: checkpoint database or output file is missing")
        db = sqlite3.connect(db_path)
        try:
            state = db.execute(
                "SELECT signature,last_completed_window,output_offset,result_json,complete "
                "FROM run_state WHERE id=1"
            ).fetchone()
        except sqlite3.Error as exc:
            db.close()
            raise ValueError("Checkpoint is invalid; start a fresh generation") from exc
        if state is None or state[0] != signature:
            db.close()
            raise ValueError("Checkpoint inputs or generation options do not match this run")
        if state[4]:
            result = json.loads(state[3])
            db.close()
            return {**result, "features": FEATURES, "output": str(output),
                    "state_dir": str(state_dir), "already_complete": True}
        last_completed_window = int(state[1])
        output_offset = int(state[2])
        result = json.loads(state[3])
        output_stream = output.open("r+", encoding="utf-8", newline="")
        output_stream.seek(output_offset)
        output_stream.truncate()
    else:
        if db_path.exists() or output.exists():
            raise FileExistsError(
                "Output/checkpoint already exists; pass --resume or choose new paths"
            )
        db = sqlite3.connect(db_path)
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA cache_size=-8192")
        db.execute("""CREATE TABLE hosts(
            host TEXT PRIMARY KEY,out_count INTEGER DEFAULT 0,
            in_count INTEGER DEFAULT 0,compromised INTEGER DEFAULT 0
        ) WITHOUT ROWID""")
        db.execute("""CREATE TABLE pairs(
            src TEXT,dst TEXT,n INTEGER,last_seen INTEGER,auth_n INTEGER,
            success_n INTEGER,PRIMARY KEY(src,dst)
        ) WITHOUT ROWID""")
        db.execute("""CREATE TABLE run_state(
            id INTEGER PRIMARY KEY CHECK(id=1),signature TEXT NOT NULL,
            last_completed_window INTEGER NOT NULL,output_offset INTEGER NOT NULL,
            result_json TEXT NOT NULL,complete INTEGER NOT NULL DEFAULT 0
        )""")
        result = {key: 0 for key in RESULT_KEYS}
        db.execute("INSERT INTO run_state VALUES(1,?,-1,0,?,0)",
                   (signature, json.dumps(result)))
        db.commit()
        last_completed_window = -1
        output_stream = output.open("x", encoding="utf-8", newline="")

    db.create_function(
        "candidate_rank", 4,
        lambda host, current_seed, current_window, source:
            zlib.crc32(f"{current_seed}:{current_window}:{source}:{host}".encode("utf-8")),
        deterministic=True,
    )
    processed_since_message = 0
    try:
        with output_stream as out:
            for window, events in groupby(limited(), key=lambda event: event["timestamp"] // width):
                if window <= last_completed_window:
                    continue

                # Build candidate features from history through the previous window.
                now = window * width
                positives: dict[str, set[str]] = defaultdict(set)
                for source, target in labels_by_window.get(window, set()):
                    known = db.execute(
                        "SELECT compromised FROM hosts WHERE host=?", (source,)
                    ).fetchone()
                    if known and known[0]:
                        positives[source].add(target)

                for source, positive_targets in positives.items():
                    n_negative = negatives * max(1, len(positive_targets))
                    candidates = [(target, 1) for target in sorted(positive_targets)]
                    exclusions = {source, *positive_targets}
                    placeholders = ",".join("?" for _ in exclusions)
                    sampled = db.execute(
                        f"SELECT host FROM hosts WHERE host NOT IN ({placeholders}) "
                        "ORDER BY candidate_rank(host,?,?,?), host LIMIT ?",
                        (*sorted(exclusions), seed, window, source, n_negative),
                    ).fetchall()
                    candidates.extend((row[0], 0) for row in sampled)
                    source_row = db.execute(
                        "SELECT out_count FROM hosts WHERE host=?", (source,)
                    ).fetchone()
                    source_out = source_row[0] if source_row else 0
                    for target, label in candidates:
                        target_row = db.execute(
                            "SELECT in_count FROM hosts WHERE host=?", (target,)
                        ).fetchone()
                        target_in = target_row[0] if target_row else 0
                        pair = db.execute(
                            "SELECT n,last_seen,auth_n,success_n FROM pairs WHERE src=? AND dst=?",
                            (source, target),
                        ).fetchone()
                        count, last_seen, auth_count, success_count = pair or (0, None, 0, 0)
                        features = [
                            float(source_out), float(target_in), float(count),
                            float(now - last_seen) if last_seen is not None else -1.0,
                            float(success_count / auth_count) if auth_count else 0.0,
                        ]
                        out.write(json.dumps({
                            "window_start": now, "source_host": source,
                            "target_host": target, "label": label,
                            "features": features,
                        }, separators=(",", ":")) + "\n")
                        result["examples"] += 1
                        result["positives"] += label
                        result["negatives"] += 1 - label

                # Make candidate rows durable before committing matching history.
                out.flush()
                os.fsync(out.fileno())
                output_offset = out.tell()

                host_delta: dict[str, list[int]] = defaultdict(lambda: [0, 0])
                pair_delta: dict[tuple[str, str], list[int]] = {}
                newly_compromised: set[str] = set()

                def flush_deltas() -> None:
                    if not host_delta and not pair_delta and not newly_compromised:
                        return
                    db.executemany("INSERT OR IGNORE INTO hosts(host) VALUES(?)",
                                   ((host,) for host in host_delta))
                    db.executemany(
                        "UPDATE hosts SET out_count=out_count+?,in_count=in_count+? WHERE host=?",
                        ((out_count, in_count, host)
                         for host, (out_count, in_count) in host_delta.items()),
                    )
                    db.executemany("""INSERT INTO pairs VALUES(?,?,?,?,?,?)
                        ON CONFLICT(src,dst) DO UPDATE SET
                        n=n+excluded.n,last_seen=MAX(last_seen,excluded.last_seen),
                        auth_n=auth_n+excluded.auth_n,success_n=success_n+excluded.success_n""",
                        ((source, target, delta[0], delta[3], delta[1], delta[2])
                         for (source, target), delta in pair_delta.items()),
                    )
                    db.executemany("UPDATE hosts SET compromised=1 WHERE host=?",
                                   ((host,) for host in newly_compromised))
                    host_delta.clear()
                    pair_delta.clear()
                    newly_compromised.clear()

                for event in events:
                    result["events_seen"] += 1
                    source, target = event.get("source_host"), event.get("target_host")
                    if not source or not target or source == "?" or target == "?":
                        continue
                    host_delta[source][0] += 1
                    host_delta[target][1] += 1
                    delta = pair_delta.setdefault((source, target), [0, 0, 0, 0])
                    delta[0] += 1
                    delta[3] = event["timestamp"]
                    if event.get("event_type", "auth") == "auth":
                        delta[1] += 1
                        delta[2] += int(str(event.get("success", "")).lower() == "success")
                    if event.get("is_redteam"):
                        newly_compromised.update((source, target))
                    if len(pair_delta) >= 50000:
                        flush_deltas()
                flush_deltas()
                result["windows"] += 1

                db.execute(
                    "UPDATE run_state SET last_completed_window=?,output_offset=?,result_json=? WHERE id=1",
                    (window, output_offset, json.dumps(result)),
                )
                db.commit()
                last_completed_window = window
                processed_since_message += 1
                if processed_since_message >= 96:
                    print(json.dumps({"status": "checkpoint", **result,
                                      "last_window": window}), flush=True)
                    processed_since_message = 0

        db.execute("UPDATE run_state SET complete=1,result_json=? WHERE id=1",
                   (json.dumps(result),))
        db.commit()
    finally:
        db.close()
    return {**result, "features": FEATURES, "output": str(output),
            "state_dir": str(state_dir), "last_completed_window": last_completed_window}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build resumable LANL next-hop examples")
    parser.add_argument("--auth", type=Path, required=True)
    parser.add_argument("--redteam", type=Path, required=True)
    parser.add_argument("--flows", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/next_hop_examples_15d.jsonl"))
    parser.add_argument("--state-dir", type=Path,
                        help="Persistent checkpoint directory (defaults beside output)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last committed window checkpoint")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--window-minutes", type=int, default=30)
    parser.add_argument("--negatives", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = generate(args.auth, args.redteam, args.output, args.flows,
                      args.days, args.window_minutes, args.negatives, args.seed,
                      args.state_dir, args.resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
