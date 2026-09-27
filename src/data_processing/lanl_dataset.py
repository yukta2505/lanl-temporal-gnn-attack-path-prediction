"""Prepare chronological LANL authentication windows and red-team labels.

The source files are comma-delimited, headerless. Auth fields are time,
source user, destination user, source computer, destination computer,
authentication type, logon type, orientation, and success/failure. Red-team
rows contain time, user, source computer, and destination computer.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
from pathlib import Path
from typing import Iterator, TextIO


def _open_text(path: Path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix.lower() == ".gz" else path.open("r", encoding="utf-8")


def _rows(path: Path) -> Iterator[list[str]]:
    with _open_text(path) as stream:
        for fields in csv.reader(stream):
            if fields and fields[0].lower() not in {"time", "timestamp"}:
                yield fields


def load_redteam(path: Path) -> set[tuple[int, str, str, str]]:
    """Load exact ground-truth keys: (second, user, source host, target host)."""
    labels: set[tuple[int, str, str, str]] = set()
    for row in _rows(path):
        if len(row) < 4:
            continue
        try:
            labels.add((int(row[0]), row[1], row[2], row[3]))
        except ValueError:
            continue
    return labels


def iter_auth_events(path: Path, labels: set[tuple[int, str, str, str]]) -> Iterator[dict]:
    """Yield normalized authentication events, skipping malformed rows."""
    for row in _rows(path):
        if len(row) < 9:
            continue
        try:
            timestamp = int(row[0])
        except ValueError:
            continue
        user, source_host, target_host = row[1], row[3], row[4]
        is_attack = (timestamp, user, source_host, target_host) in labels
        yield {
            "event_type": "auth",
            "timestamp": timestamp,
            # LANL uses relative seconds from the start of the recording, not
            # Unix epoch time; keep the raw offset and do not date-convert it.
            "source_host": source_host,
            "target_host": target_host,
            "source_user": row[1],
            "target_user": row[2],
            "auth_type": row[5],
            "logon_type": row[6],
            "orientation": row[7],
            "success": row[8],
            "is_redteam": is_attack,
        }


def iter_flow_events(path: Path) -> Iterator[dict]:
    """Yield normalized LANL flow edges (time,duration,src,src_port,dst,dst_port,protocol,packets,bytes)."""
    for row in _rows(path):
        if len(row) < 9:
            continue
        try:
            timestamp = int(row[0])
            duration = float(row[1]) if row[1] != "?" else 0.0
            packet_count = int(row[7]) if row[7] != "?" else 0
            byte_count = int(row[8]) if row[8] != "?" else 0
        except ValueError:
            continue
        yield {
            "event_type": "flow",
            "timestamp": timestamp,
            "source_host": row[2],
            "target_host": row[4],
            "source_port": row[3],
            "target_port": row[5],
            "protocol": row[6],
            "duration_seconds": duration,
            "packet_count": packet_count,
            "byte_count": byte_count,
            "is_redteam": False,
        }


def build_windows(
    auth_path: Path,
    redteam_path: Path,
    output_path: Path,
    window_minutes: int = 30,
    max_days: int | None = 15,
    flow_path: Path | None = None,
) -> dict[str, int]:
    """Write chronological, non-overlapping windows as JSON Lines.

    LANL distributes the event files in time order. The optional day limit is
    measured from the first valid authentication event.
    """
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    if max_days is not None and max_days <= 0:
        raise ValueError("max_days must be positive or None")
    labels = load_redteam(redteam_path)
    width = window_minutes * 60
    first_timestamp: int | None = None
    active_window: int | None = None
    events: list[dict] = []
    counts = {"events": 0, "auth_events": 0, "flow_events": 0, "windows": 0, "redteam_events": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_window(stream: TextIO, window_id: int, rows: list[dict]) -> None:
        stream.write(json.dumps({
            "window_start": window_id * width,
            "window_minutes": window_minutes,
            "events": rows,
        }, separators=(",", ":")) + "\n")
        counts["windows"] += 1

    with output_path.open("w", encoding="utf-8") as output:
        streams = [iter_auth_events(auth_path, labels)]
        if flow_path is not None:
            streams.append(iter_flow_events(flow_path))
        for event in heapq.merge(*streams, key=lambda item: item["timestamp"]):
            timestamp = event["timestamp"]
            if first_timestamp is None:
                first_timestamp = timestamp
            if max_days is not None and timestamp - first_timestamp >= max_days * 86400:
                break
            window_id = timestamp // width
            if active_window is None:
                active_window = window_id
            if window_id != active_window:
                write_window(output, active_window, events)
                events = []
                active_window = window_id
            events.append(event)
            counts["events"] += 1
            counts[f"{event['event_type']}_events"] += 1
            counts["redteam_events"] += int(event["is_redteam"])
        if active_window is not None and events:
            write_window(output, active_window, events)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build time-windowed LANL login data")
    parser.add_argument("--auth", type=Path, required=True, help="LANL auth.txt or auth.txt.gz")
    parser.add_argument("--redteam", type=Path, required=True, help="LANL redteam.txt or redteam.txt.gz")
    parser.add_argument("--flows", type=Path, help="Optional LANL flows.txt or flows.txt.gz")
    parser.add_argument("--output", type=Path, default=Path("data/processed/lanl_windows.jsonl"))
    parser.add_argument("--window-minutes", type=int, default=30)
    parser.add_argument("--max-days", type=int, default=15, help="Positive days from the first event")
    args = parser.parse_args()
    result = build_windows(args.auth, args.redteam, args.output, args.window_minutes,
                           args.max_days, args.flows)
    print(json.dumps({**result, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
