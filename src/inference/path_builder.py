"""Assemble ranked multi-hop paths from next-hop prediction probabilities."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Iterable


def build_paths(
    predictions: Iterable[dict],
    start_host: str,
    max_hops: int = 4,
    beam_width: int = 20,
) -> list[dict]:
    """Build cycle-free paths using products of per-hop probabilities.

    Each prediction needs ``source_host``, ``target_host``, and ``probability``.
    A log-score heap avoids numeric underflow for longer paths.
    """
    if max_hops < 1 or beam_width < 1:
        raise ValueError("max_hops and beam_width must be positive")
    outgoing: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for edge in predictions:
        src, dst = edge.get("source_host"), edge.get("target_host")
        probability = float(edge.get("probability", 0.0))
        if not src or not dst or src == dst or not 0.0 < probability <= 1.0:
            continue
        outgoing[src].append((dst, probability))

    beam = [(0.0, (start_host,), 1.0)]
    results: list[dict] = []
    for _ in range(max_hops):
        candidates = []
        for neg_log_score, path, probability in beam:
            for target, hop_probability in outgoing.get(path[-1], []):
                if target in path:
                    continue
                combined = probability * hop_probability
                extended = path + (target,)
                candidates.append((neg_log_score - math.log(hop_probability), extended, combined))
        if not candidates:
            break
        # Higher probability means lower negative log probability.
        beam = heapq.nsmallest(beam_width, candidates, key=lambda item: item[0])
        results.extend({"path": list(path), "probability": score} for _, path, score in beam)
    best: dict[tuple[str, ...], float] = {}
    for item in results:
        key = tuple(item["path"])
        best[key] = max(best.get(key, 0.0), item["probability"])
    return [
        {"path": list(path), "probability": probability}
        for path, probability in sorted(best.items(), key=lambda pair: pair[1], reverse=True)
    ]
