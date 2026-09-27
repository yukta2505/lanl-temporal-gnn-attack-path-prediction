"""Faithful edge-occlusion explanations for temporal next-hop scores."""

from __future__ import annotations

from collections.abc import Sequence

import torch


@torch.inference_mode()
def explain_by_edge_occlusion(
    model,
    node_features: torch.Tensor,
    edge_indices: Sequence[torch.Tensor],
    candidate: torch.Tensor,
    edge_metadata: Sequence[Sequence[dict]] | None = None,
    max_edges: int = 200,
    top_k: int = 10,
) -> dict:
    """Measure how removing each observed edge changes one candidate score.

    This perturbation explanation is intentionally bounded because it performs
    one additional inference per tested edge. Positive impact means the edge
    raised the predicted probability; negative impact means it lowered it.
    """
    if max_edges <= 0 or top_k <= 0:
        raise ValueError("max_edges and top_k must be positive")
    if candidate.ndim == 1:
        candidate = candidate.unsqueeze(0)
    model.eval()
    baseline = float(torch.sigmoid(model(node_features, edge_indices, candidate))[0])
    ranked: list[dict] = []
    tested = 0
    for step, edge_index in enumerate(edge_indices):
        for edge_position in range(edge_index.shape[1]):
            if tested >= max_edges:
                break
            masked = list(edge_indices)
            keep = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
            keep[edge_position] = False
            masked[step] = edge_index[:, keep]
            probability = float(torch.sigmoid(model(node_features, masked, candidate))[0])
            item = {
                "snapshot": step,
                "edge_position": edge_position,
                "probability_without_edge": probability,
                "impact": baseline - probability,
            }
            if edge_metadata is not None and step < len(edge_metadata) and edge_position < len(edge_metadata[step]):
                item["event"] = edge_metadata[step][edge_position]
            ranked.append(item)
            tested += 1
        if tested >= max_edges:
            break
    ranked.sort(key=lambda row: abs(row["impact"]), reverse=True)
    return {"baseline_probability": baseline, "tested_edges": tested, "important_edges": ranked[:top_k]}
