"""Small inductive GraphSAGE + GRU scorer for temporal next-hop candidates."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


class TemporalAttackPathModel(nn.Module):
    """Encode graph snapshots, then score source/destination host pairs.

    Inputs are node features ``[T, N, F]`` and a list of T edge-index tensors
    ``[2, E_t]`` (edge counts may differ by snapshot). Pair candidates are
    integer indices ``[B, 2]``.
    ``forward`` returns logits suitable for ``BCEWithLogitsLoss``.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input = nn.Linear(input_dim, hidden_dim)
        self.sage = SAGEConv(hidden_dim, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, node_features: torch.Tensor, edge_indices: Sequence[torch.Tensor]) -> torch.Tensor:
        if node_features.ndim != 3:
            raise ValueError("node_features must have shape [time, nodes, features]")
        if len(edge_indices) != node_features.shape[0] or any(
            edge.ndim != 2 or edge.shape[0] != 2 for edge in edge_indices
        ):
            raise ValueError("edge_indices must contain one [2, edges] tensor per snapshot")
        snapshots = []
        for features, edge_index in zip(node_features, edge_indices):
            x = F.relu(self.input(features))
            x = F.relu(self.sage(x, edge_index.long()))
            snapshots.append(x)
        temporal_input = torch.stack(snapshots, dim=1)
        _, hidden = self.temporal(temporal_input)
        return hidden[-1]

    def forward(
        self,
        node_features: torch.Tensor,
        edge_indices: Sequence[torch.Tensor],
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.encode(node_features, edge_indices)
        candidates = candidates.long()
        source = embeddings[candidates[:, 0]]
        target = embeddings[candidates[:, 1]]
        pair = torch.cat((source, target, source * target, torch.abs(source - target)), dim=-1)
        return self.pair_scorer(pair).squeeze(-1)

    @torch.inference_mode()
    def top_k(
        self,
        node_features: torch.Tensor,
        edge_indices: Sequence[torch.Tensor],
        candidates: torch.Tensor,
        k: int = 10,
    ) -> list[dict[str, float | int]]:
        if k <= 0:
            raise ValueError("k must be positive")
        self.eval()
        scores = torch.sigmoid(self(node_features, edge_indices, candidates))
        top_scores, positions = torch.topk(scores, min(k, scores.numel()))
        chosen = candidates[positions]
        return [
            {"source_index": int(pair[0]), "target_index": int(pair[1]), "probability": float(score)}
            for pair, score in zip(chosen.cpu(), top_scores.cpu())
        ]
