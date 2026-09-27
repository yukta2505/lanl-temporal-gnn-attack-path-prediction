"""Static edge-aware GraphSAGE baseline for host-pair scoring."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


class StaticEGraphSAGE(nn.Module):
    """Encode one static graph and score candidate edges with edge features."""

    def __init__(self, node_feature_dim: int, edge_feature_dim: int,
                 hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        if min(node_feature_dim, edge_feature_dim, hidden_dim) <= 0:
            raise ValueError("feature dimensions and hidden_dim must be positive")
        self.input = nn.Linear(node_feature_dim, hidden_dim)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4 + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                candidates: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        if node_features.ndim != 2 or candidates.ndim != 2 or candidates.shape[1] != 2:
            raise ValueError("expected node_features [N,F] and candidates [B,2]")
        if edge_features.ndim != 2 or edge_features.shape[0] != candidates.shape[0]:
            raise ValueError("edge_features must have one row per candidate edge")
        x = F.relu(self.input(node_features))
        x = F.relu(self.sage1(x, edge_index.long()))
        x = F.relu(self.sage2(x, edge_index.long()))
        source, target = x[candidates[:, 0].long()], x[candidates[:, 1].long()]
        pair = torch.cat((source, target, source * target, torch.abs(source - target), edge_features), dim=-1)
        return self.edge_scorer(pair).squeeze(-1)
