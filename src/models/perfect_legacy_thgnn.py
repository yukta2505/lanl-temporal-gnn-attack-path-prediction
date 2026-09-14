"""
PERFECT LEGACY THGNN - EXACT MATCH FOR 0.8545 AUC CHECKPOINT
This is the EXACT architecture that created the checkpoint.
DO NOT MODIFY - PERFECT MATCH!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, TransformerConv
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class PerfectLegacyTHGNN(nn.Module):
    """PERFECT MATCH for 0.8545 AUC checkpoint - 100% weight loading"""
    
    def __init__(self, 
                 hidden_dim: int = 256,
                 num_conv_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.2,
                 feature_dim: int = 16,
                 node_types: List[str] = None,
                 edge_types: List[Tuple[str, str, str]] = None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.feature_dim = feature_dim
        
        if node_types is None:
            node_types = ['user', 'host', 'process', 'file', 'registry']
        if edge_types is None:
            edge_types = [
                ('user', 'logs_into', 'host'),
                ('user', 'executes', 'process'),
                ('process', 'creates', 'process'),
                ('process', 'connects_to', 'host'),
                ('process', 'accesses', 'file'),
                ('process', 'modifies', 'registry')
            ]
        
        self.node_types = node_types
        self.edge_types = edge_types
        
        # Input projections - EXACT match
        # Structure: Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> LayerNorm -> ReLU -> Dropout
        # Indices:   0         1            2       3          4         5            6       7
        self.input_projections = nn.ModuleDict()
        for node_type in node_types:
            self.input_projections[node_type] = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),  # 0
                nn.BatchNorm1d(hidden_dim),           # 1
                nn.ReLU(),                            # 2
                nn.Dropout(0.1),                      # 3
                nn.Linear(hidden_dim, hidden_dim),    # 4
                nn.LayerNorm(hidden_dim),             # 5
                nn.ReLU(),                            # 6
                nn.Dropout(0.05)                      # 7
            )
        
        # Heterogeneous convolutions - EXACT match
        # Layer 0: TransformerConv
        # Layer 1: SAGEConv (NOT GATConv!)
        # Layer 2: SAGEConv
        self.convs = nn.ModuleList()
        for layer_idx in range(num_conv_layers):
            conv_dict = {}
            for src_type, rel_type, dst_type in edge_types:
                if layer_idx == 0:
                    conv_dict[(src_type, rel_type, dst_type)] = TransformerConv(
                        hidden_dim, hidden_dim // num_heads,
                        heads=num_heads, dropout=dropout,
                        concat=True, beta=True, edge_dim=None
                    )
                else:
                    # Layers 1 and 2 both use SAGEConv
                    conv_dict[(src_type, rel_type, dst_type)] = SAGEConv(
                        hidden_dim, hidden_dim, aggr='mean'
                    )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # Temporal attention - EXACT match (only this, NO GRU!)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Graph pooling - EXACT match
        # Structure: Linear -> ReLU -> Linear
        # Indices:   0         2       (no 1!)
        self.graph_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 0
            nn.ReLU(),                                # (no index in state dict)
            nn.Linear(hidden_dim // 2, hidden_dim)    # 2
        )
        
        # Link predictor - EXACT match
        # Structure: Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Linear
        # Indices:   0         1            (no)    (no)       4         (no)    (no)       7         (no)    9
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),    # 0
            nn.BatchNorm1d(hidden_dim),               # 1
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),   # 4
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # 7
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)             # 9
        )
        
        # NO layer_norms or batch_norms for nodes in checkpoint!
    
    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        """Forward pass - EXACT match"""
        # Input projection
        for node_type in self.node_types:
            if node_type in x_dict:
                x_dict[node_type] = self.input_projections[node_type](x_dict[node_type])
        
        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, edge_index_dict)
            
            # Apply activation and dropout (NO normalization!)
            for node_type in self.node_types:
                if node_type in x_dict_new:
                    x = x_dict_new[node_type]
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    
                    # Residual connection
                    if node_type in x_dict and x_dict[node_type].size() == x.size():
                        x = x + x_dict[node_type]
                    
                    x_dict_new[node_type] = x
            
            x_dict = x_dict_new
        
        return x_dict
    
    def predict_link(self, src_emb, dst_emb, edge_type_emb=None):
        """Predict link probability"""
        if edge_type_emb is not None:
            combined = torch.cat([src_emb, dst_emb, edge_type_emb, src_emb * dst_emb], dim=-1)
        else:
            combined = torch.cat([src_emb, dst_emb, src_emb * dst_emb, torch.abs(src_emb - dst_emb)], dim=-1)
        
        return torch.sigmoid(self.link_predictor(combined))
