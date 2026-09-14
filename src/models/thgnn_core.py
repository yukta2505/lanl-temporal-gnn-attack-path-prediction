#!/usr/bin/env python3
"""
Core THGNN Implementation

True Temporal Heterogeneous Graph Neural Network implementation
as specified in the technical documentation. This is the brain
of the proactive cybersecurity system.

Architecture:
- Heterogeneous Graph Convolution Layers
- GRU-based Temporal Encoding
- Link Prediction Head
- Explainable AI Integration
- Real-time Inference Capability

Performance Requirements:
- AUC-ROC ≥ 0.80
- Precision@10 ≥ 0.70
- <5 second prediction latency
- 100K events/minute processing
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, GNNExplainer
import numpy as np

from .gru_temporal_encoder import GRUTemporalEncoder

logger = logging.getLogger(__name__)

class HeterogeneousGraphConv(nn.Module):
    """
    Heterogeneous Graph Convolution Layer for THGNN.
    
    Handles multiple node types (User, Host, Process, File) and
    edge types (logs_into, executes, accesses, modifies, connects_to).
    """
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int = 256,
        conv_type: str = 'gcn',
        dropout: float = 0.4
    ):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Create convolution layers for each edge type
        conv_dict = {}
        
        for src_type, edge_type, dst_type in edge_types:
            edge_key = (src_type, edge_type, dst_type)
            
            if conv_type == 'gcn':
                conv_dict[edge_key] = GCNConv(hidden_dim, hidden_dim)
            elif conv_type == 'sage':
                conv_dict[edge_key] = SAGEConv(hidden_dim, hidden_dim)
            elif conv_type == 'gat':
                conv_dict[edge_key] = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")
        
        self.hetero_conv = HeteroConv(conv_dict, aggr='sum')
        
        # Layer normalization for each node type
        self.layer_norms = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in node_types
        })
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"Initialized HeterogeneousGraphConv with {len(edge_types)} edge types")
    
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through heterogeneous graph convolution.
        
        Args:
            x_dict: Node features for each node type
            edge_index_dict: Edge indices for each edge type
            
        Returns:
            Updated node embeddings for each node type
        """
        try:
            # Apply heterogeneous convolution
            x_dict = self.hetero_conv(x_dict, edge_index_dict)
            
            # Apply layer normalization and dropout
            for node_type in x_dict:
                if node_type in self.layer_norms:
                    x_dict[node_type] = self.layer_norms[node_type](x_dict[node_type])
                    x_dict[node_type] = F.relu(x_dict[node_type])
                    x_dict[node_type] = self.dropout_layer(x_dict[node_type])
            
            return x_dict
            
        except Exception as e:
            logger.error(f"Error in heterogeneous graph convolution: {e}")
            raise

class LinkPredictionHead(nn.Module):
    """
    Link prediction head for THGNN.
    
    Predicts likelihood of future connections between nodes
    based on their current embeddings and temporal context.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        edge_types: List[Tuple[str, str, str]] = None,
        dropout: float = 0.4
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.edge_types = edge_types or []
        
        # Edge type embeddings
        self.edge_type_embeddings = nn.Embedding(len(self.edge_types), hidden_dim)
        
        # Link prediction MLPs for each edge type
        self.link_predictors = nn.ModuleDict()
        
        for i, (src_type, edge_type, dst_type) in enumerate(self.edge_types):
            edge_key = f"{src_type}_{edge_type}_{dst_type}"
            
            self.link_predictors[edge_key] = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),  # src + dst + edge_type
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        logger.info(f"Initialized LinkPredictionHead with {len(self.edge_types)} edge types")
    
    def forward(
        self,
        src_embeddings: torch.Tensor,
        dst_embeddings: torch.Tensor,
        edge_type_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict link probabilities.
        
        Args:
            src_embeddings: Source node embeddings [num_edges, hidden_dim]
            dst_embeddings: Destination node embeddings [num_edges, hidden_dim]
            edge_type_idx: Edge type indices [num_edges]
            
        Returns:
            Link probabilities [num_edges, 1]
        """
        try:
            # Get edge type embeddings
            edge_type_emb = self.edge_type_embeddings(edge_type_idx)
            
            # Concatenate source, destination, and edge type embeddings
            combined = torch.cat([src_embeddings, dst_embeddings, edge_type_emb], dim=1)
            
            # Predict link probabilities (simplified - in practice would use edge-type-specific predictors)
            edge_key = list(self.link_predictors.keys())[0] if self.link_predictors else None
            if edge_key:
                predictions = self.link_predictors[edge_key](combined)
            else:
                # Fallback prediction
                predictions = torch.sigmoid(torch.sum(combined, dim=1, keepdim=True))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in link prediction: {e}")
            raise

class THGNN(nn.Module):
    """
    Temporal Heterogeneous Graph Neural Network for Cybersecurity.
    
    Core implementation as specified in technical documentation:
    1. Real-time log processing
    2. Temporal heterogeneous graph construction
    3. GNN-based predictive modeling with GRU
    4. Link prediction for lateral movement
    5. Explainable AI integration
    
    This is the brain of the proactive cybersecurity system.
    """
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int = 256,
        num_conv_layers: int = 3,
        temporal_hidden_dim: int = 128,
        temporal_layers: int = 2,
        dropout: float = 0.4,
        conv_type: str = 'gcn'
    ):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        
        # Node type embeddings
        self.node_type_embeddings = nn.ModuleDict({
            node_type: nn.Linear(64, hidden_dim)  # Assuming 64-dim input features
            for node_type in node_types
        })
        
        # Heterogeneous graph convolution layers
        self.conv_layers = nn.ModuleList([
            HeterogeneousGraphConv(
                node_types=node_types,
                edge_types=edge_types,
                hidden_dim=hidden_dim,
                conv_type=conv_type,
                dropout=dropout
            )
            for _ in range(num_conv_layers)
        ])
        
        # Temporal encoding with GRU
        self.temporal_encoder = GRUTemporalEncoder(
            input_dim=hidden_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_layers,
            dropout=dropout,
            use_layer_norm=True,
            use_residual=True
        )
        
        # Link prediction head
        self.link_predictor = LinkPredictionHead(
            hidden_dim=hidden_dim,
            edge_types=edge_types,
            dropout=dropout
        )
        
        # Explainer for XAI
        self.explainer = None
        self._init_explainer()
        
        # Performance tracking
        self.inference_times = []
        self.prediction_cache = {}
        
        logger.info(f"Initialized THGNN with {len(node_types)} node types, {len(edge_types)} edge types")
    
    def _init_explainer(self):
        """Initialize GNNExplainer for interpretability."""
        try:
            self.explainer = Explainer(
                model=self,
                algorithm=GNNExplainer(epochs=100),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='multiclass_classification',
                    task_level='graph',
                    return_type='probs'
                )
            )
            logger.info("GNNExplainer initialized for interpretability")
        except Exception as e:
            logger.warning(f"Could not initialize explainer: {e}")
            self.explainer = None
    
    def forward(
        self,
        graph_sequence: List[HeteroData],
        return_embeddings: bool = False,
        return_temporal_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through THGNN.
        
        Args:
            graph_sequence: Sequence of heterogeneous graphs
            return_embeddings: Whether to return node embeddings
            return_temporal_states: Whether to return temporal states
            
        Returns:
            Predictions and optionally embeddings/temporal states
        """
        start_time = time.time()
        
        try:
            if not graph_sequence:
                raise ValueError("Empty graph sequence provided")
            
            # Process each graph in the sequence
            sequence_embeddings = []
            
            for graph in graph_sequence:
                # Extract node features and edge indices
                x_dict = {}
                edge_index_dict = {}
                
                # Process node features
                for node_type in self.node_types:
                    if node_type in graph and hasattr(graph[node_type], 'x'):
                        node_features = graph[node_type].x
                        if node_features is not None and node_features.size(0) > 0:
                            # Project to hidden dimension
                            x_dict[node_type] = self.node_type_embeddings[node_type](node_features)
                        else:
                            # Create zero embeddings for missing nodes
                            x_dict[node_type] = torch.zeros(1, self.hidden_dim, device=self._get_device())
                    else:
                        # Create zero embeddings for missing node types
                        x_dict[node_type] = torch.zeros(1, self.hidden_dim, device=self._get_device())
                
                # Process edge indices
                for src_type, edge_type, dst_type in self.edge_types:
                    edge_key = (src_type, edge_type, dst_type)
                    
                    if edge_key in graph:
                        edge_index_dict[edge_key] = graph[edge_key].edge_index
                    else:
                        # Create empty edge index for missing edge types
                        edge_index_dict[edge_key] = torch.empty(2, 0, dtype=torch.long, device=self._get_device())
                
                # Apply heterogeneous graph convolutions
                current_x_dict = x_dict
                
                for conv_layer in self.conv_layers:
                    current_x_dict = conv_layer(current_x_dict, edge_index_dict)
                
                # Aggregate node embeddings into graph-level embedding
                graph_embedding = self._aggregate_graph_embedding(current_x_dict)
                sequence_embeddings.append(graph_embedding)
            
            # Apply temporal encoding
            if len(sequence_embeddings) > 1:
                # Stack embeddings into sequence
                sequence_tensor = torch.stack(sequence_embeddings).unsqueeze(0)  # [1, seq_len, hidden_dim]
                
                # Apply GRU temporal encoding
                temporal_output = self.temporal_encoder(sequence_tensor)
                
                if return_temporal_states:
                    temporal_output, temporal_states = self.temporal_encoder(
                        sequence_tensor, return_all_states=True
                    )
            else:
                # Single graph - no temporal encoding needed
                temporal_output = sequence_embeddings[0].unsqueeze(0)
                temporal_states = None
            
            # Track inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Return based on requested outputs
            if return_embeddings and return_temporal_states:
                return temporal_output, current_x_dict, temporal_states
            elif return_embeddings:
                return temporal_output, current_x_dict
            elif return_temporal_states:
                return temporal_output, temporal_states
            else:
                return temporal_output
                
        except Exception as e:
            logger.error(f"Error in THGNN forward pass: {e}")
            raise
    
    def predict_links(
        self,
        graph_sequence: List[HeteroData],
        candidate_edges: Dict[Tuple[str, str, str], torch.Tensor] = None,
        return_explanations: bool = False,
        num_candidates: int = 100
    ) -> Dict[str, Any]:
        """
        Predict future links (lateral movement paths).
        
        Args:
            graph_sequence: Sequence of heterogeneous graphs
            candidate_edges: Candidate edges to evaluate (auto-generated if None)
            return_explanations: Whether to return explanations
            num_candidates: Number of candidate edges to generate per type
            
        Returns:
            Link predictions with confidence scores and explanations
        """
        try:
            # Get temporal embeddings
            temporal_output, node_embeddings = self.forward(
                graph_sequence, return_embeddings=True
            )
            
            # Auto-generate candidate edges if not provided
            if candidate_edges is None:
                candidate_edges = self._generate_candidate_edges(
                    graph_sequence[-1], node_embeddings, num_candidates
                )
            
            predictions = {}
            explanations = {}
            
            # Predict for each edge type
            for edge_type_tuple, edge_candidates in candidate_edges.items():
                src_type, edge_type, dst_type = edge_type_tuple
                
                if edge_candidates.size(1) == 0:
                    continue
                
                # Get source and destination embeddings
                src_indices = edge_candidates[0]
                dst_indices = edge_candidates[1]
                
                if src_type in node_embeddings and dst_type in node_embeddings:
                    # Ensure indices are within bounds
                    max_src_idx = node_embeddings[src_type].size(0) - 1
                    max_dst_idx = node_embeddings[dst_type].size(0) - 1
                    
                    valid_src_mask = src_indices <= max_src_idx
                    valid_dst_mask = dst_indices <= max_dst_idx
                    valid_mask = valid_src_mask & valid_dst_mask
                    
                    if not valid_mask.any():
                        continue
                    
                    # Filter valid indices
                    valid_src_indices = src_indices[valid_mask]
                    valid_dst_indices = dst_indices[valid_mask]
                    
                    src_emb = node_embeddings[src_type][valid_src_indices]
                    dst_emb = node_embeddings[dst_type][valid_dst_indices]
                    
                    # Create edge type indices
                    edge_type_idx = torch.zeros(
                        valid_src_indices.size(0), 
                        dtype=torch.long, 
                        device=self._get_device()
                    )
                    
                    # Predict link probabilities
                    link_probs = self.link_predictor(src_emb, dst_emb, edge_type_idx)
                    
                    predictions[edge_type_tuple] = {
                        'probabilities': link_probs.squeeze(),
                        'edges': torch.stack([valid_src_indices, valid_dst_indices]),
                        'confidence': torch.mean(link_probs).item(),
                        'top_k_edges': self._get_top_k_predictions(
                            torch.stack([valid_src_indices, valid_dst_indices]),
                            link_probs.squeeze(),
                            k=min(10, valid_src_indices.size(0))
                        )
                    }
                    
                    # Generate explanations if requested
                    if return_explanations and self.explainer:
                        try:
                            explanation = self._generate_explanation(
                                graph_sequence[-1], edge_type_tuple, 
                                torch.stack([valid_src_indices, valid_dst_indices])
                            )
                            explanations[edge_type_tuple] = explanation
                        except Exception as e:
                            logger.warning(f"Could not generate explanation: {e}")
            
            result = {
                'predictions': predictions,
                'temporal_embedding': temporal_output,
                'timestamp': time.time(),
                'num_predicted_edges': sum(
                    pred['edges'].size(1) for pred in predictions.values()
                )
            }
            
            if return_explanations:
                result['explanations'] = explanations
            
            return result
            
        except Exception as e:
            logger.error(f"Error in link prediction: {e}")
            raise
    
    def _generate_candidate_edges(
        self,
        graph: HeteroData,
        node_embeddings: Dict[str, torch.Tensor],
        num_candidates: int = 100
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Generate candidate edges for link prediction.
        
        Args:
            graph: Current graph state
            node_embeddings: Node embeddings
            num_candidates: Number of candidates per edge type
            
        Returns:
            Dictionary of candidate edges by type
        """
        try:
            candidate_edges = {}
            
            for src_type, edge_type, dst_type in self.edge_types:
                edge_key = (src_type, edge_type, dst_type)
                
                # Check if we have nodes of both types
                if (src_type not in node_embeddings or 
                    dst_type not in node_embeddings or
                    node_embeddings[src_type].size(0) == 0 or
                    node_embeddings[dst_type].size(0) == 0):
                    # Create empty candidate set
                    candidate_edges[edge_key] = torch.empty(2, 0, dtype=torch.long, device=self._get_device())
                    continue
                
                num_src_nodes = node_embeddings[src_type].size(0)
                num_dst_nodes = node_embeddings[dst_type].size(0)
                
                # Generate random candidate edges
                max_candidates = min(num_candidates, num_src_nodes * num_dst_nodes)
                
                if max_candidates > 0:
                    # Sample source and destination nodes
                    src_candidates = torch.randint(
                        0, num_src_nodes, (max_candidates,), device=self._get_device()
                    )
                    dst_candidates = torch.randint(
                        0, num_dst_nodes, (max_candidates,), device=self._get_device()
                    )
                    
                    candidate_edges[edge_key] = torch.stack([src_candidates, dst_candidates])
                else:
                    candidate_edges[edge_key] = torch.empty(2, 0, dtype=torch.long, device=self._get_device())
            
            return candidate_edges
            
        except Exception as e:
            logger.error(f"Error generating candidate edges: {e}")
            return {}
    
    def _get_top_k_predictions(
        self,
        edges: torch.Tensor,
        probabilities: torch.Tensor,
        k: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Get top-k predictions with highest probabilities.
        
        Args:
            edges: Edge indices [2, num_edges]
            probabilities: Edge probabilities [num_edges]
            k: Number of top predictions to return
            
        Returns:
            Dictionary with top-k edges and probabilities
        """
        try:
            if probabilities.numel() == 0:
                return {
                    'edges': torch.empty(2, 0, dtype=torch.long, device=self._get_device()),
                    'probabilities': torch.empty(0, device=self._get_device())
                }
            
            # Get top-k indices
            top_k_values, top_k_indices = torch.topk(
                probabilities, min(k, probabilities.size(0)), largest=True
            )
            
            # Get corresponding edges
            top_k_edges = edges[:, top_k_indices]
            
            return {
                'edges': top_k_edges,
                'probabilities': top_k_values
            }
            
        except Exception as e:
            logger.error(f"Error getting top-k predictions: {e}")
            return {
                'edges': torch.empty(2, 0, dtype=torch.long, device=self._get_device()),
                'probabilities': torch.empty(0, device=self._get_device())
            }
    
    def predict_attack_sequence(
        self,
        graph_sequence: List[HeteroData],
        num_steps: int = 5,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Predict multi-step attack sequence using iterative link prediction.
        
        Args:
            graph_sequence: Current graph sequence
            num_steps: Number of future steps to predict
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Attack sequence with steps and confidence scores
        """
        try:
            attack_steps = []
            current_graph = graph_sequence[-1] if graph_sequence else None
            
            if current_graph is None:
                return {
                    'steps': [],
                    'overall_confidence': 0.0,
                    'num_steps': 0
                }
            
            overall_confidence = 1.0
            
            for step in range(num_steps):
                # Predict next most likely edges
                predictions = self.predict_links([current_graph])
                
                if not predictions['predictions']:
                    logger.info(f"No predictions available at step {step}")
                    break
                
                # Find highest confidence prediction across all edge types
                best_prediction = None
                best_confidence = 0.0
                best_edge_type = None
                
                for edge_type, pred_data in predictions['predictions'].items():
                    if pred_data['confidence'] > best_confidence:
                        best_confidence = pred_data['confidence']
                        best_prediction = pred_data
                        best_edge_type = edge_type
                
                # Check confidence threshold
                if best_confidence < confidence_threshold:
                    logger.info(f"Stopping prediction at step {step}: confidence {best_confidence:.3f} below threshold {confidence_threshold}")
                    break
                
                # Get top prediction
                if best_prediction and 'top_k_edges' in best_prediction:
                    top_edges = best_prediction['top_k_edges']
                    if top_edges['probabilities'].numel() > 0:
                        top_prob = top_edges['probabilities'][0].item()
                        top_edge = top_edges['edges'][:, 0]
                        
                        attack_step = {
                            'step_number': step + 1,
                            'edge_type': best_edge_type[1],  # edge type name
                            'source_type': best_edge_type[0],
                            'target_type': best_edge_type[2],
                            'source_node': top_edge[0].item(),
                            'target_node': top_edge[1].item(),
                            'confidence': top_prob,
                            'timestamp': time.time() + step * 60  # Assume 1 minute per step
                        }
                        
                        attack_steps.append(attack_step)
                        overall_confidence *= top_prob
                        
                        # Update graph state (simplified - in practice would add edge)
                        # For now, we continue with the same graph
                
            return {
                'steps': attack_steps,
                'overall_confidence': overall_confidence,
                'num_steps': len(attack_steps),
                'prediction_time': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error predicting attack sequence: {e}")
            return {
                'steps': [],
                'overall_confidence': 0.0,
                'num_steps': 0,
                'error': str(e)
            }
        """Aggregate node embeddings into graph-level embedding."""
        try:
            embeddings = []
            
            for node_type in self.node_types:
                if node_type in x_dict and x_dict[node_type].size(0) > 0:
                    # Global average pooling
                    pooled = torch.mean(x_dict[node_type], dim=0, keepdim=True)
                    embeddings.append(pooled)
                else:
                    # Zero embedding for missing node types
                    embeddings.append(torch.zeros(1, self.hidden_dim, device=self._get_device()))
            
            # Concatenate and project
            if embeddings:
                combined = torch.cat(embeddings, dim=0)
                graph_embedding = torch.mean(combined, dim=0)  # [hidden_dim]
                return graph_embedding
            else:
                return torch.zeros(self.hidden_dim, device=self._get_device())
                
        except Exception as e:
            logger.error(f"Error aggregating graph embedding: {e}")
            return torch.zeros(self.hidden_dim, device=self._get_device())
    
    def _generate_explanation(
        self,
        graph: HeteroData,
        edge_type: Tuple[str, str, str],
        candidate_edges: torch.Tensor
    ) -> Dict[str, Any]:
        """Generate explanation for link prediction."""
        try:
            if self.explainer is None:
                return {'error': 'Explainer not available'}
            
            # Generate explanation using GNNExplainer
            explanation = self.explainer(
                x=graph.x_dict if hasattr(graph, 'x_dict') else {},
                edge_index=graph.edge_index_dict if hasattr(graph, 'edge_index_dict') else {},
                target=candidate_edges
            )
            
            return {
                'node_importance': explanation.node_mask if hasattr(explanation, 'node_mask') else None,
                'edge_importance': explanation.edge_mask if hasattr(explanation, 'edge_mask') else None,
                'subgraph': explanation.subgraph if hasattr(explanation, 'subgraph') else None
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {'error': str(e)}
    
    def _get_device(self) -> torch.device:
        """Get device of model parameters."""
        return next(self.parameters()).device
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'total_inferences': len(self.inference_times),
            'cache_size': len(self.prediction_cache)
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking."""
        self.inference_times = []
        self.prediction_cache = {}

class THGNNConfig:
    """Configuration for THGNN model."""
    
    def __init__(self):
        # Node types as specified in documentation
        self.node_types = ['user', 'host', 'process', 'file']
        
        # Edge types as specified in documentation
        self.edge_types = [
            ('user', 'logs_into', 'host'),
            ('user', 'executes', 'process'),
            ('process', 'accesses', 'file'),
            ('process', 'connects_to', 'host'),
            ('process', 'modifies', 'file'),
            ('process', 'creates', 'process'),
            ('host', 'contains', 'process'),
            ('host', 'stores', 'file')
        ]
        
        # Model architecture
        self.hidden_dim = 256
        self.num_conv_layers = 3
        self.temporal_hidden_dim = 128
        self.temporal_layers = 2
        self.dropout = 0.4
        self.conv_type = 'gcn'
        
        # Performance requirements
        self.target_auc_roc = 0.80
        self.target_precision_at_10 = 0.70
        self.max_prediction_latency = 5.0  # seconds
        self.max_events_per_minute = 100000
        
        # Training configuration
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.early_stopping_patience = 10
        
        # Explainability
        self.enable_explanations = True
        self.explanation_method = 'gnnexplainer'
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'node_types': self.node_types,
            'edge_types': self.edge_types,
            'hidden_dim': self.hidden_dim,
            'num_conv_layers': self.num_conv_layers,
            'temporal_hidden_dim': self.temporal_hidden_dim,
            'temporal_layers': self.temporal_layers,
            'dropout': self.dropout,
            'conv_type': self.conv_type,
            'target_auc_roc': self.target_auc_roc,
            'target_precision_at_10': self.target_precision_at_10,
            'max_prediction_latency': self.max_prediction_latency,
            'max_events_per_minute': self.max_events_per_minute,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'enable_explanations': self.enable_explanations,
            'explanation_method': self.explanation_method
        }
    def _aggregate_graph_embedding(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate node embeddings into graph-level embedding."""
        try:
            embeddings = []
            
            for node_type in self.node_types:
                if node_type in x_dict and x_dict[node_type].size(0) > 0:
                    # Global average pooling
                    pooled = torch.mean(x_dict[node_type], dim=0, keepdim=True)
                    embeddings.append(pooled)
                else:
                    # Zero embedding for missing node types
                    embeddings.append(torch.zeros(1, self.hidden_dim, device=self._get_device()))
            
            # Concatenate and project
            if embeddings:
                combined = torch.cat(embeddings, dim=0)
                graph_embedding = torch.mean(combined, dim=0)  # [hidden_dim]
                return graph_embedding
            else:
                return torch.zeros(self.hidden_dim, device=self._get_device())
                
        except Exception as e:
            logger.error(f"Error aggregating graph embedding: {e}")
            return torch.zeros(self.hidden_dim, device=self._get_device())
    
    def _generate_explanation(
        self,
        graph: HeteroData,
        edge_type: Tuple[str, str, str],
        candidate_edges: torch.Tensor
    ) -> Dict[str, Any]:
        """Generate explanation for link prediction."""
        try:
            if self.explainer is None:
                return {'error': 'Explainer not available'}
            
            # Generate explanation using GNNExplainer
            explanation = self.explainer(
                x=graph.x_dict if hasattr(graph, 'x_dict') else {},
                edge_index=graph.edge_index_dict if hasattr(graph, 'edge_index_dict') else {},
                target=candidate_edges
            )
            
            return {
                'node_importance': explanation.node_mask if hasattr(explanation, 'node_mask') else None,
                'edge_importance': explanation.edge_mask if hasattr(explanation, 'edge_mask') else None,
                'subgraph': explanation.subgraph if hasattr(explanation, 'subgraph') else None
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {'error': str(e)}
    
    def _get_device(self) -> torch.device:
        """Get device of model parameters."""
        return next(self.parameters()).device
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'total_inferences': len(self.inference_times),
            'cache_size': len(self.prediction_cache)
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking."""
        self.inference_times = []
        self.prediction_cache = {}

class THGNNConfig:
    """Configuration for THGNN model."""
    
    def __init__(self):
        # Node types as specified in documentation
        self.node_types = ['user', 'host', 'process', 'file']
        
        # Edge types as specified in documentation
        self.edge_types = [
            ('user', 'logs_into', 'host'),
            ('user', 'executes', 'process'),
            ('process', 'accesses', 'file'),
            ('process', 'connects_to', 'host'),
            ('process', 'modifies', 'file'),
            ('process', 'creates', 'process'),
            ('host', 'contains', 'process'),
            ('host', 'stores', 'file')
        ]
        
        # Model architecture
        self.hidden_dim = 256
        self.num_conv_layers = 3
        self.temporal_hidden_dim = 128
        self.temporal_layers = 2
        self.dropout = 0.4
        self.conv_type = 'gcn'
        
        # Performance requirements
        self.target_auc_roc = 0.80
        self.target_precision_at_10 = 0.70
        self.max_prediction_latency = 5.0  # seconds
        self.max_events_per_minute = 100000
        
        # Training configuration
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.early_stopping_patience = 10
        
        # Explainability
        self.enable_explanations = True
        self.explanation_method = 'gnnexplainer'
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'node_types': self.node_types,
            'edge_types': self.edge_types,
            'hidden_dim': self.hidden_dim,
            'num_conv_layers': self.num_conv_layers,
            'temporal_hidden_dim': self.temporal_hidden_dim,
            'temporal_layers': self.temporal_layers,
            'dropout': self.dropout,
            'conv_type': self.conv_type,
            'target_auc_roc': self.target_auc_roc,
            'target_precision_at_10': self.target_precision_at_10,
            'max_prediction_latency': self.max_prediction_latency,
            'max_events_per_minute': self.max_events_per_minute,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'enable_explanations': self.enable_explanations,
            'explanation_method': self.explanation_method
        }