#!/usr/bin/env python3
"""
GRU Temporal Encoder for THGNN

Production-grade GRU-based temporal encoding specifically optimized for 
cybersecurity threat detection. This implementation replaces LSTM with GRU
to match PDF specifications while maintaining performance and stability.

Key Features:
- GRU-based temporal encoding for attack sequence modeling
- Memory-efficient processing for long sequences
- GPU acceleration with gradient checkpointing
- Comprehensive error handling and logging
- Real-time inference capabilities (<5 sec latency)
- Backward compatibility with existing THGNN architecture

Performance Benefits of GRU over LSTM:
- Faster training and inference (fewer parameters)
- Better performance on shorter sequences (typical in cybersecurity)
- Reduced memory usage (important for production deployment)
- Simpler architecture with comparable accuracy
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, List, Tuple, Optional, Union, Any
import math
import warnings
import time

# Suppress PyTorch warnings for production
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class GRUTemporalEncoder(nn.Module):
    """
    Production-grade GRU temporal encoder for cybersecurity attack sequences.
    
    Optimized for capturing temporal dependencies in attack patterns using
    GRU with attention mechanisms and memory optimization. Specifically
    designed for cybersecurity use cases where sequences are typically
    shorter but require fast inference.
    
    Architecture:
    - Bidirectional GRU for sequence modeling (optional)
    - Multi-head attention for temporal importance
    - Layer normalization for training stability
    - Gradient checkpointing for memory efficiency
    - Residual connections for deep networks
    - Dropout for regularization
    
    Performance Optimizations:
    - GRU uses fewer parameters than LSTM (faster training/inference)
    - Memory-efficient processing for production deployment
    - GPU acceleration with proper tensor operations
    - Batch processing support for multiple sequences
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = False,  # Disabled for stability and speed
        dropout: float = 0.4,
        use_attention: bool = False,  # Disabled for stability
        attention_heads: int = 4,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        gradient_checkpointing: bool = False,  # Disabled for stability
        temperature: float = 1.0
    ):
        """
        Initialize GRU temporal encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension (default: 256)
            num_layers: Number of GRU layers (default: 2)
            bidirectional: Use bidirectional GRU (default: False for stability)
            dropout: Dropout probability (default: 0.4)
            use_attention: Enable temporal attention (default: False for stability)
            attention_heads: Number of attention heads (default: 4)
            use_layer_norm: Enable layer normalization (default: True)
            use_residual: Enable residual connections (default: True)
            gradient_checkpointing: Enable gradient checkpointing (default: False)
            temperature: Attention temperature for sharpening (default: 1.0)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.gradient_checkpointing = gradient_checkpointing
        self.temperature = temperature
        
        # Calculate effective hidden dimension
        self.effective_hidden_dim = hidden_dim * (2 if bidirectional else 1)
        
        logger.info(f"Initializing GRUTemporalEncoder: "
                   f"hidden_dim={hidden_dim}, layers={num_layers}, "
                   f"bidirectional={bidirectional}, dropout={dropout}")
        
        # Input projection (if needed)
        if input_dim != hidden_dim:
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.input_projection = None
        
        # GRU layer - core temporal encoding
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Layer normalization for training stability
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.effective_hidden_dim)
        
        # Temporal attention mechanism (optional)
        if use_attention:
            self.attention = GRUTemporalAttention(
                hidden_dim=self.effective_hidden_dim,
                num_heads=attention_heads,
                dropout=dropout,
                temperature=temperature
            )
        
        # Output projection - always project back to input dimension
        self.output_projection = nn.Sequential(
            nn.Linear(self.effective_hidden_dim, input_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize parameters with proper initialization
        self._init_parameters()
        
        # Performance tracking
        self.inference_times = []
    
    def _init_parameters(self) -> None:
        """
        Initialize GRU parameters with Xavier/Orthogonal initialization.
        
        Proper initialization is crucial for GRU training stability and
        convergence speed in cybersecurity applications.
        """
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights: Orthogonal
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Biases: Zero initialization
                nn.init.zeros_(param.data)
        
        # Initialize projection layers
        for module in [self.input_projection, self.output_projection]:
            if module is not None:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        x_sequence: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
        return_all_states: bool = False,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through GRU temporal encoder.
        
        Optimized for cybersecurity attack sequence processing with
        comprehensive error handling and performance monitoring.
        
        Args:
            x_sequence: Input sequence [batch_size, seq_len, input_dim]
            hidden_state: Initial hidden state (optional)
            sequence_lengths: Actual sequence lengths for packing (optional)
            return_all_states: Return all hidden states or just final
            return_attention: Return attention weights for explainability
            
        Returns:
            Encoded sequence or (output, hidden_state, attention_weights) tuple
        """
        try:
            start_time = time.time()
            
            # Validate inputs
            if x_sequence.dim() != 3:
                raise ValueError(f"Expected 3D input tensor, got {x_sequence.dim()}D")
            
            batch_size, seq_len, input_dim = x_sequence.shape
            
            if input_dim != self.input_dim:
                raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {input_dim}")
            
            # Input projection to normalize dimensions
            if self.input_projection is not None:
                x_sequence = self.input_projection(x_sequence)
            
            # Store input for residual connection
            x_input = x_sequence if self.use_residual else None
            
            # Pack sequences if lengths provided (for variable-length sequences)
            packed_input = None
            if sequence_lengths is not None:
                # Sort sequences by length for packing
                sorted_lengths, sorted_indices = sequence_lengths.sort(descending=True)
                sorted_x = x_sequence[sorted_indices]
                
                packed_input = pack_padded_sequence(
                    sorted_x, sorted_lengths.cpu(),
                    batch_first=True, enforce_sorted=True
                )
                gru_input = packed_input
            else:
                gru_input = x_sequence
            
            # GRU forward pass with optional gradient checkpointing
            if self.gradient_checkpointing and self.training:
                gru_output, hidden_state = torch.utils.checkpoint.checkpoint(
                    self._gru_forward, gru_input, hidden_state
                )
            else:
                gru_output, hidden_state = self._gru_forward(gru_input, hidden_state)
            
            # Unpack sequences if packed
            if packed_input is not None:
                gru_output, _ = pad_packed_sequence(gru_output, batch_first=True)
                
                # Restore original order
                _, unsorted_indices = sorted_indices.sort()
                gru_output = gru_output[unsorted_indices]
            
            # Apply layer normalization for training stability
            if self.use_layer_norm:
                gru_output = self.layer_norm(gru_output)
            
            # Apply temporal attention (optional)
            attention_weights = None
            if self.use_attention:
                if return_attention:
                    gru_output, attention_weights = self.attention(
                        gru_output, sequence_lengths, return_weights=True
                    )
                else:
                    gru_output = self.attention(gru_output, sequence_lengths)
            
            # Apply dropout for regularization
            gru_output = self.dropout_layer(gru_output)
            
            # Output projection to match expected dimensions
            output = self.output_projection(gru_output)
            
            # Apply residual connection if enabled
            if self.use_residual and x_input is not None:
                # Ensure dimensions match for residual connection
                if output.shape == x_input.shape:
                    output = output + x_input
                else:
                    logger.warning("Skipping residual connection due to dimension mismatch")
            
            # Track inference time for performance monitoring
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Log performance metrics periodically
            if len(self.inference_times) % 100 == 0:
                avg_time = sum(self.inference_times[-100:]) / 100
                logger.info(f"GRU Temporal Encoder - Average inference time (last 100): {avg_time:.4f}s")
            
            # Return based on requested format
            if return_all_states:
                if return_attention and attention_weights is not None:
                    return output, hidden_state, attention_weights
                else:
                    return output, hidden_state
            else:
                # Return final state for each sequence
                if sequence_lengths is not None:
                    final_outputs = []
                    for i, length in enumerate(sequence_lengths):
                        final_outputs.append(output[i, length-1])
                    final_output = torch.stack(final_outputs)
                else:
                    final_output = output[:, -1]  # Last time step
                
                if return_attention and attention_weights is not None:
                    return final_output, attention_weights
                else:
                    return final_output
                    
        except Exception as e:
            logger.error(f"Error in GRUTemporalEncoder forward pass: {e}")
            logger.error(f"Input shape: {x_sequence.shape if 'x_sequence' in locals() else 'Unknown'}")
            logger.error(f"Sequence lengths: {sequence_lengths}")
            raise
    
    def _gru_forward(
        self,
        x_sequence: torch.Tensor,
        hidden_state: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GRU forward pass (separated for gradient checkpointing).
        
        Args:
            x_sequence: Input sequence or packed sequence
            hidden_state: Initial hidden state
            
        Returns:
            GRU output and final hidden state
        """
        return self.gru(x_sequence, hidden_state)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize hidden state for GRU.
        
        Args:
            batch_size: Batch size
            device: Computation device
            
        Returns:
            Initialized hidden state tensor
        """
        num_directions = 2 if self.bidirectional else 1
        
        hidden_state = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )
        
        return hidden_state
    
    def get_attention_weights(
        self,
        x_sequence: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Get attention weights for interpretability and explainability.
        
        Args:
            x_sequence: Input sequence
            sequence_lengths: Actual sequence lengths
            
        Returns:
            Attention weights tensor or None if attention disabled
        """
        if not self.use_attention:
            logger.warning("Attention is disabled. Cannot return attention weights.")
            return None
        
        # Forward pass to get attention weights
        with torch.no_grad():
            _, attention_weights = self.forward(
                x_sequence, sequence_lengths=sequence_lengths, return_attention=True
            )
            return attention_weights
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for monitoring.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {
                'avg_inference_time': 0.0,
                'min_inference_time': 0.0,
                'max_inference_time': 0.0,
                'total_inferences': 0
            }
        
        return {
            'avg_inference_time': sum(self.inference_times) / len(self.inference_times),
            'min_inference_time': min(self.inference_times),
            'max_inference_time': max(self.inference_times),
            'total_inferences': len(self.inference_times)
        }


class GRUTemporalAttention(nn.Module):
    """
    Multi-head attention mechanism optimized for GRU temporal sequences.
    
    Identifies important time steps in cybersecurity attack sequences for
    better prediction accuracy and interpretability.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Initialize GRU temporal attention.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            temperature: Attention temperature for sharpening
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply temporal attention to GRU output.
        
        Args:
            x: Input sequence [batch_size, seq_len, hidden_dim]
            sequence_lengths: Actual sequence lengths
            return_weights: Return attention weights
            
        Returns:
            Attended sequence or (attended_sequence, attention_weights)
        """
        # Create attention mask for variable length sequences
        attn_mask = None
        if sequence_lengths is not None:
            batch_size, seq_len = x.shape[:2]
            attn_mask = torch.zeros(batch_size, seq_len, device=x.device, dtype=torch.bool)
            
            for i, length in enumerate(sequence_lengths):
                if length < seq_len:
                    attn_mask[i, length:] = True
        
        # Self-attention
        attended, attention_weights = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            need_weights=True
        )
        
        # Residual connection and layer norm
        output = self.layer_norm(x + self.dropout(attended))
        
        if return_weights:
            return output, attention_weights
        else:
            return output


class GRUGraphSequenceEncoder(nn.Module):
    """
    GRU-based encoder for sequences of heterogeneous graphs.
    
    Combines node embeddings from multiple time steps and applies
    GRU temporal encoding for cybersecurity attack sequence modeling.
    
    This is a drop-in replacement for the LSTM-based GraphSequenceEncoder
    with improved performance characteristics for cybersecurity use cases.
    """
    
    def __init__(
        self,
        node_types: List[str],
        embedding_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pooling_method: str = 'mean',
        dropout: float = 0.4,
        use_attention: bool = False,
        bidirectional: bool = False,
        use_residual: bool = False  # Disabled to avoid dimension issues
    ):
        """
        Initialize GRU graph sequence encoder.
        
        Args:
            node_types: List of node types
            embedding_dim: Node embedding dimension
            hidden_dim: GRU hidden dimension
            num_layers: Number of GRU layers
            pooling_method: Graph pooling method ('mean', 'max', 'sum', 'attention')
            dropout: Dropout probability
            use_attention: Enable temporal attention
            bidirectional: Use bidirectional GRU
            use_residual: Enable residual connections
        """
        super().__init__()
        
        self.node_types = node_types
        self.embedding_dim = embedding_dim
        self.pooling_method = pooling_method
        
        logger.info(f"Initializing GRUGraphSequenceEncoder: "
                   f"node_types={len(node_types)}, embedding_dim={embedding_dim}, "
                   f"hidden_dim={hidden_dim}, pooling={pooling_method}")
        
        # Graph pooling layers
        if pooling_method == 'attention':
            self.graph_attention = nn.ModuleDict({
                node_type: nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=4,
                    batch_first=True,
                    dropout=dropout
                )
                for node_type in node_types
            })
        
        # GRU temporal encoder
        total_graph_dim = len(node_types) * embedding_dim
        self.temporal_encoder = GRUTemporalEncoder(
            input_dim=total_graph_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            use_attention=use_attention,
            use_residual=use_residual
        )
    
    def forward(
        self,
        graph_sequence: List[Dict[str, torch.Tensor]],
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Encode sequence of graphs using GRU.
        
        Args:
            graph_sequence: List of node embedding dictionaries
            return_sequence: Return full sequence or final state
            
        Returns:
            Encoded graph sequence
        """
        try:
            if not graph_sequence:
                raise ValueError("Empty graph sequence provided")
            
            # Pool each graph to fixed-size representation
            pooled_graphs = []
            
            for i, graph_embeddings in enumerate(graph_sequence):
                if not graph_embeddings:
                    logger.warning(f"Empty graph embeddings at position {i}")
                    continue
                
                pooled_features = []
                
                for node_type in self.node_types:
                    if node_type in graph_embeddings and graph_embeddings[node_type].numel() > 0:
                        node_embs = graph_embeddings[node_type]
                        
                        # Apply pooling based on method
                        if self.pooling_method == 'mean':
                            pooled = node_embs.mean(dim=0)
                        elif self.pooling_method == 'max':
                            pooled = node_embs.max(dim=0)[0]
                        elif self.pooling_method == 'sum':
                            pooled = node_embs.sum(dim=0)
                        elif self.pooling_method == 'attention':
                            # Self-attention pooling
                            if node_type in self.graph_attention:
                                attended, _ = self.graph_attention[node_type](
                                    node_embs.unsqueeze(0),
                                    node_embs.unsqueeze(0),
                                    node_embs.unsqueeze(0)
                                )
                                pooled = attended.mean(dim=1).squeeze(0)
                            else:
                                pooled = node_embs.mean(dim=0)  # Fallback to mean
                        else:
                            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
                        
                        pooled_features.append(pooled)
                    else:
                        # Zero padding for missing node types
                        device = next(iter(graph_embeddings.values())).device if graph_embeddings else torch.device('cpu')
                        pooled_features.append(torch.zeros(self.embedding_dim, device=device))
                
                if pooled_features:
                    # Concatenate all node type features
                    graph_repr = torch.cat(pooled_features, dim=0)
                    pooled_graphs.append(graph_repr)
            
            if not pooled_graphs:
                raise ValueError("No valid graphs found in sequence")
            
            # Stack into sequence tensor
            sequence_tensor = torch.stack(pooled_graphs).unsqueeze(0)  # [1, seq_len, total_dim]
            
            # Apply GRU temporal encoding
            encoded = self.temporal_encoder(
                sequence_tensor,
                return_all_states=return_sequence
            )
            
            return encoded
            
        except Exception as e:
            logger.error(f"Error in GRUGraphSequenceEncoder forward pass: {e}")
            logger.error(f"Graph sequence length: {len(graph_sequence) if graph_sequence else 0}")
            raise