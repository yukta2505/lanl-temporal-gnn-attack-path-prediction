#!/usr/bin/env python3
"""
Temporal Encoder for THGNN

Production-grade LSTM/GRU-based temporal encoding for capturing attack evolution
over time in cybersecurity graphs. Optimized for sequence modeling of graph snapshots.

Features:
- Bidirectional LSTM/GRU for temporal dependencies
- Attention mechanism for important time steps
- Memory-efficient processing for long sequences
- GPU acceleration with gradient checkpointing
- Comprehensive error handling and logging
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, List, Tuple, Optional, Union
import math
import warnings

# Suppress PyTorch warnings for production
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class TemporalEncoder(nn.Module):
    """
    Production-grade temporal encoder for graph sequences.
    
    Captures temporal dependencies in cybersecurity attack patterns using
    LSTM/GRU with attention mechanisms and memory optimization.
    
    Architecture:
    - Bidirectional LSTM/GRU for sequence modeling
    - Multi-head attention for temporal importance
    - Layer normalization for training stability
    - Gradient checkpointing for memory efficiency
    - Residual connections for deep networks
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        rnn_type: str = 'lstm',
        bidirectional: bool = True,
        dropout: float = 0.3,
        use_attention: bool = True,
        attention_heads: int = 4,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        gradient_checkpointing: bool = False
    ):
        """
        Initialize temporal encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of RNN layers
            rnn_type: RNN type ('lstm' or 'gru')
            bidirectional: Use bidirectional RNN
            dropout: Dropout probability
            use_attention: Enable temporal attention
            attention_heads: Number of attention heads
            use_layer_norm: Enable layer normalization
            use_residual: Enable residual connections
            gradient_checkpointing: Enable gradient checkpointing
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.gradient_checkpointing = gradient_checkpointing
        
        # Calculate effective hidden dimension
        self.effective_hidden_dim = hidden_dim * (2 if bidirectional else 1)
        
        logger.info(f"Initializing TemporalEncoder: {rnn_type.upper()}, "
                   f"hidden_dim={hidden_dim}, layers={num_layers}, "
                   f"bidirectional={bidirectional}")
        
        # Build RNN layer
        self.rnn = self._build_rnn()
        
        # Input projection (if needed)
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_projection = None
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.effective_hidden_dim)
        
        # Temporal attention mechanism
        if use_attention:
            self.attention = TemporalAttention(
                hidden_dim=self.effective_hidden_dim,
                num_heads=attention_heads,
                dropout=dropout
            )
        
        # Output projection
        self.output_projection = nn.Linear(
            self.effective_hidden_dim,
            input_dim  # Always project back to input dimension
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _build_rnn(self) -> nn.Module:
        """Build RNN layer (LSTM or GRU)."""
        if self.rnn_type == 'lstm':
            return nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional
            )
        elif self.rnn_type == 'gru':
            return nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
    
    def _init_parameters(self) -> None:
        """Initialize RNN parameters with Xavier initialization."""
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1 for LSTM
                if self.rnn_type == 'lstm' and 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
    
    def forward(
        self,
        x_sequence: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, ...]] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
        return_all_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through temporal encoder.
        
        Args:
            x_sequence: Input sequence [batch_size, seq_len, input_dim]
            hidden_state: Initial hidden state (optional)
            sequence_lengths: Actual sequence lengths for packing
            return_all_states: Return all hidden states or just final
            
        Returns:
            Encoded sequence or (output, hidden_state) tuple
        """
        try:
            batch_size, seq_len, _ = x_sequence.shape
            
            # Input projection
            if self.input_projection is not None:
                x_sequence = self.input_projection(x_sequence)
            
            # Store input for residual connection
            x_input = x_sequence if self.use_residual else None
            
            # Pack sequences if lengths provided
            if sequence_lengths is not None:
                x_sequence = pack_padded_sequence(
                    x_sequence, sequence_lengths.cpu(),
                    batch_first=True, enforce_sorted=False
                )
            
            # RNN forward pass with gradient checkpointing
            if self.gradient_checkpointing and self.training:
                rnn_output, hidden_state = torch.utils.checkpoint.checkpoint(
                    self._rnn_forward, x_sequence, hidden_state
                )
            else:
                rnn_output, hidden_state = self._rnn_forward(x_sequence, hidden_state)
            
            # Unpack sequences if packed
            if sequence_lengths is not None:
                rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)
            
            # Apply layer normalization
            if self.use_layer_norm:
                rnn_output = self.layer_norm(rnn_output)
            
            # Apply temporal attention
            if self.use_attention:
                rnn_output = self.attention(rnn_output, sequence_lengths)
            
            # Apply dropout
            rnn_output = self.dropout_layer(rnn_output)
            
            # Output projection
            output = self.output_projection(rnn_output)
            
            # Apply residual connection
            if self.use_residual and x_input is not None:
                output = output + x_input
            
            if return_all_states:
                return output, hidden_state
            else:
                # Return final state for each sequence
                if sequence_lengths is not None:
                    final_outputs = []
                    for i, length in enumerate(sequence_lengths):
                        final_outputs.append(output[i, length-1])
                    return torch.stack(final_outputs)
                else:
                    return output[:, -1]  # Last time step
                    
        except Exception as e:
            logger.error(f"Error in TemporalEncoder forward pass: {e}")
            raise
    
    def _rnn_forward(
        self,
        x_sequence: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, ...]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """RNN forward pass (for gradient checkpointing)."""
        return self.rnn(x_sequence, hidden_state)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """Initialize hidden state."""
        num_directions = 2 if self.bidirectional else 1
        
        if self.rnn_type == 'lstm':
            h0 = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_dim,
                device=device
            )
            c0 = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_dim,
                device=device
            )
            return (h0, c0)
        else:  # GRU
            h0 = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_dim,
                device=device
            )
            return (h0,)
    
    def get_attention_weights(
        self,
        x_sequence: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Get attention weights for interpretability."""
        if not self.use_attention:
            return None
        
        # Forward pass to get RNN output
        with torch.no_grad():
            if self.input_projection is not None:
                x_sequence = self.input_projection(x_sequence)
            
            rnn_output, _ = self.rnn(x_sequence)
            
            if self.use_layer_norm:
                rnn_output = self.layer_norm(rnn_output)
            
            # Get attention weights
            return self.attention.get_attention_weights(rnn_output, sequence_lengths)


class TemporalAttention(nn.Module):
    """
    Multi-head attention mechanism for temporal sequences.
    
    Identifies important time steps in attack sequences for better
    prediction accuracy and interpretability.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Initialize temporal attention.
        
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
        sequence_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply temporal attention.
        
        Args:
            x: Input sequence [batch_size, seq_len, hidden_dim]
            sequence_lengths: Actual sequence lengths
            
        Returns:
            Attended sequence
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
        
        return output
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get attention weights for visualization."""
        attn_mask = None
        if sequence_lengths is not None:
            batch_size, seq_len = x.shape[:2]
            attn_mask = torch.zeros(batch_size, seq_len, device=x.device, dtype=torch.bool)
            
            for i, length in enumerate(sequence_lengths):
                if length < seq_len:
                    attn_mask[i, length:] = True
        
        _, attention_weights = self.attention(
            query=x, key=x, value=x,
            attn_mask=attn_mask,
            need_weights=True
        )
        
        return attention_weights


class GraphSequenceEncoder(nn.Module):
    """
    Encoder for sequences of heterogeneous graphs.
    
    Combines node embeddings from multiple time steps and applies
    temporal encoding for attack sequence modeling.
    """
    
    def __init__(
        self,
        node_types: List[str],
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        pooling_method: str = 'mean',
        **temporal_encoder_kwargs
    ):
        """
        Initialize graph sequence encoder.
        
        Args:
            node_types: List of node types
            embedding_dim: Node embedding dimension
            hidden_dim: Temporal encoder hidden dimension
            num_layers: Number of temporal layers
            pooling_method: Graph pooling method ('mean', 'max', 'sum', 'attention')
        """
        super().__init__()
        
        self.node_types = node_types
        self.embedding_dim = embedding_dim
        self.pooling_method = pooling_method
        
        # Graph pooling layers
        if pooling_method == 'attention':
            self.graph_attention = nn.ModuleDict({
                node_type: nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=4,
                    batch_first=True
                )
                for node_type in node_types
            })
        
        # Temporal encoder
        total_graph_dim = len(node_types) * embedding_dim
        self.temporal_encoder = TemporalEncoder(
            input_dim=total_graph_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_residual=False,  # Disable residual to avoid dimension issues
            **temporal_encoder_kwargs
        )
    
    def forward(
        self,
        graph_sequence: List[Dict[str, torch.Tensor]],
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Encode sequence of graphs.
        
        Args:
            graph_sequence: List of node embedding dictionaries
            return_sequence: Return full sequence or final state
            
        Returns:
            Encoded graph sequence
        """
        # Pool each graph to fixed-size representation
        pooled_graphs = []
        
        for graph_embeddings in graph_sequence:
            pooled_features = []
            
            for node_type in self.node_types:
                if node_type in graph_embeddings:
                    node_embs = graph_embeddings[node_type]
                    
                    # Apply pooling
                    if self.pooling_method == 'mean':
                        pooled = node_embs.mean(dim=0)
                    elif self.pooling_method == 'max':
                        pooled = node_embs.max(dim=0)[0]
                    elif self.pooling_method == 'sum':
                        pooled = node_embs.sum(dim=0)
                    elif self.pooling_method == 'attention':
                        # Self-attention pooling
                        attended, _ = self.graph_attention[node_type](
                            node_embs.unsqueeze(0),
                            node_embs.unsqueeze(0),
                            node_embs.unsqueeze(0)
                        )
                        pooled = attended.mean(dim=1).squeeze(0)
                    else:
                        raise ValueError(f"Unknown pooling method: {self.pooling_method}")
                    
                    pooled_features.append(pooled)
                else:
                    # Zero padding for missing node types
                    pooled_features.append(torch.zeros(self.embedding_dim, device=next(iter(graph_embeddings.values())).device))
            
            # Concatenate all node type features
            graph_repr = torch.cat(pooled_features, dim=0)
            pooled_graphs.append(graph_repr)
        
        # Stack into sequence tensor
        sequence_tensor = torch.stack(pooled_graphs).unsqueeze(0)  # [1, seq_len, total_dim]
        
        # Apply temporal encoding
        encoded = self.temporal_encoder(
            sequence_tensor,
            return_all_states=return_sequence
        )
        
        return encoded