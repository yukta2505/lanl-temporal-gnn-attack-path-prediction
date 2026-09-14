# Temporal Modeling in Graph Neural Networks
## Time-Aware Threat Detection

**Document Version:** 1.0.0  
**Last Updated:** February 9, 2026  
**Classification:** Technical Theory

---

## Table of Contents

1. [Introduction to Temporal Graphs](#introduction)
2. [Temporal Graph Representations](#temporal-representations)
3. [Temporal Encoding Methods](#temporal-encoding)
4. [Recurrent Graph Neural Networks](#recurrent-gnns)
5. [Temporal Attention Mechanisms](#temporal-attention)
6. [Dynamic Graph Learning](#dynamic-graphs)
7. [Temporal Link Prediction](#temporal-link-prediction)
8. [Cybersecurity Applications](#cybersecurity-applications)

---

## Introduction

Cybersecurity threats unfold over time. An attacker's actions form temporal sequences: reconnaissance → initial access → lateral movement → privilege escalation → exfiltration. Traditional static graph analysis misses these temporal patterns.

### Why Temporal Modeling Matters

**Attack Characteristics**:
- **Sequential Nature**: Attacks follow kill chain stages
- **Time Dependencies**: Later actions depend on earlier successes
- **Temporal Patterns**: Specific timing signatures (e.g., off-hours access)
- **Evolution**: Attack graphs change as threats progress

**Detection Advantages**:
- Predict next attack step before it occurs
- Identify anomalous temporal patterns
- Track attack progression in real-time
- Distinguish normal vs. malicious sequences

---

## Temporal Graph Representations

### Discrete-Time Dynamic Graphs

Graph snapshots at discrete time intervals:

```
G = {G_1, G_2, ..., G_T}
```

Where G_t = (V_t, E_t, X_t) at time t.

**Advantages**:
- Simple to implement
- Clear temporal boundaries
- Efficient for batch processing

**Disadvantages**:
- Information loss between snapshots
- Arbitrary time window selection
- Cannot capture fine-grained timing

### Continuous-Time Dynamic Graphs

Events with precise timestamps:

```
G = (V, E, T)
E = {(u, v, t, e_attr) : u,v ∈ V, t ∈ ℝ}
```

**Advantages**:
- Preserves exact timing
- No information loss
- Natural for event streams

**Disadvantages**:
- More complex processing
- Higher computational cost
- Requires careful handling

### Temporal Edge Representation

Each edge carries temporal information:

```
e_ij = (source_i, target_j, timestamp_t, duration_d, features_x)
```

**Key Temporal Features**:
- **Timestamp**: When interaction occurred
- **Duration**: How long interaction lasted
- **Frequency**: Number of interactions in window
- **Recency**: Time since last interaction
- **Periodicity**: Regular timing patterns

---

## Temporal Encoding Methods

### Time Embedding Functions

Transform timestamps into learnable representations.

#### Sinusoidal Time Encoding

Based on Transformer positional encoding:

```
TE(t, 2i) = sin(t / 10000^(2i/d))
TE(t, 2i+1) = cos(t / 10000^(2i/d))
```

Where:
- t: Timestamp
- i: Dimension index
- d: Embedding dimension

**Properties**:
- Captures multiple time scales
- Smooth interpolation
- Fixed (non-learnable)

#### Learnable Time Projection

Neural network-based encoding:

```
TE(t) = MLP([t, t², sin(ωt), cos(ωt)])
```

**Advantages**:
- Adapts to data
- Captures domain-specific patterns
- More expressive

#### Relative Time Encoding

Encode time differences:

```
TE(t_i, t_j) = φ(t_i - t_j)
```

Useful for:
- Temporal proximity
- Sequence ordering
- Time-aware attention

### Time-Aware Node Features

Augment node features with temporal information:

```
x_i(t) = [x_i^static, TE(t), Δt_last, freq_window]
```

Components:
- **x_i^static**: Static node attributes
- **TE(t)**: Time embedding
- **Δt_last**: Time since last activity
- **freq_window**: Activity frequency in window

---

## Recurrent Graph Neural Networks

Combine GNNs with recurrent architectures for temporal modeling.

### GRU-Based Temporal GNN

Update node representations over time using GRU:

```
h_i^(t) = GRU(h_i^(t-1), m_i^(t))

m_i^(t) = GNN(G_t, {h_j^(t-1) : j ∈ N_i(t)})
```

**Architecture**:
```
Input: G_t, h^(t-1)
  ↓
Graph Convolution → m^(t)
  ↓
GRU Update → h^(t)
  ↓
Output: h^(t)
```

### LSTM-Based Temporal GNN

Use LSTM for longer-term dependencies:

```
(h_i^(t), c_i^(t)) = LSTM(h_i^(t-1), c_i^(t-1), m_i^(t))
```

**Cell State Update**:
```
f_t = σ(W_f · [h^(t-1), m^(t)])  # Forget gate
i_t = σ(W_i · [h^(t-1), m^(t)])  # Input gate
c̃_t = tanh(W_c · [h^(t-1), m^(t)])  # Candidate
c_t = f_t ⊙ c^(t-1) + i_t ⊙ c̃_t  # Cell state
o_t = σ(W_o · [h^(t-1), m^(t)])  # Output gate
h_t = o_t ⊙ tanh(c_t)  # Hidden state
```

### Temporal Graph Convolutional Network (T-GCN)

Combines GCN with GRU:

```
h^(t) = GRU(h^(t-1), GCN(A_t, X_t))
```

**Forward Pass**:
```python
def forward(self, X_t, A_t, h_prev):
    # Graph convolution
    m_t = self.gcn(X_t, A_t)
    
    # Temporal update
    h_t = self.gru(m_t, h_prev)
    
    return h_t
```

---

## Temporal Attention Mechanisms

Learn importance of different time steps and temporal relationships.

### Temporal Self-Attention

Attend over historical node states:

```
α_t = softmax(Q_t K^T / √d_k)
h_out = α_t V
```

Where:
```
Q_t = W_Q h_t
K = W_K [h_1, h_2, ..., h_T]
V = W_V [h_1, h_2, ..., h_T]
```

### Time-Aware Attention

Incorporate time differences in attention:

```
e_ij = a([W h_i || W h_j || TE(t_i - t_j)])
α_ij = softmax_j(e_ij)
```

**Decay Function**:
```
α_ij = α_ij × exp(-λ(t_current - t_j))
```

Recent events have higher weight.

### Multi-Scale Temporal Attention

Attend at different time scales:

```
h_out = Σ_s w_s · Attention_s(h, T_s)
```

Where:
- T_s: Time scale (hour, day, week)
- w_s: Learnable scale weights

---

## Dynamic Graph Learning

### Temporal Random Walk

Sample temporal paths respecting time ordering:

```
Path: v_1 →^(t_1) v_2 →^(t_2) v_3 →^(t_3) ...
```

Constraint: t_1 < t_2 < t_3 < ...

**Algorithm**:
```python
def temporal_random_walk(G, start_node, start_time, length):
    path = [start_node]
    current_time = start_time
    current_node = start_node
    
    for _ in range(length - 1):
        # Get future neighbors
        future_neighbors = [
            (v, t) for (u, v, t) in G.edges 
            if u == current_node and t > current_time
        ]
        
        if not future_neighbors:
            break
            
        # Sample next node
        next_node, next_time = random.choice(future_neighbors)
        path.append(next_node)
        current_node = next_node
        current_time = next_time
    
    return path
```

### Temporal Graph Sampling

Sample subgraphs preserving temporal structure:

```
G_sample = Sample(G, v_seed, t_start, t_end, k_hops)
```

**Temporal k-hop Neighborhood**:
```
N_k^t(v) = {u : ∃ path v →...→ u, length ≤ k, all edges in [t_start, t_end]}
```

---

## Temporal Link Prediction

Predict future edges based on historical graph evolution.

### Problem Formulation

Given:
- Historical graphs: G_1, G_2, ..., G_t
- Query time: t_query > t

Predict:
- Future edges: E_future at t_query

### Temporal Scoring Functions

#### Time-Decay Scoring

```
score(u, v, t) = f(z_u, z_v) × exp(-λ(t - t_last))
```

Where t_last is last interaction time.

#### Temporal Path Scoring

```
score(u, v, t) = Σ_paths P(path | G_≤t) × relevance(path)
```

Consider temporal paths between u and v.

#### Recurrent Scoring

```
score(u, v, t) = σ(z_u^(t)^T W z_v^(t))

z^(t) = RNN(z^(t-1), G_t)
```

### Temporal Negative Sampling

Sample negatives respecting temporal constraints:

**Future Negative Sampling**:
```
E_neg = {(u, v, t) : (u, v, t) ∉ E_future, u,v active at t}
```

**Historical Negative Sampling**:
```
E_neg = {(u, v, t) : (u, v, t') ∉ E for all t' ≤ t}
```

---

## Cybersecurity Applications

### Attack Sequence Prediction

Model attack progression as temporal graph:

```
G_attack = {
    (recon, t_1) → (initial_access, t_2) → 
    (lateral_movement, t_3) → (exfiltration, t_4)
}
```

**Prediction**:
```
P(action_next | G_history) = TemporalGNN(G_history)
```

### Anomalous Timing Detection

Identify unusual temporal patterns:

```
anomaly_score(e_ij, t) = |t - E[t | e_ij]| / σ[t | e_ij]
```

**Examples**:
- Login at 3 AM (unusual hour)
- Rapid sequence of actions (automated)
- Long dormancy then sudden activity

### Temporal Attack Chain Extraction

Find temporally-ordered attack paths:

```
chain = argmax_path Π_(e∈path) P(e | G_≤t_e) × temporal_coherence(path)
```

**Temporal Coherence**:
```
coherence(path) = exp(-Σ_i |t_i+1 - t_i - Δt_expected|)
```

### Real-Time Threat Forecasting

Predict threats in sliding time windows:

```
For each window [t, t+Δt]:
    G_current = G_[t-w, t]
    threats_forecast = TemporalGNN(G_current)
    alert if max(threats_forecast) > threshold
```

---

## Implementation in THGNN System

### Temporal Feature Engineering

```python
def engineer_temporal_features(edge_data, current_time):
    """
    Create temporal features for edges.
    
    Parameters
    ----------
    edge_data : pd.DataFrame
        Edge information with timestamps
    current_time : datetime
        Reference time for relative features
    
    Returns
    -------
    torch.Tensor
        Temporal feature matrix
    """
    features = []
    
    # Time since event
    time_delta = (current_time - edge_data['timestamp']).dt.total_seconds()
    features.append(np.log1p(time_delta))
    
    # Time of day (cyclical)
    hour = edge_data['timestamp'].dt.hour
    features.append(np.sin(2 * np.pi * hour / 24))
    features.append(np.cos(2 * np.pi * hour / 24))
    
    # Day of week (cyclical)
    day = edge_data['timestamp'].dt.dayofweek
    features.append(np.sin(2 * np.pi * day / 7))
    features.append(np.cos(2 * np.pi * day / 7))
    
    # Event frequency in window
    freq = edge_data.groupby(['source', 'target']).size()
    features.append(np.log1p(freq))
    
    return torch.tensor(np.column_stack(features), dtype=torch.float)
```

### Temporal Graph Construction

```python
def build_temporal_graph(events, time_window=3600):
    """
    Build temporal heterogeneous graph from events.
    
    Parameters
    ----------
    events : pd.DataFrame
        Security events with timestamps
    time_window : int
        Time window in seconds
    
    Returns
    -------
    HeteroData
        Temporal graph structure
    """
    graph = HeteroData()
    
    # Sort events by time
    events = events.sort_values('timestamp')
    
    # Create temporal edges
    for edge_type in ['logs_into', 'executes', 'connects_to']:
        edges = events[events['action'] == edge_type]
        
        # Add temporal features
        edge_attr = engineer_temporal_features(
            edges, 
            events['timestamp'].max()
        )
        
        # Store edges with timestamps
        graph[edge_type].edge_index = torch.tensor(
            edges[['source_id', 'target_id']].values.T
        )
        graph[edge_type].edge_attr = edge_attr
        graph[edge_type].timestamp = torch.tensor(
            edges['timestamp'].astype(np.int64).values
        )
    
    return graph
```

### Temporal Attention Layer

```python
class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for time-aware message passing.
    """
    
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj = nn.Linear(1, num_heads)
        
    def forward(self, x, edge_index, timestamps):
        """
        Apply temporal attention.
        
        Parameters
        ----------
        x : Tensor [num_nodes, hidden_dim]
            Node features
        edge_index : Tensor [2, num_edges]
            Edge connectivity
        timestamps : Tensor [num_edges]
            Edge timestamps
        
        Returns
        -------
        Tensor [num_nodes, hidden_dim]
            Updated node features
        """
        # Project to Q, K, V
        Q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        src, dst = edge_index
        scores = (Q[dst] * K[src]).sum(dim=-1) / np.sqrt(self.head_dim)
        
        # Add temporal bias
        time_bias = self.time_proj(timestamps.unsqueeze(-1))
        scores = scores + time_bias
        
        # Softmax over neighbors
        alpha = softmax(scores, dst, num_nodes=x.size(0))
        
        # Aggregate messages
        out = scatter(
            alpha.unsqueeze(-1) * V[src],
            dst,
            dim=0,
            dim_size=x.size(0),
            reduce='sum'
        )
        
        return out.view(-1, self.hidden_dim)
```

---

## Performance Considerations

### Computational Complexity

**Temporal GNN per layer**:
```
O(T × |E| × d × h)
```

Where:
- T: Number of time steps
- |E|: Edges per snapshot
- d: Input dimension
- h: Hidden dimension

### Memory Optimization

**Techniques**:
1. **Sliding Window**: Keep only recent history
2. **State Compression**: Compress historical states
3. **Sparse Temporal Edges**: Store only active edges
4. **Gradient Checkpointing**: Recompute activations

### Scalability Strategies

**For Large Temporal Graphs**:
- Temporal sampling: Sample time windows
- Hierarchical aggregation: Multi-resolution time
- Incremental updates: Update only changed nodes
- Distributed processing: Partition by time

---

## References

1. Xu et al. (2020). "Inductive Representation Learning on Temporal Graphs"
2. Rossi et al. (2020). "Temporal Graph Networks for Deep Learning on Dynamic Graphs"
3. Kumar et al. (2019). "Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks"
4. Trivedi et al. (2019). "DyRep: Learning Representations over Dynamic Graphs"

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-09 | Research Team | Initial release |
