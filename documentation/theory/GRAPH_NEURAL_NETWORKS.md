# Graph Neural Networks
## Theoretical Foundations for THGNN System

**Document Version:** 1.0.0  
**Last Updated:** February 9, 2026  
**Classification:** Technical Theory

---

## Table of Contents

1. [Introduction to Graph Neural Networks](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Message Passing Framework](#message-passing-framework)
4. [Graph Convolution Operations](#graph-convolution-operations)
5. [Heterogeneous Graph Neural Networks](#heterogeneous-gnns)
6. [Attention Mechanisms](#attention-mechanisms)
7. [Training and Optimization](#training-and-optimization)
8. [Applications in Cybersecurity](#cybersecurity-applications)

---

## Introduction

Graph Neural Networks (GNNs) are a class of deep learning models designed to operate on graph-structured data. Unlike traditional neural networks that process grid-like data (images, sequences), GNNs can handle irregular, non-Euclidean data structures where entities (nodes) are connected by relationships (edges).

### Why GNNs for Cybersecurity?

Security events naturally form graph structures:
- **Nodes**: Users, hosts, processes, files, network connections
- **Edges**: Interactions, dependencies, communications
- **Temporal Dimension**: Events occur over time, creating dynamic graphs

Traditional ML approaches treat events independently, missing critical relationship patterns. GNNs excel at:
- Capturing multi-hop dependencies
- Learning from graph topology
- Detecting anomalous subgraph patterns
- Predicting future interactions

---

## Mathematical Foundations

### Graph Representation

A graph G is defined as:

```
G = (V, E, X)
```

Where:
- **V**: Set of nodes (vertices), |V| = n
- **E**: Set of edges, E ⊆ V × V, |E| = m
- **X**: Node feature matrix, X ∈ ℝ^(n×d)

### Adjacency Matrix

The graph structure is represented by adjacency matrix A ∈ ℝ^(n×n):

```
A[i,j] = {
    1  if (v_i, v_j) ∈ E
    0  otherwise
}
```

### Degree Matrix

The degree matrix D is a diagonal matrix:

```
D[i,i] = Σ_j A[i,j]
```

### Normalized Adjacency

For stable training, we use normalized adjacency:

```
Ã = D^(-1/2) A D^(-1/2)
```

This symmetric normalization ensures:
- Numerical stability
- Balanced message aggregation
- Gradient flow control

---

## Message Passing Framework

GNNs operate through iterative message passing between connected nodes.

### General Message Passing Equation

At layer k, node v_i updates its representation:

```
h_i^(k) = UPDATE^(k)(h_i^(k-1), AGGREGATE^(k)({h_j^(k-1) : j ∈ N(i)}))
```

Where:
- **h_i^(k)**: Node i's representation at layer k
- **N(i)**: Neighbors of node i
- **AGGREGATE**: Aggregation function (sum, mean, max)
- **UPDATE**: Update function (neural network)

### Three-Step Process

1. **Message Construction**
   ```
   m_ij^(k) = MESSAGE^(k)(h_i^(k-1), h_j^(k-1), e_ij)
   ```
   Create messages from neighbors

2. **Message Aggregation**
   ```
   m_i^(k) = AGGREGATE^(k)({m_ij^(k) : j ∈ N(i)})
   ```
   Combine messages from all neighbors

3. **Node Update**
   ```
   h_i^(k) = UPDATE^(k)(h_i^(k-1), m_i^(k))
   ```
   Update node representation

---

## Graph Convolution Operations

### Spectral Graph Convolution

Based on graph Fourier transform:

```
g_θ ⋆ x = U g_θ(Λ) U^T x
```

Where:
- **U**: Eigenvectors of graph Laplacian L = D - A
- **Λ**: Eigenvalues of L
- **g_θ**: Learnable filter

### Spatial Graph Convolution (GCN)

Simplified spectral convolution (Kipf & Welling, 2017):

```
H^(k+1) = σ(Ã H^(k) W^(k))
```

Where:
- **Ã**: Normalized adjacency with self-loops
- **H^(k)**: Node features at layer k
- **W^(k)**: Learnable weight matrix
- **σ**: Activation function (ReLU, tanh)

### GraphSAGE

Sample and aggregate from neighbors:

```
h_i^(k) = σ(W^(k) · CONCAT(h_i^(k-1), AGG({h_j^(k-1) : j ∈ S(i)})))
```

Where S(i) is a sampled subset of neighbors.

Aggregation functions:
- **Mean**: `AGG = mean({h_j})`
- **Max**: `AGG = max({h_j})`
- **LSTM**: `AGG = LSTM({h_j})`

---

## Heterogeneous GNNs

Cybersecurity graphs contain multiple node and edge types. Heterogeneous GNNs handle this complexity.

### Heterogeneous Graph Definition

```
G = (V, E, τ_v, τ_e, X)
```

Where:
- **τ_v: V → T_v**: Node type function
- **τ_e: E → T_e**: Edge type function
- **T_v**: Set of node types (user, host, process)
- **T_e**: Set of edge types (logs_into, executes, connects_to)

### Relation-Specific Convolution

For each edge type r ∈ T_e:

```
h_i^(k,r) = σ(Σ_{j∈N_r(i)} (W_r^(k) h_j^(k-1)) / |N_r(i)|)
```

Where N_r(i) are neighbors connected via edge type r.

### Type-Specific Aggregation

Combine messages from different edge types:

```
h_i^(k) = AGG_r({h_i^(k,r) : r ∈ R(i)})
```

Common aggregation strategies:
- **Mean**: Average across edge types
- **Attention**: Weighted combination
- **Concatenation**: Preserve all information

### R-GCN (Relational GCN)

```
h_i^(k) = σ(W_0^(k) h_i^(k-1) + Σ_r Σ_{j∈N_r(i)} (W_r^(k) h_j^(k-1)) / c_{i,r})
```

Where:
- **W_0**: Self-connection weight
- **W_r**: Relation-specific weight
- **c_{i,r}**: Normalization constant

---

## Attention Mechanisms

Attention allows the model to learn importance weights for different neighbors.

### Graph Attention Networks (GAT)

Attention coefficient between nodes i and j:

```
α_ij = softmax_j(e_ij)
     = exp(e_ij) / Σ_{k∈N(i)} exp(e_ik)
```

Where:
```
e_ij = LeakyReLU(a^T [W h_i || W h_j])
```

Node update with attention:

```
h_i^(k) = σ(Σ_{j∈N(i)} α_ij W^(k) h_j^(k-1))
```

### Multi-Head Attention

Use K independent attention mechanisms:

```
h_i^(k) = ||_{k=1}^K σ(Σ_{j∈N(i)} α_ij^k W^k h_j^(k-1))
```

Where || denotes concatenation.

### Heterogeneous Attention

For heterogeneous graphs, compute attention per edge type:

```
α_ij^r = softmax_j(e_ij^r)

e_ij^r = a_r^T [W_r h_i || W_r h_j]
```

---

## Training and Optimization

### Loss Functions

#### Link Prediction Loss

Binary cross-entropy for edge prediction:

```
L = -Σ_{(i,j)∈E_pos} log(σ(z_i^T z_j)) - Σ_{(i,j)∈E_neg} log(1 - σ(z_i^T z_j))
```

Where:
- **E_pos**: Positive edges (actual connections)
- **E_neg**: Negative edges (sampled non-connections)
- **z_i**: Node embedding

#### Margin-Based Loss

```
L = Σ_{(i,j)∈E_pos} Σ_{(i,k)∈E_neg} max(0, γ - score(i,j) + score(i,k))
```

Where γ is the margin hyperparameter.

### Negative Sampling

Critical for link prediction in sparse graphs:

**Random Sampling**: Sample random node pairs
```
P(i,j) = 1 / (|V| × |V|)
```

**Degree-Based Sampling**: Favor high-degree nodes
```
P(i,j) ∝ deg(i)^α × deg(j)^α
```

**Hard Negative Mining**: Sample challenging negatives
```
P(i,j) ∝ score(i,j)  (for non-edges)
```

### Optimization Techniques

#### Gradient Clipping

Prevent exploding gradients:
```
g ← g × min(1, θ / ||g||)
```

#### Learning Rate Scheduling

Reduce learning rate over time:
```
lr_t = lr_0 × decay_rate^(epoch / decay_steps)
```

#### Dropout

Regularization for GNNs:
```
h_i^(k) = Dropout(σ(Ã H^(k-1) W^(k)))
```

---

## Cybersecurity Applications

### Attack Pattern Detection

GNNs identify anomalous subgraph patterns:

```
score(G_sub) = GNN(G_sub)
anomaly = score(G_sub) > threshold
```

### Lateral Movement Prediction

Predict next attack step:

```
P(edge_future | G_past) = LinkPredictor(GNN(G_past))
```

### Attack Chain Reconstruction

Find most likely attack path:

```
path* = argmax_path Π_{(i,j)∈path} P(edge_ij | G)
```

### Threat Prioritization

Rank entities by risk:

```
risk(v_i) = Σ_{j∈V} P(edge_ij | G) × severity(j)
```

---

## Advanced Topics

### Graph Pooling

Hierarchical graph representation:

```
G_coarse = POOL(G_fine)
```

Methods:
- **Top-K Pooling**: Keep top-k nodes by score
- **DiffPool**: Differentiable soft clustering
- **SAGPool**: Self-attention graph pooling

### Graph Generation

Generate new graphs:

```
G_new ~ P(G | z)
```

Where z is a latent code.

### Explainability

Identify important subgraphs:

```
G_important = argmax_G' MI(Y, G' | G)
```

Where MI is mutual information.

---

## Implementation Considerations

### Computational Complexity

- **GCN**: O(|E| × d × h) per layer
- **GAT**: O(|E| × d × h + |V| × h²) per layer
- **R-GCN**: O(|E| × d × h × |R|) per layer

Where:
- d: Input dimension
- h: Hidden dimension
- |R|: Number of edge types

### Memory Requirements

- **Node features**: O(|V| × d)
- **Edge indices**: O(|E|)
- **Model parameters**: O(L × d × h)

Where L is number of layers.

### Scalability Techniques

1. **Mini-batch Training**: Sample subgraphs
2. **Neighbor Sampling**: Limit neighborhood size
3. **Graph Partitioning**: Distribute across GPUs
4. **Gradient Checkpointing**: Trade compute for memory

---

## References

1. Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks"
2. Veličković et al. (2018). "Graph Attention Networks"
3. Hamilton et al. (2017). "Inductive Representation Learning on Large Graphs"
4. Schlichtkrull et al. (2018). "Modeling Relational Data with Graph Convolutional Networks"
5. Ying et al. (2018). "Graph Convolutional Neural Networks for Web-Scale Recommender Systems"

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-09 | Research Team | Initial release |
