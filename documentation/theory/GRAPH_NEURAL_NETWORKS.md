# Graph neural networks for host interaction data

## Graph representation

For this project, a graph snapshot is a directed graph `G_t = (V, E_t)` at
time window `t`. A node represents a host. An observed authentication from
source host `u` to destination host `v` contributes a directed edge `(u, v)`.
Edge attributes may include event count, time since the last interaction,
authentication outcome, and event type. Red-team labels are targets for
supervised learning; asset criticality is separate metadata.

This host-to-host representation matches the synopsis prediction task. It is
different from the older prototype's heterogeneous graph of users, processes,
and hosts.

## Message passing

A graph convolution updates each node representation using its own state and
aggregated neighbor states:

```text
h_v^(k) = UPDATE(h_v^(k-1), AGGREGATE({h_u^(k-1) : u in N(v)}))
```

GraphSAGE learns an aggregation function over sampled or complete neighbors.
It can form an embedding for a host from its local interaction structure and
node features. For directed enterprise events, the implementation must decide
whether to retain direction, add reverse edges, and how to encode edge
attributes.

## Candidate edge scoring

For source and target embeddings `z_s` and `z_t`, a pair scorer can use:

```text
[z_s, z_t, z_s * z_t, abs(z_s - z_t), edge_features]
```

The repository's `StaticEGraphSAGE` applies two GraphSAGE layers and scores
candidate pairs with edge features. It is a model component; a LANL graph
snapshot builder and training/evaluation loop are still required.

## Static and temporal models

- A **static** graph model encodes one graph assembled from a defined history
  period and has no recurrent state across snapshots.
- A **temporal** graph model encodes multiple ordered snapshots and passes
  host embeddings through a recurrent unit. The repository's
  `TemporalAttackPathModel` uses GraphSAGE per snapshot followed by a GRU, then
  scores each source/target pair.

Training should use red-team next-hop labels, sampled negatives, chronological
splits, and metrics suitable for rare positive edges. The current repository
does not yet provide that graph-model training data path; no graph-model
performance is claimed.

## Explanation and risk

An explanation should identify historical edges or input features that
affected a particular candidate score. Edge occlusion is implemented as a
bounded perturbation utility; its stability and analyst usefulness remain to
be evaluated. Risk ranking multiplies model probability by a separately
reviewed asset-criticality value. Criticality must not be learned from the
red-team target labels.
