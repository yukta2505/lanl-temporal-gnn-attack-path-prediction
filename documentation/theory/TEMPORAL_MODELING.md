# Temporal modeling for next-hop prediction

## Time windows

LANL timestamps are relative seconds from the recording start. The data
pipeline groups chronological authentication events into non-overlapping
windows. For a prediction window `t`, features must use only history available
before `t`; red-team edges in `t` provide the target labels. This prevents
future-event leakage into candidate features.

## Current baseline timeline

The streaming example builder keeps cumulative host and host-pair aggregates
in SQLite. At each window it creates examples for red-team edges whose source
host appeared in an earlier red-team event, then updates historical state with
the events in the current window. Negative target hosts are sampled from the
observed host set. The exact candidate policy and sample size must be recorded
with any reported metrics.

This is a supervised next-hop baseline with temporal history features; it is
not itself a trained temporal GNN.

## Temporal graph model

The repository's `TemporalAttackPathModel` accepts an ordered list of graph
snapshots and node features. It applies GraphSAGE to each snapshot, uses a GRU
to summarize each host's sequence of representations, and scores candidate
source/target pairs. A complete experiment still needs:

1. A graph-snapshot and candidate-label builder for the LANL sample.
2. Chronological train, validation, and test splits with positive examples in
   the relevant periods.
3. A training loop, checkpointing, threshold selection, and top-k evaluation.
4. A comparison with a static graph model and the XGBoost baseline.

## Evaluation cautions

- Keep events from later time windows out of earlier features.
- Report positive and negative candidate counts per split.
- Calculate PR-AUC and Precision@k because attack edges are rare.
- Do not interpret probability multiplied by asset criticality as model
  likelihood; it is a separate operational ranking score.
- Report path metrics only after defining the path horizon, hop limit, and
  treatment of missing ground-truth edges.

No temporal GNN results have been produced for this repository's LANL data.
