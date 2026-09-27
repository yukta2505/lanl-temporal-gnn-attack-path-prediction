# Data flow for next-hop prediction

```mermaid
sequenceDiagram
    participant A as LANL auth archive
    participant R as LANL red-team archive
    participant P as Streaming preprocessor
    participant X as XGBoost baseline
    participant I as Prediction exporter
    participant D as Analyst dashboard
    participant C as Asset inventory CSV
    A->>P: Chronological authentication events
    R->>P: Malicious edge labels
    P->>P: Update prior-window host and pair aggregates
    P->>P: Write positive and sampled negative candidate pairs
    P->>X: Candidate JSONL
    X->>I: Trained model bundle
    P->>I: Candidate feature rows
    I->>D: Probability and feature-contribution CSV
    C->>D: Host criticality values
```

## Feature/label separation

For each prediction window, feature aggregates come from events in earlier
windows. Red-team edges in the prediction window supply positive labels. The
source host must have appeared in earlier red-team activity to enter the
current candidate set. Negative destinations are sampled from hosts already
observed in the event stream.

The graph-model flow is incomplete: graph snapshots and a training pipeline
are still required. The example builder saves output offsets and aggregate
state together after each completed window. On resume it truncates uncommitted
output rows and skips completed windows; compressed input must be reread from
the start to reach that checkpoint.
