# Synopsis system architecture

## Current baseline path

```mermaid
flowchart TD
    Auth[LANL auth.txt.gz] --> Parser[LANL parser]
    Red[LANL redteam.txt.gz] --> Parser
    Parser --> Import[Optional windowed event JSONL]
    Auth --> Stream[Streaming next-hop examples]
    Red --> Stream
    Stream --> XGB[XGBoost baseline and chronological metrics]
    XGB --> Predict[Batched candidate scoring and feature contributions]
    Predict --> Dashboard[Streamlit CSV review]
    Inventory[Reviewed host criticality] --> Dashboard
```

The streaming example builder does not require the complete windowed-event
intermediate. It maintains cumulative host and pair statistics in a persistent
SQLite checkpoint and writes candidate examples to JSONL. Completed windows
and output offsets are saved together so an interrupted run can resume safely.

## Planned graph-model path

```mermaid
flowchart LR
    Windows[Chronological LANL windows] -. graph snapshot builder pending .-> Snapshots[Graph snapshots and candidate labels]
    Snapshots -. trainer pending .-> Static[Static E-GraphSAGE]
    Snapshots -. trainer pending .-> Temporal[GraphSAGE snapshots + GRU]
    Static -. future scorer .-> Review[Analyst review]
    Temporal -. future scorer .-> Review
```

The static and temporal model files currently define scorers only. They are not
trained checkpoints, and the models are not connected to the dashboard.

## Boundaries

- LANL red-team labels are used for supervised attack-edge targets; they do
  not indicate business criticality.
- The review app reads predictions and inventory from CSV files.
- Automated containment, a production REST API, streaming ingestion, and
  measured service-level objectives are outside the implemented project path.
