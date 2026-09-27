# Current project sequence diagrams

## Build and train the baseline

```mermaid
sequenceDiagram
    participant User
    participant Builder as LANL example builder
    participant Auth as auth.txt.gz
    participant Red as redteam.txt.gz
    participant Store as Persistent SQLite checkpoint
    participant Trainer as XGBoost trainer
    User->>Builder: Start 15-day example build
    Builder->>Red: Read malicious-edge labels
    Builder->>Auth: Stream chronological auth events
    loop For each prediction window
        Builder->>Store: Read prior host and pair history
        Builder->>Builder: Write positive and sampled negative pairs
        Builder->>Store: Aggregate current-window events
    end
    Builder->>Store: Save completed window and output offset
    Builder-->>User: Candidate JSONL after successful completion
    User->>Trainer: Train with chronological splits
    Trainer-->>User: Model bundle and metrics JSON
```

## Score and review candidates

```mermaid
sequenceDiagram
    participant User
    participant Predictor as Batched predictor
    participant Model as XGBoost model
    participant Dashboard as Streamlit dashboard
    participant Inventory as Asset inventory CSV
    User->>Predictor: Candidate JSONL and model bundle
    Predictor->>Model: Score feature batches
    Model-->>Predictor: Probabilities and feature contributions
    Predictor-->>User: Predictions CSV
    User->>Dashboard: Upload predictions CSV
    User->>Dashboard: Upload criticality inventory
    Dashboard->>Dashboard: Filter by source host and rank by risk
```

## Planned graph training sequence

Graph snapshot generation, static/temporal GNN training, and model-backed
dashboard inference are not implemented yet. Add them only after defining the
chronological snapshot and candidate-label format.
