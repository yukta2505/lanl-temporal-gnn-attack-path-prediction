# Project overview

## Aim

Predict likely next lateral-movement destinations from enterprise host
authentication history. Explain the prediction with model evidence and rank
alerts using both attack probability and an externally supplied asset
criticality value.

## Data and task

The current dataset path uses LANL `auth` events as directed host-to-host
interactions and `redteam` records as malicious-edge labels. Auth rows are
comma-separated and time ordered. LANL timestamps are relative seconds from
the start of the recording. A separate optional flow reader is available, but
the current local run uses auth and red-team data only.

The baseline predicts red-team edges in a time window from aggregates built
from earlier windows. Sources are restricted to hosts already observed in
prior red-team activity. Negative destination hosts are sampled from the
observed host set.

## Current implementation

- LANL parser and windowed JSONL importer.
- Streaming next-hop example generator with SQLite-backed historical counts.
- Chronological XGBoost trainer and batched explanation export.
- Static E-GraphSAGE and temporal GraphSAGE+GRU scorer modules.
- Edge-occlusion and multi-hop path utilities.
- Probability/criticality risk scorer and CSV-based Streamlit review dashboard.

The supplied local archives were read by the importer. The completed three-day
window export contains 48,889,242 auth events, 145 windows, and 25 matched
red-team events; the JSONL file is 11.8 GB. An earlier 15-day example pass was
stopped and its partial output discarded. A new checkpointable 15-day run is
active. No model has been trained on a completed 15-day sample, and no project
metrics are available yet. The static/temporal model trainers and integrated
model-to-dashboard workflow remain future work.

## Risk ranking

For probability *p* in [0, 1] and reviewed criticality *c* in [0, 5], the
dashboard computes `100 × p × c / 5`. The dataset does not provide criticality;
the project team must supply it through an asset inventory.
