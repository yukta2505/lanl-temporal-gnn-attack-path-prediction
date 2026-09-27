# Temporal Graph Learning for Lateral-Movement Prediction

This repository is being adapted to the project synopsis: predict likely next
host-to-host lateral-movement steps from LANL authentication activity, explain
candidate predictions, and rank alerts using supplied asset criticality.

## Current progress

- Added LANL auth/red-team parsing and time-windowed event import.
- Successfully generated a three-day windowed import from the local archives.
- That import contains 48,889,242 auth events across 145 windows and produced
  an 11.8 GB JSONL inspection file under `data/processed/`.
- Added a streaming, disk-backed next-hop example generator with per-window
  checkpoints. The 15-day build completed with 10,122 examples across 146
  labeled windows: 482 positives and 9,640 sampled negatives.
- Added a chronological XGBoost baseline trainer, metrics, and batched
  prediction export with local feature-contribution explanations.
- Trained the baseline once on the existing three-day example file and
  generated a prediction CSV. This is a pipeline check, not a valid final
  evaluation: validation had only five positives and the test split had none.
- Added reusable static E-GraphSAGE and GraphSAGE+GRU model components, edge
  occlusion, multi-hop path assembly, risk scoring, and a CSV-based dashboard.
- Removed the cloned generic upload/forecast/simulation application, legacy
  model-serving stack, duplicated app dependencies, and old prototype test
  scripts. Kept synopsis-related graph construction and explainability code
  for integration work.
- The 15-day XGBoost baseline is trained and scored. Its test PR-AUC is 0.600
  and F1 is 0.589 under the current red-team-conditioned candidate policy;
  this does not establish performance for arbitrary compromised hosts. Static
  and temporal GNN training, the FastAPI service, and React/Cytoscape dashboard
  remain to be implemented. Streamlit is currently a CSV review prototype.

The synopsis mentions both authentication and network-flow logs. The current
local pipeline uses authentication events and red-team labels only; no flow
archive is included in the current run.

The local `auth.txt.gz` and `redteam.txt.gz` files are excluded from Git.
Generated files under `data/processed/` and model outputs are also ignored.

## Install the project dependencies

In PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Build the LANL candidate examples

The following command reads the compressed archives directly, keeping history
in a persistent SQLite checkpoint and writing sampled candidate pairs. It may
take substantial time because the auth archive is processed sequentially.
Do not train on the existing 15-day output unless its build has completed.

```powershell
python -m src.data_processing.lanl_examples `
  --auth data/auth.txt.gz `
  --redteam data/redteam.txt.gz `
  --output data/processed/next_hop_examples_15d.jsonl `
  --days 15 `
  --window-minutes 15
```

If interrupted, resume with the same paths and options plus `--resume`. Gzip
input is reread from the beginning because it cannot seek to the checkpoint;
completed windows and their output rows are reused. The default checkpoint is
`data/processed/next_hop_examples_15d.jsonl.state/`.

The separate `src.data_processing.lanl_dataset` command writes complete
windowed event records for inspection. The example generator does not require
that large intermediate file. LANL timestamps are relative seconds from the
recording start.

## Train, score, and review

```powershell
python scripts/train_xgboost.py --input data/processed/next_hop_examples_15d.jsonl
python scripts/predict_next_hops.py --input data/processed/next_hop_examples_15d.jsonl
streamlit run app/risk_dashboard.py
```

For an initial baseline run, the existing three-day examples can be used with
`--input data/processed/next_hop_examples_3d.jsonl`; its sparse positive labels
make the resulting metrics unsuitable as final results.

The dashboard takes the prediction CSV and a reviewed asset inventory CSV with
`host,criticality` columns. Criticality must be on a 0–5 scale. The score is
`100 × probability × criticality / 5`; missing criticality remains unranked.

## Project documentation

- [Project report](THGNN_Project_Report.md): synopsis scope, implementation
  status, evaluation plan, and unreported results.
- [Alignment and remaining work](docs/PROJECT_ALIGNMENT.md): maps the synopsis
  to the repository and identifies unfinished deliverables.
- [Documentation index](documentation/README.md): project-specific guides and
  technical notes.
