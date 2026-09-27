# Quick start: LANL baseline

## 1. Prepare inputs

Place `auth.txt.gz` and `redteam.txt.gz` in `data/`. These are large local
archives and are ignored by Git. The current baseline does not require a flow
archive.

## 2. Generate candidate examples

Run from the repository root and wait for successful completion:

```powershell
python -m src.data_processing.lanl_examples --auth data/auth.txt.gz --redteam data/redteam.txt.gz --output data/processed/next_hop_examples_15d.jsonl --days 15 --window-minutes 15
```

The auth archive is processed sequentially. A partial output is not valid
training input. The builder saves a persistent checkpoint after each window.
If interrupted, rerun the same command with `--resume`. It must reread the gzip
archive from the beginning to reach the saved window, then it continues from
the last committed checkpoint.

## 3. Train and score the baseline

```powershell
python scripts/train_xgboost.py --input data/processed/next_hop_examples_15d.jsonl
python scripts/predict_next_hops.py --input data/processed/next_hop_examples_15d.jsonl --output outputs/predictions.csv
```

Metrics are written under `outputs/xgboost/`. Verify the split contains
positive labels before interpreting metrics.

## 4. Review alerts

```powershell
streamlit run app/risk_dashboard.py
```

Upload `outputs/predictions.csv` and an asset inventory CSV with
`host,criticality` columns. Supply reviewed criticality values from 0 to 5.

The static and temporal GNN modules are not trained by these commands. See the
[project alignment](../../docs/PROJECT_ALIGNMENT.md) for unfinished work.
