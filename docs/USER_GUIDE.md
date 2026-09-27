# Analyst guide

## Current workflow

The working review app is `app/risk_dashboard.py`. It accepts two CSV files:

1. **Predictions:** `source_host`, `target_host`, `probability`; `explanation`
   is optional.
2. **Asset inventory:** `host`, `criticality`, with criticality from 0 to 5.

The dashboard lets an analyst filter by source host, ranks candidates with
known criticality, and displays unranked candidates when inventory values are
missing or invalid. Its current risk formula is `100 × probability × criticality / 5`.

Start it from the repository root:

```powershell
streamlit run app/risk_dashboard.py
```

## Generate predictions

Build candidates from the local LANL archives, train the current XGBoost
baseline, and export candidate scores:

```powershell
python -m src.data_processing.lanl_examples --auth data/auth.txt.gz --redteam data/redteam.txt.gz --output data/processed/next_hop_examples_15d.jsonl --days 15 --window-minutes 15
python scripts/train_xgboost.py --input data/processed/next_hop_examples_15d.jsonl
python scripts/predict_next_hops.py --input data/processed/next_hop_examples_15d.jsonl --output outputs/predictions.csv
```

The full data pass reads the auth archive sequentially and can take a long
time. Do not train on a partial output file. This baseline's candidates are
conditioned on prior red-team compromise labels; it is not yet a live model
that accepts an arbitrary analyst-selected host.

## Limits

The dashboard consumes CSVs rather than calling a trained model. Prediction
scores and explanations have not yet been evaluated on the completed 15-day
sample. Asset criticality is an external inventory value, not a LANL label.
