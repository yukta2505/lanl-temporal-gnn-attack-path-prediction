# Project scripts

## Train the XGBoost baseline

Once the complete LANL example JSONL has been built, run:

```powershell
python scripts/train_xgboost.py --input data/processed/next_hop_examples_15d.jsonl --output-dir outputs/xgboost
```

The trainer uses chronological train, validation, and test splits and writes the model bundle and `metrics.json`. Check positive-label counts in every split before interpreting metrics. The existing three-day trial had only five validation positives and no test positives, so it is not a valid performance result.

## Export candidate predictions

```powershell
python scripts/predict_next_hops.py --model outputs/xgboost/xgboost_next_hop.joblib --input data/processed/next_hop_examples_15d.jsonl --output outputs/predictions.csv
```

The CSV includes probabilities and local XGBoost feature contributions. The current Streamlit dashboard accepts this CSV and a reviewed asset inventory CSV.

## GNN status

The E-GraphSAGE and temporal GraphSAGE+GRU modules are model components only. Snapshot data construction, trainers, integrated explanations, and evaluation are still required.
