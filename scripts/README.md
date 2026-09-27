# Project scripts

## Train the XGBoost baseline

Once the complete LANL example JSONL has been built, run:

```powershell
python scripts/train_xgboost.py --input data/processed/next_hop_examples_15d.jsonl --output-dir outputs/xgboost
```

The trainer uses chronological train, validation, and test splits and writes the model bundle and `metrics.json`. Check positive-label counts in every split before interpreting metrics. The completed 15-day run had 482 positives overall and 53 test positives; the initial XGBoost test PR-AUC was 0.600 and F1 was 0.589. This is a baseline for the current red-team-conditioned, sampled-candidate task.

## Export candidate predictions

```powershell
python scripts/predict_next_hops.py --model outputs/xgboost/xgboost_next_hop.joblib --input data/processed/next_hop_examples_15d.jsonl --output outputs/predictions.csv
```

The CSV includes probabilities and local XGBoost feature contributions. The current Streamlit dashboard accepts this CSV and a reviewed asset inventory CSV.

## GNN status

The E-GraphSAGE and temporal GraphSAGE+GRU modules are model components only. Snapshot data construction, trainers, integrated explanations, and evaluation are still required.
