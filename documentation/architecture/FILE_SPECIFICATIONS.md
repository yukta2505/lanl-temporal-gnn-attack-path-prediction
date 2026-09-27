# Data and artifact formats

## LANL inputs

- `auth.txt[.gz]`: headerless comma-separated rows with time, source user,
  destination user, source computer, destination computer, authentication
  type, logon type, orientation, and success/failure.
- `redteam.txt[.gz]`: headerless comma-separated rows with time, user, source
  computer, and destination computer.
- `flows.txt[.gz]` is optional and is not used in the current baseline run.

The repository stores the raw archives in `data/`; Git ignores the raw files
and generated `data/processed/` artifacts.

## Windowed event JSONL

`src.data_processing.lanl_dataset` writes one JSON object per non-overlapping
window. Each object has `window_start`, `window_minutes`, and an `events` list.
Timestamps remain relative seconds from the dataset start.

## Next-hop example JSONL

The streaming generator writes one candidate pair per line:

```json
{"window_start":151200,"source_host":"C17693","target_host":"C1173","label":1,"features":[15.0,2043.0,0.0,-1.0,0.0]}
```

The feature order is source historical out-events, target historical in-events,
historical pair-event count, seconds since the pair was last seen (`-1` if
unseen), and authentication success fraction. Features use earlier windows;
the label is the red-team outcome for the prediction window.

## Model and prediction artifacts

- `outputs/xgboost/xgboost_next_hop.joblib`: model plus feature-name bundle.
- `outputs/xgboost/metrics.json`: chronological split sizes, holdout metrics,
  and feature importance after training completes.
- `outputs/predictions.csv`: window, source, target, probability, and concise
  feature-contribution explanation.
- Dashboard asset inventory: CSV columns `host,criticality`; criticality is a
  reviewed value from 0 through 5.
