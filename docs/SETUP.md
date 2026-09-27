# Setup for the LANL attack-path project

## Requirements

Use Python with the packages listed in the repository's `requirements.txt`.
The model code uses PyTorch and PyTorch Geometric; the current baseline uses
XGBoost, scikit-learn, NumPy, and joblib. The review interface uses Streamlit
and pandas.

Install from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Local data

Place the LANL compressed files in the repository's `data/` directory:

```text
data/auth.txt.gz
data/redteam.txt.gz
```

These archives and generated data are excluded from Git. No flow archive is
required for the current auth-only baseline. The importer accepts plain text
files as well as `.gz` files.

## Run

From the repository root, build the candidate examples and train the baseline:

```powershell
.\.venv\Scripts\python.exe -m src.data_processing.lanl_examples --auth data/auth.txt.gz --redteam data/redteam.txt.gz --output data/processed/next_hop_examples_15d.jsonl --days 15 --window-minutes 15
.\.venv\Scripts\python.exe scripts/train_xgboost.py --input data/processed/next_hop_examples_15d.jsonl
```

The example builder streams the large auth file and stores aggregate history
in a persistent SQLite checkpoint beside the output. Allow sufficient free disk
space for intermediate aggregates and do not use the output until the command
exits successfully. If interrupted, rerun the same command with `--resume`;
gzip data must be reread from the beginning to reach the saved window.

To open the analyst review interface:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/risk_dashboard.py
```

See [the analyst guide](USER_GUIDE.md) and [synopsis alignment](PROJECT_ALIGNMENT.md)
for input formats and implementation limits.
