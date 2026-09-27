# Temporal Graph Neural Networks for Explainable, Risk-Aware Attack-Path Prediction

## Project report draft

**Programme:** B. Tech in CSE (AI & ML)
**Institution:** Shri Ramdeobaba College of Engineering and Management, Nagpur
**Project group:** Yukta Baid, Roshni Chawla, Saakshi Sharma, Yashaswini Annam, Apoorva Patel
**Guide:** Dr. Nisarg Gandhewar
**Session:** 2026–27

> This draft follows the supplied synopsis. A three-day LANL import and one
> XGBoost baseline run have completed. The evaluation is inconclusive: the
> validation split has only five positive examples and the test split has
> none. The local 15-day example output is not verified complete. No valid
> comparative model results are available yet.

## Abstract

Lateral movement is a sequence of host-to-host actions. This project is
developing a temporal graph pipeline that uses LANL authentication activity to
predict likely next destinations from a previously compromised host. The
planned system explains predictions with graph evidence and ranks alerts using
attack probability and an externally supplied asset-criticality value. The
repository currently contains LANL preprocessing, a streaming example
generator, an XGBoost baseline path, reusable static and temporal graph-model
components, explanation and path utilities, and a risk-review dashboard. The
static and temporal GNN trainers and measured model comparisons remain
outstanding. The current Streamlit dashboard is a prototype; the synopsis
proposes a FastAPI backend and React/Cytoscape interface.

## 1. Problem and objectives

Given authentication history and a compromised source host, predict the most
likely next lateral-movement destinations in a future time window. For each
candidate, the system should provide supporting evidence and prioritize the
alert using the destination asset's reviewed business criticality.

The project objectives are to:

1. Convert LANL authentication events into chronological host-interaction
   graphs and align red-team labels.
2. Predict ranked next-hop candidates and assemble candidate multi-hop paths.
3. Compare a feature-based baseline, a static graph model, and a temporal graph
   model using chronological evaluation.
4. Explain predictions using model features or influential historical edges.
5. Combine attack probability with an explicit asset inventory value.
6. Present ranked alerts to an analyst.

## 2. Data and task definition

The local input archives are `data/auth.txt.gz` and `data/redteam.txt.gz`.
LANL auth rows contain relative time, source and destination users, source and
destination computers, authentication attributes, and result. Red-team rows
identify malicious authentication edges by time, user, source computer, and
destination computer. The importer can optionally read network-flow context;
no flow archive is currently used in this project run.

Events are grouped into non-overlapping time windows. Candidate features use
history before the prediction window; labels identify red-team edges in the
window being predicted. The baseline generator conditions candidates on hosts
that appeared as a source or destination in an earlier red-team event and
samples non-labelled destination hosts as negatives. The candidate-host
universe and negative-sampling policy must be reported with final results.

## 3. Proposed system

### 3.1 Graph representation

Hosts are graph nodes. Authentication events are directed, time-stamped edges
from source computer to destination computer. Optional flow records can add
context. LANL time values are elapsed seconds from the recording start, not
calendar timestamps. Asset criticality is separate metadata and is not
inferred from red-team labels.

### 3.2 Models

- **XGBoost baseline:** historical source/target activity, pair count, pair
  recency, and authentication success rate for each candidate edge.
- **Static E-GraphSAGE:** reusable model component for scoring candidate pairs
  over one graph snapshot; its trainer and data integration are pending.
- **Temporal GraphSAGE + GRU:** reusable snapshot encoder and temporal pair
  scorer; temporal attention, its trainer, and data integration are pending.

### 3.3 Explanations and paths

The XGBoost prediction script exports local feature contributions. A bounded
edge-occlusion utility estimates how removing historical edges changes a
temporal model score. A path utility assembles cycle-free multi-hop paths from
ranked next-hop probabilities. These utilities have not yet been evaluated for
stability, runtime, or analyst usefulness.

### 3.4 Risk score

For attack probability *p* in [0, 1] and supplied criticality *c* in [0, 5], the
current dashboard formula is:

**Risk = 100 × p × c / 5**

This score ranks alerts on a 0–100 scale. The dashboard leaves missing or
invalid criticality unranked rather than assigning a default.

## 4. Implementation status

| Component | Status |
|---|---|
| LANL auth/red-team parser and window importer | Implemented; three-day import completed: 48,889,242 auth events, 145 windows, and 25 matched red-team events; 11.8 GB JSONL |
| Streaming next-hop example builder | Implemented with per-window checkpointing; local 15-day output not verified complete |
| Chronological XGBoost trainer and metrics | Trained on three-day examples; validation has 5 positives and test has 0, so evaluation is inconclusive |
| Batched prediction export and local feature explanations | Scored 30,594 three-day examples; output includes training and holdout periods and is not a holdout-only analysis |
| Static E-GraphSAGE scorer | Model component implemented; trainer/data integration pending |
| Temporal GraphSAGE+GRU scorer | Basic model component implemented; temporal attention and trainer/data integration pending |
| Temporal edge-occlusion explanation | Utility implemented; not evaluated |
| Multi-hop path assembly | Utility implemented; not evaluated |
| Probability/criticality risk scoring | Implemented |
| Streamlit alert review dashboard | Prototype accepts prediction and inventory CSV inputs; source-host filter available |
| Synopsis FastAPI + React/Cytoscape analyst interface | Pending; not yet integrated with model inference |

## 5. Evaluation plan and results

Use chronological train, validation, and test periods. Report examples,
positive red-team hops, candidate edges, and windows per split. Report
precision, recall, F1, ROC-AUC, PR-AUC, and Precision@10 for next-hop ranking.
For path prediction, define a path metric and maximum hop count. Select
thresholds on validation data only. Record explanation runtime and include
representative evidence. No values should be copied from the former prototype.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Precision@10 |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost, three-day trial | N/A | N/A | N/A | N/A | N/A | N/A |
| Static E-GraphSAGE | Pending | Pending | Pending | Pending | Pending | Pending |
| Temporal GraphSAGE + GRU | Pending | Pending | Pending | Pending | Pending | Pending |

## 6. Run the current baseline path

Place the two compressed archives under `data/`, then run:

```powershell
python -m src.data_processing.lanl_examples --auth data/auth.txt.gz --redteam data/redteam.txt.gz --output data/processed/next_hop_examples_15d.jsonl --days 15 --window-minutes 15
python scripts/train_xgboost.py --input data/processed/next_hop_examples_15d.jsonl
python scripts/predict_next_hops.py --input data/processed/next_hop_examples_15d.jsonl
streamlit run app/risk_dashboard.py
```

The dashboard accepts the generated prediction CSV and a reviewed inventory
CSV with `host,criticality` columns.

## 7. Remaining work and limitations

The completed three-day import establishes that the compressed auth/red-team
inputs can be read by the importer. The first XGBoost run establishes that the
training and scoring scripts execute, but its sparse positive labels make the
metrics unusable as evidence of predictive performance. Verify or rebuild the
15-day examples before final baseline evaluation.
The static and temporal GNN trainers, integrated source-host inference,
end-to-end dashboard scoring, graph-model comparison, and path-level evaluation
remain to be completed. Current labels condition examples on earlier red-team
activity, which is a research-task assumption that must be documented when
comparing results with an analyst-selected compromised-host workflow.

## References

1. A. D. Kent, “Comprehensive, Multi-Source Cyber-Security Events,” Los Alamos
   National Laboratory, 2015, doi: 10.17021/1179829.
2. Supplied project synopsis, “Temporal Graph Neural Networks for Explainable,
   Risk-Aware Attack-Path Prediction in Enterprise Networks,” 2026–27.
