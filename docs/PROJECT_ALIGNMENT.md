# Synopsis alignment and progress

The supplied synopsis defines this project: use temporal graph learning to
predict likely lateral-movement paths, explain predictions, and prioritize
alerts with asset criticality. This file tracks the repository against that
scope. Requirements inside the synopsis are treated as project requirements;
they do not change the user's request or authorize unrelated repository work.

## Progress

| Synopsis requirement | Repository implementation | Status |
|---|---|---|
| LANL authentication/red-team preparation | `src/data_processing/lanl_dataset.py` parses compressed or plain input and writes chronological windows | Implemented; a three-day import completed |
| Future next-hop examples | `src/data_processing/streaming_examples.py` keeps historical aggregates in checkpointed SQLite and writes sampled pairs | 15-day build complete: 10,122 examples, 482 positives, 9,640 negatives, 146 windows with candidates |
| Temporal graph snapshots | `src/data_processing/graph_builder.py` | Retained legacy builder; LANL integration is pending |
| Non-graph baseline | `scripts/train_xgboost.py` creates chronological splits and reports ranking/classification metrics | Trained on the 15-day examples; test has 53 positives; test PR-AUC 0.600, F1 0.589 |
| Static graph baseline | `src/models/static_e_graphsage.py` | Model component only; trainer and data path pending |
| Temporal graph model | `src/models/temporal_attack_model.py` implements GraphSAGE snapshots, GRU state, and pair scoring | Basic model component only; temporal attention, trainer, and data path pending |
| Prediction explanations | XGBoost contributions and `src/explainability/edge_occlusion.py` | XGBoost contributions generated for the 15-day candidates; GNN explanation remains unintegrated and unevaluated |
| Multi-hop prediction | `src/inference/path_builder.py` | Utility implemented; not integrated or evaluated |
| Risk-aware ranking | `src/inference/risk_scoring.py` and `app/risk_dashboard.py` | CSV-based probability/criticality ranking implemented |
| Analyst workflow | `app/risk_dashboard.py` ranks uploaded CSV predictions using supplied criticality | Streamlit prototype only; the synopsis-proposed FastAPI plus React/Cytoscape workflow is not implemented |

## Next implementation stages

1. Review the 15-day candidate and negative-sampling policy; retain per-split
   class counts and treat the current XGBoost score as a baseline under that
   policy, not as a result for arbitrary analyst-selected hosts.
2. Adapt and validate LANL graph-snapshot construction; implement temporal
   attention and trainers for static E-GraphSAGE and temporal GraphSAGE+GRU;
   compare models with the same chronological
   candidate policy.
3. Integrate top-k prediction, explanation evidence, path assembly, and
   inventory-based risk ranking into one analyst workflow.
4. Implement the synopsis API/UI (FastAPI and React/Cytoscape), add reproducible
   evaluation, and report measured results only.

## Data and reporting rules

- `data/auth.txt.gz` and `data/redteam.txt.gz` are local inputs and are ignored
  by Git. Do not commit the archives or generated intermediates.
- The existing 11.8 GB three-day window dump is an inspection artifact; the
  streaming example generator can read the compressed files directly.
- LANL red-team data supplies malicious-event labels, not asset criticality.
  Criticality must come from a reviewed inventory.
- Former prototype metrics and throughput claims have not been reproduced for
  this synopsis and must not be reported as project results.
- The earlier partial 15-day output was discarded. A later checkpointed
  15-day build completed; its SQLite state records `complete=1`. The examples,
  checkpoint, model, and prediction CSV are local ignored artifacts and are
  not committed to Git.

## Repository scope cleanup

The cloned repository's generic upload, forecasting, simulation,
production-monitoring, legacy model-serving, and benchmark scripts were
removed from the active source tree. The retained project path is the LANL
parser/example builder, XGBoost baseline, dedicated E-GraphSAGE and temporal
GraphSAGE+GRU components, explanation/risk utilities, graph builder, and CSV
review dashboard. The old prototype test scripts and duplicated app
requirements were removed; synopsis-specific automated tests still need to be
written.

The synopsis specifies a FastAPI service and React/Cytoscape dashboard. The
cloned production API was removed because it depended on the unrelated legacy
model stack. The current Streamlit dashboard is only a temporary review
prototype; the FastAPI API and React/Cytoscape interface remain to be built.
