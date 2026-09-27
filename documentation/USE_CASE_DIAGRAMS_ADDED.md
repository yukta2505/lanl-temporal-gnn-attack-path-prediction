# Project use cases

These use cases are limited to the synopsis project and the current prototype
workflow. They describe intended analyst and model-development tasks; they do
not claim the incomplete workflows are already operational.

## Implemented or partially implemented

| User | Use case | Current state |
|---|---|---|
| Data engineer | Prepare LANL auth and red-team events | Parser and window importer implemented; three-day import completed |
| Data engineer | Generate next-hop candidate examples | Streaming builder implemented; 15-day run not completed |
| Model developer | Train and evaluate XGBoost baseline | Trainer implemented; no full-sample run yet |
| Analyst | Review ranked predictions with host criticality | CSV-based dashboard implemented |
| Analyst | Filter predictions by source host | Dashboard filter implemented; predictions must already exist |

## Planned

- Train and compare static E-GraphSAGE and temporal GraphSAGE+GRU models.
- Start inference from an analyst-selected compromised host.
- Show edge-based explanations and multi-hop paths in the dashboard.
- Evaluate next-hop and path ranking on a chronological holdout.

See [the use-case diagram](diagrams/USE_CASE_DIAGRAMS.md) for the compact
workflow and [synopsis alignment](../docs/PROJECT_ALIGNMENT.md) for the full
status list. Previous claims about dozens of roles, production APIs, or
performance targets do not describe this project's current scope.
