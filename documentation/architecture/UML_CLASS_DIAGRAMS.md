# Current model and inference components

The diagram lists implemented components only. It does not imply that the GNN
models have been trained or integrated.

```mermaid
classDiagram
    class LANLImporter {
        +build_windows(auth, redteam, output, window_minutes, max_days)
    }
    class StreamingExamples {
        +generate(auth, redteam, output, flows, days, window_minutes)
    }
    class XGBoostTrainer {
        +train(input_path, output_dir, seed)
    }
    class PredictionExporter {
        +predict(model_path, examples_path, output_path, batch_size)
    }
    class StaticEGraphSAGE {
        +forward(node_features, edge_index, candidates, edge_features)
    }
    class TemporalAttackPathModel {
        +encode(node_features, edge_indices)
        +forward(node_features, edge_indices, candidates)
    }
    class RiskScoring {
        +score_risk(probability, criticality, max_criticality)
    }
    class PathBuilder {
        +build_paths(predictions, start_host, max_hops, beam_width)
    }
    StreamingExamples --> LANLImporter : shares event readers
    PredictionExporter --> XGBoostTrainer : loads model bundle
    StaticEGraphSAGE ..> StreamingExamples : future graph-data integration
    TemporalAttackPathModel ..> StreamingExamples : future graph-data integration
    RiskScoring --> PathBuilder : risk can rank path edges
```

Source files are listed in [the system architecture](SYSTEM_ARCHITECTURE.md)
and [synopsis alignment](../../docs/PROJECT_ALIGNMENT.md). The current
dashboard consumes CSV files and is not represented as calling model inference.
