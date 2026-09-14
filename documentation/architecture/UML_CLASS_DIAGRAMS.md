# UML Class Diagrams
## THGNN System Class Structure

**Document Version:** 1.0.0  
**Last Updated:** February 9, 2026

---

## Table of Contents

1. [Data Processing Module](#data-processing-module)
2. [Model Architecture](#model-architecture)
3. [Inference Engine](#inference-engine)
4. [Explainability Module](#explainability-module)
5. [Visualization Components](#visualization-components)
6. [API Layer](#api-layer)

---

## Data Processing Module

### Class Diagram: Log Processing

```mermaid
classDiagram
    class LogProcessor {
        -str log_format
        -dict field_mappings
        -Logger logger
        +process_logs(file_path: str) DataFrame
        +validate_logs(df: DataFrame) bool
        +normalize_timestamps(df: DataFrame) DataFrame
        -_detect_format(file_path: str) str
        -_parse_json(file_path: str) DataFrame
        -_parse_csv(file_path: str) DataFrame
    }

    class UniversalLogParser {
        -dict parsers
        -dict field_schemas
        +parse(data: Any, format: str) DataFrame
        +register_parser(format: str, parser: Callable) void
        -_validate_schema(df: DataFrame) bool
    }

    class FieldNormalizer {
        -dict normalization_rules
        +normalize_field(value: Any, field_type: str) Any
        +standardize_timestamp(ts: str) datetime
        +normalize_username(user: str) str
        +normalize_hostname(host: str) str
    }

    LogProcessor --> UniversalLogParser
    LogProcessor --> FieldNormalizer
```


### Class Diagram: Entity Extraction

```mermaid
classDiagram
    class EntityExtractor {
        -dict entity_mappings
        -EntityNormalizer normalizer
        -Logger logger
        +extract_entities(df: DataFrame) dict
        +extract_users(df: DataFrame) List~User~
        +extract_hosts(df: DataFrame) List~Host~
        +extract_processes(df: DataFrame) List~Process~
        -_assign_entity_id(entity: Entity) str
    }

    class Entity {
        <<abstract>>
        +str entity_id
        +str entity_type
        +dict attributes
        +datetime first_seen
        +datetime last_seen
        +to_dict() dict
        +from_dict(data: dict) Entity
    }

    class User {
        +str username
        +str domain
        +str sid
        +List~str~ groups
        +get_full_name() str
    }

    class Host {
        +str hostname
        +str fqdn
        +str ip_address
        +str os_type
        +get_identifier() str
    }

    class Process {
        +str process_name
        +int pid
        +str command_line
        +str parent_process
        +int parent_pid
        +get_full_path() str
    }

    class EntityNormalizer {
        +normalize_user(raw_user: str) User
        +normalize_host(raw_host: str) Host
        +normalize_process(raw_proc: dict) Process
        -_extract_domain(username: str) tuple
    }

    EntityExtractor --> EntityNormalizer
    EntityExtractor --> Entity
    Entity <|-- User
    Entity <|-- Host
    Entity <|-- Process
```

### Class Diagram: Graph Construction

```mermaid
classDiagram
    class GraphBuilder {
        -dict node_types
        -dict edge_types
        -FeatureEngineer feature_engineer
        -Logger logger
        +build_graph(entities: dict, events: DataFrame) HeteroData
        +add_nodes(graph: HeteroData, entities: dict) void
        +add_edges(graph: HeteroData, events: DataFrame) void
        +validate_graph(graph: HeteroData) bool
        -_create_node_features(entity: Entity) Tensor
        -_create_edge_features(event: dict) Tensor
    }

    class HeteroData {
        +dict node_stores
        +dict edge_stores
        +dict metadata
        +num_nodes(node_type: str) int
        +num_edges(edge_type: tuple) int
        +to(device: str) HeteroData
    }

    class FeatureEngineer {
        -dict feature_configs
        +engineer_node_features(entity: Entity) Tensor
        +engineer_edge_features(event: dict) Tensor
        +engineer_temporal_features(timestamp: datetime) Tensor
        -_one_hot_encode(value: str, categories: List) Tensor
        -_normalize_numeric(value: float) float
    }

    class TemporalEdgeCreator {
        -int time_window
        -str aggregation_method
        +create_temporal_edges(events: DataFrame) List~Edge~
        +aggregate_edges(edges: List~Edge~) List~Edge~
        -_compute_time_delta(t1: datetime, t2: datetime) float
    }

    GraphBuilder --> HeteroData
    GraphBuilder --> FeatureEngineer
    GraphBuilder --> TemporalEdgeCreator
```

---

## Model Architecture

### Class Diagram: THGNN Core

```mermaid
classDiagram
    class THGNNModel {
        -int hidden_dim
        -int num_layers
        -float dropout
        -ModuleDict convs
        -TemporalEncoder temporal_encoder
        -LinkPredictor link_predictor
        +forward(x_dict: dict, edge_index_dict: dict) Tensor
        +encode(x_dict: dict, edge_index_dict: dict) dict
        +decode(z_dict: dict, edge_label_index: Tensor) Tensor
        -_init_weights() void
    }

    class HeteroConv {
        -ModuleDict convs
        -str aggr
        +forward(x_dict: dict, edge_index_dict: dict) dict
        +reset_parameters() void
    }

    class TemporalEncoder {
        -LSTM lstm
        -Linear projection
        -int hidden_dim
        -int num_layers
        +forward(x: Tensor, timestamps: Tensor) Tensor
        +encode_sequence(x: Tensor) Tensor
        -_init_hidden(batch_size: int) tuple
    }

    class TemporalAttention {
        -Linear query_proj
        -Linear key_proj
        -Linear value_proj
        -int num_heads
        +forward(x: Tensor, timestamps: Tensor) Tensor
        -_compute_attention_weights(q: Tensor, k: Tensor) Tensor
    }

    class LinkPredictor {
        -ModuleList layers
        -str scoring_function
        +forward(z_src: Tensor, z_dst: Tensor) Tensor
        +predict_links(z_dict: dict, edge_type: tuple) Tensor
        -_dot_product_score(z1: Tensor, z2: Tensor) Tensor
        -_mlp_score(z1: Tensor, z2: Tensor) Tensor
    }

    THGNNModel --> HeteroConv
    THGNNModel --> TemporalEncoder
    THGNNModel --> LinkPredictor
    TemporalEncoder --> TemporalAttention
```

### Class Diagram: Training Components

```mermaid
classDiagram
    class Trainer {
        -THGNNModel model
        -Optimizer optimizer
        -Scheduler scheduler
        -dict config
        -Logger logger
        +train(train_loader: DataLoader, val_loader: DataLoader) dict
        +train_epoch(data_loader: DataLoader) float
        +validate(data_loader: DataLoader) dict
        +save_checkpoint(path: str) void
        +load_checkpoint(path: str) void
        -_compute_loss(pred: Tensor, target: Tensor) Tensor
    }

    class Evaluator {
        -List~str~ metrics
        -dict thresholds
        +evaluate(predictions: Tensor, targets: Tensor) dict
        +compute_auc_roc(pred: Tensor, target: Tensor) float
        +compute_precision_at_k(pred: Tensor, target: Tensor, k: int) float
        +compute_recall_at_k(pred: Tensor, target: Tensor, k: int) float
        -_compute_confusion_matrix(pred: Tensor, target: Tensor) Tensor
    }

    class ModelOptimizer {
        -dict optimization_config
        +optimize_hyperparameters(model: THGNNModel, data: HeteroData) dict
        +prune_model(model: THGNNModel, amount: float) THGNNModel
        +quantize_model(model: THGNNModel) THGNNModel
        -_grid_search(param_grid: dict) dict
    }

    Trainer --> THGNNModel
    Trainer --> Evaluator
    Trainer --> ModelOptimizer
```

---

## Inference Engine

### Class Diagram: Production Inference

```mermaid
classDiagram
    class ProductionInferenceEngine {
        -THGNNModel model
        -str device
        -ResultCache cache
        -dict config
        -Logger logger
        +predict(graph: HeteroData, top_k: int) List~Prediction~
        +batch_predict(graphs: List~HeteroData~) List~List~Prediction~~
        +warm_up() void
        +get_model_info() dict
        -_preprocess_graph(graph: HeteroData) HeteroData
        -_postprocess_predictions(scores: Tensor) List~Prediction~
    }

    class Prediction {
        +str source_entity
        +str target_entity
        +str edge_type
        +float confidence
        +datetime timestamp
        +dict metadata
        +to_dict() dict
        +from_dict(data: dict) Prediction
    }

    class ResultCache {
        -Redis redis_client
        -int ttl
        +get(key: str) Any
        +set(key: str, value: Any, ttl: int) void
        +delete(key: str) void
        +clear() void
        -_serialize(obj: Any) bytes
        -_deserialize(data: bytes) Any
    }

    class BatchProcessor {
        -int batch_size
        -int num_workers
        +process_batch(items: List) List
        +create_batches(items: List) List~List~
        -_collate_fn(batch: List) Tensor
    }

    ProductionInferenceEngine --> Prediction
    ProductionInferenceEngine --> ResultCache
    ProductionInferenceEngine --> BatchProcessor
```

---

## Explainability Module

### Class Diagram: Explanation Generation

```mermaid
classDiagram
    class ExplanationGenerator {
        -GNNExplainerWrapper explainer
        -AttackChainExtractor chain_extractor
        -MITREMapper mitre_mapper
        -Logger logger
        +explain_prediction(graph: HeteroData, prediction: Prediction) Explanation
        +generate_report(explanations: List~Explanation~) dict
        -_format_explanation(raw_explanation: dict) Explanation
    }

    class Explanation {
        +Prediction prediction
        +List~str~ important_nodes
        +List~tuple~ important_edges
        +List~str~ attack_chain
        +dict mitre_techniques
        +str narrative
        +float explanation_confidence
        +to_dict() dict
    }

    class GNNExplainerWrapper {
        -GNNExplainer explainer
        -int num_hops
        +explain_node(node_idx: int, graph: HeteroData) dict
        +explain_edge(edge_idx: tuple, graph: HeteroData) dict
        -_compute_importance_scores(graph: HeteroData) Tensor
    }

    class AttackChainExtractor {
        -dict attack_patterns
        +extract_chain(graph: HeteroData, important_edges: List) List~str~
        +identify_attack_stages(chain: List~str~) List~str~
        -_match_pattern(subgraph: HeteroData) str
    }

    class MITREMapper {
        -dict technique_db
        +map_to_mitre(attack_chain: List~str~) dict
        +get_technique_info(technique_id: str) dict
        -_load_mitre_db() dict
    }

    ExplanationGenerator --> Explanation
    ExplanationGenerator --> GNNExplainerWrapper
    ExplanationGenerator --> AttackChainExtractor
    ExplanationGenerator --> MITREMapper
```

---

## Visualization Components

### Class Diagram: Visualization Engine

```mermaid
classDiagram
    class VisualizationEngine {
        -dict themes
        -dict layout_configs
        +create_graph_visualization(graph: HeteroData) Figure
        +create_timeline(events: List) Figure
        +create_risk_heatmap(predictions: List~Prediction~) Figure
        +create_attack_path(chain: List~str~) Figure
        -_apply_theme(fig: Figure) Figure
    }

    class GraphVisualizer {
        -str layout_algorithm
        -dict node_styles
        -dict edge_styles
        +visualize(graph: HeteroData) Figure
        +highlight_nodes(graph: HeteroData, nodes: List) Figure
        +highlight_edges(graph: HeteroData, edges: List) Figure
        -_compute_layout(graph: HeteroData) dict
    }

    class TimelineVisualizer {
        -str time_format
        -dict event_colors
        +create_timeline(events: DataFrame) Figure
        +add_annotations(fig: Figure, annotations: List) Figure
        -_group_events_by_time(events: DataFrame) dict
    }

    class RiskHeatmap {
        -dict risk_thresholds
        -str color_scale
        +create_heatmap(risk_scores: dict) Figure
        +update_heatmap(fig: Figure, new_scores: dict) Figure
        -_compute_risk_score(entity: str, predictions: List) float
    }

    VisualizationEngine --> GraphVisualizer
    VisualizationEngine --> TimelineVisualizer
    VisualizationEngine --> RiskHeatmap
```

---

## API Layer

### Class Diagram: REST API

```mermaid
classDiagram
    class APIServer {
        -FastAPI app
        -ProductionInferenceEngine engine
        -dict config
        +start() void
        +stop() void
        -_setup_routes() void
        -_setup_middleware() void
    }

    class PredictionEndpoint {
        +predict(request: PredictionRequest) PredictionResponse
        +batch_predict(request: BatchPredictionRequest) BatchPredictionResponse
        -_validate_request(request: Any) bool
        -_format_response(predictions: List) dict
    }

    class UploadEndpoint {
        +upload_logs(file: UploadFile) UploadResponse
        +process_logs(file_id: str) ProcessingResponse
        -_save_file(file: UploadFile) str
        -_validate_file(file: UploadFile) bool
    }

    class ExplanationEndpoint {
        +explain(request: ExplanationRequest) ExplanationResponse
        +get_attack_chain(prediction_id: str) AttackChainResponse
        -_generate_explanation(prediction: Prediction) Explanation
    }

    class AuthMiddleware {
        -dict api_keys
        +authenticate(request: Request) bool
        +authorize(user: User, resource: str) bool
        -_validate_api_key(key: str) bool
    }

    APIServer --> PredictionEndpoint
    APIServer --> UploadEndpoint
    APIServer --> ExplanationEndpoint
    APIServer --> AuthMiddleware
```

---

## Complete System Class Hierarchy

```mermaid
classDiagram
    class THGNNSystem {
        -LogProcessor log_processor
        -EntityExtractor entity_extractor
        -GraphBuilder graph_builder
        -THGNNModel model
        -ProductionInferenceEngine inference_engine
        -ExplanationGenerator explanation_generator
        -VisualizationEngine visualization_engine
        -APIServer api_server
        +initialize() void
        +process_logs(file_path: str) HeteroData
        +predict_threats(graph: HeteroData) List~Prediction~
        +explain_predictions(predictions: List~Prediction~) List~Explanation~
        +visualize_results(predictions: List, explanations: List) dict
        +start_api_server() void
    }

    THGNNSystem --> LogProcessor
    THGNNSystem --> EntityExtractor
    THGNNSystem --> GraphBuilder
    THGNNSystem --> THGNNModel
    THGNNSystem --> ProductionInferenceEngine
    THGNNSystem --> ExplanationGenerator
    THGNNSystem --> VisualizationEngine
    THGNNSystem --> APIServer
```

---

## Design Patterns Used

### 1. Factory Pattern
- **EntityFactory**: Creates different entity types
- **ParserFactory**: Creates appropriate log parsers
- **VisualizationFactory**: Creates different visualization types

### 2. Strategy Pattern
- **ScoringStrategy**: Different link prediction scoring functions
- **AggregationStrategy**: Different edge aggregation methods
- **LayoutStrategy**: Different graph layout algorithms

### 3. Observer Pattern
- **ModelObserver**: Monitors training progress
- **CacheObserver**: Tracks cache hit/miss rates
- **MetricsObserver**: Collects performance metrics

### 4. Singleton Pattern
- **ConfigManager**: Global configuration access
- **Logger**: Centralized logging
- **CacheManager**: Shared cache instance

### 5. Builder Pattern
- **GraphBuilder**: Constructs complex graph structures
- **ExplanationBuilder**: Builds detailed explanations
- **ReportBuilder**: Generates comprehensive reports

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-09 | Development Team | Initial release |
