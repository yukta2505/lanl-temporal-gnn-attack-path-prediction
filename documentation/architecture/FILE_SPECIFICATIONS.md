# File Specifications
## Complete THGNN Codebase Reference

**Document Version:** 1.0.0  
**Last Updated:** February 9, 2026

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Core Modules](#core-modules)
3. [Data Processing](#data-processing)
4. [Model Architecture](#model-architecture)
5. [Inference Engine](#inference-engine)
6. [Visualization](#visualization)
7. [Configuration Files](#configuration-files)
8. [Scripts and Utilities](#scripts-and-utilities)

---

## Project Structure

```
THGNN/
├── src/                          # Source code
│   ├── data_processing/          # Log parsing and graph building
│   ├── models/                   # Neural network architectures
│   ├── training/                 # Training pipeline
│   ├── inference/                # Production inference
│   ├── explainability/           # Explanation generation
│   ├── visualization/            # Plotting and dashboards
│   ├── simulation/               # Attack simulation
│   ├── monitoring/               # System monitoring
│   └── utils/                    # Utility functions
├── app/                          # Streamlit dashboard
├── scripts/                      # Standalone scripts
├── config/                       # Configuration files
├── data/                         # Data storage
├── checkpoints/                  # Model checkpoints
├── logs/                         # System logs
├── tests/                        # Test suites
├── documentation/                # This documentation
├── notebooks/                    # Jupyter notebooks
└── docker/                       # Docker configurations
```

---

## Core Modules

### src/data_processing/

#### log_processor.py
**Purpose**: Parse and normalize security logs from multiple formats

**Key Classes**:
- `LogProcessor`: Main orchestrator for log processing
- `UniversalLogParser`: Multi-format parser
- `FieldNormalizer`: Standardize field names and values

**Key Functions**:
```python
process_logs(file_path: str) -> pd.DataFrame
validate_logs(df: pd.DataFrame) -> bool
normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame
```

**Input**: Raw log files (JSON, CSV, EVTX)
**Output**: Normalized pandas DataFrame

**Dependencies**: pandas, json, xml.etree, dateutil

---

#### entity_extractor.py
**Purpose**: Extract and normalize security entities from logs

**Key Classes**:
- `EntityExtractor`: Main extraction engine
- `Entity`: Base entity class
- `User`, `Host`, `Process`: Specific entity types

**Key Functions**:
```python
extract_entities(df: pd.DataFrame) -> Dict[str, List[Entity]]
extract_users(df: pd.DataFrame) -> List[User]
extract_hosts(df: pd.DataFrame) -> List[Host]
extract_processes(df: pd.DataFrame) -> List[Process]
```

**Entity Types**:
- User: username, domain, SID, groups
- Host: hostname, FQDN, IP, OS
- Process: name, PID, command line, parent

**Output**: Dictionary of entity lists by type

---

#### graph_builder.py
**Purpose**: Construct temporal heterogeneous graphs from entities

**Key Classes**:
- `GraphBuilder`: Main graph construction
- `FeatureEngineer`: Node/edge feature creation
- `TemporalEdgeCreator`: Time-aware edge generation

**Key Functions**:
```python
build_graph(entities: dict, events: pd.DataFrame) -> HeteroData
add_nodes(graph: HeteroData, entities: dict) -> None
add_edges(graph: HeteroData, events: pd.DataFrame) -> None
validate_graph(graph: HeteroData) -> bool
```

**Graph Structure**:
```python
HeteroData(
    user={'x': Tensor, 'node_id': Tensor},
    host={'x': Tensor, 'node_id': Tensor},
    process={'x': Tensor, 'node_id': Tensor},
    ('user', 'logs_into', 'host')={'edge_index': Tensor, 'edge_attr': Tensor},
    ('user', 'executes', 'process')={'edge_index': Tensor, 'edge_attr': Tensor}
)
```

---

### src/models/

#### thgnn_core.py
**Purpose**: Core THGNN model architecture

**Key Classes**:
- `THGNNModel`: Main model class
- `HeteroConv`: Heterogeneous graph convolution
- `TemporalEncoder`: LSTM-based temporal modeling
- `LinkPredictor`: Edge prediction head

**Architecture**:
```python
class THGNNModel(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, dropout=0.3):
        # Node embeddings
        self.node_embeddings = nn.ModuleDict({...})
        
        # Graph convolution layers
        self.convs = nn.ModuleList([HeteroConv(...) for _ in range(num_layers)])
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(hidden_dim)
        
        # Link predictor
        self.link_predictor = LinkPredictor(hidden_dim)
    
    def forward(self, x_dict, edge_index_dict, timestamps):
        # Encode nodes
        z_dict = self.encode(x_dict, edge_index_dict)
        
        # Temporal modeling
        z_dict = self.temporal_encoder(z_dict, timestamps)
        
        # Link prediction
        scores = self.link_predictor(z_dict)
        
        return scores
```

**Hyperparameters**:
- hidden_dim: 128
- num_layers: 2-3
- dropout: 0.3
- num_heads: 4 (attention)
- temporal_window: 3600s

---

#### temporal_encoder.py
**Purpose**: Temporal sequence modeling

**Key Classes**:
- `TemporalEncoder`: LSTM-based encoder
- `TemporalAttention`: Time-aware attention
- `TimeEmbedding`: Timestamp encoding

**Key Functions**:
```python
forward(x: Tensor, timestamps: Tensor) -> Tensor
encode_sequence(x: Tensor) -> Tensor
```

**Architecture**:
```
Input → Time Embedding → LSTM → Attention → Output
```

---

#### link_predictor.py
**Purpose**: Predict future edges in graph

**Key Classes**:
- `LinkPredictor`: Main predictor
- `ScoringFunction`: Edge scoring methods

**Scoring Functions**:
- Dot Product: `score = z_src^T · z_dst`
- MLP: `score = MLP([z_src || z_dst])`
- DistMult: `score = z_src^T · R · z_dst`

---

### src/inference/

#### production_engine.py
**Purpose**: Real-time inference for production deployment

**Key Classes**:
- `ProductionInferenceEngine`: Main inference orchestrator
- `BatchProcessor`: Batch optimization
- `ResultCache`: Redis-based caching

**Key Functions**:
```python
predict(graph: HeteroData, top_k: int = 10) -> List[Prediction]
batch_predict(graphs: List[HeteroData]) -> List[List[Prediction]]
warm_up() -> None
```

**Performance Optimizations**:
- Model warm-up and preloading
- GPU memory management
- Result caching (Redis)
- Batch processing
- Mixed precision inference

**API Example**:
```python
engine = ProductionInferenceEngine(
    model_path='checkpoints/best_model.pth',
    device='cuda',
    cache_ttl=300
)

predictions = engine.predict(graph, top_k=10)
```

---

### src/explainability/

#### explanation_generator.py
**Purpose**: Generate human-interpretable explanations

**Key Classes**:
- `ExplanationGenerator`: Main orchestrator
- `GNNExplainerWrapper`: GNNExplainer integration
- `AttackChainExtractor`: Attack sequence reconstruction
- `MITREMapper`: ATT&CK technique mapping

**Key Functions**:
```python
explain_prediction(graph: HeteroData, prediction: Prediction) -> Explanation
generate_report(explanations: List[Explanation]) -> dict
```

**Output Structure**:
```python
{
    'prediction': 'user_executes_process',
    'confidence': 0.87,
    'important_nodes': ['user_123', 'process_456'],
    'important_edges': [('user_123', 'host_789')],
    'attack_chain': ['T1078', 'T1059.001', 'T1021.002'],
    'mitre_techniques': {...},
    'narrative': 'Detected lateral movement pattern...'
}
```

---

### src/visualization/

#### graph_visualizer.py
**Purpose**: Interactive graph visualization

**Key Classes**:
- `GraphVisualizer`: Main visualizer
- `LayoutEngine`: Graph layout algorithms
- `StyleManager`: Visual styling

**Key Functions**:
```python
visualize(graph: HeteroData) -> plotly.Figure
highlight_nodes(graph: HeteroData, nodes: List[str]) -> plotly.Figure
highlight_edges(graph: HeteroData, edges: List[tuple]) -> plotly.Figure
```

**Layout Algorithms**:
- Force-directed (Fruchterman-Reingold)
- Hierarchical (Sugiyama)
- Circular
- Custom security-focused layouts

---

#### timeline_visualizer.py
**Purpose**: Temporal event visualization

**Key Functions**:
```python
create_timeline(events: pd.DataFrame) -> plotly.Figure
add_annotations(fig: plotly.Figure, annotations: List) -> plotly.Figure
```

**Features**:
- Interactive timeline
- Event clustering
- Zoom and pan
- Attack stage highlighting

---

#### risk_heatmap.py
**Purpose**: Risk scoring visualization

**Key Functions**:
```python
create_heatmap(risk_scores: dict) -> plotly.Figure
update_heatmap(fig: plotly.Figure, new_scores: dict) -> plotly.Figure
```

**Visualization Types**:
- Entity risk heatmap
- Temporal risk evolution
- Network risk distribution

---

## Configuration Files

### config/model_config.yaml
**Purpose**: Model hyperparameters and architecture settings

```yaml
model:
  hidden_dim: 128
  num_layers: 2
  dropout: 0.3
  num_heads: 4
  
temporal:
  window_size: 3600
  encoding_dim: 32
  lstm_layers: 2
  
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10
  
inference:
  top_k: 10
  confidence_threshold: 0.7
  batch_size: 64
```

---

### config/field_mappings.yaml
**Purpose**: Log field normalization mappings

```yaml
windows_event:
  timestamp: TimeCreated
  user: SubjectUserName
  host: Computer
  event_id: EventID
  
sysmon:
  timestamp: UtcTime
  user: User
  host: Computer
  process: Image
  
custom:
  timestamp: @timestamp
  user: user.name
  host: host.name
```

---

## Scripts and Utilities

### scripts/thgnn_train.py
**Purpose**: Model training script

**Usage**:
```bash
python scripts/thgnn_train.py \
    --data data/processed/graphs/ \
    --config config/model_config.yaml \
    --output checkpoints/ \
    --gpu 0
```

---

### scripts/thgnn_inference.py
**Purpose**: Batch inference script

**Usage**:
```bash
python scripts/thgnn_inference.py \
    --model checkpoints/best_model.pth \
    --input data/test_logs.json \
    --output predictions.json \
    --top-k 10
```

---

### scripts/performance_benchmarker.py
**Purpose**: System performance testing

**Metrics**:
- Throughput (events/second)
- Latency (prediction time)
- Memory usage
- GPU utilization

---

## Application Files

### app/streamlit_app.py
**Purpose**: Main Streamlit dashboard

**Pages**:
1. Upload Logs
2. Predictions
3. Visualizations
4. Forecasting
5. Reports
6. Proactive Prediction
7. Attack Simulation

**Key Features**:
- File upload interface
- Real-time predictions
- Interactive visualizations
- Attack chain display
- Risk scoring
- Export functionality

---

## Data Files

### data/raw/
**Purpose**: Original security logs

**Structure**:
```
data/raw/
├── extracted/           # Extracted from datasets
│   ├── credential_access/
│   ├── defense_evasion/
│   ├── lateral_movement/
│   └── ...
└── converted/          # Converted to standard format
```

---

### data/metadata/
**Purpose**: Entity mappings and manifests

**Files**:
- `entity_mappings.pkl`: Entity ID mappings
- `conversion_manifest.json`: Data conversion tracking
- `security_datasets_manifest.json`: Dataset metadata

---

## Test Files

### tests/test_graph_builder.py
**Purpose**: Graph construction tests

**Test Cases**:
- Empty graph handling
- Large graph processing
- Temporal edge ordering
- Feature engineering
- Graph validation

---

### tests/test_production_system.py
**Purpose**: End-to-end system tests

**Test Cases**:
- Full pipeline execution
- Performance benchmarks
- Error handling
- Cache functionality
- API endpoints

---

## File Dependencies

### Critical Dependencies

**Data Processing**:
- pandas >= 1.5.0
- numpy >= 1.23.0
- python-dateutil >= 2.8.0

**Machine Learning**:
- torch >= 2.0.0
- torch-geometric >= 2.3.0
- scikit-learn >= 1.2.0

**Visualization**:
- streamlit >= 1.28.0
- plotly >= 5.17.0
- networkx >= 3.1

**Infrastructure**:
- redis >= 4.5.0
- fastapi >= 0.104.0
- uvicorn >= 0.24.0

---

## File Size Guidelines

| File Type | Max Size | Typical Size |
|-----------|----------|--------------|
| Python Module | 1000 lines | 300-500 lines |
| Configuration | 200 lines | 50-100 lines |
| Test File | 500 lines | 200-300 lines |
| Documentation | Unlimited | 500-2000 lines |

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-09 | Development Team | Initial release |
