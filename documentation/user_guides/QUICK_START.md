# Quick Start Guide
## Get Running with THGNN in 15 Minutes

**Document Version:** 1.0.0  
**Last Updated:** February 9, 2026

---

## Prerequisites

Before starting, ensure you have:
- Python 3.10 or higher
- CUDA 11.8+ (for GPU acceleration)
- 16GB RAM minimum (32GB recommended)
- 10GB free disk space

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/thgnn-security.git
cd thgnn-security
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Model

```bash
python scripts/download_datasets.py --model-only
```

---

## First Prediction in 5 Minutes

### Option A: Using Streamlit Dashboard (Recommended)

1. **Start the dashboard**:
```bash
streamlit run app/streamlit_app.py
```

2. **Open browser**: Navigate to `http://localhost:8501`

3. **Upload logs**: 
   - Click "Upload Logs" page
   - Drag and drop your security logs (JSON, CSV, or JSONL)
   - Click "Process Logs"

4. **View predictions**:
   - Navigate to "Predictions" page
   - See top threats ranked by confidence
   - Click on predictions for detailed explanations

### Option B: Using Python API

```python
from src.inference.production_engine import ProductionInferenceEngine
from src.data_processing.log_processor import LogProcessor
from src.data_processing.graph_builder import GraphBuilder

# Initialize components
log_processor = LogProcessor()
graph_builder = GraphBuilder()
engine = ProductionInferenceEngine(
    model_path='checkpoints/best_thgnn_model.pth',
    device='cuda'  # or 'cpu'
)

# Process logs
logs_df = log_processor.process_logs('demo_security_logs.json')
entities = log_processor.extract_entities(logs_df)
graph = graph_builder.build_graph(entities, logs_df)

# Get predictions
predictions = engine.predict(graph, top_k=10)

# Display results
for pred in predictions:
    print(f"Threat: {pred.edge_type}")
    print(f"Confidence: {pred.confidence:.2%}")
    print(f"Source: {pred.source_entity} → Target: {pred.target_entity}")
    print("---")
```

### Option C: Using Command Line

```bash
python scripts/thgnn_inference.py \
    --model checkpoints/best_thgnn_model.pth \
    --input demo_security_logs.json \
    --output predictions.json \
    --top-k 10
```

---

## Understanding Your First Results

### Prediction Output

```json
{
  "prediction_id": "pred_001",
  "edge_type": "user_executes_process",
  "source_entity": "DOMAIN\\admin_user",
  "target_entity": "powershell.exe",
  "confidence": 0.87,
  "timestamp": "2026-02-09T19:30:00Z",
  "risk_score": 8.5,
  "mitre_techniques": ["T1059.001"],
  "explanation": "Detected suspicious PowerShell execution pattern..."
}
```

### Key Fields

- **edge_type**: Type of predicted interaction
- **confidence**: Model confidence (0-1)
- **risk_score**: Threat severity (0-10)
- **mitre_techniques**: Mapped ATT&CK techniques
- **explanation**: Human-readable description

---

## Common Use Cases

### Use Case 1: Detect Lateral Movement

```python
# Filter for lateral movement predictions
lateral_movement = [
    p for p in predictions 
    if 'lateral' in p.explanation.lower()
]

# Visualize attack path
from src.visualization.attack_path_visualizer import AttackPathVisualizer

visualizer = AttackPathVisualizer()
fig = visualizer.visualize_path(lateral_movement)
fig.show()
```

### Use Case 2: Real-Time Monitoring

```python
import time

while True:
    # Get latest logs
    new_logs = fetch_latest_logs()
    
    # Process and predict
    graph = process_logs_to_graph(new_logs)
    predictions = engine.predict(graph)
    
    # Alert on high-risk threats
    for pred in predictions:
        if pred.risk_score > 7.0:
            send_alert(pred)
    
    time.sleep(60)  # Check every minute
```

### Use Case 3: Historical Analysis

```python
# Load historical logs
logs_df = log_processor.process_logs('historical_logs.json')

# Build temporal graph
graph = graph_builder.build_temporal_graph(
    logs_df,
    time_window=3600  # 1 hour windows
)

# Analyze attack progression
from src.explainability.attack_chain_extractor import AttackChainExtractor

extractor = AttackChainExtractor()
attack_chains = extractor.extract_chains(graph, predictions)

for chain in attack_chains:
    print(f"Attack Chain: {' → '.join(chain.stages)}")
    print(f"Duration: {chain.duration}")
    print(f"MITRE Tactics: {chain.tactics}")
```

---

## Next Steps

### Explore the Dashboard

1. **Visualizations Page**: Interactive graph views
2. **Forecasting Page**: Predict future threats
3. **Reports Page**: Generate PDF reports
4. **Attack Simulation**: Test detection capabilities

### Customize for Your Environment

1. **Configure field mappings**: Edit `config/field_mappings.yaml`
2. **Adjust thresholds**: Modify `config/model_config.yaml`
3. **Add custom parsers**: Extend `src/data_processing/log_processor.py`

### Train on Your Data

```bash
# Prepare your logs
python scripts/prepare_training_data.py \
    --input your_logs/ \
    --output data/processed/

# Train model
python scripts/thgnn_train.py \
    --data data/processed/ \
    --config config/model_config.yaml \
    --output checkpoints/custom_model.pth
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size in config
```yaml
inference:
  batch_size: 16  # Reduce from 64
```

### Issue: Slow Predictions

**Solution**: Enable caching
```python
engine = ProductionInferenceEngine(
    model_path='...',
    cache_enabled=True,
    cache_ttl=300
)
```

### Issue: Low Prediction Accuracy

**Solution**: Check data quality
```python
from src.utils.graph_validator import GraphValidator

validator = GraphValidator()
issues = validator.validate(graph)
print(f"Found {len(issues)} issues:")
for issue in issues:
    print(f"- {issue}")
```

---

## Getting Help

- **Documentation**: See `documentation/` folder
- **Examples**: Check `notebooks/` for Jupyter examples
- **Issues**: GitHub Issues page
- **Community**: Discord server

---

## Quick Reference

### Essential Commands

```bash
# Start dashboard
streamlit run app/streamlit_app.py

# Run inference
python scripts/thgnn_inference.py --input logs.json

# Train model
python scripts/thgnn_train.py --data data/processed/

# Run tests
pytest tests/

# Check performance
python scripts/performance_benchmarker.py
```

### Essential Imports

```python
# Data processing
from src.data_processing.log_processor import LogProcessor
from src.data_processing.graph_builder import GraphBuilder

# Inference
from src.inference.production_engine import ProductionInferenceEngine

# Explainability
from src.explainability.explanation_generator import ExplanationGenerator

# Visualization
from src.visualization.graph_visualizer import GraphVisualizer
```

---

**Congratulations!** You're now ready to use THGNN for threat detection.

For detailed information, see:
- [User Manual](USER_MANUAL.md)
- [API Reference](API_REFERENCE.md)
- [System Architecture](../architecture/SYSTEM_ARCHITECTURE.md)

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-09 | Documentation Team | Initial release |
