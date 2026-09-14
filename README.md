---
title: THGNN Threat Detection
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.37.1
app_file: app.py
pinned: false
---

# THGNN - Temporal Heterogeneous Graph Neural Network for Cybersecurity Threat Detection

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AkshayHole/Thgnn-Threat-Detection)

## Overview

THGNN is an advanced cybersecurity threat detection system that uses Temporal Heterogeneous Graph Neural Networks to predict and analyze potential cyber threats in real-time.

### Key Features

- **Real-time Threat Prediction**: Analyze security logs and predict future attack steps
- **Multi-step Attack Forecasting**: Forecast attack chains up to 5 steps ahead
- **Interactive Visualizations**: Network graphs, timelines, and heatmaps
- **MITRE ATT&CK Mapping**: Automatic mapping to MITRE ATT&CK framework
- **Comprehensive Reporting**: Executive summaries and technical analysis

### Performance Metrics

- **AUC-ROC**: 92.6%
- **Precision@10**: 78%
- **Prediction Latency**: <5 seconds
- **Processing Speed**: 100K+ events/minute

## How to Use

1. **Upload Security Logs**: Upload your security logs (JSON, JSONL, CSV, or ZIP formats)
2. **Generate Predictions**: Analyze threats and generate risk-scored predictions
3. **Explore Visualizations**: View interactive network graphs and attack chains
4. **Forecast Future Attacks**: Multi-step attack path prediction
5. **Generate Reports**: Create executive summaries and technical reports

## Technical Stack

- **Framework**: PyTorch 2.0+, PyTorch Geometric 2.3+
- **Frontend**: Streamlit
- **Visualization**: Plotly, NetworkX
- **ML Architecture**: Heterogeneous GCN with LSTM temporal encoding

## Architecture

The system processes security logs through the following pipeline:

1. **Data Ingestion**: Parse security logs (Windows Event, Sysmon, custom JSON)
2. **Graph Construction**: Build temporal heterogeneous graphs (users, hosts, processes)
3. **ML Inference**: Predict future attack steps using GNN-based link prediction
4. **Explainability**: Generate explanations with GNNExplainer
5. **Visualization**: Display attack chains in interactive dashboard

## Model Details

- **Hidden Dimension**: 256
- **Convolution Layers**: 3
- **Temporal Encoding**: Bidirectional LSTM
- **Node Types**: Users, Hosts, Processes, Files, Registry
- **Edge Types**: logs_into, executes, creates, connects_to, accesses, modifies

## Citation

If you use this system in your research, please cite:

```
THGNN: Temporal Heterogeneous Graph Neural Network for Cybersecurity Threat Detection
```

## License

MIT License

## Contact

For questions or issues, please open an issue on the repository.
