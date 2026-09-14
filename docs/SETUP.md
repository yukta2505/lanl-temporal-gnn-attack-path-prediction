# THGNN Setup Guide

## System Requirements

### Hardware Requirements
- **CPU**: Intel i5/AMD Ryzen 5 or better (8+ cores recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/4060 or better)
- **Storage**: 50GB free space (SSD recommended)

### Software Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.10 or 3.11 (3.12 not yet supported by PyTorch)
- **CUDA**: 11.8 or 12.1 (for GPU acceleration)
- **Git**: Latest version

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-org/thgnn-project.git
cd thgnn-project
```

### 2. Python Environment Setup

#### Option A: Conda (Recommended)
```bash
# Create environment
conda create -n thgnn python=3.10
conda activate thgnn

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install other dependencies
pip install -r requirements.txt
```

#### Option B: Virtual Environment
```bash
# Create virtual environment
python -m venv thgnn_env

# Activate (Windows)
thgnn_env\Scripts\activate

# Activate (Linux/Mac)
source thgnn_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. GPU Setup (Optional but Recommended)

#### NVIDIA GPU Setup
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

#### CPU-Only Setup
If no GPU available, the system will automatically use CPU mode (slower but functional).

### 4. Verify Installation
```bash
# Run verification script
python scripts/verify_installation.py

# Expected output:
#  Python version: 3.10.x
#  PyTorch version: 2.1.x
#  PyTorch Geometric: 2.4.x
#  CUDA available: True
#  All dependencies installed
```

## Configuration

### 1. Model Configuration
Edit `config/model_config.yaml`:
```yaml
model:
  hidden_dim: 256        # Reduce to 128 for lower memory
  num_conv_layers: 3     # Reduce to 2 for faster training
  temporal_hidden: 128
  dropout: 0.4

training:
  batch_size: 8          # Reduce to 4 for lower memory
  learning_rate: 0.001
  num_epochs: 50
```

### 2. Data Configuration
Edit `config/data_config.yaml`:
```yaml
data:
  time_window_minutes: 5
  sequence_length: 10
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

paths:
  data_dir: data/processed/graphs
  checkpoint_dir: checkpoints
  output_dir: outputs
```

### 3. Logging Configuration
Edit `config/logging_config.yaml`:
```yaml
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/thgnn.log
```

## Quick Start

### 1. Download Sample Data
```bash
# Download Mordor datasets
python scripts/download_datasets.py

# Or use your own data
mkdir -p data/raw/custom_logs
# Copy your JSON/JSONL/CSV files here
```

### 2. Process Data
```bash
# Extract and convert data
python src/data_processing/unzip_datasets.py
python src/data_processing/jsonl_to_json.py

# Build graphs
python src/data_processing/graph_builder.py
```

### 3. Train Model (Optional)
```bash
# Train new model
python scripts/run_training.py

# Or use pre-trained model (if provided)
# Model will be loaded automatically
```

### 4. Run Dashboard
```bash
# Start Streamlit dashboard
streamlit run app/streamlit_app_perfect.py

# Open browser to: http://localhost:8501
```

### 5. CLI Inference
```bash
# Quick predictions
python scripts/run_inference.py --logs data/sample_logs.json
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Reduce `batch_size` in config (8 → 4 → 2)
- Reduce `hidden_dim` (256 → 128)
- Use CPU mode: `export CUDA_VISIBLE_DEVICES=""`

#### 2. PyTorch Geometric Installation
```
ImportError: No module named 'torch_geometric'
```
**Solutions**:
```bash
# Uninstall and reinstall
pip uninstall torch-geometric
pip install torch-geometric

# Or use conda
conda install pyg -c pyg
```

#### 3. Streamlit Port Issues
```
Port 8501 is already in use
```
**Solutions**:
```bash
# Use different port
streamlit run app/streamlit_app_perfect.py --server.port 8502

# Kill existing process
lsof -ti:8501 | xargs kill -9  # Linux/Mac
netstat -ano | findstr :8501   # Windows
```

#### 4. Data Loading Errors
```
FileNotFoundError: No such file or directory
```
**Solutions**:
- Check file paths in config files
- Ensure data directory structure exists
- Run data processing scripts first

#### 5. Model Loading Errors
```
RuntimeError: Error(s) in loading state_dict
```
**Solutions**:
- Check model architecture matches checkpoint
- Retrain model if architecture changed
- Use compatible checkpoint file

### Performance Optimization

#### For Low-Memory Systems (8GB RAM)
```yaml
# config/model_config.yaml
model:
  hidden_dim: 128
  num_conv_layers: 2

training:
  batch_size: 2
  gradient_accumulation_steps: 4
```

#### For CPU-Only Systems
```yaml
# config/model_config.yaml
training:
  batch_size: 1
  num_epochs: 20  # Fewer epochs for faster training
```

#### For High-Performance Systems
```yaml
# config/model_config.yaml
model:
  hidden_dim: 512
  num_conv_layers: 4

training:
  batch_size: 16
  num_workers: 8
```

## Directory Structure
```
thgnn-project/
├── app/                    # Streamlit dashboard
├── checkpoints/            # Trained models
├── config/                 # Configuration files
├── data/                   # Data storage
│   ├── raw/               # Original datasets
│   ├── processed/         # Processed graphs
│   └── metadata/          # Entity mappings
├── docs/                  # Documentation
├── logs/                  # Log files
├── notebooks/             # Jupyter notebooks
├── outputs/               # Results and visualizations
├── scripts/               # Utility scripts
├── src/                   # Source code
│   ├── data_processing/   # Data pipeline
│   ├── models/            # Model architecture
│   ├── training/          # Training code
│   ├── inference/         # Prediction engine
│   ├── explainability/    # Explanation system
│   ├── visualization/     # Plotting utilities
│   └── utils/             # Helper functions
└── tests/                 # Test suite
```

## Environment Variables

Create `.env` file in project root:
```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Paths
THGNN_DATA_DIR=data
THGNN_OUTPUT_DIR=outputs
THGNN_LOG_DIR=logs

# Model Settings
THGNN_MODEL_PATH=checkpoints/best_thgnn_model.pth
THGNN_DEVICE=cuda  # or 'cpu'

# Dashboard Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

## Docker Setup (Advanced)

### Build Container
```bash
# Build image
docker build -t thgnn:latest .

# Run container
docker run -p 8501:8501 -v $(pwd)/data:/app/data thgnn:latest
```

### Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Support

### Getting Help
1. Check this documentation
2. Review troubleshooting section
3. Check GitHub issues
4. Contact support team

### Reporting Issues
When reporting issues, include:
- Operating system and version
- Python version
- GPU model (if applicable)
- Error message and stack trace
- Steps to reproduce

### Performance Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor system resources
htop  # Linux/Mac
taskmgr  # Windows

# Monitor Python memory
python -m memory_profiler script.py
```

## Next Steps

After successful installation:
1. Read [User Guide](USER_GUIDE.md) for dashboard usage
2. Review [Architecture](ARCHITECTURE.md) for system details
3. Check [notebooks/](../notebooks/) for examples
4. Explore [API Documentation](API_DOCS.md)

## Version Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10-3.11 | 3.12 not yet supported |
| PyTorch | 2.1+ | GPU support required |
| PyTorch Geometric | 2.4+ | Must match PyTorch version |
| CUDA | 11.8 or 12.1 | For GPU acceleration |
| Streamlit | 1.28+ | Dashboard framework |
| NetworkX | 3.0+ | Graph visualization |
| Plotly | 5.17+ | Interactive plots |

---

**Installation complete!** 

Your THGNN system is ready for threat detection and analysis.