#!/bin/bash

set -e

echo "THGNN Environment Setup"
echo "======================"

check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: Python 3 not found"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    REQUIRED_VERSION="3.8"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        echo "ERROR: Python $REQUIRED_VERSION+ required, found $PYTHON_VERSION"
        exit 1
    fi
    
    echo "Python $PYTHON_VERSION detected"
}

setup_venv() {
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
}

install_requirements() {
    echo "Installing requirements..."
    pip install -r requirements.txt
}

check_gpu() {
    echo "Checking GPU availability..."
    python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('WARNING: CUDA not available, using CPU only')
"
}

test_imports() {
    echo "Testing critical imports..."
    python3 -c "
import torch
import torch_geometric
import pandas as pd
import numpy as np
import networkx as nx
import plotly
import streamlit
import yaml
import sklearn
print('All imports successful')
"
}

create_directories() {
    echo "Creating project structure..."
    mkdir -p data/{raw/{mordor_datasets,extracted,custom_logs},processed/{entities,graphs,splits},metadata}
    mkdir -p src/{data_processing,models,training,inference,explainability,visualization,utils}
    mkdir -p config checkpoints outputs/{predictions,visualizations,reports}
    mkdir -p tests notebooks docs scripts app/{pages,assets}
    
    find src -type d -exec touch {}/__init__.py \;
}

main() {
    check_python
    setup_venv
    install_requirements
    check_gpu
    test_imports
    create_directories
    
    echo ""
    echo "Setup completed successfully!"
    echo "Activate environment: source venv/bin/activate"
}

main "$@"
