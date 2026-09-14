# THGNN Production Docker Image
# Multi-stage build for optimized production deployment

# Base stage with Python and system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# GPU stage with CUDA support
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu-base

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-pip \
    python3.10-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Production stage
FROM gpu-base as production

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash thgnn
RUN chown -R thgnn:thgnn /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir torch-geometric
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=thgnn:thgnn src/ ./src/
COPY --chown=thgnn:thgnn app/ ./app/
COPY --chown=thgnn:thgnn config/ ./config/
COPY --chown=thgnn:thgnn scripts/ ./scripts/
COPY --chown=thgnn:thgnn notebooks/ ./notebooks/

# Copy model checkpoints if they exist
COPY --chown=thgnn:thgnn checkpoints/ ./checkpoints/ 2>/dev/null || true

# Create necessary directories
RUN mkdir -p data/raw data/processed outputs logs temp
RUN chown -R thgnn:thgnn data outputs logs temp

# Switch to non-root user
USER thgnn

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app/streamlit_app_perfect.py", "--server.port=8501", "--server.address=0.0.0.0"]