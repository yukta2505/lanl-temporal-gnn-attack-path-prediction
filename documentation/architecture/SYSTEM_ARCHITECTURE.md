# System Architecture
## THGNN Cybersecurity Threat Detection Platform

**Document Version:** 1.0.0  
**Last Updated:** February 9, 2026  
**Classification:** Technical Architecture

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture Layers](#architecture-layers)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [Integration Points](#integration-points)
7. [Scalability Design](#scalability-design)
8. [Security Architecture](#security-architecture)

---

## Executive Summary

The THGNN system implements a layered architecture designed for high-performance, real-time cybersecurity threat detection. The system processes security logs through a sophisticated pipeline that constructs temporal heterogeneous graphs, applies advanced graph neural networks for prediction, and provides explainable results through an interactive dashboard.

### Key Architectural Principles

- **Modularity**: Independent, loosely-coupled components
- **Scalability**: Horizontal scaling for high-volume processing
- **Extensibility**: Plugin architecture for new data sources and models
- **Reliability**: Fault tolerance and graceful degradation
- **Performance**: GPU-accelerated processing with sub-5-second latency

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Streamlit  │  │   REST API   │  │  WebSocket   │         │
│  │   Dashboard  │  │   Endpoints  │  │   Streaming  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Inference   │  │Explainability│  │Visualization │         │
│  │   Engine     │  │   Module     │  │   Engine     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         MODEL LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │    THGNN     │  │   Temporal   │  │     Link     │         │
│  │     Core     │  │   Encoder    │  │  Predictor   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │     Log      │  │    Entity    │  │    Graph     │         │
│  │   Processor  │  │  Extractor   │  │   Builder    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Raw Logs   │  │    Graph     │  │    Model     │         │
│  │   Storage    │  │    Cache     │  │  Checkpoints │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Layers

### 1. Presentation Layer

**Purpose**: User interaction and external system integration

**Components**:
- **Streamlit Dashboard**: Interactive web interface for security analysts
- **REST API**: Synchronous request/response endpoints
- **WebSocket API**: Real-time streaming predictions

**Technologies**:
- Streamlit 1.28+
- FastAPI 0.104+
- WebSocket (asyncio)

**Responsibilities**:
- User authentication and authorization
- Request validation and sanitization
- Response formatting and serialization
- Real-time event streaming

### 2. Application Layer

**Purpose**: Business logic and orchestration

**Components**:

#### Inference Engine
- Real-time prediction processing
- Batch prediction support
- Model version management
- Result caching and optimization

#### Explainability Module
- GNNExplainer integration
- Attack chain extraction
- MITRE ATT&CK mapping
- Pattern analysis

#### Visualization Engine
- Graph rendering
- Timeline generation
- Risk heatmap creation
- Attack path visualization

**Technologies**:
- Python 3.10+
- Redis (caching)
- Celery (task queue)

### 3. Model Layer

**Purpose**: Machine learning inference and prediction

**Components**:

#### THGNN Core
- Heterogeneous graph convolution
- Multi-head attention mechanisms
- Message passing neural networks
- Node and edge embeddings

#### Temporal Encoder
- LSTM-based temporal modeling
- Time-aware attention
- Sequence encoding
- Temporal feature extraction

#### Link Predictor
- Future edge prediction
- Scoring functions
- Negative sampling
- Confidence estimation

**Technologies**:
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- CUDA 11.8+ (GPU acceleration)

### 4. Data Processing Layer

**Purpose**: Raw data transformation and graph construction

**Components**:

#### Log Processor
- Multi-format log parsing (JSON, JSONL, CSV, Windows Event)
- Field normalization
- Timestamp standardization
- Data validation

#### Entity Extractor
- User entity extraction
- Host entity identification
- Process entity parsing
- Relationship detection

#### Graph Builder
- Heterogeneous graph construction
- Temporal edge creation
- Feature engineering
- Graph validation

**Technologies**:
- Pandas, NumPy
- NetworkX
- Custom parsers

### 5. Data Layer

**Purpose**: Persistent storage and caching

**Components**:
- **Raw Logs Storage**: Original security logs
- **Graph Cache**: Processed graph structures (Redis)
- **Model Checkpoints**: Trained model weights
- **Metadata Store**: Entity mappings and configurations

**Technologies**:
- File system (logs)
- Redis (cache)
- PostgreSQL (metadata)
- PyTorch checkpoint format

---

## Core Components

### Component: Log Processor

**Location**: `src/data_processing/log_processor.py`

**Purpose**: Parse and normalize security logs from multiple sources

**Key Classes**:
- `LogProcessor`: Main processing orchestrator
- `UniversalLogParser`: Multi-format parser
- `FieldNormalizer`: Field standardization

**Input Formats**:
- Windows Event Logs (EVTX)
- Sysmon Logs (XML/JSON)
- Custom JSON/JSONL
- CSV security logs

**Output**: Normalized pandas DataFrame with standardized fields

**Performance**: 10,000+ logs/second

---

### Component: Entity Extractor

**Location**: `src/data_processing/entity_extractor.py`

**Purpose**: Extract and normalize security entities from logs

**Key Classes**:
- `EntityExtractor`: Main extraction engine
- `EntityNormalizer`: Entity standardization
- `EntityMapper`: ID assignment and tracking

**Entity Types**:
- **User**: domain\username format
- **Host**: FQDN or hostname
- **Process**: executable name, PID, command line
- **File**: full path
- **Network**: IP addresses, ports

**Output**: Entity dictionaries with unique IDs and metadata

---

### Component: Graph Builder

**Location**: `src/data_processing/graph_builder.py`

**Purpose**: Construct temporal heterogeneous graphs from entities

**Key Classes**:
- `GraphBuilder`: Main graph construction
- `TemporalEdgeCreator`: Time-aware edge generation
- `FeatureEngineer`: Node and edge feature creation

**Graph Structure**:
```python
HeteroData(
    user={
        x: [num_users, user_features],
        node_id: [num_users]
    },
    host={
        x: [num_hosts, host_features],
        node_id: [num_hosts]
    },
    process={
        x: [num_processes, process_features],
        node_id: [num_processes]
    },
    (user, logs_into, host)={
        edge_index: [2, num_edges],
        edge_attr: [num_edges, edge_features],
        timestamp: [num_edges]
    },
    (user, executes, process)={...},
    (process, connects_to, host)={...}
)
```

**Performance**: 1,000+ nodes/second

---

### Component: THGNN Model

**Location**: `src/models/thgnn_core.py`

**Purpose**: Core graph neural network for threat prediction

**Architecture**:
```
Input Graph
    ↓
Node Feature Embedding (Linear)
    ↓
Heterogeneous Graph Convolution Layer 1
    ↓
Temporal Attention Layer
    ↓
Heterogeneous Graph Convolution Layer 2
    ↓
Temporal Encoding (LSTM)
    ↓
Link Prediction Head
    ↓
Output Scores
```

**Key Parameters**:
- Hidden dimensions: 128
- Number of layers: 2-3
- Attention heads: 4
- Dropout: 0.3
- Temporal window: 3600 seconds

**Performance**: 2.3 second inference latency

---

### Component: Inference Engine

**Location**: `src/inference/production_engine.py`

**Purpose**: Real-time prediction and scoring

**Key Classes**:
- `ProductionInferenceEngine`: Main inference orchestrator
- `BatchProcessor`: Batch prediction optimization
- `ResultCache`: Prediction caching

**Features**:
- Model warm-up and preloading
- GPU memory management
- Result caching (Redis)
- Confidence thresholding
- Top-K prediction filtering

**API**:
```python
engine = ProductionInferenceEngine(model_path, device='cuda')
predictions = engine.predict(graph_data, top_k=10)
```

---

### Component: Explainability Module

**Location**: `src/explainability/explanation_generator.py`

**Purpose**: Generate human-interpretable explanations

**Key Classes**:
- `ExplanationGenerator`: Main explanation orchestrator
- `GNNExplainerWrapper`: GNNExplainer integration
- `AttackChainExtractor`: Attack sequence reconstruction
- `MITREMapper`: ATT&CK technique mapping

**Output**:
```python
{
    'prediction': 'user_executes_process',
    'confidence': 0.87,
    'important_nodes': ['user_123', 'process_456'],
    'important_edges': [('user_123', 'host_789')],
    'attack_chain': ['T1078', 'T1059.001', 'T1021.002'],
    'explanation': 'Detected lateral movement pattern...'
}
```

---

## Data Flow

### End-to-End Processing Pipeline

```
1. Log Ingestion
   ├─ File upload (Streamlit)
   ├─ API submission (REST)
   └─ Stream ingestion (WebSocket)
        ↓
2. Log Processing
   ├─ Format detection
   ├─ Parsing and validation
   └─ Field normalization
        ↓
3. Entity Extraction
   ├─ User extraction
   ├─ Host identification
   ├─ Process parsing
   └─ Relationship detection
        ↓
4. Graph Construction
   ├─ Node creation
   ├─ Edge generation
   ├─ Feature engineering
   └─ Temporal ordering
        ↓
5. Model Inference
   ├─ Graph encoding
   ├─ Message passing
   ├─ Temporal attention
   └─ Link prediction
        ↓
6. Explainability
   ├─ GNNExplainer analysis
   ├─ Attack chain extraction
   └─ MITRE mapping
        ↓
7. Visualization
   ├─ Graph rendering
   ├─ Timeline generation
   └─ Risk scoring
        ↓
8. Result Delivery
   ├─ Dashboard display
   ├─ API response
   └─ Alert generation
```

### Processing Time Breakdown

| Stage | Average Time | Percentage |
|-------|--------------|------------|
| Log Processing | 0.3s | 13% |
| Entity Extraction | 0.2s | 9% |
| Graph Construction | 0.4s | 17% |
| Model Inference | 1.0s | 43% |
| Explainability | 0.3s | 13% |
| Visualization | 0.1s | 4% |
| **Total** | **2.3s** | **100%** |

---

## Integration Points

### External System Integration

#### 1. SIEM Integration
- **Splunk**: Forward logs via HTTP Event Collector
- **Elastic**: Logstash pipeline integration
- **QRadar**: Custom log source configuration

#### 2. Threat Intelligence
- **MISP**: Threat indicator enrichment
- **STIX/TAXII**: Threat intelligence feeds
- **VirusTotal**: File and IP reputation

#### 3. Ticketing Systems
- **ServiceNow**: Automated incident creation
- **Jira**: Security task management
- **PagerDuty**: Alert escalation

#### 4. Authentication
- **LDAP/Active Directory**: User authentication
- **OAuth 2.0**: API authentication
- **SAML**: Single sign-on

---

## Scalability Design

### Horizontal Scaling Strategy

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                         │
│                    (NGINX/HAProxy)                       │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  API Server  │  │  API Server  │  │  API Server  │
│   Instance 1 │  │   Instance 2 │  │   Instance 3 │
└──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                ┌──────────────────┐
                │   Redis Cluster  │
                │   (Shared Cache) │
                └──────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  GPU Worker  │  │  GPU Worker  │  │  GPU Worker  │
│   (Model 1)  │  │   (Model 2)  │  │   (Model 3)  │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Performance Optimization

#### Caching Strategy
- **L1 Cache**: In-memory Python dictionaries (hot data)
- **L2 Cache**: Redis (processed graphs, embeddings)
- **L3 Cache**: Disk (model checkpoints, historical data)

#### GPU Optimization
- Batch processing for multiple predictions
- Mixed precision training (FP16)
- Gradient checkpointing for memory efficiency
- Model parallelism for large graphs

#### Database Optimization
- Connection pooling
- Query optimization and indexing
- Partitioning by timestamp
- Read replicas for analytics

---

## Security Architecture

### Security Layers

#### 1. Network Security
- TLS 1.3 for all communications
- API rate limiting
- DDoS protection
- Firewall rules

#### 2. Application Security
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CSRF tokens

#### 3. Authentication & Authorization
- Multi-factor authentication
- Role-based access control (RBAC)
- API key management
- Session management

#### 4. Data Security
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- Secure key management
- Data anonymization

#### 5. Audit & Compliance
- Comprehensive audit logging
- Access tracking
- Compliance reporting (SOC 2, GDPR)
- Security monitoring

---

## Deployment Architecture

### Production Deployment

```
┌─────────────────────────────────────────────────────────┐
│                    Cloud Provider (AWS/Azure/GCP)        │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │              Kubernetes Cluster                 │    │
│  │                                                 │    │
│  │  ┌──────────────┐  ┌──────────────┐           │    │
│  │  │   Ingress    │  │   Service    │           │    │
│  │  │  Controller  │  │     Mesh     │           │    │
│  │  └──────────────┘  └──────────────┘           │    │
│  │                                                 │    │
│  │  ┌──────────────────────────────────────┐     │    │
│  │  │         Application Pods             │     │    │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐   │     │    │
│  │  │  │  API   │ │Inference│ │Worker │   │     │    │
│  │  │  │  Pod   │ │  Pod    │ │  Pod  │   │     │    │
│  │  │  └────────┘ └────────┘ └────────┘   │     │    │
│  │  └──────────────────────────────────────┘     │    │
│  │                                                 │    │
│  │  ┌──────────────────────────────────────┐     │    │
│  │  │         Data Services                │     │    │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐   │     │    │
│  │  │  │ Redis  │ │Postgres│ │  S3    │   │     │    │
│  │  │  │Cluster │ │   DB   │ │Storage │   │     │    │
│  │  │  └────────┘ └────────┘ └────────┘   │     │    │
│  │  └──────────────────────────────────────┘     │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Container Architecture

**Docker Images**:
- `thgnn-api:latest` - REST API server
- `thgnn-inference:latest` - Model inference worker
- `thgnn-dashboard:latest` - Streamlit dashboard
- `thgnn-worker:latest` - Background task processor

**Resource Requirements**:
- API Server: 2 CPU, 4GB RAM
- Inference Worker: 4 CPU, 16GB RAM, 1 GPU (16GB VRAM)
- Dashboard: 1 CPU, 2GB RAM
- Redis: 2 CPU, 8GB RAM
- PostgreSQL: 2 CPU, 8GB RAM

---

## Monitoring and Observability

### Metrics Collection

**System Metrics**:
- CPU, memory, disk, network utilization
- GPU utilization and memory
- Container health and restarts

**Application Metrics**:
- Request rate and latency
- Error rates and types
- Prediction accuracy and confidence
- Cache hit rates

**Business Metrics**:
- Threats detected per hour
- False positive rate
- Mean time to detection (MTTD)
- Analyst response time

### Logging Strategy

**Log Levels**:
- ERROR: System failures and exceptions
- WARN: Degraded performance or anomalies
- INFO: Normal operations and milestones
- DEBUG: Detailed diagnostic information

**Log Aggregation**:
- Centralized logging (ELK Stack)
- Structured JSON logging
- Correlation IDs for request tracking
- Log retention: 90 days

---

## Disaster Recovery

### Backup Strategy

**Data Backups**:
- Model checkpoints: Daily, retained 30 days
- Configuration: Real-time replication
- Logs: Continuous streaming to archive
- Metadata: Hourly snapshots

**Recovery Objectives**:
- Recovery Time Objective (RTO): 1 hour
- Recovery Point Objective (RPO): 15 minutes

### High Availability

- Multi-zone deployment
- Automated failover
- Health checks and auto-healing
- Database replication

---

## Future Architecture Enhancements

### Planned Improvements

1. **Federated Learning**: Multi-tenant model training
2. **Edge Deployment**: On-premise inference nodes
3. **AutoML Integration**: Automated model optimization
4. **Streaming Architecture**: Apache Kafka integration
5. **Advanced Caching**: Distributed cache with Hazelcast

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-09 | Architecture Team | Initial release |

**Review Schedule**: Quarterly

**Next Review**: May 9, 2026
