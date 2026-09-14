# 🛡️ THGNN – Lateral Movement Detection

### Temporal Heterogeneous Graph Neural Network for Cybersecurity Threat Detection

## 📌 Overview

THGNN (Temporal Heterogeneous Graph Neural Network) is a cybersecurity
threat detection system designed to analyze security events and identify
potential lateral movement and attack activity.

The system models relationships between different entities such as
users, hosts, processes, files, and registry objects over time using a
temporal heterogeneous graph.

## 🎯 Objectives

- Detect suspicious lateral movement activity.
- Analyze relationships between entities in security logs.
- Predict potential future attack steps.
- Map detected activities to MITRE ATT&CK techniques.
- Provide visual and understandable security analysis.

## ✨ Key Features

- 🔍 Threat detection and prediction
- 🔗 Multi-step attack forecasting
- 📊 Interactive network graph visualization
- 🛡️ MITRE ATT&CK technique mapping
- 🧠 Temporal heterogeneous graph modeling
- 📑 Technical and executive reporting
- 🔎 Model explainability

## 🧠 How It Works

```text
Security Logs
      ↓
Data Ingestion
      ↓
Graph Construction
      ↓
Temporal Heterogeneous Graph
      ↓
THGNN Model
      ↓
Threat / Attack Prediction
      ↓
Explainability
      ↓
Interactive Visualization
      ↓
Security Reports

## 🏗️ System Architecture

The THGNN system consists of multiple components that work together
to process security events, construct a temporal heterogeneous graph,
detect potential threats, and present the results for analysis.

The main components are:

- **Data Ingestion** – Collects and processes security event data.
- **Data Preprocessing** – Cleans and prepares the input data.
- **Graph Construction** – Represents security entities and their
  relationships as a heterogeneous graph.
- **Temporal Modeling** – Captures changes in entity relationships
  over time.
- **THGNN Model** – Analyzes the graph to identify suspicious
  activity and predict potential attack steps.
- **Explainability** – Helps understand the factors contributing
  to model predictions.
- **Visualization** – Displays security events, relationships,
  and attack paths.
- **Reporting** – Provides security analysis and prediction results.


## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Graph Learning | PyTorch Geometric |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Graph Processing | NetworkX |
| Temporal Modeling | LSTM |
| Explainability | GNNExplainer |
| Containerization | Docker |


## 🧩 Model Details

### Node Types

The graph represents different cybersecurity entities, including:

- Users
- Hosts
- Processes
- Files
- Registry objects

### Relationship Types

The system models relationships such as:

- `logs_into`
- `executes`
- `creates`
- `connects_to`
- `accesses`
- `modifies`

### Model Components

- Heterogeneous graph representation
- Graph neural network layers
- Temporal encoding
- Bidirectional LSTM
- Threat prediction
- Explainability using GNNExplainer

## 📊 Performance Results

The current project reports the following performance metrics:

| Metric | Result |
|---|---:|
| AUC-ROC | 92.6% |
| Precision@10 | 78% |
| Prediction Latency | < 5 seconds |
| Processing Speed | 100K+ events/minute |

## 📂 Project Structure

```text
THGNN-Lateral-Movement-Detection/
│
├── app/
├── config/
├── docs/
├── documentation/
├── notebooks/
├── scripts/
├── src/
├── tests/
│
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── environment.yml
├── requirements.txt
├── README.md
├── THGNN_Project_Documentation.txt
└── THGNN_Project_Report.md

## ⚙️ Installation

### 1. Clone the Repository
git clone https://github.com/123shraddha555/THGNN-Lateral-Movement-Detection.git

### 2. Navigate to the Project Directory
cd THGNN-Lateral-Movement-Detection

pip install -r requirements.txt

## 🛡️ MITRE ATT&CK Mapping

- **T1078 – Valid Accounts**


## 🚀 Future Scope

- Improve real-time threat detection.
- Support more MITRE ATT&CK techniques.
- Enhance model accuracy and scalability.

## 👤 Author

**Shraddha Ghaytadkar**

GitHub: https://github.com/123shraddha555
