# Data Flow Architecture
## THGNN End-to-End Pipeline

**Document Version:** 1.0.0  
**Last Updated:** February 9, 2026

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Stage 1: Log Ingestion](#stage-1-log-ingestion)
3. [Stage 2: Data Processing](#stage-2-data-processing)
4. [Stage 3: Graph Construction](#stage-3-graph-construction)
5. [Stage 4: Model Inference](#stage-4-model-inference)
6. [Stage 5: Explainability](#stage-5-explainability)
7. [Stage 6: Visualization](#stage-6-visualization)
8. [Data Formats](#data-formats)
9. [Error Handling](#error-handling)

---

## Pipeline Overview

The THGNN system processes security data through six distinct stages, transforming raw logs into actionable threat intelligence.

### High-Level Flow

```
Raw Logs → Parsing → Entity Extraction → Graph Building → 
ML Inference → Explanation → Visualization → Alerts
```

### Processing Time Budget

| Stage | Target Time | Max Time |
|-------|-------------|----------|
| Log Ingestion | 0.1s | 0.3s |
| Data Processing | 0.3s | 0.5s |
| Graph Construction | 0.4s | 0.6s |
| Model Inference | 1.0s | 1.5s |
| Explainability | 0.3s | 0.5s |
| Visualization | 0.2s | 0.3s |
| **Total** | **2.3s** | **3.7s** |

---

## Stage 1: Log Ingestion

### Input Sources

**File Upload**:
```
User → Streamlit UI → File Upload → Temp Storage
```

**API Submission**:
```
External System → REST API → Validation → Queue
```

**Stream Ingestion**:
```
SIEM → WebSocket → Buffer → Processing Queue
```

### Supported Formats

- Windows Event Logs (EVTX, JSON)
- Sysmon Logs (XML, JSON)
- Custom JSON/JSONL
- CSV with security fields
- Splunk/Elastic exports

### Validation Steps

1. **Format Detection**: Identify log format
2. **Schema Validation**: Check required fields
3. **Size Check**: Ensure within limits (< 100MB)
4. **Encoding Verification**: UTF-8 validation
5. **Malware Scan**: Optional virus check

