# THGNN User Guide

## Overview

The Temporal Heterogeneous Graph Neural Network (THGNN) system is a professional cybersecurity threat detection platform that analyzes security logs to predict and explain potential attack chains.

## Getting Started

### Launching the Dashboard
```bash
# Start the application
streamlit run app/streamlit_app_perfect.py

# Open browser to: http://localhost:8501
```

### Dashboard Navigation

The THGNN dashboard consists of 5 main pages:

1. **📤 Upload Logs** - Import security data
2. **🔍 Predictions** - View threat predictions
3. **📊 Visualizations** - Interactive graphs and charts
4. **🔮 Forecasting** - Future attack predictions
5. **📋 Reports** - Generate and export reports

## Page-by-Page Guide

### 1. Upload Logs Page

#### Supported File Formats
- **JSON**: Single array of events `[{event1}, {event2}, ...]`
- **JSONL**: One event per line `{event1}\n{event2}\n...`
- **CSV**: Tabular format with headers
- **ZIP**: Compressed files (auto-extracted)

#### Required Fields
Your log files should contain these fields:
- `timestamp`: Event time (ISO 8601 format)
- `user`: Username (e.g., "DOMAIN\\alice")
- `host`: Hostname (e.g., "WORKSTATION01")
- `process`: Process name (e.g., "powershell.exe")
- `event_id`: Event identifier (e.g., 4624, 4688)

#### Example Log Entry
```json
{
  "timestamp": "2026-01-11T10:30:00Z",
  "event_id": 4624,
  "user": "DOMAIN\\alice",
  "host": "WORKSTATION01",
  "process": "winlogon.exe",
  "action": "logon",
  "source_ip": "192.168.1.100"
}
```

#### Upload Process
1. **Select File**: Click "Browse files" or drag & drop
2. **Format Detection**: System auto-detects file format
3. **Validation**: Checks for required fields
4. **Processing**: Extracts entities and builds graphs
5. **Confirmation**: Shows processing results

#### Status Indicators
- 🟢 **Success**: File processed successfully
- 🟡 **Warning**: Some issues found but processing continued
- 🔴 **Error**: Processing failed, check file format

### 2. Predictions Page

#### Threat Predictions Table
Displays top-risk predictions with columns:
- **Source Entity**: Origin of potential attack
- **Target Entity**: Destination/target
- **Edge Type**: Type of connection (logs_into, executes, etc.)
- **Risk Score**: Probability of threat (0-1)
- **Confidence**: Model confidence (0-1)
- **Explanation**: Human-readable description

#### Filtering Options
- **Risk Threshold**: Show only predictions above threshold
- **Entity Type**: Filter by users, hosts, or processes
- **Time Range**: Filter by time window
- **Edge Type**: Filter by connection type

#### Sorting and Export
- **Sort**: Click column headers to sort
- **Export**: Download as CSV or JSON
- **Details**: Click row for detailed explanation

#### Risk Score Interpretation
- **0.9-1.0**: Critical threat (immediate attention)
- **0.7-0.9**: High risk (investigate soon)
- **0.5-0.7**: Medium risk (monitor)
- **0.0-0.5**: Low risk (baseline activity)

### 3. Visualizations Page

#### Interactive Network Graph
- **Nodes**: Represent entities (users, hosts, processes)
- **Edges**: Represent connections/relationships
- **Colors**: Indicate entity types and risk levels
- **Size**: Proportional to activity level

#### Graph Controls
- **Zoom**: Mouse wheel or zoom controls
- **Pan**: Click and drag to move
- **Select**: Click nodes/edges for details
- **Filter**: Hide/show entity types
- **Layout**: Choose graph layout algorithm

#### Timeline View
- **X-axis**: Time progression
- **Y-axis**: Entities
- **Events**: Markers show activities
- **Predictions**: Dashed lines for forecasts

#### Process Tree
- **Hierarchy**: Parent-child process relationships
- **Commands**: Hover for command lines
- **Risk**: Color-coded by suspicion level
- **Expansion**: Click to expand/collapse branches

#### Risk Heatmap
- **Rows**: Entities (users/hosts)
- **Columns**: Time windows
- **Colors**: Risk intensity
- **Patterns**: Identify hotspots and trends

### 4. Forecasting Page

#### Multi-Step Prediction
- **Horizon**: Select 1-5 future time windows
- **Confidence**: Decreases with prediction distance
- **Scenarios**: Multiple possible attack paths

#### Attack Path Analysis
- **Critical Paths**: Most likely attack sequences
- **Target Analysis**: Probable final targets
- **Mitigation**: Suggested countermeasures

#### Forecast Controls
- **Time Horizon**: 5-30 minutes ahead
- **Confidence Threshold**: Minimum prediction confidence
- **Max Paths**: Number of attack paths to show
- **Update Frequency**: Auto-refresh interval

### 5. Reports Page

#### Executive Summary
- **Threat Level**: Overall risk assessment
- **Key Findings**: Top threats and patterns
- **Recommendations**: Actionable security advice
- **Metrics**: Performance statistics

#### Technical Report
- **Detailed Analysis**: In-depth threat breakdown
- **Model Performance**: Accuracy metrics
- **Data Quality**: Processing statistics
- **Methodology**: Technical approach

#### Export Options
- **PDF**: Professional formatted report
- **HTML**: Interactive web report
- **JSON**: Raw data export
- **CSV**: Tabular data for analysis

## Advanced Features

### Custom Thresholds
Adjust risk thresholds based on your environment:
- **High Security**: Lower thresholds (0.5+)
- **Normal Operations**: Medium thresholds (0.7+)
- **Noisy Environment**: Higher thresholds (0.8+)

### Time Window Configuration
- **Short Windows** (1-2 min): Real-time monitoring
- **Medium Windows** (5-10 min): Balanced analysis
- **Long Windows** (15-30 min): Pattern detection

### Entity Filtering
Focus analysis on specific entities:
- **VIP Users**: High-privilege accounts
- **Critical Hosts**: Servers and domain controllers
- **Suspicious Processes**: Known attack tools

## Interpreting Results

### Understanding Predictions
Each prediction includes:
- **Context**: Why this connection is suspicious
- **Historical**: Comparison to normal behavior
- **Techniques**: MITRE ATT&CK mapping
- **Confidence**: Model certainty level

### Common Attack Patterns
The system detects these patterns:
- **Lateral Movement**: User accessing multiple hosts
- **Privilege Escalation**: Process spawning with higher privileges
- **Persistence**: Unusual startup processes
- **Exfiltration**: Large data transfers

### False Positive Management
- **Whitelist**: Mark legitimate activities
- **Baseline**: Update normal behavior patterns
- **Feedback**: Improve model accuracy over time

## Best Practices

### Data Quality
- **Complete Logs**: Ensure all required fields present
- **Consistent Format**: Use standardized log formats
- **Regular Updates**: Process logs frequently
- **Clean Data**: Remove corrupted entries

### Analysis Workflow
1. **Upload** fresh log data
2. **Review** high-risk predictions first
3. **Investigate** suspicious patterns
4. **Forecast** potential next steps
5. **Generate** reports for stakeholders

### Performance Optimization
- **Batch Processing**: Upload larger files less frequently
- **Time Windows**: Adjust based on environment
- **Filtering**: Focus on relevant entities
- **Caching**: System caches processed data

## Troubleshooting

### Common Issues

#### Upload Problems
**Issue**: File upload fails
**Solutions**:
- Check file format (JSON/JSONL/CSV)
- Verify required fields present
- Ensure file size < 100MB
- Check file encoding (UTF-8)

#### No Predictions
**Issue**: No threats detected
**Solutions**:
- Lower risk threshold
- Check data quality
- Verify time range
- Ensure sufficient data volume

#### Slow Performance
**Issue**: Dashboard responds slowly
**Solutions**:
- Reduce data size
- Increase time windows
- Clear browser cache
- Check system resources

#### Visualization Issues
**Issue**: Graphs don't display
**Solutions**:
- Refresh browser page
- Check JavaScript enabled
- Try different browser
- Clear browser cache

### Error Messages

#### "Invalid file format"
- File is not JSON, JSONL, or CSV
- File is corrupted or empty
- Encoding issues (use UTF-8)

#### "Missing required fields"
- Log entries lack timestamp, user, or host
- Field names don't match expected format
- Data types are incorrect

#### "Processing failed"
- Insufficient memory
- Data too large
- System overloaded

## Keyboard Shortcuts

### Navigation
- **Ctrl+1**: Upload page
- **Ctrl+2**: Predictions page
- **Ctrl+3**: Visualizations page
- **Ctrl+4**: Forecasting page
- **Ctrl+5**: Reports page

### Actions
- **Ctrl+R**: Refresh current page
- **Ctrl+E**: Export current data
- **Ctrl+F**: Open search/filter
- **Esc**: Close dialogs

## API Integration

### REST Endpoints
```bash
# Upload logs
POST /api/upload
Content-Type: multipart/form-data

# Get predictions
GET /api/predictions?threshold=0.7

# Generate forecast
POST /api/forecast
{"horizon": 3, "confidence": 0.8}

# Export report
GET /api/report?format=pdf
```

### Python SDK
```python
from thgnn_client import THGNNClient

client = THGNNClient("http://localhost:8501")
predictions = client.get_predictions(threshold=0.8)
```

## Security Considerations

### Data Privacy
- Logs processed locally (no cloud transmission)
- Sensitive data can be anonymized
- Access controls available
- Audit logging enabled

### Network Security
- HTTPS encryption supported
- Authentication integration available
- IP whitelisting configurable
- VPN compatibility

## Support and Maintenance

### Regular Maintenance
- **Model Updates**: Retrain monthly
- **Data Cleanup**: Archive old logs
- **Performance Monitoring**: Check system health
- **Backup**: Save configurations and models

### Getting Help
- **Documentation**: Check this guide first
- **Logs**: Review system logs for errors
- **Support**: Contact technical support
- **Community**: Join user forums

## Appendix

### Supported Log Sources
- Windows Event Logs
- Sysmon
- Linux Audit Logs
- Custom JSON formats
- SIEM exports
- Network logs

### MITRE ATT&CK Coverage
The system maps predictions to MITRE ATT&CK techniques:
- **T1059**: Command and Scripting Interpreter
- **T1021**: Remote Services
- **T1078**: Valid Accounts
- **T1055**: Process Injection
- **T1003**: OS Credential Dumping

### Performance Benchmarks
- **Processing Speed**: 100K+ events/minute
- **Prediction Latency**: <5 seconds
- **Memory Usage**: 8GB recommended
- **Storage**: 1GB per 1M events

---

**Happy threat hunting!** 🛡️

For technical support, contact: support@thgnn-project.com