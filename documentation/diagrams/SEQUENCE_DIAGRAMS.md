# Sequence Diagrams
## THGNN System Interactions

**Document Version:** 1.0.0  
**Last Updated:** February 9, 2026

---

## Table of Contents

1. [Log Processing Sequence](#log-processing-sequence)
2. [Prediction Flow](#prediction-flow)
3. [Explanation Generation](#explanation-generation)
4. [Real-Time Monitoring](#real-time-monitoring)
5. [API Request Flow](#api-request-flow)
6. [Training Pipeline](#training-pipeline)

---

## Log Processing Sequence

### Complete Log-to-Graph Pipeline

```mermaid
sequenceDiagram
    participant User
    participant Dashboard
    participant LogProcessor
    participant EntityExtractor
    participant GraphBuilder
    participant Cache
    
    User->>Dashboard: Upload security logs
    Dashboard->>LogProcessor: process_logs(file)
    
    LogProcessor->>LogProcessor: detect_format()
    LogProcessor->>LogProcessor: parse_logs()
    LogProcessor->>LogProcessor: validate_schema()
    LogProcessor->>LogProcessor: normalize_fields()
    LogProcessor-->>Dashboard: DataFrame
    
    Dashboard->>EntityExtractor: extract_entities(df)
    EntityExtractor->>EntityExtractor: extract_users()
    EntityExtractor->>EntityExtractor: extract_hosts()
    EntityExtractor->>EntityExtractor: extract_processes()
    EntityExtractor->>EntityExtractor: assign_entity_ids()
    EntityExtractor-->>Dashboard: entities_dict
    
    Dashboard->>GraphBuilder: build_graph(entities, df)
    GraphBuilder->>GraphBuilder: create_nodes()
    GraphBuilder->>GraphBuilder: create_edges()
    GraphBuilder->>GraphBuilder: engineer_features()
    GraphBuilder->>GraphBuilder: validate_graph()
    GraphBuilder-->>Dashboard: HeteroData
    
    Dashboard->>Cache: store_graph(graph_id, graph)
    Cache-->>Dashboard: success
    
    Dashboard-->>User: Graph ready for prediction
```

---

## Prediction Flow

### End-to-End Inference Sequence

```mermaid
sequenceDiagram
    participant User
    participant Dashboard
    participant InferenceEngine
    participant Model
    participant Cache
    participant Explainer
    
    User->>Dashboard: Request predictions
    Dashboard->>Cache: get_cached_predictions(graph_id)
    
    alt Cache Hit
        Cache-->>Dashboard: cached_predictions
    else Cache Miss
        Dashboard->>InferenceEngine: predict(graph, top_k=10)
        
        InferenceEngine->>InferenceEngine: preprocess_graph()
        InferenceEngine->>Model: forward(graph)
        
        Model->>Model: encode_nodes()
        Model->>Model: message_passing()
        Model->>Model: temporal_encoding()
        Model->>Model: link_prediction()
        Model-->>InferenceEngine: prediction_scores
        
        InferenceEngine->>InferenceEngine: postprocess_scores()
        InferenceEngine->>InferenceEngine: filter_top_k()
        InferenceEngine->>InferenceEngine: compute_confidence()
        InferenceEngine-->>Dashboard: predictions
        
        Dashboard->>Cache: store_predictions(graph_id, predictions)
    end
    
    Dashboard->>Explainer: explain_predictions(predictions)
    Explainer-->>Dashboard: explanations
    
    Dashboard-->>User: Display predictions + explanations
```

---

## Explanation Generation

### GNNExplainer Integration

```mermaid
sequenceDiagram
    participant Dashboard
    participant ExplanationGenerator
    participant GNNExplainer
    participant AttackChainExtractor
    participant MITREMapper
    participant Database
    
    Dashboard->>ExplanationGenerator: explain_prediction(graph, prediction)
    
    ExplanationGenerator->>GNNExplainer: explain_edge(edge_index)
    GNNExplainer->>GNNExplainer: compute_importance_scores()
    GNNExplainer->>GNNExplainer: identify_important_nodes()
    GNNExplainer->>GNNExplainer: identify_important_edges()
    GNNExplainer-->>ExplanationGenerator: importance_dict
    
    ExplanationGenerator->>AttackChainExtractor: extract_chain(important_edges)
    AttackChainExtractor->>AttackChainExtractor: build_subgraph()
    AttackChainExtractor->>AttackChainExtractor: find_temporal_path()
    AttackChainExtractor->>AttackChainExtractor: identify_stages()
    AttackChainExtractor-->>ExplanationGenerator: attack_chain
    
    ExplanationGenerator->>MITREMapper: map_to_mitre(attack_chain)
    MITREMapper->>Database: query_techniques(chain_patterns)
    Database-->>MITREMapper: technique_matches
    MITREMapper->>MITREMapper: rank_techniques()
    MITREMapper-->>ExplanationGenerator: mitre_techniques
    
    ExplanationGenerator->>ExplanationGenerator: generate_narrative()
    ExplanationGenerator->>ExplanationGenerator: compute_confidence()
    ExplanationGenerator-->>Dashboard: Explanation
```

---

## Real-Time Monitoring

### Streaming Prediction Pipeline

```mermaid
sequenceDiagram
    participant SIEM
    participant WebSocket
    participant StreamProcessor
    participant InferenceEngine
    participant AlertManager
    participant Analyst
    
    loop Every Event
        SIEM->>WebSocket: stream_event(log_entry)
        WebSocket->>StreamProcessor: buffer_event(log_entry)
        
        alt Buffer Full or Timeout
            StreamProcessor->>StreamProcessor: aggregate_events()
            StreamProcessor->>StreamProcessor: build_incremental_graph()
            
            StreamProcessor->>InferenceEngine: predict_streaming(graph_update)
            InferenceEngine->>InferenceEngine: update_embeddings()
            InferenceEngine->>InferenceEngine: compute_predictions()
            InferenceEngine-->>StreamProcessor: predictions
            
            alt High Risk Detected
                StreamProcessor->>AlertManager: create_alert(prediction)
                AlertManager->>AlertManager: enrich_alert()
                AlertManager->>AlertManager: prioritize()
                AlertManager->>Analyst: send_notification(alert)
                Analyst-->>AlertManager: acknowledge
            end
        end
    end
```

---

## API Request Flow

### REST API Prediction Endpoint

```mermaid
sequenceDiagram
    participant Client
    participant APIGateway
    participant AuthMiddleware
    participant PredictionEndpoint
    participant InferenceEngine
    participant Database
    
    Client->>APIGateway: POST /api/v1/predict
    APIGateway->>AuthMiddleware: authenticate(request)
    
    alt Invalid Auth
        AuthMiddleware-->>Client: 401 Unauthorized
    else Valid Auth
        AuthMiddleware->>PredictionEndpoint: forward_request()
        
        PredictionEndpoint->>PredictionEndpoint: validate_request()
        
        alt Invalid Request
            PredictionEndpoint-->>Client: 400 Bad Request
        else Valid Request
            PredictionEndpoint->>InferenceEngine: predict(graph_data)
            InferenceEngine-->>PredictionEndpoint: predictions
            
            PredictionEndpoint->>Database: log_request(metadata)
            PredictionEndpoint->>PredictionEndpoint: format_response()
            PredictionEndpoint-->>Client: 200 OK + predictions
        end
    end
```

---

## Training Pipeline

### Model Training Sequence

```mermaid
sequenceDiagram
    participant Trainer
    participant DataLoader
    participant Model
    participant Optimizer
    participant Evaluator
    participant Checkpoint
    
    Trainer->>DataLoader: initialize(train_data)
    Trainer->>Model: initialize(config)
    Trainer->>Optimizer: initialize(model.parameters())
    
    loop For Each Epoch
        loop For Each Batch
            DataLoader->>Trainer: next_batch()
            Trainer->>Model: forward(batch)
            Model-->>Trainer: predictions
            
            Trainer->>Trainer: compute_loss(predictions, targets)
            Trainer->>Optimizer: zero_grad()
            Trainer->>Trainer: loss.backward()
            Trainer->>Optimizer: step()
        end
        
        Trainer->>Evaluator: evaluate(val_data)
        Evaluator->>Model: forward(val_batch)
        Evaluator->>Evaluator: compute_metrics()
        Evaluator-->>Trainer: metrics
        
        alt Best Model
            Trainer->>Checkpoint: save_checkpoint(model, metrics)
        end
        
        alt Early Stopping
            Trainer->>Trainer: stop_training()
        end
    end
    
    Trainer->>Checkpoint: load_best_model()
    Trainer->>Evaluator: final_evaluation(test_data)
    Evaluator-->>Trainer: final_metrics
```

---

## User Interaction Flows

### Dashboard Navigation

```mermaid
sequenceDiagram
    participant User
    participant UploadPage
    participant PredictionsPage
    participant VisualizationsPage
    participant ReportsPage
    
    User->>UploadPage: Navigate to Upload
    User->>UploadPage: Select file
    User->>UploadPage: Click "Process"
    UploadPage->>UploadPage: process_logs()
    UploadPage-->>User: "Processing complete"
    
    User->>PredictionsPage: Navigate to Predictions
    PredictionsPage->>PredictionsPage: load_predictions()
    PredictionsPage-->>User: Display predictions table
    
    User->>PredictionsPage: Click prediction
    PredictionsPage->>PredictionsPage: load_explanation()
    PredictionsPage-->>User: Show detailed explanation
    
    User->>VisualizationsPage: Navigate to Visualizations
    VisualizationsPage->>VisualizationsPage: render_graph()
    VisualizationsPage-->>User: Interactive graph
    
    User->>VisualizationsPage: Select nodes
    VisualizationsPage->>VisualizationsPage: highlight_selection()
    VisualizationsPage-->>User: Updated visualization
    
    User->>ReportsPage: Navigate to Reports
    User->>ReportsPage: Click "Generate Report"
    ReportsPage->>ReportsPage: compile_report()
    ReportsPage->>ReportsPage: export_pdf()
    ReportsPage-->>User: Download report
```

---

## Error Handling Flows

### Graceful Degradation

```mermaid
sequenceDiagram
    participant User
    participant System
    participant ErrorHandler
    participant Logger
    participant Fallback
    
    User->>System: Request operation
    System->>System: execute_operation()
    
    alt Operation Fails
        System->>ErrorHandler: handle_error(exception)
        ErrorHandler->>Logger: log_error(details)
        
        alt Recoverable Error
            ErrorHandler->>Fallback: use_fallback_method()
            Fallback-->>ErrorHandler: fallback_result
            ErrorHandler-->>User: Success (with warning)
        else Unrecoverable Error
            ErrorHandler->>ErrorHandler: format_error_message()
            ErrorHandler-->>User: Error message + suggestions
        end
    else Operation Succeeds
        System-->>User: Success result
    end
```

---

## Performance Monitoring

### Metrics Collection Flow

```mermaid
sequenceDiagram
    participant System
    participant MetricsCollector
    participant Prometheus
    participant Grafana
    participant AlertManager
    
    loop Continuous
        System->>MetricsCollector: record_metric(name, value)
        MetricsCollector->>MetricsCollector: aggregate_metrics()
        MetricsCollector->>Prometheus: push_metrics()
        
        Prometheus->>Prometheus: store_time_series()
        Grafana->>Prometheus: query_metrics()
        Prometheus-->>Grafana: metric_data
        Grafana->>Grafana: render_dashboard()
        
        alt Threshold Exceeded
            Prometheus->>AlertManager: trigger_alert()
            AlertManager->>AlertManager: evaluate_rules()
            AlertManager->>AlertManager: send_notification()
        end
    end
```

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-09 | Architecture Team | Initial release |
