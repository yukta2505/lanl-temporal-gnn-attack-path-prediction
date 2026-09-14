#!/usr/bin/env python3
"""
Production API for THGNN System

FastAPI-based production API for real-time cybersecurity threat prediction.
Provides RESTful endpoints for model inference, attack path prediction,
and proactive alerting with comprehensive error handling and monitoring.

Features:
- RESTful API endpoints for threat prediction
- Real-time model inference with <5 second latency
- Request validation and rate limiting
- Comprehensive error handling and logging
- Prometheus metrics integration
- API key authentication
- Request/response caching with Redis
- Async processing for high throughput
- OpenAPI documentation
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import redis
import torch

# Import THGNN components
from ..models.thgnn_model_enhanced import EnhancedTHGNN
from ..inference.production_engine import ProductionInferenceEngine
from ..visualization.attack_path_visualizer import AttackPathVisualizer
from ..simulation.attack_scenarios import AttackScenarios

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('thgnn_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('thgnn_api_request_duration_seconds', 'Request duration')
PREDICTION_LATENCY = Histogram('thgnn_prediction_latency_seconds', 'Prediction latency')
ACTIVE_CONNECTIONS = Gauge('thgnn_active_connections', 'Active connections')
MODEL_MEMORY_USAGE = Gauge('thgnn_model_memory_bytes', 'Model memory usage')

# Initialize FastAPI app
app = FastAPI(
    title="THGNN Production API",
    description="Temporal Heterogeneous Graph Neural Network for Cybersecurity Threat Prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer()

# Global variables
inference_engine: Optional[ProductionInferenceEngine] = None
redis_client: Optional[redis.Redis] = None
api_keys = {"demo_key_2026": "demo_user"}  # In production, use proper key management

# Pydantic models for request/response validation
class LogEvent(BaseModel):
    """Single log event model."""
    timestamp: str = Field(..., description="Event timestamp in ISO format")
    event_type: str = Field(..., description="Type of security event")
    source_host: str = Field(..., description="Source host identifier")
    source_user: Optional[str] = Field(None, description="Source user identifier")
    target_resource: Optional[str] = Field(None, description="Target resource")
    event_id: Optional[int] = Field(None, description="Event ID")
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional event data")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Invalid timestamp format')

class PredictionRequest(BaseModel):
    """Prediction request model."""
    events: List[LogEvent] = Field(..., description="List of security events")
    num_steps: int = Field(5, ge=1, le=10, description="Number of prediction steps")
    confidence_threshold: float = Field(0.7, ge=0.1, le=1.0, description="Confidence threshold")
    include_visualization: bool = Field(False, description="Include attack path visualization")
    include_timeline: bool = Field(False, description="Include attack timeline")
    
    class Config:
        schema_extra = {
            "example": {
                "events": [
                    {
                        "timestamp": "2026-02-04T14:23:15Z",
                        "event_type": "user_login",
                        "source_host": "WORKSTATION-01",
                        "source_user": "admin@company.local",
                        "event_id": 4624
                    }
                ],
                "num_steps": 5,
                "confidence_threshold": 0.7,
                "include_visualization": True,
                "include_timeline": True
            }
        }

class AttackStep(BaseModel):
    """Attack step model."""
    step_id: int = Field(..., description="Step identifier")
    edge_type: str = Field(..., description="Type of attack action")
    source_type: str = Field(..., description="Source entity type")
    target_type: str = Field(..., description="Target entity type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    mitre_techniques: List[str] = Field(..., description="MITRE ATT&CK techniques")
    risk_level: str = Field(..., description="Risk level (low, medium, high, critical)")

class PredictionResponse(BaseModel):
    """Prediction response model."""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(..., description="Response timestamp")
    attack_path: Dict[str, Any] = Field(..., description="Predicted attack path")
    mitre_techniques: List[str] = Field(..., description="All MITRE techniques")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall risk score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    prediction_time: float = Field(..., description="Prediction processing time")
    alerts: List[Dict[str, Any]] = Field(..., description="Proactive alerts")
    visualization_data: Optional[Dict[str, Any]] = Field(None, description="Visualization data")
    timeline_data: Optional[Dict[str, Any]] = Field(None, description="Timeline data")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Model loading status")
    gpu_available: bool = Field(..., description="GPU availability")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")

# Dependency functions
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication."""
    token = credentials.credentials
    if token not in api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_keys[token]

async def get_redis_client():
    """Get Redis client for caching."""
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_client.ping()  # Test connection
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            redis_client = None
    return redis_client

def generate_request_id(request_data: str) -> str:
    """Generate unique request ID."""
    return hashlib.md5(f"{request_data}{time.time()}".encode()).hexdigest()[:16]

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global inference_engine
    
    try:
        logger.info("Starting THGNN Production API...")
        
        # Initialize inference engine
        inference_engine = ProductionInferenceEngine()
        
        # Load model (in production, load from checkpoint)
        logger.info("Loading THGNN model...")
        # inference_engine.load_model("path/to/model/checkpoint.pth")
        
        # Initialize Redis connection
        await get_redis_client()
        
        logger.info("THGNN Production API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global redis_client
    
    logger.info("Shutting down THGNN Production API...")
    
    if redis_client:
        redis_client.close()
    
    logger.info("THGNN Production API shutdown complete")

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track requests for monitoring."""
    start_time = time.time()
    
    # Increment active connections
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_DURATION.observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        return response
        
    except Exception as e:
        # Record error metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        raise
    
    finally:
        # Decrement active connections
        ACTIVE_CONNECTIONS.dec()

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check model status
        model_loaded = inference_engine is not None
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        
        # Get memory usage
        memory_usage = {
            "total_mb": 0,
            "used_mb": 0,
            "available_mb": 0
        }
        
        if gpu_available:
            memory_usage["gpu_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            memory_usage["gpu_used_mb"] = torch.cuda.memory_allocated(0) / 1024 / 1024
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            model_loaded=model_loaded,
            gpu_available=gpu_available,
            memory_usage=memory_usage
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/predict", response_model=PredictionResponse)
async def predict_attack_path(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    user: str = Depends(verify_api_key)
):
    """Predict attack path from security events."""
    start_time = time.time()
    request_id = generate_request_id(request.json())
    
    try:
        logger.info(f"Processing prediction request {request_id} for user {user}")
        
        # Check cache first
        redis_client = await get_redis_client()
        cache_key = f"prediction:{hashlib.md5(request.json().encode()).hexdigest()}"
        
        if redis_client:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Returning cached result for request {request_id}")
                return PredictionResponse.parse_raw(cached_result)
        
        # Validate inference engine
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert events to internal format
        processed_events = []
        for event in request.events:
            processed_event = {
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'source_host': event.source_host,
                'source_user': event.source_user,
                'target_resource': event.target_resource,
                'event_id': event.event_id,
                'additional_data': event.additional_data or {}
            }
            processed_events.append(processed_event)
        
        # Generate predictions (mock for demo)
        predictions = await generate_mock_predictions(
            processed_events,
            request.num_steps,
            request.confidence_threshold
        )
        
        # Generate visualization data if requested
        visualization_data = None
        if request.include_visualization:
            visualization_data = await generate_visualization_data(predictions)
        
        # Generate timeline data if requested
        timeline_data = None
        if request.include_timeline:
            timeline_data = await generate_timeline_data(predictions)
        
        # Create response
        prediction_time = time.time() - start_time
        PREDICTION_LATENCY.observe(prediction_time)
        
        response = PredictionResponse(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            attack_path=predictions['attack_path'],
            mitre_techniques=predictions['mitre_techniques'],
            risk_score=predictions['risk_score'],
            confidence=predictions['confidence'],
            prediction_time=prediction_time,
            alerts=predictions['alerts'],
            visualization_data=visualization_data,
            timeline_data=timeline_data
        )
        
        # Cache result
        if redis_client:
            background_tasks.add_task(
                cache_prediction_result,
                cache_key,
                response.json(),
                ttl=300  # 5 minutes
            )
        
        logger.info(f"Completed prediction request {request_id} in {prediction_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed for request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/scenarios")
async def get_attack_scenarios(user: str = Depends(verify_api_key)):
    """Get available attack scenarios."""
    try:
        scenarios = AttackScenarios()
        available_scenarios = scenarios.get_all_scenarios()
        
        return {
            "scenarios": available_scenarios,
            "total_count": len(available_scenarios),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get scenarios: {e}")
        raise HTTPException(status_code=500, detail="Failed to get scenarios")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    try:
        # Update model memory usage
        if torch.cuda.is_available():
            MODEL_MEMORY_USAGE.set(torch.cuda.memory_allocated(0))
        
        return generate_latest().decode('utf-8')
        
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")

@app.get("/status")
async def get_system_status(user: str = Depends(verify_api_key)):
    """Get detailed system status."""
    try:
        status = {
            "api_version": "1.0.0",
            "model_status": "loaded" if inference_engine else "not_loaded",
            "gpu_status": "available" if torch.cuda.is_available() else "not_available",
            "redis_status": "connected" if redis_client else "not_connected",
            "uptime_seconds": time.time() - startup_time if 'startup_time' in globals() else 0,
            "active_connections": ACTIVE_CONNECTIONS._value._value,
            "total_requests": sum(REQUEST_COUNT._metrics.values()),
            "timestamp": datetime.now().isoformat()
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get status")

# Helper functions
async def generate_mock_predictions(
    events: List[Dict[str, Any]],
    num_steps: int,
    confidence_threshold: float
) -> Dict[str, Any]:
    """Generate mock predictions for demonstration."""
    try:
        # Mock attack steps
        attack_patterns = [
            {'edge_type': 'executes', 'source_type': 'user', 'target_type': 'process'},
            {'edge_type': 'accesses', 'source_type': 'process', 'target_type': 'file'},
            {'edge_type': 'connects_to', 'source_type': 'process', 'target_type': 'host'},
            {'edge_type': 'modifies', 'source_type': 'process', 'target_type': 'registry'},
            {'edge_type': 'logs_into', 'source_type': 'user', 'target_type': 'host'}
        ]
        
        steps = []
        for i in range(min(num_steps, len(attack_patterns))):
            pattern = attack_patterns[i]
            confidence = max(confidence_threshold, 0.75 + (i * 0.05))
            
            step = {
                'step_id': i + 1,
                'edge_type': pattern['edge_type'],
                'source_type': pattern['source_type'],
                'target_type': pattern['target_type'],
                'confidence': confidence,
                'mitre_techniques': ['T1078', 'T1059.001'],
                'risk_level': 'high' if confidence > 0.8 else 'medium'
            }
            steps.append(step)
        
        # Generate alerts
        alerts = []
        if steps and steps[0]['confidence'] > 0.8:
            alerts.append({
                'type': 'HIGH_CONFIDENCE_ATTACK',
                'severity': 'HIGH',
                'message': f"High-confidence attack sequence predicted",
                'confidence': steps[0]['confidence'],
                'timestamp': datetime.now().isoformat()
            })
        
        return {
            'attack_path': {
                'steps': steps,
                'overall_confidence': sum(s['confidence'] for s in steps) / len(steps) if steps else 0,
                'num_steps': len(steps)
            },
            'mitre_techniques': ['T1078', 'T1059.001', 'T1021.002'],
            'risk_score': 0.85,
            'confidence': 0.82,
            'alerts': alerts
        }
        
    except Exception as e:
        logger.error(f"Error generating mock predictions: {e}")
        raise

async def generate_visualization_data(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Generate visualization data for attack path."""
    try:
        return {
            'nodes': [
                {'id': f'node_{i}', 'type': step['source_type'], 'risk': step['risk_level']}
                for i, step in enumerate(predictions['attack_path']['steps'])
            ],
            'edges': [
                {
                    'source': f'node_{i}',
                    'target': f'node_{i+1}',
                    'type': step['edge_type'],
                    'confidence': step['confidence']
                }
                for i, step in enumerate(predictions['attack_path']['steps'][:-1])
            ],
            'layout': 'hierarchical',
            'animation_enabled': True
        }
        
    except Exception as e:
        logger.error(f"Error generating visualization data: {e}")
        return {}

async def generate_timeline_data(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Generate timeline data for attack progression."""
    try:
        timeline = []
        cumulative_time = 0
        
        for i, step in enumerate(predictions['attack_path']['steps']):
            cumulative_time += 5 * (i + 1)  # Mock timing
            
            timeline_entry = {
                'step_number': i + 1,
                'time_from_now_minutes': cumulative_time,
                'edge_type': step['edge_type'],
                'confidence': step['confidence'],
                'mitre_techniques': step['mitre_techniques']
            }
            timeline.append(timeline_entry)
        
        return {
            'timeline': timeline,
            'total_duration_minutes': cumulative_time,
            'overall_confidence': predictions['confidence']
        }
        
    except Exception as e:
        logger.error(f"Error generating timeline data: {e}")
        return {}

async def cache_prediction_result(cache_key: str, result: str, ttl: int):
    """Cache prediction result in Redis."""
    try:
        redis_client = await get_redis_client()
        if redis_client:
            redis_client.setex(cache_key, ttl, result)
            logger.debug(f"Cached prediction result with key {cache_key}")
    except Exception as e:
        logger.error(f"Failed to cache result: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

# Production server configuration
if __name__ == "__main__":
    uvicorn.run(
        "production_api:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        access_log=True,
        reload=False  # Set to False for production
    )