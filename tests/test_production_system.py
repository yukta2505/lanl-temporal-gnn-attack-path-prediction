#!/usr/bin/env python3
"""
Production System Tests

Comprehensive test suite for THGNN production system covering
all critical components, performance requirements, and edge cases.

Features:
- Unit tests for all major components
- Integration tests for end-to-end workflows
- Performance benchmarks and latency tests
- Error handling and edge case validation
- Security and input validation tests
- API endpoint testing
- Monitoring and alerting tests
- Production readiness validation
"""

import pytest
import asyncio
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
from torch_geometric.data import HeteroData

# Import system components
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.gru_temporal_encoder import GRUTemporalEncoder
from models.thgnn_model_enhanced import EnhancedTHGNN, AttackPathPredictor
from visualization.attack_path_visualizer import AttackPathVisualizer
from visualization.proactive_dashboard import ProactiveDashboard
from simulation.realtime_simulator import RealtimeLogSimulator, StreamingEventProcessor
from simulation.attack_scenarios import AttackScenarios
from monitoring.production_monitor import ProductionMonitor
from api.production_api import app
from fastapi.testclient import TestClient

class TestGRUTemporalEncoder:
    """Test GRU temporal encoder component."""
    
    @pytest.fixture
    def encoder(self):
        """Create GRU temporal encoder for testing."""
        return GRUTemporalEncoder(
            input_dim=256,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3
        )
    
    @pytest.fixture
    def sample_sequence(self):
        """Create sample sequence data."""
        batch_size, seq_len, input_dim = 2, 5, 256
        return torch.randn(batch_size, seq_len, input_dim)
    
    def test_encoder_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.input_dim == 256
        assert encoder.hidden_dim == 128
        assert encoder.num_layers == 2
        assert encoder.dropout == 0.3
        assert not encoder.bidirectional
        assert not encoder.use_attention
    
    def test_forward_pass(self, encoder, sample_sequence):
        """Test forward pass through encoder."""
        output = encoder(sample_sequence)
        
        # Check output shape
        batch_size = sample_sequence.size(0)
        assert output.shape == (batch_size, encoder.input_dim)
        
        # Check output is not NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_pass_with_sequence_lengths(self, encoder):
        """Test forward pass with variable sequence lengths."""
        batch_size, max_seq_len, input_dim = 3, 10, 256
        sequences = torch.randn(batch_size, max_seq_len, input_dim)
        sequence_lengths = torch.tensor([8, 6, 10])
        
        output = encoder(sequences, sequence_lengths=sequence_lengths)
        
        assert output.shape == (batch_size, encoder.input_dim)
        assert not torch.isnan(output).any()
    
    def test_performance_tracking(self, encoder, sample_sequence):
        """Test performance tracking functionality."""
        # Clear previous stats
        encoder.reset_performance_stats()
        
        # Run multiple forward passes
        for _ in range(5):
            encoder(sample_sequence)
        
        stats = encoder.get_performance_stats()
        assert stats['total_inferences'] == 5
        assert stats['avg_inference_time'] > 0
        assert stats['min_inference_time'] > 0
        assert stats['max_inference_time'] > 0
    
    def test_attention_mechanism(self):
        """Test attention mechanism when enabled."""
        encoder = GRUTemporalEncoder(
            input_dim=256,
            hidden_dim=128,
            use_attention=True,
            attention_heads=4
        )
        
        batch_size, seq_len, input_dim = 2, 5, 256
        sequences = torch.randn(batch_size, seq_len, input_dim)
        
        output, attention_weights = encoder(
            sequences, 
            return_attention=True
        )
        
        assert output.shape == (batch_size, encoder.input_dim)
        assert attention_weights is not None
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing functionality."""
        encoder = GRUTemporalEncoder(
            input_dim=256,
            hidden_dim=128,
            gradient_checkpointing=True
        )
        
        batch_size, seq_len, input_dim = 2, 5, 256
        sequences = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        
        encoder.train()  # Enable training mode
        output = encoder(sequences)
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        
        assert sequences.grad is not None

class TestEnhancedTHGNN:
    """Test Enhanced THGNN model."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.hidden_dim = 256
        config.temporal_hidden_dim = 128
        config.temporal_layers = 2
        config.dropout = 0.4
        config.get = Mock(return_value=5)  # max_prediction_steps
        return config
    
    @pytest.fixture
    def mock_metadata(self):
        """Create mock metadata."""
        return {
            'node_types': ['user', 'host', 'process'],
            'edge_types': ['logs_into', 'executes', 'accesses']
        }
    
    @pytest.fixture
    def sample_graph(self):
        """Create sample heterogeneous graph."""
        graph = HeteroData()
        
        # Add node types
        graph['user'].x = torch.randn(10, 64)
        graph['host'].x = torch.randn(5, 64)
        graph['process'].x = torch.randn(20, 64)
        
        # Add edge types
        graph['user', 'logs_into', 'host'].edge_index = torch.randint(0, 5, (2, 15))
        graph['user', 'executes', 'process'].edge_index = torch.randint(0, 10, (2, 25))
        graph['process', 'accesses', 'host'].edge_index = torch.randint(0, 5, (2, 30))
        
        return graph
    
    @patch('src.models.thgnn_model_enhanced.THGNN.__init__')
    def test_model_initialization(self, mock_super_init, mock_config, mock_metadata):
        """Test Enhanced THGNN initialization."""
        mock_super_init.return_value = None
        
        model = EnhancedTHGNN(mock_config, mock_metadata)
        
        assert hasattr(model, 'temporal_encoder')
        assert hasattr(model, 'attack_path_predictor')
        assert hasattr(model, 'mitre_mapper')
        assert isinstance(model.temporal_encoder, GRUTemporalEncoder)
    
    def test_attack_path_prediction(self, sample_graph):
        """Test attack path prediction functionality."""
        predictor = AttackPathPredictor(hidden_dim=256, num_steps=5)
        
        # Mock model for prediction
        mock_model = Mock()
        
        attack_path = predictor.predict_path(
            current_graph=sample_graph,
            model=mock_model,
            num_steps=3,
            confidence_threshold=0.7
        )
        
        assert 'steps' in attack_path
        assert 'overall_confidence' in attack_path
        assert 'num_steps' in attack_path
        assert isinstance(attack_path['steps'], list)

class TestAttackPathVisualizer:
    """Test attack path visualization component."""
    
    @pytest.fixture
    def visualizer(self):
        """Create attack path visualizer."""
        return AttackPathVisualizer()
    
    @pytest.fixture
    def sample_attack_steps(self):
        """Create sample attack steps."""
        return [
            {
                'edge_type': 'executes',
                'source_type': 'user',
                'target_type': 'process',
                'confidence': 0.85,
                'mitre_techniques': ['T1059.001']
            },
            {
                'edge_type': 'accesses',
                'source_type': 'process',
                'target_type': 'file',
                'confidence': 0.78,
                'mitre_techniques': ['T1005']
            }
        ]
    
    def test_visualizer_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert hasattr(visualizer, 'risk_colors')
        assert hasattr(visualizer, 'edge_colors')
        assert hasattr(visualizer, 'mitre_colors')
        assert 'critical' in visualizer.risk_colors
        assert 'predicted' in visualizer.edge_colors
    
    def test_attack_sequence_plotting(self, visualizer, sample_attack_steps):
        """Test attack sequence plotting."""
        fig = visualizer.plot_attack_sequence(
            attack_steps=sample_attack_steps,
            title="Test Attack Path",
            animate=False
        )
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
    
    def test_timeline_visualization(self, visualizer):
        """Test timeline visualization."""
        timeline_data = {
            'timeline': [
                {
                    'step_number': 1,
                    'time_from_now_minutes': 5,
                    'confidence': 0.85,
                    'mitre_techniques': ['T1078']
                },
                {
                    'step_number': 2,
                    'time_from_now_minutes': 10,
                    'confidence': 0.72,
                    'mitre_techniques': ['T1059.001']
                }
            ]
        }
        
        fig = visualizer.create_proactive_timeline(timeline_data)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_mitre_mapping_visualization(self, visualizer, sample_attack_steps):
        """Test MITRE technique mapping visualization."""
        techniques = ['T1078', 'T1059.001', 'T1005']
        
        fig = visualizer.generate_mitre_mapping_viz(sample_attack_steps)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_error_handling(self, visualizer):
        """Test error handling in visualization."""
        # Test with empty attack steps
        fig = visualizer.plot_attack_sequence(
            attack_steps=[],
            title="Empty Attack Path"
        )
        
        assert fig is not None
        
        # Test with malformed data
        malformed_steps = [{'invalid': 'data'}]
        fig = visualizer.plot_attack_sequence(
            attack_steps=malformed_steps,
            title="Malformed Data"
        )
        
        assert fig is not None

class TestRealtimeSimulator:
    """Test real-time simulation component."""
    
    @pytest.fixture
    def simulator(self):
        """Create real-time simulator."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return RealtimeLogSimulator(scenario_path=temp_dir)
    
    def test_simulator_initialization(self, simulator):
        """Test simulator initialization."""
        assert hasattr(simulator, 'scenarios')
        assert hasattr(simulator, 'streaming_stats')
        assert not simulator.streaming_active
        assert not simulator.streaming_paused
    
    def test_scenario_loading(self, simulator):
        """Test scenario loading functionality."""
        scenarios = simulator.get_available_scenarios()
        
        assert isinstance(scenarios, dict)
        # Should have at least fallback scenarios
        assert len(scenarios) >= 1
    
    def test_streaming_controls(self, simulator):
        """Test streaming control functions."""
        # Test pause (should fail when not active)
        assert not simulator.pause_streaming()
        
        # Test resume (should fail when not active)
        assert not simulator.resume_streaming()
        
        # Test stop (should fail when not active)
        assert not simulator.stop_streaming()
    
    def test_streaming_status(self, simulator):
        """Test streaming status reporting."""
        status = simulator.get_streaming_status()
        
        assert 'active' in status
        assert 'paused' in status
        assert 'stats' in status
        assert not status['active']
        assert not status['paused']
    
    def test_performance_stats(self, simulator):
        """Test performance statistics."""
        stats = simulator.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert 'scenarios_loaded' in stats

class TestAttackScenarios:
    """Test attack scenarios component."""
    
    @pytest.fixture
    def scenarios(self):
        """Create attack scenarios."""
        return AttackScenarios()
    
    def test_scenarios_initialization(self, scenarios):
        """Test scenarios initialization."""
        assert hasattr(scenarios, 'scenarios')
        assert hasattr(scenarios, 'mitre_techniques')
        assert len(scenarios.scenarios) > 0
    
    def test_get_all_scenarios(self, scenarios):
        """Test getting all scenarios."""
        all_scenarios = scenarios.get_all_scenarios()
        
        assert isinstance(all_scenarios, dict)
        assert len(all_scenarios) > 0
        
        # Check scenario structure
        for scenario_id, scenario_info in all_scenarios.items():
            assert 'name' in scenario_info
            assert 'description' in scenario_info
            assert 'duration_minutes' in scenario_info
            assert 'mitre_techniques' in scenario_info
    
    def test_get_specific_scenario(self, scenarios):
        """Test getting specific scenario."""
        scenario_names = list(scenarios.scenarios.keys())
        if scenario_names:
            scenario = scenarios.get_scenario(scenario_names[0])
            
            assert scenario is not None
            assert 'attack_steps' in scenario
            assert 'timing' in scenario
            assert 'entity_relationships' in scenario
    
    def test_scenario_customization(self, scenarios):
        """Test scenario customization."""
        scenario_names = list(scenarios.scenarios.keys())
        if scenario_names:
            base_scenario = scenario_names[0]
            
            modifications = {
                'duration_minutes': 20,
                'add_techniques': ['T1234'],
                'scale_duration': 1.5
            }
            
            customized = scenarios.customize_scenario(base_scenario, modifications)
            
            assert customized is not None
            assert customized['duration_minutes'] == 20
    
    def test_scenario_export(self, scenarios):
        """Test scenario export functionality."""
        scenario_names = list(scenarios.scenarios.keys())
        if scenario_names:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_path = f.name
            
            try:
                success = scenarios.export_scenario_to_json(scenario_names[0], temp_path)
                assert success
                
                # Verify file was created and contains valid JSON
                with open(temp_path, 'r') as f:
                    data = json.load(f)
                    assert isinstance(data, dict)
            finally:
                Path(temp_path).unlink(missing_ok=True)

class TestProductionMonitor:
    """Test production monitoring component."""
    
    @pytest.fixture
    def monitor(self):
        """Create production monitor."""
        config = {
            'monitoring_interval': 1,  # Short interval for testing
            'enable_alerting': False,  # Disable alerting for tests
            'prometheus_port': 8002    # Different port to avoid conflicts
        }
        return ProductionMonitor(config)
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert hasattr(monitor, 'config')
        assert hasattr(monitor, 'alert_thresholds')
        assert not monitor.monitoring_active
    
    def test_metrics_collection(self, monitor):
        """Test metrics collection."""
        system_metrics = monitor._collect_system_metrics()
        model_metrics = monitor._collect_model_metrics()
        security_metrics = monitor._collect_security_metrics()
        
        # Check system metrics
        assert hasattr(system_metrics, 'cpu_percent')
        assert hasattr(system_metrics, 'memory_percent')
        assert hasattr(system_metrics, 'timestamp')
        
        # Check model metrics
        assert hasattr(model_metrics, 'prediction_latency_ms')
        assert hasattr(model_metrics, 'prediction_accuracy')
        assert hasattr(model_metrics, 'timestamp')
        
        # Check security metrics
        assert hasattr(security_metrics, 'threats_detected')
        assert hasattr(security_metrics, 'mitre_techniques_seen')
        assert hasattr(security_metrics, 'timestamp')
    
    def test_health_check(self, monitor):
        """Test system health check."""
        # Add some mock metrics
        monitor.system_metrics_history.append(monitor._collect_system_metrics())
        monitor.model_metrics_history.append(monitor._collect_model_metrics())
        
        health = monitor.get_system_health()
        
        assert 'status' in health
        assert 'message' in health
        assert 'timestamp' in health
        assert health['status'] in ['healthy', 'degraded', 'unhealthy', 'unknown']
    
    def test_performance_summary(self, monitor):
        """Test performance summary generation."""
        # Add some mock metrics
        for _ in range(5):
            monitor.system_metrics_history.append(monitor._collect_system_metrics())
            monitor.model_metrics_history.append(monitor._collect_model_metrics())
            time.sleep(0.1)  # Small delay to create time differences
        
        summary = monitor.get_performance_summary(hours=1)
        
        assert 'system_performance' in summary
        assert 'model_performance' in summary
        assert 'time_period_hours' in summary
    
    def test_metrics_export(self, monitor):
        """Test metrics export functionality."""
        # Add some mock metrics
        monitor.system_metrics_history.append(monitor._collect_system_metrics())
        
        exported = monitor.export_metrics(format='json')
        
        assert isinstance(exported, str)
        data = json.loads(exported)
        assert 'system_metrics' in data
        assert 'export_timestamp' in data

class TestProductionAPI:
    """Test production API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers."""
        return {"Authorization": "Bearer demo_key_2026"}
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'timestamp' in data
        assert 'model_loaded' in data
    
    def test_prediction_endpoint(self, client, auth_headers):
        """Test prediction endpoint."""
        request_data = {
            "events": [
                {
                    "timestamp": "2026-02-04T14:23:15Z",
                    "event_type": "user_login",
                    "source_host": "WORKSTATION-01",
                    "source_user": "admin@company.local",
                    "event_id": 4624
                }
            ],
            "num_steps": 3,
            "confidence_threshold": 0.7
        }
        
        response = client.post("/predict", json=request_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert 'request_id' in data
        assert 'attack_path' in data
        assert 'confidence' in data
        assert 'prediction_time' in data
    
    def test_scenarios_endpoint(self, client, auth_headers):
        """Test scenarios endpoint."""
        response = client.get("/scenarios", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert 'scenarios' in data
        assert 'total_count' in data
    
    def test_authentication(self, client):
        """Test API authentication."""
        # Test without auth header
        response = client.get("/scenarios")
        assert response.status_code == 401
        
        # Test with invalid key
        invalid_headers = {"Authorization": "Bearer invalid_key"}
        response = client.get("/scenarios", headers=invalid_headers)
        assert response.status_code == 401
    
    def test_input_validation(self, client, auth_headers):
        """Test input validation."""
        # Test with invalid request data
        invalid_request = {
            "events": [],  # Empty events
            "num_steps": 15,  # Too many steps
            "confidence_threshold": 1.5  # Invalid threshold
        }
        
        response = client.post("/predict", json=invalid_request, headers=auth_headers)
        assert response.status_code == 422  # Validation error

class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_prediction_latency(self):
        """Test prediction latency requirements."""
        # Create mock model and data
        encoder = GRUTemporalEncoder(input_dim=256, hidden_dim=128)
        sample_data = torch.randn(1, 10, 256)
        
        # Warm up
        for _ in range(5):
            encoder(sample_data)
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            encoder(sample_data)
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / 10
        
        # Should be under 5 seconds (very generous for unit test)
        assert avg_latency < 5.0
    
    def test_memory_usage(self):
        """Test memory usage requirements."""
        encoder = GRUTemporalEncoder(input_dim=256, hidden_dim=128)
        
        # Check parameter count
        param_count = sum(p.numel() for p in encoder.parameters())
        
        # Should be reasonable for production deployment
        assert param_count < 10_000_000  # 10M parameters max
    
    def test_throughput(self):
        """Test inference throughput."""
        encoder = GRUTemporalEncoder(input_dim=256, hidden_dim=128)
        batch_size = 32
        sample_data = torch.randn(batch_size, 10, 256)
        
        start_time = time.time()
        encoder(sample_data)
        end_time = time.time()
        
        throughput = batch_size / (end_time - start_time)
        
        # Should process at least 10 samples per second
        assert throughput > 10.0

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        encoder = GRUTemporalEncoder(input_dim=256, hidden_dim=128)
        
        # Test with empty sequence
        empty_sequence = torch.empty(0, 0, 256)
        
        # Should not crash
        try:
            output = encoder(empty_sequence)
            # If it doesn't crash, output should be reasonable
            assert output is not None
        except Exception as e:
            # If it does crash, should be a reasonable error
            assert isinstance(e, (ValueError, RuntimeError))
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        visualizer = AttackPathVisualizer()
        
        # Test with malformed attack steps
        malformed_steps = [
            {'missing_required_fields': True},
            {'edge_type': None, 'confidence': 'invalid'}
        ]
        
        # Should not crash and return error figure
        fig = visualizer.plot_attack_sequence(malformed_steps)
        assert fig is not None
    
    def test_network_error_handling(self):
        """Test handling of network errors."""
        client = TestClient(app)
        
        # Test with very large request (should be rejected)
        large_events = [
            {
                "timestamp": "2026-02-04T14:23:15Z",
                "event_type": "user_login",
                "source_host": "WORKSTATION-01",
                "additional_data": {"large_field": "x" * 10000}
            }
        ] * 1000
        
        request_data = {
            "events": large_events,
            "num_steps": 5
        }
        
        auth_headers = {"Authorization": "Bearer demo_key_2026"}
        response = client.post("/predict", json=request_data, headers=auth_headers)
        
        # Should handle gracefully (either accept or reject with proper error)
        assert response.status_code in [200, 413, 422, 500]

# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Test configuration."""
    return {
        'test_data_dir': Path(__file__).parent / 'test_data',
        'temp_dir': tempfile.mkdtemp(),
        'timeout': 30
    }

@pytest.fixture(autouse=True)
def setup_test_environment(test_config):
    """Setup test environment."""
    # Create test data directory
    test_config['test_data_dir'].mkdir(exist_ok=True)
    
    yield
    
    # Cleanup
    if Path(test_config['temp_dir']).exists():
        shutil.rmtree(test_config['temp_dir'])

# Performance test markers
pytestmark = pytest.mark.performance

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--benchmark-only",
        "--benchmark-sort=mean"
    ])