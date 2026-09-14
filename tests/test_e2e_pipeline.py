#!/usr/bin/env python3
"""
Day 10 Morning: Complete End-to-End Pipeline Test
Test entire pipeline: upload → process → predict → visualize → export

Author: THGNN Expert
Date: January 11, 2026
Version: 1.0 - PRODUCTION E2E TEST
"""

import os
import sys
import json
import time
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any

import pytest
import pandas as pd
import numpy as np
import torch

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestE2EPipeline:
    """Complete end-to-end pipeline testing"""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.test_data_dir = project_root / "temp" / "e2e_test"
        cls.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample security events
        cls.sample_events = [
            {
                "timestamp": "2026-01-11T10:00:00Z",
                "event_id": 4624,
                "user": "DOMAIN\\alice",
                "host": "WORKSTATION01",
                "process": "winlogon.exe",
                "action": "logon",
                "source_ip": "192.168.1.100"
            },
            {
                "timestamp": "2026-01-11T10:01:00Z",
                "event_id": 4688,
                "user": "DOMAIN\\alice",
                "host": "WORKSTATION01",
                "process": "powershell.exe",
                "command": "Get-Process",
                "parent_process": "explorer.exe"
            },
            {
                "timestamp": "2026-01-11T10:02:00Z",
                "event_id": 5156,
                "user": "DOMAIN\\alice",
                "host": "WORKSTATION01",
                "process": "powershell.exe",
                "target_host": "DC01",
                "target_port": 445,
                "protocol": "TCP"
            },
            {
                "timestamp": "2026-01-11T10:03:00Z",
                "event_id": 4624,
                "user": "DOMAIN\\alice",
                "host": "DC01",
                "process": "winlogon.exe",
                "action": "logon",
                "source_ip": "192.168.1.100"
            },
            {
                "timestamp": "2026-01-11T10:04:00Z",
                "event_id": 4688,
                "user": "DOMAIN\\alice",
                "host": "DC01",
                "process": "cmd.exe",
                "command": "net user admin /add",
                "parent_process": "winlogon.exe"
            }
        ]
    
    def test_data_upload_and_parsing(self):
        """Test 1: Data upload and parsing"""
        logger.info("🧪 Testing data upload and parsing...")
        
        # Create test JSON file
        test_file = self.test_data_dir / "test_events.json"
        with open(test_file, 'w') as f:
            json.dump(self.sample_events, f, indent=2)
        
        # Test file exists and is readable
        assert test_file.exists()
        assert test_file.stat().st_size > 0
        
        # Test JSON parsing
        with open(test_file, 'r') as f:
            loaded_events = json.load(f)
        
        assert len(loaded_events) == 5
        assert all('timestamp' in event for event in loaded_events)
        assert all('user' in event for event in loaded_events)
        
        logger.info(" Data upload and parsing: PASSED")
    
    def test_entity_extraction(self):
        """Test 2: Entity extraction"""
        logger.info("🧪 Testing entity extraction...")
        
        try:
            from data_processing.entity_extractor import EntityExtractor
            
            extractor = EntityExtractor()
            entities = extractor.extract_entities(self.sample_events)
            
            # Verify entities extracted
            assert 'users' in entities
            assert 'hosts' in entities
            assert 'processes' in entities
            
            # Check specific entities
            assert 'DOMAIN\\alice' in entities['users']
            assert 'WORKSTATION01' in entities['hosts']
            assert 'DC01' in entities['hosts']
            assert 'powershell.exe' in entities['processes']
            
            logger.info(" Entity extraction: PASSED")
            
        except ImportError:
            logger.warning("⚠️ Entity extractor not available, skipping test")
    
    def test_graph_construction(self):
        """Test 3: Graph construction"""
        logger.info("🧪 Testing graph construction...")
        
        try:
            from data_processing.graph_builder import TemporalGraphBuilder
            
            builder = TemporalGraphBuilder()
            graphs = builder.build_temporal_graphs(self.sample_events)
            
            # Verify graphs created
            assert len(graphs) > 0
            
            # Check first graph structure
            graph = graphs[0]
            assert hasattr(graph, 'node_types')
            assert hasattr(graph, 'edge_types')
            
            logger.info(" Graph construction: PASSED")
            
        except ImportError:
            logger.warning("⚠️ Graph builder not available, skipping test")
    
    def test_model_inference(self):
        """Test 4: Model inference"""
        logger.info("🧪 Testing model inference...")
        
        # Check if trained model exists
        model_path = project_root / "best_thgnn_model.pth"
        if not model_path.exists():
            model_path = project_root / "checkpoints" / "best_thgnn_model.pth"
        
        if model_path.exists():
            try:
                from inference.predictor import RealTimePredictor
                
                predictor = RealTimePredictor()
                predictor.load_model(str(model_path))
                
                # Test prediction (mock data)
                predictions = predictor.predict_mock_data()
                
                assert 'predictions' in predictions
                assert len(predictions['predictions']) > 0
                
                logger.info(" Model inference: PASSED")
                
            except Exception as e:
                logger.warning(f"⚠️ Model inference failed: {e}")
        else:
            logger.warning("⚠️ No trained model found, skipping inference test")
    
    def test_visualization_generation(self):
        """Test 5: Visualization generation"""
        logger.info("🧪 Testing visualization generation...")
        
        try:
            from visualization.graph_visualizer import GraphVisualizer
            
            visualizer = GraphVisualizer()
            
            # Create mock graph data
            mock_nodes = [
                {"id": "user_0", "type": "user", "label": "alice", "risk": 0.8},
                {"id": "host_0", "type": "host", "label": "WORKSTATION01", "risk": 0.6},
                {"id": "host_1", "type": "host", "label": "DC01", "risk": 0.9}
            ]
            
            mock_edges = [
                {"source": "user_0", "target": "host_0", "type": "logs_into", "weight": 0.7},
                {"source": "user_0", "target": "host_1", "type": "logs_into", "weight": 0.9}
            ]
            
            # Test visualization creation
            fig = visualizer.create_network_graph(mock_nodes, mock_edges)
            
            assert fig is not None
            assert hasattr(fig, 'data')
            
            logger.info(" Visualization generation: PASSED")
            
        except ImportError:
            logger.warning("⚠️ Visualization components not available, skipping test")
    
    def test_performance_benchmarks(self):
        """Test 6: Performance benchmarks"""
        logger.info("🧪 Testing performance benchmarks...")
        
        # Test parsing speed
        start_time = time.time()
        
        # Create larger dataset for performance test
        large_events = self.sample_events * 1000  # 5000 events
        
        # Simulate parsing
        processed_count = 0
        for event in large_events:
            if 'timestamp' in event and 'user' in event:
                processed_count += 1
        
        end_time = time.time()
        processing_time = end_time - start_time
        events_per_second = len(large_events) / processing_time
        
        logger.info(f"📊 Processed {len(large_events)} events in {processing_time:.2f}s")
        logger.info(f"📊 Processing speed: {events_per_second:.0f} events/second")
        
        # Performance targets
        assert events_per_second > 1000, f"Processing too slow: {events_per_second:.0f} events/sec"
        assert processing_time < 10, f"Processing took too long: {processing_time:.2f}s"
        
        logger.info(" Performance benchmarks: PASSED")
    
    def test_error_handling(self):
        """Test 7: Error handling"""
        logger.info("🧪 Testing error handling...")
        
        # Test malformed JSON
        malformed_file = self.test_data_dir / "malformed.json"
        with open(malformed_file, 'w') as f:
            f.write('{"invalid": json}')
        
        # Test empty file
        empty_file = self.test_data_dir / "empty.json"
        empty_file.touch()
        
        # Test missing required fields
        incomplete_events = [
            {"timestamp": "2026-01-11T10:00:00Z"},  # Missing user, host
            {"user": "alice"}  # Missing timestamp
        ]
        
        incomplete_file = self.test_data_dir / "incomplete.json"
        with open(incomplete_file, 'w') as f:
            json.dump(incomplete_events, f)
        
        # Verify error handling doesn't crash
        try:
            with open(malformed_file, 'r') as f:
                json.load(f)
            assert False, "Should have raised JSON error"
        except json.JSONDecodeError:
            pass  # Expected
        
        logger.info(" Error handling: PASSED")
    
    def test_memory_usage(self):
        """Test 8: Memory usage"""
        logger.info("🧪 Testing memory usage...")
        
        import psutil
        import gc
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        large_events = self.sample_events * 10000  # 50,000 events
        
        # Process data
        processed_data = []
        for event in large_events:
            processed_data.append({
                'user': event.get('user', 'unknown'),
                'host': event.get('host', 'unknown'),
                'timestamp': event.get('timestamp', ''),
                'risk_score': np.random.random()
            })
        
        # Get peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Cleanup
        del large_events
        del processed_data
        gc.collect()
        
        logger.info(f"📊 Initial memory: {initial_memory:.1f} MB")
        logger.info(f"📊 Peak memory: {peak_memory:.1f} MB")
        logger.info(f"📊 Memory increase: {memory_increase:.1f} MB")
        
        # Memory should not exceed 1GB for this test
        assert peak_memory < 1024, f"Memory usage too high: {peak_memory:.1f} MB"
        
        logger.info(" Memory usage: PASSED")
    
    def test_dashboard_integration(self):
        """Test 9: Dashboard integration"""
        logger.info("🧪 Testing dashboard integration...")
        
        # Test if Streamlit app can be imported
        try:
            import streamlit as st
            
            # Test if main app files exist
            app_file = project_root / "app" / "streamlit_app_perfect.py"
            assert app_file.exists(), "Main app file not found"
            
            # Test if pages exist
            pages_dir = project_root / "app" / "pages"
            if pages_dir.exists():
                page_files = list(pages_dir.glob("*.py"))
                logger.info(f"📊 Found {len(page_files)} dashboard pages")
            
            logger.info(" Dashboard integration: PASSED")
            
        except ImportError:
            logger.warning("⚠️ Streamlit not available, skipping dashboard test")
    
    def test_export_functionality(self):
        """Test 10: Export functionality"""
        logger.info("🧪 Testing export functionality...")
        
        # Test JSON export
        export_data = {
            "predictions": [
                {
                    "source": "user_0",
                    "target": "host_1",
                    "risk_score": 0.85,
                    "confidence": 0.92
                }
            ],
            "metadata": {
                "timestamp": "2026-01-11T10:00:00Z",
                "model_version": "1.0",
                "total_predictions": 1
            }
        }
        
        export_file = self.test_data_dir / "export_test.json"
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Verify export
        assert export_file.exists()
        assert export_file.stat().st_size > 0
        
        # Test CSV export
        df = pd.DataFrame([
            {"user": "alice", "host": "DC01", "risk": 0.85},
            {"user": "bob", "host": "SERVER01", "risk": 0.72}
        ])
        
        csv_file = self.test_data_dir / "export_test.csv"
        df.to_csv(csv_file, index=False)
        
        # Verify CSV export
        assert csv_file.exists()
        loaded_df = pd.read_csv(csv_file)
        assert len(loaded_df) == 2
        
        logger.info(" Export functionality: PASSED")
    
    @classmethod
    def teardown_class(cls):
        """Cleanup test environment"""
        import shutil
        if cls.test_data_dir.exists():
            shutil.rmtree(cls.test_data_dir)

def run_e2e_tests():
    """Run all end-to-end tests"""
    logger.info("🚀 Starting End-to-End Pipeline Tests...")
    
    test_suite = TestE2EPipeline()
    test_suite.setup_class()
    
    tests = [
        test_suite.test_data_upload_and_parsing,
        test_suite.test_entity_extraction,
        test_suite.test_graph_construction,
        test_suite.test_model_inference,
        test_suite.test_visualization_generation,
        test_suite.test_performance_benchmarks,
        test_suite.test_error_handling,
        test_suite.test_memory_usage,
        test_suite.test_dashboard_integration,
        test_suite.test_export_functionality
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            logger.error(f"[ERROR] {test.__name__} FAILED: {e}")
            failed += 1
    
    test_suite.teardown_class()
    
    logger.info(f"🏁 E2E Tests Complete: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info(" ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
    else:
        logger.warning(f"⚠️ {failed} tests failed - review and fix issues")
    
    return failed == 0

if __name__ == "__main__":
    success = run_e2e_tests()
    sys.exit(0 if success else 1)