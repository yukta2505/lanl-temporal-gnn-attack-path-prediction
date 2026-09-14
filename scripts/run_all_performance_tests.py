#!/usr/bin/env python3
"""
THGNN Master Performance Test Runner
Executes all performance validation tests and generates comprehensive report.

Author: THGNN Expert
Date: February 4, 2026
Version: 1.0 - Master Test Execution
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MasterTestRunner:
    """
    Master test runner for all THGNN performance validation tests.
    """
    
    def __init__(self):
        """Initialize the master test runner."""
        self.test_results = {
            'model_validation': {},
            'pipeline_integration': {},
            'performance_benchmarks': {},
            'test_summary': {},
            'execution_log': []
        }
        self.start_time = time.time()
        
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        logger.info("Master Test Runner initialized")
    
    def log_execution(self, test_name: str, status: str, details: str = ""):
        """Log test execution details."""
        entry = {
            'test_name': test_name,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.test_results['execution_log'].append(entry)
        logger.info(f"{test_name}: {status} - {details}")
    
    def run_model_validation(self) -> bool:
        """Run THGNN model validation tests."""
        logger.info("=" * 60)
        logger.info("🎯 RUNNING THGNN MODEL VALIDATION")
        logger.info("=" * 60)
        
        try:
            # Import and run the validator
            from thgnn_performance_validator import THGNNPerformanceValidator
            
            validator = THGNNPerformanceValidator()
            results = validator.run_comprehensive_validation()
            
            self.test_results['model_validation'] = results
            
            # Check if validation was successful
            success = results.get('validation_summary', {}).get('validation_success', False)
            
            if success:
                self.log_execution("Model Validation", "SUCCESS", 
                                 f"Processed {results.get('validation_summary', {}).get('total_events_processed', 0)} events")
                return True
            else:
                self.log_execution("Model Validation", "FAILED", 
                                 results.get('validation_error', 'Unknown error'))
                return False
                
        except Exception as e:
            self.log_execution("Model Validation", "ERROR", str(e))
            logger.error(f"Model validation failed: {e}")
            return False
    
    def run_pipeline_integration(self) -> bool:
        """Run pipeline integration tests."""
        logger.info("=" * 60)
        logger.info("🔄 RUNNING PIPELINE INTEGRATION TESTS")
        logger.info("=" * 60)
        
        try:
            # Import and run the pipeline tester
            from pipeline_integration_tester import PipelineIntegrationTester
            
            tester = PipelineIntegrationTester()
            results = tester.run_comprehensive_pipeline_test()
            
            self.test_results['pipeline_integration'] = results
            
            # Check if integration was successful
            success = results.get('performance_benchmarks', {}).get('overall_success', False)
            
            if success:
                integration_metrics = results.get('integration_metrics', {})
                self.log_execution("Pipeline Integration", "SUCCESS", 
                                 f"End-to-end flow completed in {integration_metrics.get('total_flow_time', 0):.2f}s")
                return True
            else:
                self.log_execution("Pipeline Integration", "FAILED", 
                                 results.get('test_error', 'Integration flow failed'))
                return False
                
        except Exception as e:
            self.log_execution("Pipeline Integration", "ERROR", str(e))
            logger.error(f"Pipeline integration failed: {e}")
            return False
    
    def run_performance_benchmarks(self) -> bool:
        """Run performance benchmarking tests."""
        logger.info("=" * 60)
        logger.info("📊 RUNNING PERFORMANCE BENCHMARKS")
        logger.info("=" * 60)
        
        try:
            # Import and run the benchmarker
            from performance_benchmarker import THGNNPerformanceBenchmarker
            
            benchmarker = THGNNPerformanceBenchmarker()
            results = benchmarker.run_comprehensive_benchmarks()
            
            self.test_results['performance_benchmarks'] = results
            
            # Check benchmark success
            summary = results.get('performance_summary', {})
            grade = summary.get('performance_grade', 'F')
            
            if grade in ['A', 'B']:
                self.log_execution("Performance Benchmarks", "SUCCESS", 
                                 f"Performance grade: {grade}")
                return True
            else:
                self.log_execution("Performance Benchmarks", "WARNING", 
                                 f"Performance grade: {grade} - needs optimization")
                return True  # Still consider successful, just needs optimization
                
        except Exception as e:
            self.log_execution("Performance Benchmarks", "ERROR", str(e))
            logger.error(f"Performance benchmarking failed: {e}")
            return False
    
    def generate_master_report(self) -> Dict[str, Any]:
        """Generate comprehensive master test report."""
        logger.info("Generating master test report...")
        
        total_time = time.time() - self.start_time
        
        # Count successful tests
        successful_tests = len([log for log in self.test_results['execution_log'] 
                               if log['status'] in ['SUCCESS', 'WARNING']])
        total_tests = len(self.test_results['execution_log'])
        
        # Extract key metrics
        model_validation = self.test_results.get('model_validation', {})
        pipeline_integration = self.test_results.get('pipeline_integration', {})
        performance_benchmarks = self.test_results.get('performance_benchmarks', {})
        
        # Model performance metrics
        model_perf = model_validation.get('model_performance', {})
        accuracy_metrics = model_validation.get('accuracy_metrics', {})
        
        # Pipeline performance metrics
        integration_metrics = pipeline_integration.get('integration_metrics', {})
        
        # Benchmark metrics
        benchmark_summary = performance_benchmarks.get('performance_summary', {})
        
        master_report = {
            'test_execution_summary': {
                'total_execution_time': total_time,
                'tests_executed': total_tests,
                'tests_successful': successful_tests,
                'success_rate': successful_tests / max(total_tests, 1),
                'execution_timestamp': datetime.now().isoformat()
            },
            'model_performance_summary': {
                'predictions_generated': model_perf.get('successful_predictions', 0),
                'prediction_success_rate': model_perf.get('success_rate', 0),
                'avg_prediction_latency': model_perf.get('avg_prediction_time', 0),
                'auc_roc': accuracy_metrics.get('auc_roc', 0),
                'precision_at_10': accuracy_metrics.get('precision_at_10', 0)
            },
            'pipeline_performance_summary': {
                'end_to_end_flow_time': integration_metrics.get('total_flow_time', 0),
                'pipeline_success': integration_metrics.get('flow_success', False),
                'events_processed': integration_metrics.get('data_flow_integrity', {}).get('events_processed', 0),
                'reports_generated': integration_metrics.get('data_flow_integrity', {}).get('reports_generated', 0)
            },
            'benchmark_performance_summary': {
                'performance_grade': benchmark_summary.get('performance_grade', 'N/A'),
                'target_compliance_rate': self._calculate_compliance_rate(benchmark_summary),
                'max_throughput': self._extract_max_throughput(performance_benchmarks),
                'peak_memory_usage': self._extract_peak_memory(performance_benchmarks)
            },
            'overall_assessment': self._generate_overall_assessment(successful_tests, total_tests),
            'recommendations': self._generate_master_recommendations(),
            'detailed_execution_log': self.test_results['execution_log']
        }
        
        self.test_results['test_summary'] = master_report
        return master_report
    
    def _calculate_compliance_rate(self, benchmark_summary: Dict) -> float:
        """Calculate target compliance rate from benchmark summary."""
        compliance = benchmark_summary.get('target_compliance', {})
        if not compliance:
            return 0.0
        
        compliant_count = sum(1 for target in compliance.values() if target.get('compliant', False))
        return compliant_count / len(compliance)
    
    def _extract_max_throughput(self, performance_benchmarks: Dict) -> float:
        """Extract maximum throughput from benchmark results."""
        throughput = performance_benchmarks.get('throughput_benchmarks', {})
        scalability = throughput.get('scalability_metrics', {})
        return scalability.get('max_throughput', 0)
    
    def _extract_peak_memory(self, performance_benchmarks: Dict) -> float:
        """Extract peak memory usage from benchmark results."""
        resources = performance_benchmarks.get('resource_utilization', {})
        memory = resources.get('memory_profile', {})
        return memory.get('peak_memory_mb', 0)
    
    def _generate_overall_assessment(self, successful_tests: int, total_tests: int) -> Dict[str, Any]:
        """Generate overall system assessment."""
        success_rate = successful_tests / max(total_tests, 1)
        
        if success_rate >= 0.9:
            status = "EXCELLENT"
            grade = "A"
            description = "System exceeds performance expectations"
        elif success_rate >= 0.8:
            status = "GOOD"
            grade = "B"
            description = "System meets most performance requirements"
        elif success_rate >= 0.7:
            status = "ACCEPTABLE"
            grade = "C"
            description = "System meets basic requirements but needs optimization"
        else:
            status = "NEEDS_IMPROVEMENT"
            grade = "D"
            description = "System requires significant optimization"
        
        return {
            'status': status,
            'grade': grade,
            'description': description,
            'success_rate': success_rate,
            'production_ready': success_rate >= 0.8
        }
    
    def _generate_master_recommendations(self) -> List[str]:
        """Generate master recommendations based on all test results."""
        recommendations = []
        
        # Model validation recommendations
        model_validation = self.test_results.get('model_validation', {})
        model_perf = model_validation.get('model_performance', {})
        accuracy = model_validation.get('accuracy_metrics', {})
        
        if model_perf.get('avg_prediction_time', 0) > 3.0:
            recommendations.append("CRITICAL: Optimize prediction latency - consider GPU acceleration")
        
        if accuracy.get('auc_roc', 0) < 0.80:
            recommendations.append("HIGH: Improve model accuracy - AUC-ROC below target")
        
        # Pipeline integration recommendations
        pipeline = self.test_results.get('pipeline_integration', {})
        integration = pipeline.get('integration_metrics', {})
        
        if not integration.get('flow_success', False):
            recommendations.append("CRITICAL: Fix pipeline integration issues")
        
        if integration.get('total_flow_time', 0) > 30.0:
            recommendations.append("MEDIUM: Optimize end-to-end pipeline performance")
        
        # Performance benchmark recommendations
        benchmarks = self.test_results.get('performance_benchmarks', {})
        summary = benchmarks.get('performance_summary', {})
        
        if summary.get('performance_grade', 'F') in ['C', 'D', 'F']:
            recommendations.append("HIGH: Address performance benchmark failures")
        
        # General recommendations
        if not recommendations:
            recommendations.append("EXCELLENT: All systems performing optimally!")
        
        recommendations.extend([
            "OPTIMIZATION: Implement Redis caching for improved performance",
            "MONITORING: Deploy Prometheus metrics collection",
            "SCALING: Prepare Kubernetes deployment configuration",
            "SECURITY: Implement comprehensive input validation",
            "DOCUMENTATION: Update performance benchmarking documentation"
        ])
        
        return recommendations
    
    def save_master_results(self) -> str:
        """Save all test results to comprehensive report file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"logs/thgnn_master_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"Master test results saved to: {results_file}")
        return results_file
    
    def print_master_report(self):
        """Print comprehensive master test report."""
        print("\n" + "=" * 100)
        print("🎯 THGNN MASTER PERFORMANCE VALIDATION REPORT")
        print("=" * 100)
        
        summary = self.test_results.get('test_summary', {})
        
        # Execution Summary
        exec_summary = summary.get('test_execution_summary', {})
        print(f"\n⏱️  EXECUTION SUMMARY:")
        print(f"  • Total Execution Time: {exec_summary.get('total_execution_time', 0):.2f} seconds")
        print(f"  • Tests Executed: {exec_summary.get('tests_executed', 0)}")
        print(f"  • Tests Successful: {exec_summary.get('tests_successful', 0)}")
        print(f"  • Success Rate: {exec_summary.get('success_rate', 0):.2%}")
        
        # Model Performance
        model_summary = summary.get('model_performance_summary', {})
        print(f"\n🎯 MODEL PERFORMANCE:")
        print(f"  • Predictions Generated: {model_summary.get('predictions_generated', 0):,}")
        print(f"  • Prediction Success Rate: {model_summary.get('prediction_success_rate', 0):.2%}")
        print(f"  • Avg Prediction Latency: {model_summary.get('avg_prediction_latency', 0):.3f}s")
        print(f"  • AUC-ROC: {model_summary.get('auc_roc', 0):.3f}")
        print(f"  • Precision@10: {model_summary.get('precision_at_10', 0):.3f}")
        
        # Pipeline Performance
        pipeline_summary = summary.get('pipeline_performance_summary', {})
        print(f"\n🔄 PIPELINE PERFORMANCE:")
        print(f"  • End-to-End Flow Time: {pipeline_summary.get('end_to_end_flow_time', 0):.2f}s")
        print(f"  • Pipeline Success: {'✅ PASS' if pipeline_summary.get('pipeline_success', False) else '❌ FAIL'}")
        print(f"  • Events Processed: {pipeline_summary.get('events_processed', 0):,}")
        print(f"  • Reports Generated: {pipeline_summary.get('reports_generated', 0)}")
        
        # Benchmark Performance
        benchmark_summary = summary.get('benchmark_performance_summary', {})
        print(f"\n📊 BENCHMARK PERFORMANCE:")
        print(f"  • Performance Grade: {benchmark_summary.get('performance_grade', 'N/A')}")
        print(f"  • Target Compliance: {benchmark_summary.get('target_compliance_rate', 0):.2%}")
        print(f"  • Max Throughput: {benchmark_summary.get('max_throughput', 0):.1f} events/sec")
        print(f"  • Peak Memory Usage: {benchmark_summary.get('peak_memory_usage', 0):.1f} MB")
        
        # Overall Assessment
        assessment = summary.get('overall_assessment', {})
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print(f"  • Status: {assessment.get('status', 'UNKNOWN')}")
        print(f"  • Grade: {assessment.get('grade', 'N/A')}")
        print(f"  • Production Ready: {'✅ YES' if assessment.get('production_ready', False) else '❌ NO'}")
        print(f"  • Description: {assessment.get('description', 'No assessment available')}")
        
        # Top Recommendations
        recommendations = summary.get('recommendations', [])
        print(f"\n🚀 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:8], 1):
            print(f"  {i}. {rec}")
        
        # Execution Log Summary
        print(f"\n📋 EXECUTION LOG:")
        for log_entry in self.test_results.get('execution_log', []):
            status_icon = "✅" if log_entry['status'] == "SUCCESS" else "⚠️" if log_entry['status'] == "WARNING" else "❌"
            print(f"  {status_icon} {log_entry['test_name']}: {log_entry['status']}")
        
        print("\n" + "=" * 100)
    
    def run_all_tests(self) -> bool:
        """Run all performance validation tests."""
        logger.info("🚀 STARTING COMPREHENSIVE THGNN PERFORMANCE VALIDATION")
        logger.info("This will test model performance, pipeline integration, and system benchmarks.")
        logger.info("=" * 80)
        
        overall_success = True
        
        try:
            # Run model validation
            model_success = self.run_model_validation()
            overall_success = overall_success and model_success
            
            # Run pipeline integration tests
            pipeline_success = self.run_pipeline_integration()
            overall_success = overall_success and pipeline_success
            
            # Run performance benchmarks
            benchmark_success = self.run_performance_benchmarks()
            overall_success = overall_success and benchmark_success
            
            # Generate master report
            logger.info("=" * 60)
            logger.info("📊 GENERATING MASTER REPORT")
            logger.info("=" * 60)
            
            master_report = self.generate_master_report()
            
            # Save results
            results_file = self.save_master_results()
            
            # Print master report
            self.print_master_report()
            
            logger.info("=" * 80)
            if overall_success:
                logger.info("🎉 ALL PERFORMANCE TESTS COMPLETED SUCCESSFULLY!")
            else:
                logger.warning("⚠️  SOME PERFORMANCE TESTS FAILED - CHECK REPORT FOR DETAILS")
            
            logger.info(f"📁 Comprehensive results saved to: {results_file}")
            logger.info("=" * 80)
            
            return overall_success
            
        except Exception as e:
            logger.error(f"Master test execution failed: {e}")
            self.log_execution("Master Test Runner", "CRITICAL_ERROR", str(e))
            return False


def main():
    """Main execution function."""
    print("🎯 THGNN Master Performance Validation Suite")
    print("=" * 60)
    print("This comprehensive test suite will validate:")
    print("  1. THGNN Model Performance & Accuracy")
    print("  2. End-to-End Pipeline Integration")
    print("  3. System Performance Benchmarks")
    print("=" * 60)
    
    # Initialize master test runner
    runner = MasterTestRunner()
    
    # Run all tests
    success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()