#!/usr/bin/env python3
"""
THGNN Ultimate System Management - Production Grade
The definitive system management script for THGNN operations.

Features:
- System validation and health checks
- Environment setup and configuration
- Data pipeline management
- Model deployment and monitoring
- Performance benchmarking
- Cleanup and maintenance

Commands:
- validate: Check system requirements and configuration
- setup: Initialize environment and data directories
- benchmark: Run performance benchmarks
- deploy: Deploy model for production inference
- monitor: Monitor system health and performance
- cleanup: Clean temporary files and optimize storage

Author: THGNN Expert
Version: 2.0 - ULTIMATE PRODUCTION
Date: February 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils.path_resolver import (
    get_project_root, get_data_dir, get_graphs_dir, 
    get_checkpoints_dir, validate_project_structure
)
import logging
import time
import warnings
import shutil
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

import torch
import numpy as np
import pandas as pd

# Suppress warnings for production
warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/thgnn_system.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemStatus:
    """System status information"""
    timestamp: str
    python_version: str
    pytorch_version: str
    cuda_available: bool
    cuda_version: str
    gpu_count: int
    gpu_memory_gb: float
    cpu_count: int
    ram_gb: float
    disk_space_gb: float
    data_directory_exists: bool
    model_files_found: int
    graph_files_found: int
    system_health: str
    performance_score: float

class UltimateTHGNNSystem:
    """
    Ultimate THGNN system management for production operations.
    
    Provides comprehensive system management:
    - Environment validation and setup
    - Performance monitoring and benchmarking
    - Data pipeline management
    - Model deployment and monitoring
    - System maintenance and cleanup
    """
    
    def __init__(self):
        """Initialize the ultimate system manager"""
        logger.info("🚀 Initializing Ultimate THGNN System Manager")
        
        self.project_root = project_root
        self.data_dir = get_data_dir()
        self.output_dir = project_root / "outputs"
        self.logs_dir = project_root / "logs"
        
        # Ensure directories exist
        for directory in [self.output_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ System manager initialized successfully")
    
    def validate_system(self) -> SystemStatus:
        """Comprehensive system validation"""
        logger.info("🔍 Running comprehensive system validation")
        
        # Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # PyTorch version and CUDA
        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        
        # GPU memory
        gpu_memory_gb = 0.0
        if cuda_available and gpu_count > 0:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # System resources
        cpu_count = psutil.cpu_count()
        ram_gb = psutil.virtual_memory().total / 1e9
        disk_space_gb = shutil.disk_usage(str(self.project_root)).free / 1e9
        
        # Data directory validation
        data_directory_exists = self.data_dir.exists()
        
        # Count model and graph files
        model_files_found = len(list(self.project_root.glob("**/*.pth")))
        graph_files_found = len(list((self.data_dir / "graphs").glob("*.pth"))) if data_directory_exists else 0
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(
            cuda_available, gpu_memory_gb, cpu_count, ram_gb
        )
        
        # Determine system health
        system_health = self._determine_system_health(
            python_version, cuda_available, gpu_memory_gb, ram_gb, 
            disk_space_gb, data_directory_exists
        )
        
        status = SystemStatus(
            timestamp=datetime.now().isoformat(),
            python_version=python_version,
            pytorch_version=pytorch_version,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            cpu_count=cpu_count,
            ram_gb=ram_gb,
            disk_space_gb=disk_space_gb,
            data_directory_exists=data_directory_exists,
            model_files_found=model_files_found,
            graph_files_found=graph_files_found,
            system_health=system_health,
            performance_score=performance_score
        )
        
        # Log validation results
        self._log_validation_results(status)
        
        return status
    
    def _calculate_performance_score(self, cuda_available: bool, gpu_memory_gb: float, 
                                   cpu_count: int, ram_gb: float) -> float:
        """Calculate system performance score (0-100)"""
        score = 0.0
        
        # GPU score (40 points)
        if cuda_available:
            if gpu_memory_gb >= 8:
                score += 40
            elif gpu_memory_gb >= 4:
                score += 30
            else:
                score += 20
        else:
            score += 10  # CPU only
        
        # CPU score (20 points)
        if cpu_count >= 8:
            score += 20
        elif cpu_count >= 4:
            score += 15
        else:
            score += 10
        
        # RAM score (20 points)
        if ram_gb >= 16:
            score += 20
        elif ram_gb >= 8:
            score += 15
        else:
            score += 10
        
        # Python version score (10 points)
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if python_version in ["3.10", "3.11"]:
            score += 10
        elif python_version == "3.9":
            score += 8
        else:
            score += 5
        
        # PyTorch version score (10 points)
        pytorch_version = torch.__version__
        if pytorch_version.startswith("2."):
            score += 10
        elif pytorch_version.startswith("1.13") or pytorch_version.startswith("1.12"):
            score += 8
        else:
            score += 5
        
        return min(score, 100.0)
    
    def _determine_system_health(self, python_version: str, cuda_available: bool, 
                               gpu_memory_gb: float, ram_gb: float, disk_space_gb: float,
                               data_directory_exists: bool) -> str:
        """Determine overall system health"""
        issues = []
        
        # Check Python version
        if not python_version.startswith(("3.10", "3.11")):
            issues.append("Python version not optimal (recommend 3.10 or 3.11)")
        
        # Check GPU
        if not cuda_available:
            issues.append("CUDA not available (training will be slower)")
        elif gpu_memory_gb < 4:
            issues.append("GPU memory insufficient (<4GB)")
        
        # Check RAM
        if ram_gb < 8:
            issues.append("RAM insufficient (<8GB)")
        
        # Check disk space
        if disk_space_gb < 10:
            issues.append("Low disk space (<10GB)")
        
        # Check data directory
        if not data_directory_exists:
            issues.append("Data directory not found")
        
        if len(issues) == 0:
            return "EXCELLENT"
        elif len(issues) <= 2:
            return "GOOD"
        elif len(issues) <= 4:
            return "FAIR"
        else:
            return "POOR"
    
    def _log_validation_results(self, status: SystemStatus) -> None:
        """Log detailed validation results"""
        logger.info("📊 System Validation Results:")
        logger.info(f"   Python: {status.python_version}")
        logger.info(f"   PyTorch: {status.pytorch_version}")
        logger.info(f"   CUDA: {'✅' if status.cuda_available else '❌'} {status.cuda_version}")
        logger.info(f"   GPU: {status.gpu_count} devices, {status.gpu_memory_gb:.1f}GB")
        logger.info(f"   CPU: {status.cpu_count} cores")
        logger.info(f"   RAM: {status.ram_gb:.1f}GB")
        logger.info(f"   Disk: {status.disk_space_gb:.1f}GB free")
        logger.info(f"   Data Dir: {'✅' if status.data_directory_exists else '❌'}")
        logger.info(f"   Models: {status.model_files_found} files")
        logger.info(f"   Graphs: {status.graph_files_found} files")
        logger.info(f"   Health: {status.system_health}")
        logger.info(f"   Performance Score: {status.performance_score:.1f}/100")
    
    def setup_environment(self) -> bool:
        """Setup THGNN environment and directories"""
        logger.info("🛠️  Setting up THGNN environment")
        
        try:
            # Create data directories
            directories = [
                self.data_dir,
                self.data_dir / "graphs",
                self.data_dir / "metadata",
                self.data_dir / "engineered",
                self.project_root / "checkpoints",
                self.project_root / "outputs" / "training",
                self.project_root / "outputs" / "inference",
                self.project_root / "outputs" / "reports",
                self.project_root / "logs",
                self.project_root / "temp" / "uploads"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created directory: {directory}")
            
            # Create configuration files if they don't exist
            self._create_default_configs()
            
            # Create sample data if needed
            self._create_sample_data()
            
            logger.info("✅ Environment setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment setup failed: {e}")
            return False
    
    def _create_default_configs(self) -> None:
        """Create default configuration files"""
        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Model configuration
        model_config = {
            "model": {
                "hidden_dim": 256,
                "num_conv_layers": 3,
                "temporal_hidden": 128,
                "dropout": 0.4,
                "use_uncertainty": True
            },
            "training": {
                "learning_rate": 0.001,
                "batch_size": 8,
                "num_epochs": 50,
                "early_stopping_patience": 10
            },
            "data": {
                "data_dir": "data",
                "graphs_dir": "data/graphs",
                "time_window_minutes": 5,
                "sequence_length": 10
            }
        }
        
        model_config_path = config_dir / "model_config.yaml"
        if not model_config_path.exists():
            import yaml
            with open(model_config_path, 'w') as f:
                yaml.dump(model_config, f, default_flow_style=False)
            logger.info(f"✅ Created model config: {model_config_path}")
    
    def _create_sample_data(self) -> None:
        """Create sample data for testing"""
        sample_file = self.project_root / "demo_final.json"
        if not sample_file.exists():
            sample_data = {
                "events": [
                    {
                        "timestamp": "2026-02-04T10:00:00Z",
                        "event_type": "process_creation",
                        "user": "alice",
                        "host": "WORKSTATION01",
                        "process": "powershell.exe",
                        "command_line": "powershell.exe -ExecutionPolicy Bypass",
                        "risk_score": 0.7
                    },
                    {
                        "timestamp": "2026-02-04T10:01:00Z",
                        "event_type": "network_connection",
                        "user": "alice",
                        "host": "WORKSTATION01",
                        "process": "powershell.exe",
                        "destination": "192.168.1.100",
                        "port": 445,
                        "risk_score": 0.8
                    }
                ]
            }
            
            with open(sample_file, 'w') as f:
                json.dump(sample_data, f, indent=2)
            logger.info(f"✅ Created sample data: {sample_file}")
    
    def benchmark_system(self) -> Dict[str, Any]:
        """Run comprehensive system benchmarks"""
        logger.info("⚡ Running system performance benchmarks")
        
        benchmarks = {}
        
        # CPU benchmark
        logger.info("🔄 Running CPU benchmark...")
        cpu_start = time.time()
        
        # Matrix multiplication benchmark
        size = 1000
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        c = np.dot(a, b)
        
        cpu_time = time.time() - cpu_start
        benchmarks['cpu_matrix_multiply_seconds'] = cpu_time
        
        # GPU benchmark (if available)
        if torch.cuda.is_available():
            logger.info("🔄 Running GPU benchmark...")
            gpu_start = time.time()
            
            device = torch.device('cuda')
            a_gpu = torch.randn(size, size, device=device)
            b_gpu = torch.randn(size, size, device=device)
            c_gpu = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
            
            gpu_time = time.time() - gpu_start
            benchmarks['gpu_matrix_multiply_seconds'] = gpu_time
            benchmarks['gpu_speedup'] = cpu_time / gpu_time if gpu_time > 0 else 0
        
        # Memory benchmark
        logger.info("🔄 Running memory benchmark...")
        memory_start = time.time()
        
        # Allocate and process large array
        large_array = np.random.rand(10000, 1000)
        result = np.sum(large_array, axis=1)
        
        memory_time = time.time() - memory_start
        benchmarks['memory_processing_seconds'] = memory_time
        
        # I/O benchmark
        logger.info("🔄 Running I/O benchmark...")
        io_start = time.time()
        
        # Write and read test file
        test_file = self.project_root / "temp" / "benchmark_test.json"
        test_file.parent.mkdir(exist_ok=True)
        
        test_data = {"data": list(range(10000))}
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        test_file.unlink()  # Clean up
        
        io_time = time.time() - io_start
        benchmarks['io_processing_seconds'] = io_time
        
        # Calculate overall performance score
        performance_score = self._calculate_benchmark_score(benchmarks)
        benchmarks['overall_performance_score'] = performance_score
        
        logger.info("✅ Benchmark completed successfully")
        self._log_benchmark_results(benchmarks)
        
        return benchmarks
    
    def _calculate_benchmark_score(self, benchmarks: Dict[str, Any]) -> float:
        """Calculate overall performance score from benchmarks"""
        score = 100.0
        
        # CPU performance (lower is better)
        cpu_time = benchmarks.get('cpu_matrix_multiply_seconds', 10)
        if cpu_time > 5:
            score -= 20
        elif cpu_time > 2:
            score -= 10
        
        # GPU performance (if available)
        if 'gpu_speedup' in benchmarks:
            speedup = benchmarks['gpu_speedup']
            if speedup > 10:
                score += 10
            elif speedup > 5:
                score += 5
        else:
            score -= 15  # No GPU penalty
        
        # Memory performance
        memory_time = benchmarks.get('memory_processing_seconds', 5)
        if memory_time > 2:
            score -= 10
        
        # I/O performance
        io_time = benchmarks.get('io_processing_seconds', 2)
        if io_time > 1:
            score -= 10
        
        return max(score, 0.0)
    
    def _log_benchmark_results(self, benchmarks: Dict[str, Any]) -> None:
        """Log benchmark results"""
        logger.info("📊 Benchmark Results:")
        logger.info(f"   CPU Matrix Multiply: {benchmarks.get('cpu_matrix_multiply_seconds', 0):.3f}s")
        
        if 'gpu_matrix_multiply_seconds' in benchmarks:
            logger.info(f"   GPU Matrix Multiply: {benchmarks['gpu_matrix_multiply_seconds']:.3f}s")
            logger.info(f"   GPU Speedup: {benchmarks['gpu_speedup']:.1f}x")
        
        logger.info(f"   Memory Processing: {benchmarks.get('memory_processing_seconds', 0):.3f}s")
        logger.info(f"   I/O Processing: {benchmarks.get('io_processing_seconds', 0):.3f}s")
        logger.info(f"   Overall Score: {benchmarks.get('overall_performance_score', 0):.1f}/100")
    
    def cleanup_system(self) -> Dict[str, Any]:
        """Clean up temporary files and optimize storage"""
        logger.info("🧹 Starting system cleanup")
        
        cleanup_stats = {
            'files_removed': 0,
            'space_freed_mb': 0,
            'directories_cleaned': []
        }
        
        # Directories to clean
        cleanup_dirs = [
            self.project_root / "temp",
            self.project_root / "__pycache__",
            self.project_root / "src" / "__pycache__",
            self.project_root / "scripts" / "__pycache__",
            self.logs_dir
        ]
        
        for cleanup_dir in cleanup_dirs:
            if cleanup_dir.exists():
                initial_size = self._get_directory_size(cleanup_dir)
                
                if cleanup_dir.name == "logs":
                    # Keep only recent log files
                    self._cleanup_logs(cleanup_dir)
                else:
                    # Remove entire directory
                    shutil.rmtree(cleanup_dir, ignore_errors=True)
                    cleanup_dir.mkdir(exist_ok=True)
                
                final_size = self._get_directory_size(cleanup_dir) if cleanup_dir.exists() else 0
                space_freed = (initial_size - final_size) / 1024 / 1024  # MB
                
                cleanup_stats['space_freed_mb'] += space_freed
                cleanup_stats['directories_cleaned'].append(str(cleanup_dir))
                
                logger.info(f"✅ Cleaned {cleanup_dir}: {space_freed:.1f}MB freed")
        
        # Clean temporary model files
        temp_models = list(self.project_root.glob("**/checkpoint_epoch_*.pth"))
        for model_file in temp_models[:-5]:  # Keep last 5 checkpoints
            file_size = model_file.stat().st_size / 1024 / 1024
            model_file.unlink()
            cleanup_stats['files_removed'] += 1
            cleanup_stats['space_freed_mb'] += file_size
        
        logger.info(f"✅ Cleanup completed: {cleanup_stats['files_removed']} files, {cleanup_stats['space_freed_mb']:.1f}MB freed")
        
        return cleanup_stats
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except (OSError, PermissionError):
            pass
        return total_size
    
    def _cleanup_logs(self, logs_dir: Path) -> None:
        """Clean up old log files, keeping only recent ones"""
        if not logs_dir.exists():
            return
        
        # Keep logs from last 7 days
        cutoff_time = time.time() - (7 * 24 * 60 * 60)
        
        for log_file in logs_dir.glob("*.log"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
            except (OSError, PermissionError):
                pass
    
    def monitor_system(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Monitor system performance for specified duration"""
        logger.info(f"📊 Monitoring system performance for {duration_seconds} seconds")
        
        monitoring_data = {
            'duration_seconds': duration_seconds,
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'disk_usage': [],
            'timestamps': []
        }
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            current_time = time.time()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            monitoring_data['cpu_usage'].append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            monitoring_data['memory_usage'].append(memory_percent)
            
            # GPU usage (if available)
            gpu_percent = 0
            if torch.cuda.is_available():
                try:
                    # This would require nvidia-ml-py for real GPU monitoring
                    # For now, use memory allocation as proxy
                    gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                    gpu_percent = min(gpu_memory, 100)
                except:
                    gpu_percent = 0
            
            monitoring_data['gpu_usage'].append(gpu_percent)
            
            # Disk usage
            disk = shutil.disk_usage(str(self.project_root))
            disk_percent = (disk.used / disk.total) * 100
            monitoring_data['disk_usage'].append(disk_percent)
            
            monitoring_data['timestamps'].append(current_time)
            
            time.sleep(1)
        
        # Calculate statistics
        monitoring_stats = {
            'avg_cpu_usage': np.mean(monitoring_data['cpu_usage']),
            'max_cpu_usage': np.max(monitoring_data['cpu_usage']),
            'avg_memory_usage': np.mean(monitoring_data['memory_usage']),
            'max_memory_usage': np.max(monitoring_data['memory_usage']),
            'avg_gpu_usage': np.mean(monitoring_data['gpu_usage']),
            'max_gpu_usage': np.max(monitoring_data['gpu_usage']),
            'avg_disk_usage': np.mean(monitoring_data['disk_usage'])
        }
        
        logger.info("📊 Monitoring Results:")
        logger.info(f"   CPU: {monitoring_stats['avg_cpu_usage']:.1f}% avg, {monitoring_stats['max_cpu_usage']:.1f}% max")
        logger.info(f"   Memory: {monitoring_stats['avg_memory_usage']:.1f}% avg, {monitoring_stats['max_memory_usage']:.1f}% max")
        logger.info(f"   GPU: {monitoring_stats['avg_gpu_usage']:.1f}% avg, {monitoring_stats['max_gpu_usage']:.1f}% max")
        logger.info(f"   Disk: {monitoring_stats['avg_disk_usage']:.1f}% used")
        
        return {**monitoring_data, **monitoring_stats}

def main():
    """Main system management function"""
    parser = argparse.ArgumentParser(description='Ultimate THGNN System Management')
    parser.add_argument('command', choices=['validate', 'setup', 'benchmark', 'cleanup', 'monitor'],
                       help='System management command')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--duration', type=int, default=60, help='Monitoring duration in seconds')
    
    args = parser.parse_args()
    
    try:
        # Create system manager
        system = UltimateTHGNNSystem()
        
        # Execute command
        if args.command == 'validate':
            result = system.validate_system()
            print(f"\n🎯 System Health: {result.system_health}")
            print(f"📊 Performance Score: {result.performance_score:.1f}/100")
            
        elif args.command == 'setup':
            success = system.setup_environment()
            if success:
                print("\n✅ Environment setup completed successfully!")
            else:
                print("\n❌ Environment setup failed!")
                sys.exit(1)
                
        elif args.command == 'benchmark':
            result = system.benchmark_system()
            print(f"\n⚡ Performance Score: {result['overall_performance_score']:.1f}/100")
            
        elif args.command == 'cleanup':
            result = system.cleanup_system()
            print(f"\n🧹 Cleanup completed: {result['space_freed_mb']:.1f}MB freed")
            
        elif args.command == 'monitor':
            result = system.monitor_system(args.duration)
            print(f"\n📊 Monitoring completed: {args.duration}s")
            print(f"   CPU: {result['avg_cpu_usage']:.1f}% avg")
            print(f"   Memory: {result['avg_memory_usage']:.1f}% avg")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(result) if hasattr(result, '__dict__') else result, 
                         f, indent=2, default=str)
            print(f"📄 Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"❌ System management failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()