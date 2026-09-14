#!/usr/bin/env python3
"""
THGNN Master Control Script - Ultimate Production Interface
The definitive command-line interface for all THGNN operations.

This is the single entry point for all THGNN functionality:
- Training: Train new models with optimal configurations
- Inference: Run threat predictions on security logs
- Reports: Generate professional analysis reports
- System: Manage system health and performance
- Dashboard: Launch the interactive web interface

Usage Examples:
  python thgnn.py train --data data --epochs 50
  python thgnn.py predict --logs security_events.json --output threats.json
  python thgnn.py report --input threats.json --format pdf
  python thgnn.py system validate
  python thgnn.py dashboard --port 8501

Author: THGNN Expert
Version: 2.0 - ULTIMATE PRODUCTION
Date: February 2026
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class THGNNMasterController:
    """
    Master controller for all THGNN operations.
    
    Provides a unified interface to:
    - Training pipeline (thgnn_train.py)
    - Inference engine (thgnn_inference.py)
    - Report generator (thgnn_report.py)
    - System management (thgnn_system.py)
    - Dashboard application (streamlit_app_perfect.py)
    """
    
    def __init__(self):
        """Initialize the master controller"""
        self.project_root = project_root
        self.scripts_dir = project_root / "scripts"
        self.app_dir = project_root / "app"
        
        # Script mappings
        self.scripts = {
            'train': self.scripts_dir / 'thgnn_train.py',
            'predict': self.scripts_dir / 'thgnn_inference.py',
            'report': self.scripts_dir / 'thgnn_report.py',
            'system': self.scripts_dir / 'thgnn_system.py',
            'dashboard': self.app_dir / 'streamlit_app_perfect.py'
        }
        
        # Validate scripts exist
        self._validate_scripts()
    
    def _validate_scripts(self) -> None:
        """Validate that all required scripts exist"""
        missing_scripts = []
        
        for name, script_path in self.scripts.items():
            if not script_path.exists():
                missing_scripts.append(f"{name}: {script_path}")
        
        if missing_scripts:
            logger.error("❌ Missing required scripts:")
            for script in missing_scripts:
                logger.error(f"   {script}")
            sys.exit(1)
        
        logger.info("✅ All THGNN scripts validated successfully")
    
    def run_training(self, args: argparse.Namespace) -> int:
        """Execute training pipeline"""
        logger.info("🚀 Starting THGNN Training Pipeline")
        
        cmd = [sys.executable, str(self.scripts['train'])]
        
        # Add training arguments
        if hasattr(args, 'config') and args.config:
            cmd.extend(['--config', args.config])
        if hasattr(args, 'data_dir') and args.data_dir:
            cmd.extend(['--data-dir', args.data_dir])
        if hasattr(args, 'epochs') and args.epochs:
            cmd.extend(['--epochs', str(args.epochs)])
        if hasattr(args, 'batch_size') and args.batch_size:
            cmd.extend(['--batch-size', str(args.batch_size)])
        if hasattr(args, 'learning_rate') and args.learning_rate:
            cmd.extend(['--learning-rate', str(args.learning_rate)])
        if hasattr(args, 'gpu') and args.gpu:
            cmd.append('--gpu')
        
        return subprocess.run(cmd).returncode
    
    def run_inference(self, args: argparse.Namespace) -> int:
        """Execute inference pipeline"""
        logger.info("🔮 Starting THGNN Inference Pipeline")
        
        cmd = [sys.executable, str(self.scripts['predict'])]
        
        # Add inference arguments
        if hasattr(args, 'logs') and args.logs:
            cmd.extend(['--logs', args.logs])
        if hasattr(args, 'output') and args.output:
            cmd.extend(['--output', args.output])
        if hasattr(args, 'model') and args.model:
            cmd.extend(['--model', args.model])
        if hasattr(args, 'batch') and args.batch:
            cmd.append('--batch')
        if hasattr(args, 'top_k') and args.top_k:
            cmd.extend(['--top-k', str(args.top_k)])
        if hasattr(args, 'risk_threshold') and args.risk_threshold:
            cmd.extend(['--risk-threshold', str(args.risk_threshold)])
        if hasattr(args, 'verbose') and args.verbose:
            cmd.append('--verbose')
        
        return subprocess.run(cmd).returncode
    
    def run_reporting(self, args: argparse.Namespace) -> int:
        """Execute report generation"""
        logger.info("📊 Starting THGNN Report Generation")
        
        cmd = [sys.executable, str(self.scripts['report'])]
        
        # Add reporting arguments
        if hasattr(args, 'input') and args.input:
            cmd.extend(['--input', args.input])
        if hasattr(args, 'output') and args.output:
            cmd.extend(['--output', args.output])
        if hasattr(args, 'title') and args.title:
            cmd.extend(['--title', args.title])
        if hasattr(args, 'organization') and args.organization:
            cmd.extend(['--organization', args.organization])
        if hasattr(args, 'classification') and args.classification:
            cmd.extend(['--classification', args.classification])
        if hasattr(args, 'formats') and args.formats:
            cmd.extend(['--formats'] + args.formats)
        if hasattr(args, 'no_executive') and args.no_executive:
            cmd.append('--no-executive')
        if hasattr(args, 'no_technical') and args.no_technical:
            cmd.append('--no-technical')
        
        return subprocess.run(cmd).returncode
    
    def run_system_management(self, args: argparse.Namespace) -> int:
        """Execute system management"""
        logger.info("🛠️  Starting THGNN System Management")
        
        cmd = [sys.executable, str(self.scripts['system'])]
        
        # Add system arguments
        if hasattr(args, 'system_command') and args.system_command:
            cmd.append(args.system_command)
        if hasattr(args, 'output') and args.output:
            cmd.extend(['--output', args.output])
        if hasattr(args, 'duration') and args.duration:
            cmd.extend(['--duration', str(args.duration)])
        
        return subprocess.run(cmd).returncode
    
    def run_dashboard(self, args: argparse.Namespace) -> int:
        """Launch dashboard application"""
        logger.info("🌐 Starting THGNN Dashboard")
        
        cmd = ['streamlit', 'run', str(self.scripts['dashboard'])]
        
        # Add dashboard arguments
        if hasattr(args, 'port') and args.port:
            cmd.extend(['--server.port', str(args.port)])
        if hasattr(args, 'host') and args.host:
            cmd.extend(['--server.address', args.host])
        
        return subprocess.run(cmd).returncode
    
    def show_status(self) -> None:
        """Show THGNN system status"""
        print("\n" + "="*80)
        print("🛡️  THGNN - Temporal Heterogeneous Graph Neural Network")
        print("    Ultimate Cybersecurity Threat Detection System")
        print("="*80)
        
        # Check system components
        print("\n📊 System Components:")
        for name, script_path in self.scripts.items():
            status = "✅" if script_path.exists() else "❌"
            print(f"   {status} {name.capitalize()}: {script_path.name}")
        
        # Check data directory - DYNAMIC RESOLUTION
        from utils.path_resolver import get_data_dir
        data_dir = get_data_dir()
        data_status = "✅" if data_dir.exists() else "❌"
        print(f"   {data_status} Data Directory: {data_dir}")
        
        # Check for trained models
        model_files = list(project_root.glob("**/*.pth"))
        model_status = "✅" if model_files else "❌"
        print(f"   {model_status} Trained Models: {len(model_files)} found")
        
        print("\n🚀 Quick Start Commands:")
        print("   python thgnn.py system validate    # Check system health")
        print("   python thgnn.py system setup       # Setup environment")
        print("   python thgnn.py train              # Train new model")
        print("   python thgnn.py predict --logs demo_final.json  # Run inference")
        print("   python thgnn.py dashboard          # Launch web interface")
        
        print("\n📚 Documentation:")
        print("   README.md           - Project overview and setup")
        print("   docs/USER_GUIDE.md  - Detailed user guide")
        print("   docs/SETUP.md       - Installation instructions")
        
        print("="*80)

def create_parser() -> argparse.ArgumentParser:
    """Create the master argument parser"""
    parser = argparse.ArgumentParser(
        description='THGNN - Ultimate Cybersecurity Threat Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # System management
  python thgnn.py system validate
  python thgnn.py system setup
  python thgnn.py system benchmark
  
  # Training
  python thgnn.py train --epochs 50 --batch-size 8
  python thgnn.py train --config config/model_config.yaml
  
  # Inference
  python thgnn.py predict --logs security_events.json
  python thgnn.py predict --logs data/ --batch --output results/
  
  # Reporting
  python thgnn.py report --input results.json --format pdf html
  python thgnn.py report --input results.json --title "Security Analysis"
  
  # Dashboard
  python thgnn.py dashboard
  python thgnn.py dashboard --port 8080
  
  # Status
  python thgnn.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='THGNN commands')
    
    # Training subparser
    train_parser = subparsers.add_parser('train', help='Train THGNN model')
    train_parser.add_argument('--config', type=str, help='Configuration file path')
    train_parser.add_argument('--data-dir', type=str, default='data', help='Data directory (relative to project root)')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    
    # Inference subparser
    predict_parser = subparsers.add_parser('predict', help='Run threat prediction')
    predict_parser.add_argument('--logs', type=str, required=True, help='Log file or directory')
    predict_parser.add_argument('--output', type=str, help='Output file path')
    predict_parser.add_argument('--model', type=str, help='Model file path')
    predict_parser.add_argument('--batch', action='store_true', help='Batch processing')
    predict_parser.add_argument('--top-k', type=int, default=20, help='Top K threats')
    predict_parser.add_argument('--risk-threshold', type=float, default=0.3, help='Risk threshold')
    predict_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Reporting subparser
    report_parser = subparsers.add_parser('report', help='Generate analysis reports')
    report_parser.add_argument('--input', type=str, required=True, help='Inference results file')
    report_parser.add_argument('--output', type=str, help='Output file prefix')
    report_parser.add_argument('--title', type=str, default='THGNN Threat Analysis', help='Report title')
    report_parser.add_argument('--organization', type=str, default='Security Operations Center', help='Organization')
    report_parser.add_argument('--classification', type=str, default='CONFIDENTIAL', help='Classification')
    report_parser.add_argument('--formats', nargs='+', default=['html', 'pdf'], 
                              choices=['html', 'pdf', 'json', 'csv'], help='Output formats')
    report_parser.add_argument('--no-executive', action='store_true', help='Skip executive summary')
    report_parser.add_argument('--no-technical', action='store_true', help='Skip technical report')
    
    # System subparser
    system_parser = subparsers.add_parser('system', help='System management')
    system_parser.add_argument('system_command', choices=['validate', 'setup', 'benchmark', 'cleanup', 'monitor'],
                              help='System command')
    system_parser.add_argument('--output', type=str, help='Output file')
    system_parser.add_argument('--duration', type=int, default=60, help='Monitor duration (seconds)')
    
    # Dashboard subparser
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch web dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8501, help='Port number')
    dashboard_parser.add_argument('--host', type=str, default='localhost', help='Host address')
    
    # Status subparser
    subparsers.add_parser('status', help='Show system status')
    
    return parser

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create master controller
    controller = THGNNMasterController()
    
    # Handle commands
    if args.command == 'train':
        return controller.run_training(args)
    
    elif args.command == 'predict':
        return controller.run_inference(args)
    
    elif args.command == 'report':
        return controller.run_reporting(args)
    
    elif args.command == 'system':
        return controller.run_system_management(args)
    
    elif args.command == 'dashboard':
        return controller.run_dashboard(args)
    
    elif args.command == 'status':
        controller.show_status()
        return 0
    
    else:
        # Show status by default
        controller.show_status()
        return 0

if __name__ == "__main__":
    sys.exit(main())