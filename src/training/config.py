#!/usr/bin/env python3
"""
Training Configuration Management for THGNN

Production-grade configuration system with validation, type safety, and
comprehensive parameter management for cybersecurity threat detection training.

Features:
- Type-safe dataclass configuration
- YAML configuration loading with validation
- Command-line argument override support
- Environment variable integration
- Configuration versioning and tracking
- Hyperparameter validation and constraints
"""

import logging
import yaml
import os
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_dim: int = 256
    num_conv_layers: int = 3
    temporal_hidden: int = 128
    temporal_layers: int = 2
    conv_type: str = 'sage'
    num_attention_heads: int = 4
    dropout: float = 0.4
    use_layer_norm: bool = True
    use_residual: bool = True
    use_uncertainty: bool = True
    temperature: float = 1.0
    
    def __post_init__(self):
        """Validate model configuration."""
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_conv_layers > 0, "num_conv_layers must be positive"
        assert self.temporal_hidden > 0, "temporal_hidden must be positive"
        assert self.conv_type in ['gcn', 'gat', 'sage'], f"Invalid conv_type: {self.conv_type}"
        assert 0.0 <= self.dropout <= 1.0, "dropout must be in [0, 1]"
        assert self.temperature > 0, "temperature must be positive"

@dataclass
@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 50
    
    # Scheduling
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.001
    
    # Gradient management
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Sampling
    negative_sampling_ratio: float = 1.0
    negative_sampling_strategy: str = 'mixed'
    max_edges_per_type: int = 100  # CRITICAL FIX: Limit edges per type for memory efficiency
    
    # Loss function
    pos_weight: float = 1.0
    neg_weight: float = 1.0
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    uncertainty_weight: float = 0.1
    
    # Regularization
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    
    def __post_init__(self):
        """Validate training configuration."""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.scheduler in ['cosine', 'step', 'plateau', 'none'], f"Invalid scheduler: {self.scheduler}"
        assert self.early_stopping_patience > 0, "early_stopping_patience must be positive"
        assert self.gradient_clip > 0, "gradient_clip must be positive"
        assert self.negative_sampling_ratio > 0, "negative_sampling_ratio must be positive"
        assert self.negative_sampling_strategy in ['random', 'hard', 'mixed'], f"Invalid sampling strategy: {self.negative_sampling_strategy}"
        assert self.max_edges_per_type > 0, "max_edges_per_type must be positive"

@dataclass
class DataConfig:
    """Data processing and loading configuration."""
    # Paths - DYNAMICALLY RESOLVED (NO HARDCODING!)
    data_dir: str = 'data'
    graphs_dir: str = 'data/graphs'
    metadata_dir: str = 'data/metadata'
    
    # Graph settings
    time_window_minutes: int = 5
    sequence_length: int = 10
    max_nodes_per_type: int = 10000
    max_edges_per_type: int = 100000
    
    # Data splits (temporal)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Node feature dimensions
    node_feature_dims: Dict[str, int] = field(default_factory=lambda: {
        'user': 128,
        'host': 128,
        'process': 128,
        'file': 128,
        'registry': 128
    })
    
    # Edge types
    edge_types: List[List[str]] = field(default_factory=lambda: [
        ['user', 'logs_into', 'host'],
        ['user', 'executes', 'process'],
        ['process', 'creates', 'process'],
        ['process', 'connects_to', 'host'],
        ['process', 'accesses', 'file'],
        ['process', 'modifies', 'registry']
    ])
    
    def __post_init__(self):
        """Validate data configuration."""
        assert self.time_window_minutes > 0, "time_window_minutes must be positive"
        assert self.sequence_length > 0, "sequence_length must be positive"
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        assert all(ratio > 0 for ratio in [self.train_ratio, self.val_ratio, self.test_ratio]), "All split ratios must be positive"
        assert self.num_workers >= 0, "num_workers must be non-negative"

@dataclass
class SystemConfig:
    """System and hardware configuration."""
    # Device settings
    device: str = 'auto'
    gpu_memory_fraction: float = 0.8
    mixed_precision: bool = True
    
    # Parallel processing
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging
    log_level: str = 'INFO'
    log_interval: int = 100
    save_interval: int = 1000
    
    # Monitoring
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = 'thgnn-cybersecurity'
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = False
    benchmark: bool = True
    
    def __post_init__(self):
        """Validate system configuration."""
        assert self.device in ['auto', 'cuda', 'cpu'], f"Invalid device: {self.device}"
        assert 0.0 < self.gpu_memory_fraction <= 1.0, "gpu_memory_fraction must be in (0, 1]"
        assert self.log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR'], f"Invalid log_level: {self.log_level}"
        assert self.log_interval > 0, "log_interval must be positive"
        assert self.save_interval > 0, "save_interval must be positive"

@dataclass
class PerformanceConfig:
    """Performance requirements configuration."""
    target_auc_roc: float = 0.80
    target_precision_at_10: float = 0.70
    target_recall_at_100: float = 0.75
    max_prediction_latency: float = 5.0
    max_training_time: float = 28800  # 8 hours
    min_events_per_minute: int = 100000
    min_snapshots_per_minute: int = 50
    
    def __post_init__(self):
        """Validate performance configuration."""
        assert 0.0 <= self.target_auc_roc <= 1.0, "target_auc_roc must be in [0, 1]"
        assert 0.0 <= self.target_precision_at_10 <= 1.0, "target_precision_at_10 must be in [0, 1]"
        assert self.max_prediction_latency > 0, "max_prediction_latency must be positive"

@dataclass
class CybersecurityConfig:
    """Cybersecurity domain configuration."""
    mitre_techniques: Dict[str, str] = field(default_factory=lambda: {
        'T1059.001': 'PowerShell',
        'T1021.002': 'SMB/Windows Admin Shares',
        'T1078': 'Valid Accounts',
        'T1055': 'Process Injection',
        'T1003': 'OS Credential Dumping'
    })
    
    high_risk_processes: List[str] = field(default_factory=lambda: [
        'powershell.exe',
        'cmd.exe',
        'psexec.exe',
        'wmic.exe',
        'mimikatz.exe',
        'rundll32.exe',
        'regsvr32.exe'
    ])
    
    critical_registry_paths: List[str] = field(default_factory=lambda: [
        'hklm\\software\\microsoft\\windows\\currentversion\\run',
        'hklm\\system\\currentcontrolset\\services',
        'hklm\\software\\microsoft\\windows nt\\currentversion\\winlogon'
    ])
    
    suspicious_ports: List[int] = field(default_factory=lambda: [445, 135, 139, 3389, 22, 23, 80, 443])
    executable_extensions: List[str] = field(default_factory=lambda: ['.exe', '.bat', '.cmd', '.ps1', '.vbs', '.js'])

@dataclass
class PathsConfig:
    """File paths and directories configuration."""
    # Model checkpoints
    checkpoint_dir: str = 'checkpoints'
    best_model_path: str = 'checkpoints/best_model.pth'
    
    # Logging
    log_dir: str = 'outputs/logs'
    tensorboard_dir: str = 'outputs/tensorboard'
    
    # Outputs
    predictions_dir: str = 'outputs/predictions'
    visualizations_dir: str = 'outputs/visualizations'
    reports_dir: str = 'outputs/reports'
    
    # Temporary
    temp_dir: str = 'temp'
    cache_dir: str = 'cache'
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for attr_name in dir(self):
            if not attr_name.startswith('_') and attr_name.endswith('_dir'):
                path = Path(getattr(self, attr_name))
                path.mkdir(parents=True, exist_ok=True)

@dataclass
class TrainingConfiguration:
    """Complete training configuration combining all components."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    cybersecurity: CybersecurityConfig = field(default_factory=CybersecurityConfig)
    
    # Metadata
    version: str = '1.0.0'
    description: str = 'THGNN Cybersecurity Training Configuration'
    created_by: str = 'THGNN Expert System'
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'TrainingConfiguration':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Create configuration with validation
            config = cls()
            
            # Update each section if present
            if 'model' in config_dict:
                config.model = ModelConfig(**config_dict['model'])
            
            if 'training' in config_dict:
                config.training = TrainingConfig(**config_dict['training'])
            
            if 'data' in config_dict:
                config.data = DataConfig(**config_dict['data'])
            
            if 'system' in config_dict:
                config.system = SystemConfig(**config_dict['system'])
            
            if 'paths' in config_dict:
                config.paths = PathsConfig(**config_dict['paths'])
            
            if 'performance' in config_dict:
                config.performance = PerformanceConfig(**config_dict['performance'])
            
            if 'cybersecurity' in config_dict:
                config.cybersecurity = CybersecurityConfig(**config_dict['cybersecurity'])
            
            # Update metadata
            for key in ['version', 'description', 'created_by']:
                if key in config_dict:
                    setattr(config, key, config_dict[key])
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self)
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {output_path}: {e}")
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        """Update configuration from command-line arguments."""
        # Model parameters
        if hasattr(args, 'hidden_dim') and args.hidden_dim is not None:
            self.model.hidden_dim = args.hidden_dim
        
        if hasattr(args, 'num_conv_layers') and args.num_conv_layers is not None:
            self.model.num_conv_layers = args.num_conv_layers
        
        if hasattr(args, 'dropout') and args.dropout is not None:
            self.model.dropout = args.dropout
        
        # Training parameters
        if hasattr(args, 'learning_rate') and args.learning_rate is not None:
            self.training.learning_rate = args.learning_rate
        
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            self.training.batch_size = args.batch_size
        
        if hasattr(args, 'num_epochs') and args.num_epochs is not None:
            self.training.num_epochs = args.num_epochs
        
        # System parameters
        if hasattr(args, 'device') and args.device is not None:
            self.system.device = args.device
        
        if hasattr(args, 'num_workers') and args.num_workers is not None:
            self.system.num_workers = args.num_workers
        
        logger.info("Configuration updated from command-line arguments")
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        # Training parameters
        if 'THGNN_LEARNING_RATE' in os.environ:
            self.training.learning_rate = float(os.environ['THGNN_LEARNING_RATE'])
        
        if 'THGNN_BATCH_SIZE' in os.environ:
            self.training.batch_size = int(os.environ['THGNN_BATCH_SIZE'])
        
        if 'THGNN_NUM_EPOCHS' in os.environ:
            self.training.num_epochs = int(os.environ['THGNN_NUM_EPOCHS'])
        
        # System parameters
        if 'THGNN_DEVICE' in os.environ:
            self.system.device = os.environ['THGNN_DEVICE']
        
        if 'THGNN_NUM_WORKERS' in os.environ:
            self.system.num_workers = int(os.environ['THGNN_NUM_WORKERS'])
        
        # Data paths
        if 'THGNN_DATA_DIR' in os.environ:
            self.data.data_dir = os.environ['THGNN_DATA_DIR']
        
        logger.info("Configuration updated from environment variables")
    
    def get_device(self) -> torch.device:
        """Get the appropriate torch device."""
        if self.system.device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                logger.info("Using CPU")
        else:
            device = torch.device(self.system.device)
            logger.info(f"Using device: {device}")
        
        return device
    
    def setup_reproducibility(self) -> None:
        """Setup reproducible training."""
        import random
        import numpy as np
        
        # Set seeds
        random.seed(self.system.random_seed)
        np.random.seed(self.system.random_seed)
        torch.manual_seed(self.system.random_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.system.random_seed)
            torch.cuda.manual_seed_all(self.system.random_seed)
        
        # Set deterministic behavior
        if self.system.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("Deterministic mode enabled")
        else:
            torch.backends.cudnn.benchmark = self.system.benchmark
            logger.info(f"Benchmark mode: {self.system.benchmark}")
        
        logger.info(f"Reproducibility setup with seed: {self.system.random_seed}")
    
    def validate(self) -> bool:
        """Validate the complete configuration."""
        try:
            # Validate each component (post_init methods handle this)
            self.model.__post_init__()
            self.training.__post_init__()
            self.data.__post_init__()
            self.system.__post_init__()
            self.paths.__post_init__()
            self.performance.__post_init__()
            # cybersecurity config doesn't need validation
            
            # Additional cross-component validation
            if self.training.batch_size > self.data.max_nodes_per_type:
                logger.warning("Batch size larger than max nodes per type")
            
            if self.model.hidden_dim < 32:
                logger.warning("Hidden dimension might be too small for complex patterns")
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging."""
        return {
            'model': {
                'hidden_dim': self.model.hidden_dim,
                'num_conv_layers': self.model.num_conv_layers,
                'conv_type': self.model.conv_type,
                'dropout': self.model.dropout
            },
            'training': {
                'learning_rate': self.training.learning_rate,
                'batch_size': self.training.batch_size,
                'num_epochs': self.training.num_epochs,
                'scheduler': self.training.scheduler
            },
            'data': {
                'data_dir': self.data.data_dir,
                'time_window_minutes': self.data.time_window_minutes,
                'sequence_length': self.data.sequence_length
            },
            'system': {
                'device': self.system.device,
                'mixed_precision': self.system.mixed_precision,
                'random_seed': self.system.random_seed
            }
        }


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for training configuration."""
    parser = argparse.ArgumentParser(
        description='THGNN Training Configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--hidden-dim', type=int, help='Hidden dimension')
    model_group.add_argument('--num-conv-layers', type=int, help='Number of convolution layers')
    model_group.add_argument('--conv-type', choices=['gcn', 'gat', 'sage'], help='Convolution type')
    model_group.add_argument('--dropout', type=float, help='Dropout probability')
    
    # Training arguments
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--learning-rate', type=float, help='Learning rate')
    train_group.add_argument('--batch-size', type=int, help='Batch size')
    train_group.add_argument('--num-epochs', type=int, help='Number of epochs')
    train_group.add_argument('--scheduler', choices=['cosine', 'step', 'plateau', 'none'], help='Learning rate scheduler')
    
    # System arguments
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument('--device', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    system_group.add_argument('--num-workers', type=int, help='Number of data loading workers')
    system_group.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training')
    
    # Configuration file
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--save-config', type=str, help='Save configuration to YAML file')
    
    return parser


if __name__ == "__main__":
    # Example usage and testing
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = TrainingConfiguration.from_yaml(args.config)
    else:
        config = TrainingConfiguration()
    
    # Update from command line
    config.update_from_args(args)
    config.update_from_env()
    
    # Validate
    if config.validate():
        print("Configuration is valid")
        print(f"Summary: {config.get_summary()}")
        
        # Save if requested
        if args.save_config:
            config.to_yaml(args.save_config)
    else:
        print("Configuration validation failed")
        exit(1)