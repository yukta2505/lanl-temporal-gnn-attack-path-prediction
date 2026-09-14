#!/usr/bin/env python3
"""
Configuration Management System for THGNN Visualization
Production-ready configuration loader with validation and caching.

Author: THGNN Expert
Date: January 9, 2026
Version: 1.0 - CONFIGURATION MANAGEMENT
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from functools import lru_cache
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Type-safe configuration class for visualization settings"""
    
    # Global settings
    default_width: int = 1200
    default_height: int = 800
    max_nodes_display: int = 1000
    max_edges_display: int = 5000
    
    # Color schemes
    security_entity_colors: Dict[str, str] = field(default_factory=dict)
    risk_level_colors: Dict[str, str] = field(default_factory=dict)
    prediction_confidence_colors: Dict[str, str] = field(default_factory=dict)
    mitre_tactic_colors: Dict[str, str] = field(default_factory=dict)
    ui_theme_colors: Dict[str, str] = field(default_factory=dict)
    
    # Graph visualization
    node_size_range: List[int] = field(default_factory=lambda: [10, 50])
    edge_width_range: List[int] = field(default_factory=lambda: [1, 8])
    node_opacity: float = 0.8
    edge_opacity: float = 0.6
    
    # Timeline visualization
    event_marker_size: int = 12
    prediction_marker_size: int = 15
    risk_timeline_width: int = 2
    
    # Process tree
    process_base_size: int = 15
    process_max_size: int = 40
    suspicious_processes: List[str] = field(default_factory=list)
    
    # Performance
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_memory_mb: int = 2048
    
    # Security
    max_string_length: int = 10000
    max_array_length: int = 100000
    max_file_size_mb: int = 100

class ConfigManager:
    """
    Thread-safe configuration manager with validation and caching.
    Provides centralized access to all visualization configuration.
    """
    
    _instance: Optional['ConfigManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ConfigManager':
        """Singleton pattern implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize configuration manager"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._config_cache: Dict[str, Any] = {}
        self._config_file_path: Optional[Path] = None
        self._config_data: Dict[str, Any] = {}
        
        # Load default configuration
        self._load_configuration()
        
        logger.info("Configuration manager initialized successfully")
    
    def _load_configuration(self) -> None:
        """Load configuration from YAML file with fallback to defaults"""
        try:
            # Try to find configuration file
            config_paths = [
                Path('config/visualization_config.yaml'),
                Path('../config/visualization_config.yaml'),
                Path('../../config/visualization_config.yaml'),
                Path.cwd() / 'config' / 'visualization_config.yaml'
            ]
            
            config_file = None
            for path in config_paths:
                if path.exists():
                    config_file = path
                    break
            
            if config_file is None:
                logger.warning("Configuration file not found, using defaults")
                self._config_data = self._get_default_config()
                return
            
            # Load YAML configuration
            with open(config_file, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
            
            self._config_file_path = config_file
            
            # Validate configuration
            self._validate_configuration()
            
            logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")
            self._config_data = self._get_default_config()
    
    def _validate_configuration(self) -> None:
        """Validate configuration data"""
        try:
            required_sections = [
                'global', 'color_schemes', 'graph_visualization',
                'timeline_visualization', 'process_tree', 'risk_heatmap'
            ]
            
            for section in required_sections:
                if section not in self._config_data:
                    logger.warning(f"Missing configuration section: {section}")
                    self._config_data[section] = {}
            
            # Validate color schemes
            color_schemes = self._config_data.get('color_schemes', {})
            for scheme_name, colors in color_schemes.items():
                if not isinstance(colors, dict):
                    logger.warning(f"Invalid color scheme format: {scheme_name}")
                    continue
                
                for color_name, color_value in colors.items():
                    if not self._is_valid_color(color_value):
                        logger.warning(f"Invalid color value: {color_name}={color_value}")
            
            # Validate numeric ranges
            self._validate_numeric_ranges()
            
            logger.info("Configuration validation completed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
    
    def _validate_numeric_ranges(self) -> None:
        """Validate numeric configuration values"""
        try:
            # Validate global settings
            global_config = self._config_data.get('global', {})
            
            width = global_config.get('default_width', 1200)
            height = global_config.get('default_height', 800)
            
            if not (100 <= width <= 5000):
                logger.warning(f"Invalid default_width: {width}, using 1200")
                global_config['default_width'] = 1200
            
            if not (100 <= height <= 3000):
                logger.warning(f"Invalid default_height: {height}, using 800")
                global_config['default_height'] = 800
            
            # Validate node size ranges
            graph_config = self._config_data.get('graph_visualization', {})
            nodes_config = graph_config.get('nodes', {})
            
            size_range = nodes_config.get('size_range', [10, 50])
            if not (isinstance(size_range, list) and len(size_range) == 2):
                logger.warning(f"Invalid node size_range: {size_range}")
                nodes_config['size_range'] = [10, 50]
            
        except Exception as e:
            logger.error(f"Numeric validation failed: {e}")
    
    def _is_valid_color(self, color: str) -> bool:
        """Validate color format (hex or named color)"""
        try:
            if not isinstance(color, str):
                return False
            
            # Check hex color format
            if color.startswith('#') and len(color) == 7:
                try:
                    int(color[1:], 16)
                    return True
                except ValueError:
                    return False
            
            # Check named colors (basic validation)
            named_colors = {
                'red', 'green', 'blue', 'yellow', 'orange', 'purple',
                'black', 'white', 'gray', 'grey', 'pink', 'brown'
            }
            
            return color.lower() in named_colors
            
        except Exception:
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when file is not available"""
        return {
            'global': {
                'default_width': 1200,
                'default_height': 800,
                'max_nodes_display': 1000,
                'max_edges_display': 5000,
                'enable_caching': True,
                'cache_ttl_seconds': 3600
            },
            'color_schemes': {
                'security_entities': {
                    'user': '#2E86AB',
                    'host': '#A23B72',
                    'process': '#F18F01',
                    'file': '#C73E1D',
                    'registry': '#8B5A3C',
                    'unknown': '#95A5A6'
                },
                'risk_levels': {
                    'low': '#27AE60',
                    'medium': '#F39C12',
                    'high': '#E74C3C',
                    'critical': '#8E44AD',
                    'unknown': '#95A5A6'
                },
                'mitre_tactics': {
                    'TA0001': '#3498DB',
                    'TA0002': '#E67E22',
                    'TA0003': '#2ECC71',
                    'TA0004': '#F1C40F',
                    'TA0005': '#9B59B6',
                    'TA0006': '#E74C3C',
                    'TA0007': '#1ABC9C',
                    'TA0008': '#8E44AD',
                    'default': '#95A5A6'
                }
            },
            'graph_visualization': {
                'nodes': {
                    'size_range': [10, 50],
                    'opacity': 0.8,
                    'border_width': 2
                },
                'edges': {
                    'width_range': [1, 8],
                    'opacity': 0.6
                }
            },
            'process_tree': {
                'suspicious_indicators': [
                    'powershell.exe', 'cmd.exe', 'wmic.exe', 'psexec.exe'
                ]
            }
        }
    
    @lru_cache(maxsize=128)
    def get_config(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value with caching.
        
        Parameters
        ----------
        section : str
            Configuration section name
        key : str, optional
            Specific key within section
        default : Any, optional
            Default value if key not found
            
        Returns
        -------
        Any
            Configuration value
        """
        try:
            section_data = self._config_data.get(section, {})
            
            if key is None:
                return section_data
            
            return section_data.get(key, default)
            
        except Exception as e:
            logger.error(f"Failed to get config {section}.{key}: {e}")
            return default
    
    def get_color_scheme(self, scheme_name: str) -> Dict[str, str]:
        """
        Get color scheme by name.
        
        Parameters
        ----------
        scheme_name : str
            Name of color scheme
            
        Returns
        -------
        Dict[str, str]
            Color mapping
        """
        try:
            color_schemes = self.get_config('color_schemes', default={})
            return color_schemes.get(scheme_name, {})
            
        except Exception as e:
            logger.error(f"Failed to get color scheme {scheme_name}: {e}")
            return {}
    
    def get_entity_color(self, entity_type: str) -> str:
        """
        Get color for entity type.
        
        Parameters
        ----------
        entity_type : str
            Type of entity (user, host, process, etc.)
            
        Returns
        -------
        str
            Hex color code
        """
        try:
            entity_colors = self.get_color_scheme('security_entities')
            return entity_colors.get(entity_type, entity_colors.get('unknown', '#95A5A6'))
            
        except Exception as e:
            logger.error(f"Failed to get entity color for {entity_type}: {e}")
            return '#95A5A6'
    
    def get_risk_color(self, risk_level: str) -> str:
        """
        Get color for risk level.
        
        Parameters
        ----------
        risk_level : str
            Risk level (low, medium, high, critical)
            
        Returns
        -------
        str
            Hex color code
        """
        try:
            risk_colors = self.get_color_scheme('risk_levels')
            return risk_colors.get(risk_level, risk_colors.get('unknown', '#95A5A6'))
            
        except Exception as e:
            logger.error(f"Failed to get risk color for {risk_level}: {e}")
            return '#95A5A6'
    
    def get_mitre_color(self, tactic_id: str) -> str:
        """
        Get color for MITRE ATT&CK tactic.
        
        Parameters
        ----------
        tactic_id : str
            MITRE tactic ID (e.g., TA0001)
            
        Returns
        -------
        str
            Hex color code
        """
        try:
            mitre_colors = self.get_color_scheme('mitre_tactics')
            return mitre_colors.get(tactic_id, mitre_colors.get('default', '#95A5A6'))
            
        except Exception as e:
            logger.error(f"Failed to get MITRE color for {tactic_id}: {e}")
            return '#95A5A6'
    
    def get_visualization_config(self) -> VisualizationConfig:
        """
        Get typed visualization configuration object.
        
        Returns
        -------
        VisualizationConfig
            Typed configuration object
        """
        try:
            # Extract configuration values
            global_config = self.get_config('global', default={})
            graph_config = self.get_config('graph_visualization', default={})
            timeline_config = self.get_config('timeline_visualization', default={})
            process_config = self.get_config('process_tree', default={})
            performance_config = self.get_config('performance', default={})
            security_config = self.get_config('security', default={})
            
            # Create typed configuration
            config = VisualizationConfig(
                default_width=global_config.get('default_width', 1200),
                default_height=global_config.get('default_height', 800),
                max_nodes_display=global_config.get('max_nodes_display', 1000),
                max_edges_display=global_config.get('max_edges_display', 5000),
                
                security_entity_colors=self.get_color_scheme('security_entities'),
                risk_level_colors=self.get_color_scheme('risk_levels'),
                prediction_confidence_colors=self.get_color_scheme('prediction_confidence'),
                mitre_tactic_colors=self.get_color_scheme('mitre_tactics'),
                ui_theme_colors=self.get_color_scheme('ui_theme'),
                
                node_size_range=graph_config.get('nodes', {}).get('size_range', [10, 50]),
                edge_width_range=graph_config.get('edges', {}).get('width_range', [1, 8]),
                node_opacity=graph_config.get('nodes', {}).get('opacity', 0.8),
                edge_opacity=graph_config.get('edges', {}).get('opacity', 0.6),
                
                event_marker_size=timeline_config.get('events', {}).get('marker_size', 12),
                prediction_marker_size=timeline_config.get('predictions', {}).get('marker_size', 15),
                risk_timeline_width=timeline_config.get('risk_timeline', {}).get('line_width', 2),
                
                process_base_size=process_config.get('processes', {}).get('base_size', 15),
                process_max_size=process_config.get('processes', {}).get('max_size', 40),
                suspicious_processes=process_config.get('suspicious_indicators', []),
                
                enable_caching=performance_config.get('cache', {}).get('enable', True),
                cache_ttl_seconds=performance_config.get('cache', {}).get('ttl_seconds', 3600),
                max_memory_mb=performance_config.get('memory', {}).get('max_memory_mb', 2048),
                
                max_string_length=security_config.get('validation', {}).get('max_string_length', 10000),
                max_array_length=security_config.get('validation', {}).get('max_array_length', 100000),
                max_file_size_mb=security_config.get('files', {}).get('max_size_mb', 100)
            )
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to create visualization config: {e}")
            return VisualizationConfig()  # Return default config
    
    def reload_configuration(self) -> bool:
        """
        Reload configuration from file.
        
        Returns
        -------
        bool
            True if reload successful
        """
        try:
            # Clear cache
            self.get_config.cache_clear()
            self._config_cache.clear()
            
            # Reload configuration
            self._load_configuration()
            
            logger.info("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def validate_input(self, value: Any, value_type: str) -> bool:
        """
        Validate input based on security configuration.
        
        Parameters
        ----------
        value : Any
            Value to validate
        value_type : str
            Type of validation ('string', 'array', 'file_size')
            
        Returns
        -------
        bool
            True if valid
        """
        try:
            security_config = self.get_config('security', default={})
            validation_config = security_config.get('validation', {})
            
            if value_type == 'string':
                if not isinstance(value, str):
                    return False
                max_length = validation_config.get('max_string_length', 10000)
                return len(value) <= max_length
            
            elif value_type == 'array':
                if not isinstance(value, (list, tuple)):
                    return False
                max_length = validation_config.get('max_array_length', 100000)
                return len(value) <= max_length
            
            elif value_type == 'file_size':
                if not isinstance(value, (int, float)):
                    return False
                max_size_mb = security_config.get('files', {}).get('max_size_mb', 100)
                return value <= max_size_mb * 1024 * 1024  # Convert to bytes
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False


# Global configuration manager instance
config_manager = ConfigManager()

def get_config(section: str, key: Optional[str] = None, default: Any = None) -> Any:
    """
    Convenience function to get configuration values.
    
    Parameters
    ----------
    section : str
        Configuration section
    key : str, optional
        Configuration key
    default : Any, optional
        Default value
        
    Returns
    -------
    Any
        Configuration value
    """
    return config_manager.get_config(section, key, default)

def get_entity_color(entity_type: str) -> str:
    """Get color for entity type"""
    return config_manager.get_entity_color(entity_type)

def get_risk_color(risk_level: str) -> str:
    """Get color for risk level"""
    return config_manager.get_risk_color(risk_level)

def get_mitre_color(tactic_id: str) -> str:
    """Get color for MITRE tactic"""
    return config_manager.get_mitre_color(tactic_id)

def get_visualization_config() -> VisualizationConfig:
    """Get typed visualization configuration"""
    return config_manager.get_visualization_config()

def validate_input(value: Any, value_type: str) -> bool:
    """Validate input value"""
    return config_manager.validate_input(value, value_type)


def test_config_manager():
    """Test the configuration manager"""
    
    logger.info("Testing Configuration Manager...")
    
    try:
        # Test singleton pattern
        cm1 = ConfigManager()
        cm2 = ConfigManager()
        assert cm1 is cm2, "Singleton pattern failed"
        
        # Test configuration loading
        config = get_visualization_config()
        assert isinstance(config, VisualizationConfig), "Config type validation failed"
        
        # Test color retrieval
        user_color = get_entity_color('user')
        assert user_color.startswith('#'), f"Invalid user color: {user_color}"
        
        risk_color = get_risk_color('high')
        assert risk_color.startswith('#'), f"Invalid risk color: {risk_color}"
        
        mitre_color = get_mitre_color('TA0001')
        assert mitre_color.startswith('#'), f"Invalid MITRE color: {mitre_color}"
        
        # Test input validation
        assert validate_input("test", "string"), "String validation failed"
        assert validate_input([1, 2, 3], "array"), "Array validation failed"
        assert validate_input(1024, "file_size"), "File size validation failed"
        
        # Test configuration sections
        global_config = get_config('global')
        assert isinstance(global_config, dict), "Global config not dict"
        
        width = get_config('global', 'default_width', 1200)
        assert isinstance(width, int), "Width not integer"
        
        logger.info(" All configuration manager tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Configuration manager test failed: {e}")
        return False


if __name__ == '__main__':
    """Test the configuration manager"""
    
    success = test_config_manager()
    
    if success:
        logger.info(" Configuration Manager implementation complete!")
        logger.info(" Ready for visualization code fixes")
    else:
        logger.error("[ERROR] Configuration Manager needs fixes")
        sys.exit(1)