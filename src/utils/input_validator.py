#!/usr/bin/env python3
"""
Input Validation System for THGNN Visualization
Production-ready input validation with security focus.

Author: THGNN Expert
Date: January 9, 2026
Version: 1.0 - INPUT VALIDATION SYSTEM
"""

import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_value: Any = None

class InputValidator:
    """
    Production-ready input validator with security focus.
    Validates all inputs to visualization components.
    """
    
    def __init__(self):
        """Initialize input validator"""
        
        # Security patterns
        self.malicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                # JavaScript URLs
            r'on\w+\s*=',                 # Event handlers
            r'eval\s*\(',                 # eval() calls
            r'exec\s*\(',                 # exec() calls
            r'import\s+os',               # OS imports
            r'__import__',                # Dynamic imports
            r'\.\./',                     # Path traversal
            r'\\.\\.\\',                  # Windows path traversal
        ]
        
        # Allowed characters for different contexts
        self.allowed_chars = {
            'alphanumeric': r'^[a-zA-Z0-9_\-\.]+$',
            'alphanumeric_extended': r'^[a-zA-Z0-9_\-\.\s\(\)\[\]]+$',
            'filename': r'^[a-zA-Z0-9_\-\.]+$',
            'entity_name': r'^[a-zA-Z0-9_\-\.\\\:]+$',  # Allow domain\user format
            'mitre_technique': r'^T\d{4}(\.\d{3})?$',   # MITRE technique format
            'hex_color': r'^#[0-9A-Fa-f]{6}$',          # Hex color format
        }
        
        # Maximum lengths for different data types
        self.max_lengths = {
            'short_string': 100,
            'medium_string': 1000,
            'long_string': 10000,
            'description': 5000,
            'command_line': 2000,
            'file_path': 500,
            'entity_name': 200,
            'technique_id': 20,
        }
        
        # Numeric ranges
        self.numeric_ranges = {
            'confidence': (0.0, 1.0),
            'risk_score': (0.0, 1.0),
            'importance': (0.0, 1.0),
            'width': (100, 5000),
            'height': (100, 3000),
            'node_size': (1, 100),
            'edge_width': (1, 20),
            'opacity': (0.0, 1.0),
            'timestamp': (0, 2147483647),  # Unix timestamp range
        }
        
        logger.info("Input validator initialized successfully")
    
    def validate_string(
        self, 
        value: Any, 
        context: str = 'alphanumeric_extended',
        max_length: Optional[int] = None,
        required: bool = True
    ) -> ValidationResult:
        """
        Validate string input with context-specific rules.
        
        Parameters
        ----------
        value : Any
            Value to validate
        context : str
            Validation context (alphanumeric, filename, etc.)
        max_length : int, optional
            Maximum allowed length
        required : bool
            Whether the field is required
            
        Returns
        -------
        ValidationResult
            Validation result with sanitized value
        """
        try:
            errors = []
            warnings = []
            
            # Check if value is provided when required
            if value is None or value == '':
                if required:
                    errors.append("Required field is empty")
                    return ValidationResult(False, errors, warnings)
                else:
                    return ValidationResult(True, errors, warnings, '')
            
            # Convert to string
            if not isinstance(value, str):
                try:
                    value = str(value)
                    warnings.append("Value converted to string")
                except Exception:
                    errors.append("Cannot convert value to string")
                    return ValidationResult(False, errors, warnings)
            
            # Check for malicious patterns
            for pattern in self.malicious_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    errors.append(f"Potentially malicious pattern detected: {pattern}")
                    return ValidationResult(False, errors, warnings)
            
            # Check length
            if max_length is None:
                max_length = self.max_lengths.get('medium_string', 1000)
            
            if len(value) > max_length:
                errors.append(f"String too long: {len(value)} > {max_length}")
                return ValidationResult(False, errors, warnings)
            
            # Check character pattern
            if context in self.allowed_chars:
                pattern = self.allowed_chars[context]
                if not re.match(pattern, value):
                    errors.append(f"Invalid characters for context '{context}'")
                    return ValidationResult(False, errors, warnings)
            
            # Sanitize value
            sanitized_value = self._sanitize_string(value)
            
            return ValidationResult(True, errors, warnings, sanitized_value)
            
        except Exception as e:
            logger.error(f"String validation failed: {e}")
            return ValidationResult(False, [f"Validation error: {e}"], [])
    
    def validate_numeric(
        self, 
        value: Any, 
        context: str = 'general',
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        required: bool = True
    ) -> ValidationResult:
        """
        Validate numeric input with range checking.
        
        Parameters
        ----------
        value : Any
            Value to validate
        context : str
            Validation context for range lookup
        min_val : float, optional
            Minimum allowed value
        max_val : float, optional
            Maximum allowed value
        required : bool
            Whether the field is required
            
        Returns
        -------
        ValidationResult
            Validation result with sanitized value
        """
        try:
            errors = []
            warnings = []
            
            # Check if value is provided when required
            if value is None:
                if required:
                    errors.append("Required numeric field is None")
                    return ValidationResult(False, errors, warnings)
                else:
                    return ValidationResult(True, errors, warnings, 0.0)
            
            # Convert to numeric
            try:
                if isinstance(value, (int, float)):
                    numeric_value = float(value)
                elif isinstance(value, str):
                    numeric_value = float(value)
                else:
                    errors.append(f"Cannot convert {type(value)} to numeric")
                    return ValidationResult(False, errors, warnings)
            except (ValueError, TypeError):
                errors.append(f"Invalid numeric value: {value}")
                return ValidationResult(False, errors, warnings)
            
            # Check for special values
            if np.isnan(numeric_value):
                errors.append("Value is NaN")
                return ValidationResult(False, errors, warnings)
            
            if np.isinf(numeric_value):
                errors.append("Value is infinite")
                return ValidationResult(False, errors, warnings)
            
            # Get range from context
            if context in self.numeric_ranges:
                context_min, context_max = self.numeric_ranges[context]
                if min_val is None:
                    min_val = context_min
                if max_val is None:
                    max_val = context_max
            
            # Check range
            if min_val is not None and numeric_value < min_val:
                errors.append(f"Value {numeric_value} below minimum {min_val}")
                return ValidationResult(False, errors, warnings)
            
            if max_val is not None and numeric_value > max_val:
                errors.append(f"Value {numeric_value} above maximum {max_val}")
                return ValidationResult(False, errors, warnings)
            
            return ValidationResult(True, errors, warnings, numeric_value)
            
        except Exception as e:
            logger.error(f"Numeric validation failed: {e}")
            return ValidationResult(False, [f"Validation error: {e}"], [])
    
    def validate_array(
        self, 
        value: Any, 
        element_type: str = 'any',
        max_length: Optional[int] = None,
        required: bool = True
    ) -> ValidationResult:
        """
        Validate array/list input.
        
        Parameters
        ----------
        value : Any
            Value to validate
        element_type : str
            Expected type of array elements
        max_length : int, optional
            Maximum array length
        required : bool
            Whether the field is required
            
        Returns
        -------
        ValidationResult
            Validation result with sanitized array
        """
        try:
            errors = []
            warnings = []
            
            # Check if value is provided when required
            if value is None:
                if required:
                    errors.append("Required array field is None")
                    return ValidationResult(False, errors, warnings)
                else:
                    return ValidationResult(True, errors, warnings, [])
            
            # Convert to list if needed
            if not isinstance(value, (list, tuple, np.ndarray)):
                errors.append(f"Expected array, got {type(value)}")
                return ValidationResult(False, errors, warnings)
            
            # Convert to list
            array_value = list(value)
            
            # Check length
            if max_length is None:
                max_length = 100000  # Default max length
            
            if len(array_value) > max_length:
                errors.append(f"Array too long: {len(array_value)} > {max_length}")
                return ValidationResult(False, errors, warnings)
            
            # Validate elements
            sanitized_array = []
            for i, element in enumerate(array_value):
                if element_type == 'string':
                    result = self.validate_string(element, required=False)
                elif element_type == 'numeric':
                    result = self.validate_numeric(element, required=False)
                elif element_type == 'dict':
                    result = self.validate_dict(element, required=False)
                else:
                    # Accept any type
                    result = ValidationResult(True, [], [], element)
                
                if not result.is_valid:
                    errors.extend([f"Element {i}: {error}" for error in result.errors])
                else:
                    sanitized_array.append(result.sanitized_value)
                    warnings.extend(result.warnings)
            
            if errors:
                return ValidationResult(False, errors, warnings)
            
            return ValidationResult(True, errors, warnings, sanitized_array)
            
        except Exception as e:
            logger.error(f"Array validation failed: {e}")
            return ValidationResult(False, [f"Validation error: {e}"], [])
    
    def validate_dict(
        self, 
        value: Any, 
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
        required: bool = True
    ) -> ValidationResult:
        """
        Validate dictionary input.
        
        Parameters
        ----------
        value : Any
            Value to validate
        required_keys : List[str], optional
            Keys that must be present
        optional_keys : List[str], optional
            Keys that are allowed but not required
        required : bool
            Whether the field is required
            
        Returns
        -------
        ValidationResult
            Validation result with sanitized dict
        """
        try:
            errors = []
            warnings = []
            
            # Check if value is provided when required
            if value is None:
                if required:
                    errors.append("Required dict field is None")
                    return ValidationResult(False, errors, warnings)
                else:
                    return ValidationResult(True, errors, warnings, {})
            
            # Check if it's a dictionary
            if not isinstance(value, dict):
                errors.append(f"Expected dict, got {type(value)}")
                return ValidationResult(False, errors, warnings)
            
            # Check required keys
            if required_keys:
                for key in required_keys:
                    if key not in value:
                        errors.append(f"Missing required key: {key}")
            
            # Check for unexpected keys
            if optional_keys is not None:
                allowed_keys = set(required_keys or []) | set(optional_keys)
                for key in value.keys():
                    if key not in allowed_keys:
                        warnings.append(f"Unexpected key: {key}")
            
            # Sanitize dictionary
            sanitized_dict = {}
            for key, val in value.items():
                # Validate key
                key_result = self.validate_string(key, 'alphanumeric_extended', 100, True)
                if not key_result.is_valid:
                    errors.extend([f"Key '{key}': {error}" for error in key_result.errors])
                    continue
                
                # For now, accept any value type (could be extended)
                sanitized_dict[key_result.sanitized_value] = val
            
            if errors:
                return ValidationResult(False, errors, warnings)
            
            return ValidationResult(True, errors, warnings, sanitized_dict)
            
        except Exception as e:
            logger.error(f"Dict validation failed: {e}")
            return ValidationResult(False, [f"Validation error: {e}"], [])
    
    def validate_graph_data(self, graph_data: Any) -> ValidationResult:
        """
        Validate PyTorch Geometric graph data.
        
        Parameters
        ----------
        graph_data : Any
            Graph data to validate
            
        Returns
        -------
        ValidationResult
            Validation result
        """
        try:
            errors = []
            warnings = []
            
            # Check if it's a HeteroData object
            if not hasattr(graph_data, 'node_types'):
                errors.append("Invalid graph data: missing node_types")
                return ValidationResult(False, errors, warnings)
            
            if not hasattr(graph_data, 'edge_types'):
                errors.append("Invalid graph data: missing edge_types")
                return ValidationResult(False, errors, warnings)
            
            # Validate node types
            node_types = graph_data.node_types
            if not isinstance(node_types, (list, tuple)):
                errors.append("Invalid node_types format")
                return ValidationResult(False, errors, warnings)
            
            # Check for reasonable number of nodes
            total_nodes = 0
            for node_type in node_types:
                if hasattr(graph_data, 'x_dict') and node_type in graph_data.x_dict:
                    num_nodes = graph_data.x_dict[node_type].size(0)
                    total_nodes += num_nodes
            
            if total_nodes > 10000:  # Configurable limit
                warnings.append(f"Large graph with {total_nodes} nodes may impact performance")
            
            # Validate edge types
            edge_types = graph_data.edge_types
            if not isinstance(edge_types, (list, tuple)):
                errors.append("Invalid edge_types format")
                return ValidationResult(False, errors, warnings)
            
            # Check edge format
            for edge_type in edge_types:
                if not isinstance(edge_type, tuple) or len(edge_type) != 3:
                    errors.append(f"Invalid edge type format: {edge_type}")
                    continue
                
                src_type, relation, dst_type = edge_type
                
                # Validate entity types
                for entity_type in [src_type, dst_type]:
                    result = self.validate_string(entity_type, 'alphanumeric', 50, True)
                    if not result.is_valid:
                        errors.extend([f"Edge type '{entity_type}': {error}" for error in result.errors])
                
                # Validate relation
                result = self.validate_string(relation, 'alphanumeric_extended', 100, True)
                if not result.is_valid:
                    errors.extend([f"Edge relation '{relation}': {error}" for error in result.errors])
            
            if errors:
                return ValidationResult(False, errors, warnings)
            
            return ValidationResult(True, errors, warnings, graph_data)
            
        except Exception as e:
            logger.error(f"Graph data validation failed: {e}")
            return ValidationResult(False, [f"Validation error: {e}"], [])
    
    def validate_prediction_data(self, predictions: Any) -> ValidationResult:
        """
        Validate prediction data structure.
        
        Parameters
        ----------
        predictions : Any
            Prediction data to validate
            
        Returns
        -------
        ValidationResult
            Validation result with sanitized predictions
        """
        try:
            errors = []
            warnings = []
            
            # Validate as array
            array_result = self.validate_array(predictions, 'dict', 1000, False)
            if not array_result.is_valid:
                return array_result
            
            predictions_list = array_result.sanitized_value
            sanitized_predictions = []
            
            # Validate each prediction
            required_keys = ['source_entity', 'target_entity', 'edge_type']
            optional_keys = ['confidence', 'risk_score', 'importance_score', 'mitre_technique']
            
            for i, pred in enumerate(predictions_list):
                # Validate prediction structure
                dict_result = self.validate_dict(pred, required_keys, optional_keys, True)
                if not dict_result.is_valid:
                    errors.extend([f"Prediction {i}: {error}" for error in dict_result.errors])
                    continue
                
                pred_dict = dict_result.sanitized_value
                
                # Validate specific fields
                # Source entity
                source_result = self.validate_string(
                    pred_dict.get('source_entity'), 'entity_name', 200, True
                )
                if not source_result.is_valid:
                    errors.extend([f"Prediction {i} source_entity: {error}" for error in source_result.errors])
                    continue
                
                # Target entity
                target_result = self.validate_string(
                    pred_dict.get('target_entity'), 'entity_name', 200, True
                )
                if not target_result.is_valid:
                    errors.extend([f"Prediction {i} target_entity: {error}" for error in target_result.errors])
                    continue
                
                # Edge type
                edge_result = self.validate_string(
                    pred_dict.get('edge_type'), 'alphanumeric_extended', 100, True
                )
                if not edge_result.is_valid:
                    errors.extend([f"Prediction {i} edge_type: {error}" for error in edge_result.errors])
                    continue
                
                # Confidence (optional)
                if 'confidence' in pred_dict:
                    conf_result = self.validate_numeric(
                        pred_dict['confidence'], 'confidence', required=False
                    )
                    if not conf_result.is_valid:
                        errors.extend([f"Prediction {i} confidence: {error}" for error in conf_result.errors])
                        continue
                    pred_dict['confidence'] = conf_result.sanitized_value
                
                # Risk score (optional)
                if 'risk_score' in pred_dict:
                    risk_result = self.validate_numeric(
                        pred_dict['risk_score'], 'risk_score', required=False
                    )
                    if not risk_result.is_valid:
                        errors.extend([f"Prediction {i} risk_score: {error}" for error in risk_result.errors])
                        continue
                    pred_dict['risk_score'] = risk_result.sanitized_value
                
                # MITRE technique (optional)
                if 'mitre_technique' in pred_dict:
                    mitre_result = self.validate_string(
                        pred_dict['mitre_technique'], 'mitre_technique', 20, False
                    )
                    if not mitre_result.is_valid:
                        warnings.extend([f"Prediction {i} mitre_technique: {error}" for error in mitre_result.errors])
                    else:
                        pred_dict['mitre_technique'] = mitre_result.sanitized_value
                
                sanitized_predictions.append(pred_dict)
            
            if errors:
                return ValidationResult(False, errors, warnings)
            
            return ValidationResult(True, errors, warnings, sanitized_predictions)
            
        except Exception as e:
            logger.error(f"Prediction data validation failed: {e}")
            return ValidationResult(False, [f"Validation error: {e}"], [])
    
    def _sanitize_string(self, value: str) -> str:
        """
        Sanitize string value for safe use.
        
        Parameters
        ----------
        value : str
            String to sanitize
            
        Returns
        -------
        str
            Sanitized string
        """
        try:
            # Remove null bytes
            sanitized = value.replace('\x00', '')
            
            # Normalize whitespace
            sanitized = ' '.join(sanitized.split())
            
            # Remove control characters except newlines and tabs
            sanitized = ''.join(char for char in sanitized 
                              if ord(char) >= 32 or char in '\n\t')
            
            return sanitized
            
        except Exception as e:
            logger.error(f"String sanitization failed: {e}")
            return value


def test_input_validator():
    """Test the input validator"""
    
    logger.info("Testing Input Validator...")
    
    try:
        validator = InputValidator()
        
        # Test string validation
        result = validator.validate_string("test_string", "alphanumeric")
        assert result.is_valid, f"String validation failed: {result.errors}"
        
        # Test malicious string detection
        result = validator.validate_string("<script>alert('xss')</script>", "alphanumeric")
        assert not result.is_valid, "Malicious string not detected"
        
        # Test numeric validation
        result = validator.validate_numeric(0.85, "confidence")
        assert result.is_valid, f"Numeric validation failed: {result.errors}"
        assert result.sanitized_value == 0.85, "Numeric value not preserved"
        
        # Test array validation
        result = validator.validate_array([1, 2, 3], "numeric", 10)
        assert result.is_valid, f"Array validation failed: {result.errors}"
        
        # Test dictionary validation
        test_dict = {"key1": "value1", "key2": "value2"}
        result = validator.validate_dict(test_dict, ["key1"], ["key2"])
        assert result.is_valid, f"Dict validation failed: {result.errors}"
        
        # Test prediction validation
        predictions = [
            {
                "source_entity": "user_1",
                "target_entity": "host_2",
                "edge_type": "logs_into",
                "confidence": 0.85,
                "risk_score": 0.9
            }
        ]
        result = validator.validate_prediction_data(predictions)
        assert result.is_valid, f"Prediction validation failed: {result.errors}"
        
        logger.info(" All input validator tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Input validator test failed: {e}")
        return False


if __name__ == '__main__':
    """Test the input validator"""
    
    success = test_input_validator()
    
    if success:
        logger.info(" Input Validator implementation complete!")
        logger.info(" Ready for visualization code fixes")
    else:
        logger.error("[ERROR] Input Validator needs fixes")
        sys.exit(1)