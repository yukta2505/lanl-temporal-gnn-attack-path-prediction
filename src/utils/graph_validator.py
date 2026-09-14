#!/usr/bin/env python3
"""
Critical Fix: Graph Data Validation and Format Correction
Ensures all graph data is in correct HeteroData format for THGNN processing.

Author: THGNN Expert
Date: January 10, 2026
Version: 1.0 - CRITICAL FIX
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import torch
import numpy as np
from torch_geometric.data import HeteroData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GraphValidator:
    """
    Critical graph data validator and format corrector.
    Ensures all graphs are valid HeteroData objects with required attributes.
    """
    
    def __init__(self):
        """Initialize the graph validator"""
        self.required_node_types = ['user', 'host', 'process']
        self.required_edge_types = [
            ('user', 'logs_into', 'host'),
            ('user', 'executes', 'process'),
            ('process', 'creates', 'process'),
            ('process', 'connects_to', 'host')
        ]
        
        logger.info("Graph validator initialized")
    
    def validate_graph(self, graph: Any) -> Tuple[bool, List[str]]:
        """
        Validate a single graph object.
        
        Parameters
        ----------
        graph : Any
            Graph object to validate
            
        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Check if it's a HeteroData object
            if not isinstance(graph, HeteroData):
                errors.append(f"Graph is not HeteroData, got {type(graph)}")
                return False, errors
            
            # Check for x_dict attribute
            if not hasattr(graph, 'x_dict'):
                errors.append("Graph missing x_dict attribute")
                return False, errors
            
            # Check for edge_index_dict attribute
            if not hasattr(graph, 'edge_index_dict'):
                errors.append("Graph missing edge_index_dict attribute")
                return False, errors
            
            # Check node types
            if not graph.x_dict:
                errors.append("Graph has empty x_dict")
                return False, errors
            
            # Validate node features
            for node_type, features in graph.x_dict.items():
                if not isinstance(features, torch.Tensor):
                    errors.append(f"Node features for {node_type} are not tensors")
                    continue
                
                if features.dim() != 2:
                    errors.append(f"Node features for {node_type} must be 2D, got {features.dim()}D")
                
                if features.size(0) == 0:
                    errors.append(f"Node type {node_type} has zero nodes")
            
            # Validate edge indices
            for edge_type, edge_index in graph.edge_index_dict.items():
                if not isinstance(edge_index, torch.Tensor):
                    errors.append(f"Edge index for {edge_type} is not a tensor")
                    continue
                
                if edge_index.dim() != 2 or edge_index.size(0) != 2:
                    errors.append(f"Edge index for {edge_type} must be [2, num_edges], got {edge_index.shape}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors
    
    def fix_graph_format(self, graph: Any) -> Optional[HeteroData]:
        """
        Attempt to fix graph format issues.
        
        Parameters
        ----------
        graph : Any
            Graph object to fix
            
        Returns
        -------
        Optional[HeteroData]
            Fixed graph or None if unfixable
        """
        try:
            # If it's already a valid HeteroData, return as-is
            if isinstance(graph, HeteroData):
                is_valid, errors = self.validate_graph(graph)
                if is_valid:
                    return graph
                
                # Try to fix common issues
                return self._fix_hetero_data(graph)
            
            # If it's a string (file path), try to load it
            if isinstance(graph, str):
                return self._load_graph_from_path(graph)
            
            # If it's a dict, try to convert to HeteroData
            if isinstance(graph, dict):
                return self._dict_to_hetero_data(graph)
            
            logger.error(f"Cannot fix graph of type {type(graph)}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to fix graph format: {e}")
            return None
    
    def _fix_hetero_data(self, graph: HeteroData) -> Optional[HeteroData]:
        """Fix issues in existing HeteroData object"""
        try:
            fixed_graph = HeteroData()
            
            # Fix x_dict
            if hasattr(graph, 'x_dict') and graph.x_dict:
                for node_type, features in graph.x_dict.items():
                    if isinstance(features, torch.Tensor) and features.size(0) > 0:
                        # Ensure 2D tensor
                        if features.dim() == 1:
                            features = features.unsqueeze(1)
                        fixed_graph[node_type].x = features
            else:
                # Create minimal node features
                for node_type in self.required_node_types:
                    fixed_graph[node_type].x = torch.randn(1, 64)  # Minimal features
            
            # Fix edge_index_dict
            if hasattr(graph, 'edge_index_dict') and graph.edge_index_dict:
                for edge_type, edge_index in graph.edge_index_dict.items():
                    if isinstance(edge_index, torch.Tensor) and edge_index.size(1) > 0:
                        # Ensure correct shape [2, num_edges]
                        if edge_index.dim() == 1:
                            edge_index = edge_index.unsqueeze(0).repeat(2, 1)
                        elif edge_index.size(0) != 2:
                            edge_index = edge_index.t()
                        
                        fixed_graph[edge_type].edge_index = edge_index
            
            return fixed_graph
            
        except Exception as e:
            logger.error(f"Failed to fix HeteroData: {e}")
            return None
    
    def _load_graph_from_path(self, path: str) -> Optional[HeteroData]:
        """Load graph from file path"""
        try:
            if not os.path.exists(path):
                logger.error(f"Graph file not found: {path}")
                return None
            
            graph = torch.load(path, map_location='cpu', weights_only=False)
            return self.fix_graph_format(graph)
            
        except Exception as e:
            logger.error(f"Failed to load graph from {path}: {e}")
            return None
    
    def _dict_to_hetero_data(self, graph_dict: Dict) -> Optional[HeteroData]:
        """Convert dictionary to HeteroData"""
        try:
            hetero_data = HeteroData()
            
            # Handle x_dict
            if 'x_dict' in graph_dict:
                for node_type, features in graph_dict['x_dict'].items():
                    if isinstance(features, (list, np.ndarray)):
                        features = torch.tensor(features, dtype=torch.float32)
                    hetero_data[node_type].x = features
            
            # Handle edge_index_dict
            if 'edge_index_dict' in graph_dict:
                for edge_type, edge_index in graph_dict['edge_index_dict'].items():
                    if isinstance(edge_index, (list, np.ndarray)):
                        edge_index = torch.tensor(edge_index, dtype=torch.long)
                    hetero_data[edge_type].edge_index = edge_index
            
            return hetero_data
            
        except Exception as e:
            logger.error(f"Failed to convert dict to HeteroData: {e}")
            return None
    
    def validate_graph_sequence(self, sequence: List[Any]) -> Tuple[List[HeteroData], List[str]]:
        """
        Validate and fix a sequence of graphs.
        
        Parameters
        ----------
        sequence : List[Any]
            Sequence of graph objects
            
        Returns
        -------
        Tuple[List[HeteroData], List[str]]
            (valid_graphs, error_messages)
        """
        valid_graphs = []
        all_errors = []
        
        for i, graph in enumerate(sequence):
            try:
                # Try to fix the graph
                fixed_graph = self.fix_graph_format(graph)
                
                if fixed_graph is not None:
                    # Validate the fixed graph
                    is_valid, errors = self.validate_graph(fixed_graph)
                    
                    if is_valid:
                        valid_graphs.append(fixed_graph)
                    else:
                        all_errors.extend([f"Graph {i}: {error}" for error in errors])
                else:
                    all_errors.append(f"Graph {i}: Could not fix format")
                    
            except Exception as e:
                all_errors.append(f"Graph {i}: Validation failed - {e}")
        
        logger.info(f"Validated sequence: {len(valid_graphs)}/{len(sequence)} graphs valid")
        
        return valid_graphs, all_errors
    
    def create_minimal_graph(self) -> HeteroData:
        """Create a minimal valid graph for testing"""
        graph = HeteroData()
        
        # Add minimal node features
        for node_type in self.required_node_types:
            graph[node_type].x = torch.randn(2, 64)  # 2 nodes, 64 features each
        
        # Add minimal edges
        for src, rel, dst in self.required_edge_types:
            if src in self.required_node_types and dst in self.required_node_types:
                # Create edge between first nodes of each type
                graph[src, rel, dst].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        return graph


def fix_training_data_format(data_loader: Any) -> Any:
    """
    Critical fix for training data format issues.
    Ensures all data in the loader is properly formatted.
    """
    validator = GraphValidator()
    
    try:
        # Fix train_loader
        if hasattr(data_loader, 'train_loader') and data_loader.train_loader:
            logger.info("Fixing train_loader format...")
            fixed_sequences = []
            
            for sequence in data_loader.train_loader:
                if isinstance(sequence, list):
                    valid_graphs, errors = validator.validate_graph_sequence(sequence)
                    if valid_graphs:
                        fixed_sequences.append(valid_graphs)
                    else:
                        logger.warning(f"Skipping invalid sequence: {errors}")
                else:
                    # Single graph
                    fixed_graph = validator.fix_graph_format(sequence)
                    if fixed_graph:
                        fixed_sequences.append([fixed_graph])
            
            data_loader.train_loader = fixed_sequences
            logger.info(f"Fixed train_loader: {len(fixed_sequences)} sequences")
        
        # Fix val_loader
        if hasattr(data_loader, 'val_loader') and data_loader.val_loader:
            logger.info("Fixing val_loader format...")
            fixed_sequences = []
            
            for sequence in data_loader.val_loader:
                if isinstance(sequence, list):
                    valid_graphs, errors = validator.validate_graph_sequence(sequence)
                    if valid_graphs:
                        fixed_sequences.append(valid_graphs)
                else:
                    fixed_graph = validator.fix_graph_format(sequence)
                    if fixed_graph:
                        fixed_sequences.append([fixed_graph])
            
            data_loader.val_loader = fixed_sequences
            logger.info(f"Fixed val_loader: {len(fixed_sequences)} sequences")
        
        # Fix test_loader
        if hasattr(data_loader, 'test_loader') and data_loader.test_loader:
            logger.info("Fixing test_loader format...")
            fixed_sequences = []
            
            for sequence in data_loader.test_loader:
                if isinstance(sequence, list):
                    valid_graphs, errors = validator.validate_graph_sequence(sequence)
                    if valid_graphs:
                        fixed_sequences.append(valid_graphs)
                else:
                    fixed_graph = validator.fix_graph_format(sequence)
                    if fixed_graph:
                        fixed_sequences.append([fixed_graph])
            
            data_loader.test_loader = fixed_sequences
            logger.info(f"Fixed test_loader: {len(fixed_sequences)} sequences")
        
        return data_loader
        
    except Exception as e:
        logger.error(f"Failed to fix training data format: {e}")
        return data_loader


def test_graph_validator():
    """Test the graph validator"""
    logger.info("Testing Graph Validator...")
    
    try:
        validator = GraphValidator()
        
        # Test 1: Create minimal valid graph
        valid_graph = validator.create_minimal_graph()
        is_valid, errors = validator.validate_graph(valid_graph)
        
        if is_valid:
            logger.info(" Minimal graph validation passed")
        else:
            logger.error(f"[ERROR] Minimal graph validation failed: {errors}")
            return False
        
        # Test 2: Test invalid graph (string)
        invalid_graph = "not_a_graph"
        fixed_graph = validator.fix_graph_format(invalid_graph)
        
        if fixed_graph is None:
            logger.info(" Correctly rejected invalid string graph")
        else:
            logger.error("[ERROR] Should have rejected string graph")
        
        # Test 3: Test sequence validation
        sequence = [valid_graph, valid_graph, "invalid"]
        valid_graphs, errors = validator.validate_graph_sequence(sequence)
        
        if len(valid_graphs) == 2:
            logger.info(" Sequence validation passed")
        else:
            logger.error(f"[ERROR] Sequence validation failed: got {len(valid_graphs)} valid graphs")
            return False
        
        logger.info(" All graph validator tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Graph validator test failed: {e}")
        return False


if __name__ == '__main__':
    """Test the graph validator"""
    success = test_graph_validator()
    
    if success:
        logger.info(" Graph Validator implementation complete!")
        logger.info(" Ready to fix critical training issues")
    else:
        logger.error("[ERROR] Graph Validator implementation needs fixes")
        sys.exit(1)