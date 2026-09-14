#!/usr/bin/env python3
"""
Day 7 Afternoon: Interactive Graph Visualization for THGNN
Production-ready graph visualization using NetworkX and Plotly for security analysts.

Author: THGNN Expert
Date: January 9, 2026
Version: 1.0 - INTERACTIVE GRAPH VISUALIZATION
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GraphVisualizationConfig:
    """Configuration for graph visualization"""
    width: int = 1200
    height: int = 800
    node_size_range: Tuple[int, int] = (10, 50)
    edge_width_range: Tuple[int, int] = (1, 8)
    color_scheme: str = 'security'  # 'security', 'risk', 'entity_type'
    layout_algorithm: str = 'spring'  # 'spring', 'circular', 'kamada_kawai'
    show_labels: bool = True
    show_edge_labels: bool = False
    interactive: bool = True
    export_format: str = 'html'  # 'html', 'png', 'svg'

class GraphVisualizer:
    """
    Production-ready graph visualizer for THGNN predictions.
    Creates interactive network visualizations for security analysts.
    """
    
    def __init__(self, config: Optional[GraphVisualizationConfig] = None):
        """
        Initialize the graph visualizer.
        
        Parameters
        ----------
        config : GraphVisualizationConfig, optional
            Visualization configuration
        """
        self.config = config or GraphVisualizationConfig()
        
        # Security-focused color schemes
        self.color_schemes = {
            'security': {
                'user': '#2E86AB',      # Blue for users
                'host': '#A23B72',      # Purple for hosts  
                'process': '#F18F01',   # Orange for processes
                'file': '#C73E1D',      # Red for files
                'registry': '#8B5A3C',  # Brown for registry
                'background': '#FFFFFF',
                'text': '#2C3E50',
                'grid': '#ECF0F1'
            },
            'risk': {
                'low': '#27AE60',       # Green for low risk
                'medium': '#F39C12',    # Orange for medium risk
                'high': '#E74C3C',      # Red for high risk
                'critical': '#8E44AD'   # Purple for critical risk
            },
            'prediction': {
                'existing': '#95A5A6',  # Gray for existing edges
                'predicted': '#E74C3C', # Red for predicted edges
                'confident': '#C0392B', # Dark red for high confidence
                'uncertain': '#F39C12'  # Orange for low confidence
            }
        }
        
        # Layout algorithms
        self.layout_algorithms = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'shell': nx.shell_layout,
            'spectral': nx.spectral_layout
        }
        
        logger.info("Graph visualizer initialized successfully")
    
    def plot_graph_snapshot(
        self,
        graph: HeteroData,
        title: str = "Network Graph Snapshot",
        predictions: Optional[List[Dict]] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive graph visualization of a single snapshot.
        
        Parameters
        ----------
        graph : HeteroData
            PyTorch Geometric heterogeneous graph
        title : str
            Plot title
        predictions : List[Dict], optional
            Predicted edges to highlight
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        try:
            logger.info(f"Creating graph visualization: {title}")
            
            # Convert PyG graph to NetworkX
            G = self._convert_to_networkx(graph)
            
            if len(G.nodes()) == 0:
                logger.warning("Empty graph provided")
                return self._create_empty_plot(title)
            
            # Compute layout
            pos = self._compute_layout(G)
            
            # Create node traces
            node_traces = self._create_node_traces(G, pos, graph)
            
            # Create edge traces
            edge_traces = self._create_edge_traces(G, pos, predictions)
            
            # Create figure
            fig = go.Figure()
            
            # Add edge traces first (so they appear behind nodes)
            for trace in edge_traces:
                fig.add_trace(trace)
            
            # Add node traces
            for trace in node_traces:
                fig.add_trace(trace)
            
            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': self.color_schemes['security']['text']}
                },
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Hover over nodes and edges for details. Click and drag to explore.",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color=self.color_schemes['security']['text'], size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor=self.color_schemes['security']['background'],
                width=self.config.width,
                height=self.config.height
            )
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            logger.info(f"Graph visualization created successfully with {len(G.nodes())} nodes and {len(G.edges())} edges")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create graph visualization: {e}")
            return self._create_error_plot(title, str(e))
    
    def plot_temporal_evolution(
        self,
        graph_sequence: List[HeteroData],
        title: str = "Temporal Graph Evolution",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create animated visualization showing graph evolution over time.
        
        Parameters
        ----------
        graph_sequence : List[HeteroData]
            Sequence of graph snapshots
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        go.Figure
            Plotly figure with animation
        """
        try:
            logger.info(f"Creating temporal evolution visualization with {len(graph_sequence)} snapshots")
            
            if not graph_sequence:
                return self._create_empty_plot(title)
            
            # Convert all graphs to NetworkX
            nx_graphs = [self._convert_to_networkx(g) for g in graph_sequence]
            
            # Get union of all nodes for consistent positioning
            all_nodes = set()
            for G in nx_graphs:
                all_nodes.update(G.nodes())
            
            # Create master graph for layout
            master_graph = nx.Graph()
            master_graph.add_nodes_from(all_nodes)
            
            # Add all edges from all snapshots
            for G in nx_graphs:
                master_graph.add_edges_from(G.edges())
            
            # Compute consistent layout
            pos = self._compute_layout(master_graph)
            
            # Create frames for animation
            frames = []
            for i, G in enumerate(nx_graphs):
                frame_data = []
                
                # Add edge traces for this frame
                edge_traces = self._create_edge_traces(G, pos)
                frame_data.extend(edge_traces)
                
                # Add node traces for this frame
                node_traces = self._create_node_traces(G, pos, graph_sequence[i])
                frame_data.extend(node_traces)
                
                frames.append(go.Frame(
                    data=frame_data,
                    name=f"snapshot_{i}",
                    layout=go.Layout(
                        title=f"{title} - Snapshot {i+1}/{len(graph_sequence)}"
                    )
                ))
            
            # Create initial figure with first frame
            fig = go.Figure(
                data=frames[0].data if frames else [],
                frames=frames
            )
            
            # Add animation controls
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 1000, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 300}
                            }]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        }
                    ]
                }],
                sliders=[{
                    'steps': [
                        {
                            'args': [[f"snapshot_{i}"], {
                                'frame': {'duration': 300, 'redraw': True},
                                'mode': 'immediate',
                                'transition': {'duration': 300}
                            }],
                            'label': f"T+{i}",
                            'method': 'animate'
                        }
                        for i in range(len(graph_sequence))
                    ],
                    'active': 0,
                    'currentvalue': {'prefix': 'Time Window: '},
                    'len': 0.9,
                    'x': 0.1,
                    'xanchor': 'left',
                    'y': 0,
                    'yanchor': 'top'
                }],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor=self.color_schemes['security']['background'],
                width=self.config.width,
                height=self.config.height + 100  # Extra space for controls
            )
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            logger.info("Temporal evolution visualization created successfully")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create temporal evolution visualization: {e}")
            return self._create_error_plot(title, str(e))
    
    def plot_explanation_subgraph(
        self,
        explanation_result,
        graph: HeteroData,
        title: str = "Attack Explanation Subgraph",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Visualize explanation subgraph with highlighted important nodes/edges.
        
        Parameters
        ----------
        explanation_result : ExplanationResult
            Result from GNN explainer
        graph : HeteroData
            Original graph
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        go.Figure
            Plotly figure
        """
        try:
            logger.info("Creating explanation subgraph visualization")
            
            # Convert to NetworkX
            G = self._convert_to_networkx(graph)
            
            # Filter to subgraph nodes
            subgraph_nodes = explanation_result.subgraph_nodes
            if subgraph_nodes:
                # Map node IDs to actual node identifiers
                actual_nodes = []
                for node_id in subgraph_nodes:
                    if node_id in G.nodes():
                        actual_nodes.append(node_id)
                
                if actual_nodes:
                    G = G.subgraph(actual_nodes).copy()
            
            if len(G.nodes()) == 0:
                logger.warning("Empty explanation subgraph")
                return self._create_empty_plot(title)
            
            # Compute layout
            pos = self._compute_layout(G)
            
            # Create node traces with importance highlighting
            node_traces = self._create_explanation_node_traces(
                G, pos, explanation_result
            )
            
            # Create edge traces with importance highlighting
            edge_traces = self._create_explanation_edge_traces(
                G, pos, explanation_result
            )
            
            # Create figure
            fig = go.Figure()
            
            # Add traces
            for trace in edge_traces:
                fig.add_trace(trace)
            for trace in node_traces:
                fig.add_trace(trace)
            
            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                showlegend=True,
                hovermode='closest',
                annotations=[
                    dict(
                        text=f"Explanation for prediction: {explanation_result.prediction_id}<br>"
                             f"Confidence: {explanation_result.confidence:.1%}<br>"
                             f"Risk Score: {explanation_result.risk_score:.1f}",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        xanchor='left', yanchor='top',
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="black",
                        borderwidth=1,
                        font=dict(size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor=self.color_schemes['security']['background'],
                width=self.config.width,
                height=self.config.height
            )
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            logger.info("Explanation subgraph visualization created successfully")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create explanation subgraph: {e}")
            return self._create_error_plot(title, str(e))
    
    def _convert_to_networkx(self, graph: HeteroData) -> nx.Graph:
        """Convert PyG HeteroData to NetworkX graph"""
        try:
            # Create empty graph
            G = nx.Graph()
            
            # Add nodes with attributes
            for node_type in graph.node_types:
                if node_type in graph.x_dict:
                    num_nodes = graph.x_dict[node_type].size(0)
                    for i in range(num_nodes):
                        node_id = f"{node_type}_{i}"
                        G.add_node(node_id, 
                                  node_type=node_type,
                                  features=graph.x_dict[node_type][i].tolist() if graph.x_dict[node_type].dim() > 1 else [graph.x_dict[node_type][i].item()])
            
            # Add edges with attributes
            for edge_type in graph.edge_types:
                if edge_type in graph.edge_index_dict:
                    edge_index = graph.edge_index_dict[edge_type]
                    src_type, relation, dst_type = edge_type
                    
                    for i in range(edge_index.size(1)):
                        src_idx = edge_index[0, i].item()
                        dst_idx = edge_index[1, i].item()
                        
                        src_node = f"{src_type}_{src_idx}"
                        dst_node = f"{dst_type}_{dst_idx}"
                        
                        if src_node in G.nodes() and dst_node in G.nodes():
                            G.add_edge(src_node, dst_node,
                                     edge_type=relation,
                                     source_type=src_type,
                                     target_type=dst_type)
            
            return G
            
        except Exception as e:
            logger.error(f"Failed to convert graph to NetworkX: {e}")
            return nx.Graph()  # Return empty graph
    
    def _compute_layout(self, G: nx.Graph) -> Dict:
        """Compute node positions using specified layout algorithm"""
        try:
            if len(G.nodes()) == 0:
                return {}
            
            layout_func = self.layout_algorithms.get(
                self.config.layout_algorithm, 
                nx.spring_layout
            )
            
            # Use spring layout with good parameters for security graphs
            if self.config.layout_algorithm == 'spring':
                pos = layout_func(G, k=3, iterations=50, seed=42)
            else:
                pos = layout_func(G)
            
            return pos
            
        except Exception as e:
            logger.error(f"Failed to compute layout: {e}")
            # Fallback to simple circular layout
            return nx.circular_layout(G)
    
    def _create_node_traces(
        self, 
        G: nx.Graph, 
        pos: Dict, 
        graph: HeteroData
    ) -> List[go.Scatter]:
        """Create node traces grouped by entity type"""
        traces = []
        
        try:
            # Group nodes by type
            nodes_by_type = {}
            for node in G.nodes(data=True):
                node_id, node_data = node
                node_type = node_data.get('node_type', 'unknown')
                
                if node_type not in nodes_by_type:
                    nodes_by_type[node_type] = []
                nodes_by_type[node_type].append((node_id, node_data))
            
            # Create trace for each node type
            for node_type, nodes in nodes_by_type.items():
                if not nodes:
                    continue
                
                x_coords = []
                y_coords = []
                hover_texts = []
                node_sizes = []
                
                for node_id, node_data in nodes:
                    if node_id in pos:
                        x_coords.append(pos[node_id][0])
                        y_coords.append(pos[node_id][1])
                        
                        # Create hover text
                        hover_text = f"<b>{node_id}</b><br>"
                        hover_text += f"Type: {node_type}<br>"
                        hover_text += f"Connections: {G.degree(node_id)}"
                        hover_texts.append(hover_text)
                        
                        # Node size based on degree
                        degree = G.degree(node_id)
                        size = max(self.config.node_size_range[0], 
                                 min(self.config.node_size_range[1], 
                                     10 + degree * 3))
                        node_sizes.append(size)
                
                if x_coords:  # Only create trace if we have coordinates
                    trace = go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='markers+text' if self.config.show_labels else 'markers',
                        text=[node_id.split('_')[0] for node_id, _ in nodes] if self.config.show_labels else None,
                        textposition="middle center",
                        textfont=dict(size=8, color='white'),
                        hovertext=hover_texts,
                        hoverinfo='text',
                        marker=dict(
                            size=node_sizes,
                            color=self.color_schemes['security'].get(node_type, '#95A5A6'),
                            line=dict(width=2, color='white'),
                            opacity=0.8
                        ),
                        name=f"{node_type.title()} Nodes",
                        showlegend=True
                    )
                    traces.append(trace)
            
            return traces
            
        except Exception as e:
            logger.error(f"Failed to create node traces: {e}")
            return []
    
    def _create_edge_traces(
        self, 
        G: nx.Graph, 
        pos: Dict, 
        predictions: Optional[List[Dict]] = None
    ) -> List[go.Scatter]:
        """Create edge traces"""
        traces = []
        
        try:
            # Group edges by type
            edges_by_type = {}
            predicted_edges = set()
            
            # Mark predicted edges
            if predictions:
                for pred in predictions:
                    src = pred.get('source_entity', '')
                    dst = pred.get('target_entity', '')
                    predicted_edges.add((src, dst))
                    predicted_edges.add((dst, src))  # Undirected
            
            for edge in G.edges(data=True):
                src, dst, edge_data = edge
                edge_type = edge_data.get('edge_type', 'unknown')
                
                # Check if this is a predicted edge
                is_predicted = (src, dst) in predicted_edges or (dst, src) in predicted_edges
                category = 'predicted' if is_predicted else edge_type
                
                if category not in edges_by_type:
                    edges_by_type[category] = []
                edges_by_type[category].append((src, dst, edge_data))
            
            # Create trace for each edge type
            for edge_type, edges in edges_by_type.items():
                if not edges:
                    continue
                
                x_coords = []
                y_coords = []
                hover_texts = []
                
                for src, dst, edge_data in edges:
                    if src in pos and dst in pos:
                        # Add line coordinates
                        x_coords.extend([pos[src][0], pos[dst][0], None])
                        y_coords.extend([pos[src][1], pos[dst][1], None])
                        
                        # Hover text for middle of edge
                        hover_text = f"<b>{src} → {dst}</b><br>"
                        hover_text += f"Relation: {edge_data.get('edge_type', 'unknown')}"
                        hover_texts.append(hover_text)
                
                if x_coords:  # Only create trace if we have coordinates
                    # Different styling for predicted vs existing edges
                    if edge_type == 'predicted':
                        line_color = self.color_schemes['prediction']['predicted']
                        line_width = 4
                        line_dash = 'dash'
                        opacity = 0.8
                    else:
                        line_color = self.color_schemes['security'].get(edge_type, '#95A5A6')
                        line_width = 2
                        line_dash = 'solid'
                        opacity = 0.6
                    
                    trace = go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        line=dict(
                            width=line_width,
                            color=line_color,
                            dash=line_dash
                        ),
                        opacity=opacity,
                        name=f"{edge_type.replace('_', ' ').title()} Edges",
                        showlegend=True,
                        hoverinfo='skip'
                    )
                    traces.append(trace)
            
            return traces
            
        except Exception as e:
            logger.error(f"Failed to create edge traces: {e}")
            return []
    
    def _create_explanation_node_traces(
        self, 
        G: nx.Graph, 
        pos: Dict, 
        explanation_result
    ) -> List[go.Scatter]:
        """Create node traces with explanation importance highlighting"""
        traces = []
        
        try:
            # Get importance scores
            node_importance = explanation_result.node_importance
            influential_nodes = set(explanation_result.influential_nodes)
            
            # Group nodes by importance level
            high_importance = []
            medium_importance = []
            low_importance = []
            
            for node in G.nodes():
                importance = node_importance.get(node, 0.0)
                
                if node in influential_nodes or importance > 0.7:
                    high_importance.append(node)
                elif importance > 0.3:
                    medium_importance.append(node)
                else:
                    low_importance.append(node)
            
            # Create traces for each importance level
            importance_groups = [
                (high_importance, 'High Importance', '#E74C3C', 30),
                (medium_importance, 'Medium Importance', '#F39C12', 20),
                (low_importance, 'Low Importance', '#95A5A6', 15)
            ]
            
            for nodes, name, color, base_size in importance_groups:
                if not nodes:
                    continue
                
                x_coords = []
                y_coords = []
                hover_texts = []
                sizes = []
                
                for node in nodes:
                    if node in pos:
                        x_coords.append(pos[node][0])
                        y_coords.append(pos[node][1])
                        
                        importance = node_importance.get(node, 0.0)
                        hover_text = f"<b>{node}</b><br>"
                        hover_text += f"Importance: {importance:.3f}<br>"
                        hover_text += f"Connections: {G.degree(node)}"
                        hover_texts.append(hover_text)
                        
                        # Size based on importance
                        size = base_size + (importance * 20)
                        sizes.append(size)
                
                if x_coords:
                    trace = go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='markers',
                        hovertext=hover_texts,
                        hoverinfo='text',
                        marker=dict(
                            size=sizes,
                            color=color,
                            line=dict(width=2, color='white'),
                            opacity=0.8
                        ),
                        name=name,
                        showlegend=True
                    )
                    traces.append(trace)
            
            return traces
            
        except Exception as e:
            logger.error(f"Failed to create explanation node traces: {e}")
            return []
    
    def _create_explanation_edge_traces(
        self, 
        G: nx.Graph, 
        pos: Dict, 
        explanation_result
    ) -> List[go.Scatter]:
        """Create edge traces with explanation importance highlighting"""
        traces = []
        
        try:
            # Get edge importance
            edge_importance = explanation_result.edge_importance
            influential_edges = explanation_result.influential_edges
            
            # Create mapping of influential edges
            influential_edge_set = set()
            for edge_info in influential_edges:
                src = edge_info.get('source', '')
                dst = edge_info.get('target', '')
                influential_edge_set.add((src, dst))
                influential_edge_set.add((dst, src))  # Undirected
            
            # Group edges by importance
            high_importance_edges = []
            low_importance_edges = []
            
            for src, dst, edge_data in G.edges(data=True):
                edge_key = f"{src}_{dst}"
                importance = edge_importance.get(edge_key, 0.0)
                
                if (src, dst) in influential_edge_set or importance > 0.5:
                    high_importance_edges.append((src, dst, edge_data, importance))
                else:
                    low_importance_edges.append((src, dst, edge_data, importance))
            
            # Create traces
            edge_groups = [
                (high_importance_edges, 'Important Edges', '#E74C3C', 4, 'solid'),
                (low_importance_edges, 'Other Edges', '#95A5A6', 1, 'solid')
            ]
            
            for edges, name, color, width, dash in edge_groups:
                if not edges:
                    continue
                
                x_coords = []
                y_coords = []
                
                for src, dst, edge_data, importance in edges:
                    if src in pos and dst in pos:
                        x_coords.extend([pos[src][0], pos[dst][0], None])
                        y_coords.extend([pos[src][1], pos[dst][1], None])
                
                if x_coords:
                    trace = go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        line=dict(
                            width=width,
                            color=color,
                            dash=dash
                        ),
                        opacity=0.7,
                        name=name,
                        showlegend=True,
                        hoverinfo='skip'
                    )
                    traces.append(trace)
            
            return traces
            
        except Exception as e:
            logger.error(f"Failed to create explanation edge traces: {e}")
            return []
    
    def _create_empty_plot(self, title: str) -> go.Figure:
        """Create empty plot with message"""
        fig = go.Figure()
        fig.add_annotation(
            text="No graph data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=self.config.width,
            height=self.config.height
        )
        return fig
    
    def _create_error_plot(self, title: str, error_msg: str) -> go.Figure:
        """Create error plot with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization:<br>{error_msg}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=self.config.width,
            height=self.config.height
        )
        return fig
    
    def _save_figure(self, fig: go.Figure, save_path: str):
        """Save figure to file"""
        try:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.export_format == 'html':
                fig.write_html(str(path))
            elif self.config.export_format == 'png':
                fig.write_image(str(path))
            elif self.config.export_format == 'svg':
                fig.write_image(str(path))
            
            logger.info(f"Figure saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save figure: {e}")


class InteractiveGraphPlotter:
    """
    Simplified interface for creating interactive graph plots.
    Designed for easy integration with Streamlit and other frameworks.
    """
    
    def __init__(self):
        """Initialize the interactive graph plotter"""
        self.visualizer = GraphVisualizer()
        logger.info("Interactive graph plotter initialized")
    
    def plot(
        self,
        graph: HeteroData,
        title: str = "Network Graph",
        predictions: Optional[List[Dict]] = None,
        width: int = 1200,
        height: int = 800
    ) -> go.Figure:
        """
        Create interactive graph plot with minimal configuration.
        
        Parameters
        ----------
        graph : HeteroData
            PyTorch Geometric graph
        title : str
            Plot title
        predictions : List[Dict], optional
            Predicted edges to highlight
        width : int
            Plot width
        height : int
            Plot height
            
        Returns
        -------
        go.Figure
            Plotly figure
        """
        # Update config
        self.visualizer.config.width = width
        self.visualizer.config.height = height
        
        return self.visualizer.plot_graph_snapshot(
            graph=graph,
            title=title,
            predictions=predictions
        )


def test_graph_visualizer():
    """Test the graph visualizer with sample data"""
    
    logger.info("Testing Graph Visualizer...")
    
    try:
        # Create sample graph
        from torch_geometric.data import HeteroData
        import torch
        
        graph = HeteroData()
        graph['user'].x = torch.randn(5, 64)
        graph['host'].x = torch.randn(3, 64)
        graph['process'].x = torch.randn(4, 64)
        
        graph[('user', 'logs_into', 'host')].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
        graph[('user', 'executes', 'process')].edge_index = torch.tensor([[0, 1], [0, 1]])
        graph[('process', 'connects_to', 'host')].edge_index = torch.tensor([[0, 1], [0, 1]])
        
        # Initialize visualizer
        visualizer = GraphVisualizer()
        
        # Test basic graph visualization
        fig1 = visualizer.plot_graph_snapshot(
            graph=graph,
            title="Test Graph Snapshot"
        )
        logger.info(" Basic graph visualization created")
        
        # Test with predictions
        predictions = [
            {
                'source_entity': 'user_0',
                'target_entity': 'host_1',
                'edge_type': 'logs_into',
                'risk_score': 85.0,
                'confidence': 0.92
            }
        ]
        
        fig2 = visualizer.plot_graph_snapshot(
            graph=graph,
            title="Graph with Predictions",
            predictions=predictions
        )
        logger.info(" Graph with predictions created")
        
        # Test temporal evolution
        graph_sequence = [graph, graph, graph]  # Same graph for simplicity
        fig3 = visualizer.plot_temporal_evolution(
            graph_sequence=graph_sequence,
            title="Temporal Evolution Test"
        )
        logger.info(" Temporal evolution visualization created")
        
        # Test interactive plotter
        plotter = InteractiveGraphPlotter()
        fig4 = plotter.plot(
            graph=graph,
            title="Interactive Plot Test",
            predictions=predictions
        )
        logger.info(" Interactive plotter test completed")
        
        # Save test outputs
        output_dir = Path('outputs/visualizations/graphs')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig1.write_html(str(output_dir / 'test_basic_graph.html'))
        fig2.write_html(str(output_dir / 'test_graph_with_predictions.html'))
        fig3.write_html(str(output_dir / 'test_temporal_evolution.html'))
        fig4.write_html(str(output_dir / 'test_interactive_plot.html'))
        
        logger.info(" All graph visualizer tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Graph visualizer test failed: {e}")
        return False


if __name__ == '__main__':
    """Test the graph visualizer"""
    
    # Ensure output directory exists
    os.makedirs('outputs/visualizations/graphs', exist_ok=True)
    
    # Run tests
    success = test_graph_visualizer()
    
    if success:
        logger.info(" Day 7 Afternoon Graph Visualizer implementation complete!")
        logger.info(" Ready for timeline visualization")
    else:
        logger.error("[ERROR] Graph Visualizer implementation needs fixes")
        sys.exit(1)