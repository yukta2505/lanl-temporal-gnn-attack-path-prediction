#!/usr/bin/env python3
"""
Day 7 Afternoon: Process Tree Visualization for THGNN
Production-ready hierarchical process tree visualization for security analysts.

Author: THGNN Expert
Date: January 9, 2026
Version: 1.0 - PROCESS TREE VISUALIZATION
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
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessTreeConfig:
    """Configuration for process tree visualization"""
    width: int = 1200
    height: int = 800
    show_command_lines: bool = True
    highlight_suspicious: bool = True
    interactive: bool = True
    layout_algorithm: str = 'hierarchical'  # 'hierarchical', 'radial', 'tree'
    export_format: str = 'html'

class ProcessTreeVisualizer:
    """
    Production-ready process tree visualizer for security analysis.
    Creates hierarchical visualizations of process relationships.
    """
    
    def __init__(self, config: Optional[ProcessTreeConfig] = None):
        """
        Initialize the process tree visualizer.
        
        Parameters
        ----------
        config : ProcessTreeConfig, optional
            Visualization configuration
        """
        self.config = config or ProcessTreeConfig()
        
        # Suspicious process indicators
        self.suspicious_processes = {
            'powershell.exe', 'cmd.exe', 'wmic.exe', 'psexec.exe',
            'mimikatz.exe', 'procdump.exe', 'rundll32.exe',
            'regsvr32.exe', 'mshta.exe', 'cscript.exe', 'wscript.exe'
        }
        
        # Process risk levels
        self.process_risk_colors = {
            'critical': '#C0392B',    # Dark red
            'high': '#E74C3C',        # Red
            'medium': '#F39C12',      # Orange
            'low': '#27AE60',         # Green
            'unknown': '#95A5A6'      # Gray
        }
        
        # Process type colors
        self.process_type_colors = {
            'system': '#3498DB',      # Blue
            'service': '#9B59B6',     # Purple
            'application': '#2ECC71', # Green
            'script': '#E67E22',      # Orange
            'suspicious': '#E74C3C',  # Red
            'unknown': '#95A5A6'      # Gray
        }
        
        logger.info("Process tree visualizer initialized successfully")
    
    def build_process_tree(
        self,
        events: List[Dict[str, Any]],
        title: str = "Process Tree",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Build hierarchical process tree from security events.
        
        Parameters
        ----------
        events : List[Dict]
            List of process creation events
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        try:
            logger.info(f"Building process tree from {len(events)} events")
            
            if not events:
                return self._create_empty_tree(title)
            
            # Build process hierarchy
            process_tree = self._build_hierarchy(events)
            
            if not process_tree:
                return self._create_empty_tree(title)
            
            # Create NetworkX graph for layout
            G = self._create_networkx_tree(process_tree)
            
            # Compute hierarchical layout
            pos = self._compute_tree_layout(G)
            
            # Create visualization
            fig = self._create_tree_visualization(G, pos, process_tree, title)
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            logger.info("Process tree visualization created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to build process tree: {e}")
            return self._create_error_tree(title, str(e))
    
    def visualize_attack_process_tree(
        self,
        attack_chain,
        title: str = "Attack Process Tree",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Visualize process tree specifically for attack chains.
        
        Parameters
        ----------
        attack_chain : AttackChain
            Attack chain from explainability module
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        try:
            logger.info(f"Creating attack process tree for {attack_chain.chain_id}")
            
            if not attack_chain.attack_steps:
                return self._create_empty_tree(title)
            
            # Extract process events from attack steps
            process_events = []
            for step in attack_chain.attack_steps:
                if 'process' in step.edge_type or step.edge_type == 'executes':
                    event = {
                        'timestamp': step.timestamp,
                        'process_name': step.target_entity,
                        'parent_process': step.source_entity,
                        'command_line': getattr(step, 'command_line', ''),
                        'pid': getattr(step, 'pid', 0),
                        'parent_pid': getattr(step, 'parent_pid', 0),
                        'user': getattr(step, 'user', 'unknown'),
                        'importance_score': step.importance_score,
                        'mitre_technique': step.mitre_technique,
                        'is_attack_step': True
                    }
                    process_events.append(event)
            
            if not process_events:
                return self._create_empty_tree(title)
            
            # Build and visualize tree
            return self.build_process_tree(
                events=process_events,
                title=f"{title} - {attack_chain.attack_pattern or 'Unknown Pattern'}",
                save_path=save_path
            )
            
        except Exception as e:
            logger.error(f"Failed to create attack process tree: {e}")
            return self._create_error_tree(title, str(e))
    
    def _build_hierarchy(self, events: List[Dict]) -> Dict[str, Any]:
        """Build process hierarchy from events"""
        try:
            processes = {}
            parent_child_map = defaultdict(list)
            
            # Process events to build hierarchy
            for event in events:
                process_name = event.get('process_name', event.get('target_entity', 'unknown'))
                parent_name = event.get('parent_process', event.get('source_entity', ''))
                
                # Create process node
                process_id = f"{process_name}_{event.get('pid', len(processes))}"
                
                if process_id not in processes:
                    processes[process_id] = {
                        'name': process_name,
                        'pid': event.get('pid', 0),
                        'parent_pid': event.get('parent_pid', 0),
                        'command_line': event.get('command_line', ''),
                        'user': event.get('user', 'unknown'),
                        'timestamp': event.get('timestamp', datetime.now()),
                        'risk_level': self._assess_process_risk(process_name, event),
                        'is_suspicious': self._is_suspicious_process(process_name),
                        'is_attack_step': event.get('is_attack_step', False),
                        'importance_score': event.get('importance_score', 0.0),
                        'mitre_technique': event.get('mitre_technique', ''),
                        'children': []
                    }
                
                # Build parent-child relationships
                if parent_name:
                    parent_id = f"{parent_name}_{event.get('parent_pid', 0)}"
                    parent_child_map[parent_id].append(process_id)
            
            # Build tree structure
            for parent_id, children in parent_child_map.items():
                if parent_id in processes:
                    processes[parent_id]['children'] = children
            
            # Find root processes (no parents or parent not in our data)
            roots = []
            for process_id, process_info in processes.items():
                parent_pid = process_info['parent_pid']
                has_parent_in_data = any(
                    p['pid'] == parent_pid for p in processes.values()
                )
                if not has_parent_in_data or parent_pid == 0:
                    roots.append(process_id)
            
            return {
                'processes': processes,
                'roots': roots,
                'parent_child_map': dict(parent_child_map)
            }
            
        except Exception as e:
            logger.error(f"Failed to build hierarchy: {e}")
            return {}
    
    def _create_networkx_tree(self, process_tree: Dict) -> nx.DiGraph:
        """Create NetworkX directed graph from process tree"""
        try:
            G = nx.DiGraph()
            
            processes = process_tree.get('processes', {})
            
            # Add nodes
            for process_id, process_info in processes.items():
                G.add_node(
                    process_id,
                    **process_info
                )
            
            # Add edges (parent -> child relationships)
            for process_id, process_info in processes.items():
                for child_id in process_info.get('children', []):
                    if child_id in processes:
                        G.add_edge(process_id, child_id)
            
            return G
            
        except Exception as e:
            logger.error(f"Failed to create NetworkX tree: {e}")
            return nx.DiGraph()
    
    def _compute_tree_layout(self, G: nx.DiGraph) -> Dict:
        """Compute hierarchical layout for tree"""
        try:
            if len(G.nodes()) == 0:
                return {}
            
            if self.config.layout_algorithm == 'hierarchical':
                # Use graphviz-style hierarchical layout
                try:
                    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
                except:
                    # Fallback to spring layout with hierarchy hints
                    pos = self._hierarchical_spring_layout(G)
            elif self.config.layout_algorithm == 'radial':
                # Radial tree layout
                pos = self._radial_tree_layout(G)
            else:
                # Default spring layout
                pos = nx.spring_layout(G, k=3, iterations=50)
            
            return pos
            
        except Exception as e:
            logger.error(f"Failed to compute tree layout: {e}")
            return nx.spring_layout(G) if len(G.nodes()) > 0 else {}
    
    def _hierarchical_spring_layout(self, G: nx.DiGraph) -> Dict:
        """Create hierarchical layout using spring algorithm with level constraints"""
        try:
            # Compute levels for each node
            levels = {}
            roots = [n for n in G.nodes() if G.in_degree(n) == 0]
            
            # BFS to assign levels
            queue = [(root, 0) for root in roots]
            visited = set()
            
            while queue:
                node, level = queue.pop(0)
                if node in visited:
                    continue
                
                visited.add(node)
                levels[node] = level
                
                for child in G.successors(node):
                    if child not in visited:
                        queue.append((child, level + 1))
            
            # Create initial positions based on levels
            pos = {}
            level_counts = defaultdict(int)
            
            for node, level in levels.items():
                level_counts[level] += 1
            
            level_positions = defaultdict(int)
            
            for node, level in levels.items():
                x = level_positions[level] - (level_counts[level] - 1) / 2
                y = -level  # Negative to have root at top
                pos[node] = (x, y)
                level_positions[level] += 1
            
            return pos
            
        except Exception as e:
            logger.error(f"Failed to create hierarchical spring layout: {e}")
            return nx.spring_layout(G)
    
    def _radial_tree_layout(self, G: nx.DiGraph) -> Dict:
        """Create radial tree layout"""
        try:
            # Find root nodes
            roots = [n for n in G.nodes() if G.in_degree(n) == 0]
            
            if not roots:
                return nx.spring_layout(G)
            
            # Use first root as center
            root = roots[0]
            pos = {root: (0, 0)}
            
            # Assign positions in concentric circles
            visited = {root}
            queue = [(root, 0, 0)]  # node, level, angle_start
            
            while queue:
                node, level, angle_start = queue.pop(0)
                children = list(G.successors(node))
                
                if not children:
                    continue
                
                # Calculate angle step for children
                angle_step = 2 * np.pi / max(len(children), 1)
                radius = (level + 1) * 2
                
                for i, child in enumerate(children):
                    if child not in visited:
                        angle = angle_start + i * angle_step
                        x = radius * np.cos(angle)
                        y = radius * np.sin(angle)
                        pos[child] = (x, y)
                        visited.add(child)
                        queue.append((child, level + 1, angle))
            
            return pos
            
        except Exception as e:
            logger.error(f"Failed to create radial tree layout: {e}")
            return nx.spring_layout(G)
    
    def _create_tree_visualization(
        self, 
        G: nx.DiGraph, 
        pos: Dict, 
        process_tree: Dict,
        title: str
    ) -> go.Figure:
        """Create the tree visualization"""
        try:
            fig = go.Figure()
            
            # Add edges first (so they appear behind nodes)
            self._add_tree_edges(fig, G, pos)
            
            # Add nodes
            self._add_tree_nodes(fig, G, pos)
            
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
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=self.config.width,
                height=self.config.height,
                margin=dict(l=50, r=50, t=80, b=50),
                annotations=[
                    dict(
                        text="Hover over processes for details. Red nodes indicate suspicious processes.",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12, color='gray')
                    )
                ]
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create tree visualization: {e}")
            return self._create_error_tree(title, str(e))
    
    def _add_tree_edges(self, fig: go.Figure, G: nx.DiGraph, pos: Dict):
        """Add edges to tree visualization"""
        try:
            edge_x = []
            edge_y = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines',
                name='Process Relationships',
                showlegend=False
            ))
            
        except Exception as e:
            logger.error(f"Failed to add tree edges: {e}")
    
    def _add_tree_nodes(self, fig: go.Figure, G: nx.DiGraph, pos: Dict):
        """Add nodes to tree visualization"""
        try:
            # Group nodes by risk level
            risk_groups = defaultdict(list)
            
            for node in G.nodes(data=True):
                node_id, node_data = node
                risk_level = node_data.get('risk_level', 'unknown')
                risk_groups[risk_level].append((node_id, node_data))
            
            # Create trace for each risk level
            for risk_level, nodes in risk_groups.items():
                if not nodes:
                    continue
                
                node_x = []
                node_y = []
                node_text = []
                hover_text = []
                node_sizes = []
                
                for node_id, node_data in nodes:
                    if node_id in pos:
                        x, y = pos[node_id]
                        node_x.append(x)
                        node_y.append(y)
                        
                        # Node label
                        process_name = node_data.get('name', node_id)
                        node_text.append(process_name)
                        
                        # Hover text with details
                        hover = f"<b>{process_name}</b><br>"
                        hover += f"PID: {node_data.get('pid', 'N/A')}<br>"
                        hover += f"User: {node_data.get('user', 'N/A')}<br>"
                        hover += f"Risk: {risk_level}<br>"
                        
                        if self.config.show_command_lines and node_data.get('command_line'):
                            cmd = node_data['command_line']
                            if len(cmd) > 100:
                                cmd = cmd[:100] + "..."
                            hover += f"Command: {cmd}<br>"
                        
                        if node_data.get('mitre_technique'):
                            hover += f"MITRE: {node_data['mitre_technique']}<br>"
                        
                        hover_text.append(hover)
                        
                        # Node size based on importance or children count
                        importance = node_data.get('importance_score', 0.0)
                        children_count = len(node_data.get('children', []))
                        size = max(15, min(40, 15 + importance * 20 + children_count * 3))
                        node_sizes.append(size)
                
                if node_x:  # Only create trace if we have nodes
                    fig.add_trace(go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_text,
                        textposition="middle center",
                        textfont=dict(size=8, color='white'),
                        hovertext=hover_text,
                        hoverinfo='text',
                        marker=dict(
                            size=node_sizes,
                            color=self.process_risk_colors.get(risk_level, '#95A5A6'),
                            line=dict(width=2, color='white'),
                            opacity=0.8
                        ),
                        name=f"{risk_level.title()} Risk Processes",
                        showlegend=True
                    ))
            
        except Exception as e:
            logger.error(f"Failed to add tree nodes: {e}")
    
    def _assess_process_risk(self, process_name: str, event: Dict) -> str:
        """Assess risk level of a process"""
        try:
            # Check if explicitly marked as attack step
            if event.get('is_attack_step', False):
                return 'critical'
            
            # Check importance score
            importance = event.get('importance_score', 0.0)
            if importance > 0.8:
                return 'high'
            elif importance > 0.6:
                return 'medium'
            
            # Check if suspicious process
            if self._is_suspicious_process(process_name):
                return 'high'
            
            # Check command line for suspicious patterns
            cmd_line = event.get('command_line', '').lower()
            suspicious_patterns = [
                'powershell', 'cmd /c', 'wmic', 'net user', 'net group',
                'mimikatz', 'procdump', 'psexec', 'rundll32'
            ]
            
            if any(pattern in cmd_line for pattern in suspicious_patterns):
                return 'medium'
            
            return 'low'
            
        except Exception as e:
            logger.error(f"Failed to assess process risk: {e}")
            return 'unknown'
    
    def _is_suspicious_process(self, process_name: str) -> bool:
        """Check if process is known to be suspicious"""
        try:
            return process_name.lower() in self.suspicious_processes
        except Exception as e:
            logger.error(f"Failed to check suspicious process: {e}")
            return False
    
    def _create_empty_tree(self, title: str) -> go.Figure:
        """Create empty tree with message"""
        fig = go.Figure()
        fig.add_annotation(
            text="No process tree data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=title,
            width=self.config.width,
            height=self.config.height
        )
        return fig
    
    def _create_error_tree(self, title: str, error_msg: str) -> go.Figure:
        """Create error tree with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating process tree:<br>{error_msg}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title=title,
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
            
            logger.info(f"Process tree saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save process tree: {e}")


class HierarchicalTreePlotter:
    """
    Simplified interface for creating process tree plots.
    Designed for easy integration with Streamlit and other frameworks.
    """
    
    def __init__(self):
        """Initialize the hierarchical tree plotter"""
        self.visualizer = ProcessTreeVisualizer()
        logger.info("Hierarchical tree plotter initialized")
    
    def plot_process_tree(
        self,
        events: List[Dict],
        title: str = "Process Tree",
        width: int = 1200,
        height: int = 800
    ) -> go.Figure:
        """
        Create process tree plot with minimal configuration.
        
        Parameters
        ----------
        events : List[Dict]
            Process events
        title : str
            Plot title
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
        
        return self.visualizer.build_process_tree(
            events=events,
            title=title
        )


def test_process_tree_visualizer():
    """Test the process tree visualizer with sample data"""
    
    logger.info("Testing Process Tree Visualizer...")
    
    try:
        # Create sample process events
        base_time = datetime.now() - timedelta(hours=1)
        
        events = [
            {
                'timestamp': base_time,
                'process_name': 'explorer.exe',
                'parent_process': 'winlogon.exe',
                'pid': 1000,
                'parent_pid': 500,
                'user': 'SYSTEM',
                'command_line': 'C:\\Windows\\explorer.exe',
                'importance_score': 0.1
            },
            {
                'timestamp': base_time + timedelta(minutes=5),
                'process_name': 'cmd.exe',
                'parent_process': 'explorer.exe',
                'pid': 1500,
                'parent_pid': 1000,
                'user': 'DOMAIN\\alice',
                'command_line': 'cmd.exe /c whoami',
                'importance_score': 0.6
            },
            {
                'timestamp': base_time + timedelta(minutes=10),
                'process_name': 'powershell.exe',
                'parent_process': 'cmd.exe',
                'pid': 2000,
                'parent_pid': 1500,
                'user': 'DOMAIN\\alice',
                'command_line': 'powershell.exe -ExecutionPolicy Bypass',
                'importance_score': 0.9,
                'is_attack_step': True,
                'mitre_technique': 'T1059.001'
            },
            {
                'timestamp': base_time + timedelta(minutes=15),
                'process_name': 'net.exe',
                'parent_process': 'powershell.exe',
                'pid': 2500,
                'parent_pid': 2000,
                'user': 'DOMAIN\\alice',
                'command_line': 'net user administrator /domain',
                'importance_score': 0.8,
                'is_attack_step': True,
                'mitre_technique': 'T1087.002'
            }
        ]
        
        # Initialize visualizer
        visualizer = ProcessTreeVisualizer()
        
        # Test process tree
        fig1 = visualizer.build_process_tree(
            events=events,
            title="Test Process Tree"
        )
        logger.info(" Process tree created")
        
        # Test hierarchical tree plotter
        plotter = HierarchicalTreePlotter()
        fig2 = plotter.plot_process_tree(
            events=events,
            title="Hierarchical Tree Plotter Test"
        )
        logger.info(" Hierarchical tree plotter test completed")
        
        # Save test outputs
        output_dir = Path('outputs/visualizations/process_trees')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig1.write_html(str(output_dir / 'test_process_tree.html'))
        fig2.write_html(str(output_dir / 'test_hierarchical_plotter.html'))
        
        logger.info(" All process tree visualizer tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Process tree visualizer test failed: {e}")
        return False


if __name__ == '__main__':
    """Test the process tree visualizer"""
    
    # Ensure output directory exists
    os.makedirs('outputs/visualizations/process_trees', exist_ok=True)
    
    # Run tests
    success = test_process_tree_visualizer()
    
    if success:
        logger.info(" Day 7 Afternoon Process Tree Visualizer implementation complete!")
        logger.info(" Ready for risk heatmap visualization")
    else:
        logger.error("[ERROR] Process Tree Visualizer implementation needs fixes")
        sys.exit(1)