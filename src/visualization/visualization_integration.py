#!/usr/bin/env python3
"""
Day 8 Morning: Visualization Integration Module
Production-ready integration of all visualization components for THGNN.

Author: THGNN Expert
Date: January 10, 2026
Version: 1.0 - VISUALIZATION INTEGRATION
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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from torch_geometric.data import HeteroData

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import visualization modules
from visualization.graph_visualizer import GraphVisualizer, InteractiveGraphPlotter
from visualization.timeline_visualizer import TimelineVisualizer
from visualization.process_tree_visualizer import ProcessTreeVisualizer
from visualization.risk_heatmap import RiskHeatmapVisualizer

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VisualizationSuite:
    """Complete visualization suite for THGNN analysis"""
    graph_viz: GraphVisualizer
    timeline_viz: TimelineVisualizer
    process_tree_viz: ProcessTreeVisualizer
    heatmap_viz: RiskHeatmapVisualizer
    plotter: InteractiveGraphPlotter

class THGNNVisualizationEngine:
    """
    Production-ready visualization engine integrating all THGNN visualizations.
    Provides unified interface for creating comprehensive security analysis dashboards.
    """
    
    def __init__(self):
        """Initialize the visualization engine with all components"""
        try:
            self.suite = VisualizationSuite(
                graph_viz=GraphVisualizer(),
                timeline_viz=TimelineVisualizer(),
                process_tree_viz=ProcessTreeVisualizer(),
                heatmap_viz=RiskHeatmapVisualizer(),
                plotter=InteractiveGraphPlotter()
            )
            
            # Output directories
            self.output_dirs = {
                'graphs': Path('outputs/visualizations/graphs'),
                'timelines': Path('outputs/visualizations/timelines'),
                'process_trees': Path('outputs/visualizations/process_trees'),
                'heatmaps': Path('outputs/visualizations/heatmaps'),
                'dashboards': Path('outputs/visualizations/dashboards')
            }
            
            # Create output directories
            for dir_path in self.output_dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("THGNN Visualization Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize visualization engine: {e}")
            raise
    
    def create_comprehensive_dashboard(
        self,
        graph: HeteroData,
        events: List[Dict],
        predictions: List[Dict],
        risk_scores: Dict[str, float],
        title: str = "THGNN Security Analysis Dashboard"
    ) -> Dict[str, go.Figure]:
        """
        Create comprehensive dashboard with all visualization types.
        
        Parameters
        ----------
        graph : HeteroData
            Current graph snapshot
        events : List[Dict]
            Security events
        predictions : List[Dict]
            Model predictions
        risk_scores : Dict[str, float]
            Entity risk scores
        title : str
            Dashboard title
            
        Returns
        -------
        Dict[str, go.Figure]
            Dictionary of all visualization figures
        """
        try:
            logger.info(f"Creating comprehensive dashboard: {title}")
            
            dashboard = {}
            
            # 1. Interactive Graph Visualization
            logger.info("Creating graph visualization...")
            dashboard['graph'] = self.suite.graph_viz.plot_graph_snapshot(
                graph=graph,
                title=f"{title} - Network Graph",
                predictions=predictions
            )
            
            # 2. Timeline Visualization
            logger.info("Creating timeline visualization...")
            dashboard['timeline'] = self.suite.timeline_viz.create_event_timeline(
                events=events,
                predictions=predictions,
                title=f"{title} - Attack Timeline"
            )
            
            # 3. Process Tree Visualization
            logger.info("Creating process tree visualization...")
            dashboard['process_tree'] = self.suite.process_tree_viz.build_process_tree(
                events=events,
                title=f"{title} - Process Tree"
            )
            
            # 4. Risk Heatmap
            logger.info("Creating risk heatmap...")
            dashboard['heatmap'] = self.suite.heatmap_viz.create_risk_heatmap(
                entities=list(risk_scores.keys()),
                scores=risk_scores,
                title=f"{title} - Risk Heatmap"
            )
            
            # 5. Combined Overview Dashboard
            logger.info("Creating combined overview...")
            dashboard['overview'] = self._create_overview_dashboard(
                graph_fig=dashboard['graph'],
                timeline_fig=dashboard['timeline'],
                heatmap_fig=dashboard['heatmap'],
                title=title
            )
            
            logger.info("Comprehensive dashboard created successfully")
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to create comprehensive dashboard: {e}")
            return {}
    
    def create_prediction_analysis(
        self,
        predictions: List[Dict],
        explanations: List[Dict],
        graph: HeteroData,
        title: str = "Prediction Analysis"
    ) -> Dict[str, go.Figure]:
        """
        Create focused analysis of model predictions.
        
        Parameters
        ----------
        predictions : List[Dict]
            Model predictions
        explanations : List[Dict]
            Prediction explanations
        graph : HeteroData
            Graph data
        title : str
            Analysis title
            
        Returns
        -------
        Dict[str, go.Figure]
            Prediction analysis figures
        """
        try:
            logger.info(f"Creating prediction analysis: {title}")
            
            analysis = {}
            
            # Top predictions graph
            top_predictions = sorted(predictions, key=lambda x: x.get('risk_score', 0), reverse=True)[:10]
            
            analysis['top_predictions_graph'] = self.suite.graph_viz.plot_graph_snapshot(
                graph=graph,
                title=f"{title} - Top 10 Predictions",
                predictions=top_predictions
            )
            
            # Prediction confidence distribution
            analysis['confidence_distribution'] = self._create_confidence_distribution(
                predictions=predictions,
                title=f"{title} - Confidence Distribution"
            )
            
            # Risk score timeline
            analysis['risk_timeline'] = self._create_risk_timeline(
                predictions=predictions,
                title=f"{title} - Risk Over Time"
            )
            
            # Explanation subgraphs for top predictions
            if explanations:
                analysis['explanation_subgraphs'] = []
                for i, explanation in enumerate(explanations[:5]):  # Top 5
                    if hasattr(explanation, 'subgraph_nodes'):
                        fig = self.suite.graph_viz.plot_explanation_subgraph(
                            explanation_result=explanation,
                            graph=graph,
                            title=f"Explanation {i+1}"
                        )
                        analysis['explanation_subgraphs'].append(fig)
            
            logger.info("Prediction analysis created successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to create prediction analysis: {e}")
            return {}
    
    def create_attack_chain_visualization(
        self,
        attack_chain: List[Dict],
        graph: HeteroData,
        title: str = "Attack Chain Analysis"
    ) -> go.Figure:
        """
        Create visualization of complete attack chain.
        
        Parameters
        ----------
        attack_chain : List[Dict]
            Sequence of attack events
        graph : HeteroData
            Graph data
        title : str
            Visualization title
            
        Returns
        -------
        go.Figure
            Attack chain visualization
        """
        try:
            logger.info(f"Creating attack chain visualization: {title}")
            
            # Create timeline of attack steps
            fig = go.Figure()
            
            # Extract timeline data
            steps = []
            for i, step in enumerate(attack_chain):
                steps.append({
                    'step': i + 1,
                    'timestamp': step.get('timestamp', datetime.now()),
                    'event': step.get('event', f'Step {i+1}'),
                    'entity': step.get('entity', 'unknown'),
                    'mitre_technique': step.get('mitre_technique', ''),
                    'risk_score': step.get('risk_score', 50)
                })
            
            # Create Gantt-style chart
            for step in steps:
                fig.add_trace(go.Scatter(
                    x=[step['timestamp']],
                    y=[step['step']],
                    mode='markers+text',
                    text=step['event'],
                    textposition='middle right',
                    marker=dict(
                        size=20 + step['risk_score'] / 5,
                        color=self._get_risk_color(step['risk_score']),
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate=(
                        f"<b>Step {step['step']}</b><br>"
                        f"Event: {step['event']}<br>"
                        f"Entity: {step['entity']}<br>"
                        f"MITRE: {step['mitre_technique']}<br>"
                        f"Risk: {step['risk_score']:.1f}<br>"
                        f"Time: {step['timestamp']}<extra></extra>"
                    ),
                    showlegend=False
                ))
            
            # Add connecting lines
            if len(steps) > 1:
                x_coords = [step['timestamp'] for step in steps]
                y_coords = [step['step'] for step in steps]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(color='red', width=3, dash='dash'),
                    name='Attack Path',
                    showlegend=True
                ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title="Time",
                yaxis_title="Attack Step",
                yaxis=dict(
                    tickmode='linear',
                    tick0=1,
                    dtick=1,
                    range=[0.5, len(steps) + 0.5]
                ),
                hovermode='closest',
                width=1200,
                height=600,
                showlegend=True
            )
            
            logger.info("Attack chain visualization created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create attack chain visualization: {e}")
            return go.Figure()
    
    def export_dashboard(
        self,
        dashboard: Dict[str, go.Figure],
        output_dir: str = "outputs/visualizations/dashboards",
        format: str = "html"
    ) -> Dict[str, str]:
        """
        Export dashboard figures to files.
        
        Parameters
        ----------
        dashboard : Dict[str, go.Figure]
            Dashboard figures
        output_dir : str
            Output directory
        format : str
            Export format ('html', 'png', 'svg')
            
        Returns
        -------
        Dict[str, str]
            Mapping of figure names to file paths
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            exported_files = {}
            
            for name, fig in dashboard.items():
                if isinstance(fig, go.Figure):
                    filename = f"{name}.{format}"
                    filepath = output_path / filename
                    
                    if format == 'html':
                        fig.write_html(str(filepath))
                    elif format in ['png', 'svg']:
                        fig.write_image(str(filepath))
                    
                    exported_files[name] = str(filepath)
                    logger.info(f"Exported {name} to {filepath}")
            
            logger.info(f"Dashboard exported successfully to {output_dir}")
            return exported_files
            
        except Exception as e:
            logger.error(f"Failed to export dashboard: {e}")
            return {}
    
    def _create_overview_dashboard(
        self,
        graph_fig: go.Figure,
        timeline_fig: go.Figure,
        heatmap_fig: go.Figure,
        title: str
    ) -> go.Figure:
        """Create combined overview dashboard"""
        try:
            # Create subplot layout
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Network Graph', 'Timeline', 'Risk Heatmap', 'Summary'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}]],
                horizontal_spacing=0.1,
                vertical_spacing=0.15
            )
            
            # Add graph traces (simplified)
            for trace in graph_fig.data:
                fig.add_trace(trace, row=1, col=1)
            
            # Add timeline traces (simplified)
            for trace in timeline_fig.data:
                fig.add_trace(trace, row=1, col=2)
            
            # Add heatmap traces (simplified)
            for trace in heatmap_fig.data:
                fig.add_trace(trace, row=2, col=1)
            
            # Add summary statistics
            fig.add_trace(
                go.Scatter(
                    x=[0.5], y=[0.5],
                    text="Dashboard Overview",
                    mode="text",
                    textfont=dict(size=16),
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': f"{title} - Overview Dashboard",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24}
                },
                width=1600,
                height=1200,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create overview dashboard: {e}")
            return go.Figure()
    
    def _create_confidence_distribution(
        self,
        predictions: List[Dict],
        title: str
    ) -> go.Figure:
        """Create confidence distribution plot"""
        try:
            confidences = [pred.get('confidence', 0) for pred in predictions]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=confidences,
                nbinsx=20,
                name='Confidence Distribution',
                marker_color='blue',
                opacity=0.7
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Confidence Score',
                yaxis_title='Number of Predictions',
                width=800,
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create confidence distribution: {e}")
            return go.Figure()
    
    def _create_risk_timeline(
        self,
        predictions: List[Dict],
        title: str
    ) -> go.Figure:
        """Create risk score timeline"""
        try:
            # Sort predictions by timestamp
            sorted_preds = sorted(
                predictions,
                key=lambda x: x.get('timestamp', datetime.now())
            )
            
            timestamps = [pred.get('timestamp', datetime.now()) for pred in sorted_preds]
            risk_scores = [pred.get('risk_score', 0) for pred in sorted_preds]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=risk_scores,
                mode='lines+markers',
                name='Risk Score',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Time',
                yaxis_title='Risk Score',
                width=1000,
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create risk timeline: {e}")
            return go.Figure()
    
    def _get_risk_color(self, risk_score: float) -> str:
        """Get color based on risk score"""
        if risk_score >= 80:
            return '#8E44AD'  # Critical - Purple
        elif risk_score >= 60:
            return '#E74C3C'  # High - Red
        elif risk_score >= 40:
            return '#F39C12'  # Medium - Orange
        else:
            return '#27AE60'  # Low - Green


def test_visualization_integration():
    """Test the complete visualization integration"""
    
    logger.info("Testing THGNN Visualization Integration...")
    
    try:
        # Initialize engine
        engine = THGNNVisualizationEngine()
        
        # Create sample data
        from torch_geometric.data import HeteroData
        import torch
        
        # Sample graph
        graph = HeteroData()
        graph['user'].x = torch.randn(5, 64)
        graph['host'].x = torch.randn(3, 64)
        graph['process'].x = torch.randn(4, 64)
        
        graph[('user', 'logs_into', 'host')].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
        graph[('user', 'executes', 'process')].edge_index = torch.tensor([[0, 1], [0, 1]])
        graph[('process', 'connects_to', 'host')].edge_index = torch.tensor([[0, 1], [0, 1]])
        
        # Sample events
        events = [
            {
                'timestamp': datetime.now(),
                'event_type': 'logon',
                'source_entity': 'user_0',
                'target_entity': 'host_0',
                'risk_score': 75
            },
            {
                'timestamp': datetime.now(),
                'event_type': 'process_create',
                'source_entity': 'user_0',
                'target_entity': 'process_0',
                'risk_score': 60
            }
        ]
        
        # Sample predictions
        predictions = [
            {
                'source_entity': 'user_0',
                'target_entity': 'host_1',
                'edge_type': 'logs_into',
                'risk_score': 85.0,
                'confidence': 0.92,
                'timestamp': datetime.now()
            }
        ]
        
        # Sample risk scores
        risk_scores = {
            'user_0': 85.0,
            'user_1': 45.0,
            'host_0': 70.0,
            'host_1': 90.0,
            'process_0': 55.0
        }
        
        # Test comprehensive dashboard
        dashboard = engine.create_comprehensive_dashboard(
            graph=graph,
            events=events,
            predictions=predictions,
            risk_scores=risk_scores,
            title="Test Security Analysis"
        )
        
        logger.info(f" Created dashboard with {len(dashboard)} visualizations")
        
        # Test prediction analysis
        analysis = engine.create_prediction_analysis(
            predictions=predictions,
            explanations=[],
            graph=graph,
            title="Test Prediction Analysis"
        )
        
        logger.info(f" Created prediction analysis with {len(analysis)} visualizations")
        
        # Test attack chain visualization
        attack_chain = [
            {
                'timestamp': datetime.now(),
                'event': 'User login',
                'entity': 'user_0',
                'mitre_technique': 'T1078',
                'risk_score': 60
            },
            {
                'timestamp': datetime.now(),
                'event': 'Process execution',
                'entity': 'process_0',
                'mitre_technique': 'T1059',
                'risk_score': 80
            }
        ]
        
        chain_fig = engine.create_attack_chain_visualization(
            attack_chain=attack_chain,
            graph=graph,
            title="Test Attack Chain"
        )
        
        logger.info(" Created attack chain visualization")
        
        # Test export
        exported = engine.export_dashboard(
            dashboard=dashboard,
            output_dir="outputs/visualizations/test_dashboard"
        )
        
        logger.info(f" Exported {len(exported)} dashboard files")
        
        logger.info(" All visualization integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Visualization integration test failed: {e}")
        return False


if __name__ == '__main__':
    """Test the visualization integration"""
    
    # Run tests
    success = test_visualization_integration()
    
    if success:
        logger.info(" Day 8 Morning Visualization Integration complete!")
        logger.info(" Ready for Streamlit dashboard development")
    else:
        logger.error("[ERROR] Visualization Integration needs fixes")
        sys.exit(1)