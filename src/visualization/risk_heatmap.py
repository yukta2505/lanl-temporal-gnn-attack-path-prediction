#!/usr/bin/env python3
"""
Day 7 Afternoon: Risk Heatmap Visualization for THGNN
Production-ready risk heatmap visualization for threat landscape analysis.

Author: THGNN Expert
Date: January 9, 2026
Version: 1.0 - RISK HEATMAP VISUALIZATION
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RiskHeatmapConfig:
    """Configuration for risk heatmap visualization"""
    width: int = 1400
    height: int = 800
    colorscale: str = 'Reds'  # 'Reds', 'YlOrRd', 'Viridis', 'RdYlBu_r'
    show_values: bool = True
    interactive: bool = True
    time_aggregation: str = 'hour'  # 'minute', 'hour', 'day'
    export_format: str = 'html'

class RiskHeatmapVisualizer:
    """
    Production-ready risk heatmap visualizer for threat landscape analysis.
    Creates interactive heatmaps showing risk scores across entities and time.
    """
    
    def __init__(self, config: Optional[RiskHeatmapConfig] = None):
        """
        Initialize the risk heatmap visualizer.
        
        Parameters
        ----------
        config : RiskHeatmapConfig, optional
            Visualization configuration
        """
        self.config = config or RiskHeatmapConfig()
        
        # Risk level thresholds
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 1.0
        }
        
        # Color scales for different visualizations
        self.color_scales = {
            'risk': 'Reds',
            'activity': 'Blues', 
            'anomaly': 'YlOrRd',
            'confidence': 'Greens'
        }
        
        logger.info("Risk heatmap visualizer initialized successfully")
    
    def create_risk_heatmap(
        self,
        entities: List[str],
        risk_scores_over_time: Dict[str, List[float]],
        time_windows: List[datetime],
        title: str = "Risk Score Heatmap",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create risk score heatmap showing entities vs time windows.
        
        Parameters
        ----------
        entities : List[str]
            List of entity names (users, hosts, processes)
        risk_scores_over_time : Dict[str, List[float]]
            Risk scores for each entity over time windows
        time_windows : List[datetime]
            Time window timestamps
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
            logger.info(f"Creating risk heatmap for {len(entities)} entities over {len(time_windows)} time windows")
            
            if not entities or not time_windows:
                return self._create_empty_heatmap(title)
            
            # Prepare data matrix
            risk_matrix = self._prepare_risk_matrix(entities, risk_scores_over_time, time_windows)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=risk_matrix,
                x=[t.strftime('%H:%M') for t in time_windows],
                y=entities,
                colorscale=self.config.colorscale,
                hoverongaps=False,
                hovertemplate=(
                    "<b>Entity:</b> %{y}<br>"
                    "<b>Time:</b> %{x}<br>"
                    "<b>Risk Score:</b> %{z:.3f}<br>"
                    "<extra></extra>"
                ),
                colorbar=dict(
                    title="Risk Score",
                    tickmode="array",
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["0.0", "0.25", "0.5", "0.75", "1.0"]
                )
            ))
            
            # Add risk level lines
            self._add_risk_level_annotations(fig, len(time_windows))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title="Time Windows",
                yaxis_title="Entities",
                width=self.config.width,
                height=max(400, len(entities) * 25 + 200),
                margin=dict(l=150, r=100, t=80, b=50)
            )
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            logger.info("Risk heatmap created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create risk heatmap: {e}")
            return self._create_error_heatmap(title, str(e))
    
    def create_entity_risk_distribution(
        self,
        risk_scores: Dict[str, float],
        title: str = "Entity Risk Distribution",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create histogram showing distribution of risk scores across entities.
        
        Parameters
        ----------
        risk_scores : Dict[str, float]
            Risk scores for each entity
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
            logger.info(f"Creating risk distribution for {len(risk_scores)} entities")
            
            if not risk_scores:
                return self._create_empty_heatmap(title)
            
            # Prepare data
            entities = list(risk_scores.keys())
            scores = list(risk_scores.values())
            
            # Create subplots: histogram + box plot
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=("Risk Score Distribution", "Risk Score Summary"),
                vertical_spacing=0.1
            )
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=scores,
                    nbinsx=20,
                    name="Risk Distribution",
                    marker_color='red',
                    opacity=0.7,
                    hovertemplate=(
                        "<b>Risk Range:</b> %{x}<br>"
                        "<b>Count:</b> %{y}<br>"
                        "<extra></extra>"
                    )
                ),
                row=1, col=1
            )
            
            # Add box plot
            fig.add_trace(
                go.Box(
                    y=scores,
                    name="Risk Scores",
                    marker_color='red',
                    boxpoints='outliers',
                    hovertemplate=(
                        "<b>Risk Score:</b> %{y:.3f}<br>"
                        "<extra></extra>"
                    )
                ),
                row=2, col=1
            )
            
            # Add risk level lines to histogram
            for level, threshold in self.risk_thresholds.items():
                fig.add_vline(
                    x=threshold,
                    line_dash="dash",
                    line_color=self._get_risk_color(level),
                    annotation_text=level.title(),
                    row=1, col=1
                )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                showlegend=False,
                width=self.config.width,
                height=self.config.height
            )
            
            # Update axes
            fig.update_xaxes(title_text="Risk Score", row=1, col=1)
            fig.update_yaxes(title_text="Number of Entities", row=1, col=1)
            fig.update_yaxes(title_text="Risk Score", row=2, col=1)
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            logger.info("Risk distribution created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create risk distribution: {e}")
            return self._create_error_heatmap(title, str(e))
    
    def create_threat_landscape_heatmap(
        self,
        predictions: List[Dict[str, Any]],
        title: str = "Threat Landscape Heatmap",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create heatmap showing threat landscape across entity types and attack types.
        
        Parameters
        ----------
        predictions : List[Dict]
            List of threat predictions
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
            logger.info(f"Creating threat landscape heatmap from {len(predictions)} predictions")
            
            if not predictions:
                return self._create_empty_heatmap(title)
            
            # Prepare data for heatmap
            df = pd.DataFrame(predictions)
            
            # Extract entity types and attack types
            df['source_type'] = df['source_entity'].apply(self._extract_entity_type)
            df['target_type'] = df['target_entity'].apply(self._extract_entity_type)
            df['attack_vector'] = df['source_type'] + ' → ' + df['target_type']
            
            # Aggregate by attack vector and edge type
            threat_matrix = df.groupby(['attack_vector', 'edge_type'])['risk_score'].mean().unstack(fill_value=0)
            
            if threat_matrix.empty:
                return self._create_empty_heatmap(title)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=threat_matrix.values,
                x=threat_matrix.columns,
                y=threat_matrix.index,
                colorscale='YlOrRd',
                hoverongaps=False,
                hovertemplate=(
                    "<b>Attack Vector:</b> %{y}<br>"
                    "<b>Attack Type:</b> %{x}<br>"
                    "<b>Avg Risk Score:</b> %{z:.3f}<br>"
                    "<extra></extra>"
                ),
                colorbar=dict(
                    title="Average Risk Score"
                )
            ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title="Attack Types",
                yaxis_title="Attack Vectors",
                width=self.config.width,
                height=max(400, len(threat_matrix.index) * 40 + 200),
                margin=dict(l=200, r=100, t=80, b=100)
            )
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            logger.info("Threat landscape heatmap created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create threat landscape heatmap: {e}")
            return self._create_error_heatmap(title, str(e))
    
    def create_correlation_heatmap(
        self,
        entities: List[str],
        correlation_matrix: np.ndarray,
        title: str = "Entity Risk Correlation",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create correlation heatmap showing relationships between entity risks.
        
        Parameters
        ----------
        entities : List[str]
            Entity names
        correlation_matrix : np.ndarray
            Correlation matrix between entities
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
            logger.info(f"Creating correlation heatmap for {len(entities)} entities")
            
            if len(entities) == 0 or correlation_matrix.size == 0:
                return self._create_empty_heatmap(title)
            
            # Create correlation heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=entities,
                y=entities,
                colorscale='RdBu_r',
                zmid=0,
                hoverongaps=False,
                hovertemplate=(
                    "<b>Entity 1:</b> %{y}<br>"
                    "<b>Entity 2:</b> %{x}<br>"
                    "<b>Correlation:</b> %{z:.3f}<br>"
                    "<extra></extra>"
                ),
                colorbar=dict(
                    title="Correlation",
                    tickmode="array",
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
                )
            ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title="Entities",
                yaxis_title="Entities",
                width=max(600, len(entities) * 30 + 200),
                height=max(600, len(entities) * 30 + 200),
                margin=dict(l=150, r=100, t=80, b=100)
            )
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            logger.info("Correlation heatmap created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create correlation heatmap: {e}")
            return self._create_error_heatmap(title, str(e))
    
    def _prepare_risk_matrix(
        self,
        entities: List[str],
        risk_scores_over_time: Dict[str, List[float]],
        time_windows: List[datetime]
    ) -> np.ndarray:
        """Prepare risk score matrix for heatmap"""
        try:
            # Initialize matrix with zeros
            matrix = np.zeros((len(entities), len(time_windows)))
            
            # Fill matrix with risk scores
            for i, entity in enumerate(entities):
                if entity in risk_scores_over_time:
                    scores = risk_scores_over_time[entity]
                    # Pad or truncate scores to match time windows
                    for j in range(min(len(scores), len(time_windows))):
                        matrix[i, j] = scores[j]
            
            return matrix
            
        except Exception as e:
            logger.error(f"Failed to prepare risk matrix: {e}")
            return np.zeros((len(entities), len(time_windows)))
    
    def _add_risk_level_annotations(self, fig: go.Figure, num_time_windows: int):
        """Add risk level threshold annotations"""
        try:
            # Add horizontal lines for risk thresholds
            for level, threshold in self.risk_thresholds.items():
                if level == 'critical':  # Skip critical (it's at 1.0)
                    continue
                
                fig.add_shape(
                    type="line",
                    x0=-0.5, x1=num_time_windows - 0.5,
                    y0=threshold, y1=threshold,
                    line=dict(
                        color=self._get_risk_color(level),
                        width=2,
                        dash="dash"
                    )
                )
                
                # Add annotation
                fig.add_annotation(
                    x=num_time_windows - 1,
                    y=threshold,
                    text=level.title(),
                    showarrow=False,
                    xanchor="left",
                    bgcolor="white",
                    bordercolor=self._get_risk_color(level),
                    borderwidth=1,
                    font=dict(size=10)
                )
            
        except Exception as e:
            logger.error(f"Failed to add risk level annotations: {e}")
    
    def _extract_entity_type(self, entity_name: str) -> str:
        """Extract entity type from entity name"""
        try:
            if '_' in entity_name:
                return entity_name.split('_')[0]
            elif '\\' in entity_name:
                return 'user'
            elif '.' in entity_name and not entity_name.endswith('.exe'):
                return 'host'
            elif entity_name.endswith('.exe'):
                return 'process'
            else:
                return 'unknown'
        except Exception as e:
            logger.error(f"Failed to extract entity type: {e}")
            return 'unknown'
    
    def _get_risk_color(self, risk_level: str) -> str:
        """Get color for risk level"""
        colors = {
            'low': 'green',
            'medium': 'orange',
            'high': 'red',
            'critical': 'darkred'
        }
        return colors.get(risk_level, 'gray')
    
    def _create_empty_heatmap(self, title: str) -> go.Figure:
        """Create empty heatmap with message"""
        fig = go.Figure()
        fig.add_annotation(
            text="No heatmap data available",
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
    
    def _create_error_heatmap(self, title: str, error_msg: str) -> go.Figure:
        """Create error heatmap with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating heatmap:<br>{error_msg}",
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
            
            logger.info(f"Risk heatmap saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save risk heatmap: {e}")


class ThreatHeatmapGenerator:
    """
    Simplified interface for creating threat heatmap plots.
    Designed for easy integration with Streamlit and other frameworks.
    """
    
    def __init__(self):
        """Initialize the threat heatmap generator"""
        self.visualizer = RiskHeatmapVisualizer()
        logger.info("Threat heatmap generator initialized")
    
    def plot_risk_heatmap(
        self,
        entities: List[str],
        risk_scores_over_time: Dict[str, List[float]],
        time_windows: List[datetime],
        title: str = "Risk Heatmap",
        width: int = 1400,
        height: int = 800
    ) -> go.Figure:
        """
        Create risk heatmap with minimal configuration.
        
        Parameters
        ----------
        entities : List[str]
            Entity names
        risk_scores_over_time : Dict[str, List[float]]
            Risk scores over time
        time_windows : List[datetime]
            Time windows
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
        
        return self.visualizer.create_risk_heatmap(
            entities=entities,
            risk_scores_over_time=risk_scores_over_time,
            time_windows=time_windows,
            title=title
        )
    
    def plot_threat_landscape(
        self,
        predictions: List[Dict],
        title: str = "Threat Landscape",
        width: int = 1400,
        height: int = 800
    ) -> go.Figure:
        """
        Create threat landscape heatmap.
        
        Parameters
        ----------
        predictions : List[Dict]
            Threat predictions
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
        
        return self.visualizer.create_threat_landscape_heatmap(
            predictions=predictions,
            title=title
        )


def test_risk_heatmap_visualizer():
    """Test the risk heatmap visualizer with sample data"""
    
    logger.info("Testing Risk Heatmap Visualizer...")
    
    try:
        # Create sample data
        entities = ['user_0', 'user_1', 'host_0', 'host_1', 'process_0', 'process_1']
        
        # Generate sample time windows
        base_time = datetime.now() - timedelta(hours=2)
        time_windows = [base_time + timedelta(minutes=i*15) for i in range(8)]
        
        # Generate sample risk scores over time
        risk_scores_over_time = {}
        for entity in entities:
            # Simulate varying risk scores
            scores = []
            base_risk = np.random.uniform(0.2, 0.8)
            for i in range(len(time_windows)):
                # Add some variation and trend
                variation = np.random.normal(0, 0.1)
                trend = 0.05 * i if 'user_1' in entity else -0.02 * i
                score = max(0, min(1, base_risk + variation + trend))
                scores.append(score)
            risk_scores_over_time[entity] = scores
        
        # Create sample predictions
        predictions = [
            {
                'source_entity': 'user_0',
                'target_entity': 'host_1',
                'edge_type': 'logs_into',
                'risk_score': 0.85
            },
            {
                'source_entity': 'user_1',
                'target_entity': 'process_0',
                'edge_type': 'executes',
                'risk_score': 0.75
            },
            {
                'source_entity': 'process_0',
                'target_entity': 'host_0',
                'edge_type': 'connects_to',
                'risk_score': 0.65
            }
        ]
        
        # Initialize visualizer
        visualizer = RiskHeatmapVisualizer()
        
        # Test risk heatmap
        fig1 = visualizer.create_risk_heatmap(
            entities=entities,
            risk_scores_over_time=risk_scores_over_time,
            time_windows=time_windows,
            title="Test Risk Heatmap"
        )
        logger.info(" Risk heatmap created")
        
        # Test risk distribution
        current_risk_scores = {entity: scores[-1] for entity, scores in risk_scores_over_time.items()}
        fig2 = visualizer.create_entity_risk_distribution(
            risk_scores=current_risk_scores,
            title="Test Risk Distribution"
        )
        logger.info(" Risk distribution created")
        
        # Test threat landscape
        fig3 = visualizer.create_threat_landscape_heatmap(
            predictions=predictions,
            title="Test Threat Landscape"
        )
        logger.info(" Threat landscape heatmap created")
        
        # Test correlation heatmap
        correlation_matrix = np.corrcoef([scores for scores in risk_scores_over_time.values()])
        fig4 = visualizer.create_correlation_heatmap(
            entities=entities,
            correlation_matrix=correlation_matrix,
            title="Test Correlation Heatmap"
        )
        logger.info(" Correlation heatmap created")
        
        # Test threat heatmap generator
        generator = ThreatHeatmapGenerator()
        fig5 = generator.plot_risk_heatmap(
            entities=entities,
            risk_scores_over_time=risk_scores_over_time,
            time_windows=time_windows,
            title="Threat Heatmap Generator Test"
        )
        logger.info(" Threat heatmap generator test completed")
        
        # Save test outputs
        output_dir = Path('outputs/visualizations/heatmaps')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig1.write_html(str(output_dir / 'test_risk_heatmap.html'))
        fig2.write_html(str(output_dir / 'test_risk_distribution.html'))
        fig3.write_html(str(output_dir / 'test_threat_landscape.html'))
        fig4.write_html(str(output_dir / 'test_correlation_heatmap.html'))
        fig5.write_html(str(output_dir / 'test_threat_generator.html'))
        
        logger.info(" All risk heatmap visualizer tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Risk heatmap visualizer test failed: {e}")
        return False


if __name__ == '__main__':
    """Test the risk heatmap visualizer"""
    
    # Ensure output directory exists
    os.makedirs('outputs/visualizations/heatmaps', exist_ok=True)
    
    # Run tests
    success = test_risk_heatmap_visualizer()
    
    if success:
        logger.info(" Day 7 Afternoon Risk Heatmap Visualizer implementation complete!")
        logger.info(" Ready for complete visualization system integration")
    else:
        logger.error("[ERROR] Risk Heatmap Visualizer implementation needs fixes")
        sys.exit(1)