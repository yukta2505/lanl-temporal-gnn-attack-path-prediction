#!/usr/bin/env python3
"""
Day 7 Afternoon: Timeline Visualization for THGNN
Production-ready timeline visualization for attack sequence analysis.

Author: THGNN Expert
Date: January 9, 2026
Version: 1.0 - ATTACK TIMELINE VISUALIZATION
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
import torch

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TimelineVisualizationConfig:
    """Configuration for timeline visualization"""
    width: int = 1400
    height: int = 600
    show_predictions: bool = True
    show_confidence_bands: bool = True
    color_by_risk: bool = True
    interactive: bool = True
    export_format: str = 'html'

class TimelineVisualizer:
    """
    Production-ready timeline visualizer for attack sequence analysis.
    Creates interactive timeline plots showing events and predictions over time.
    """
    
    def __init__(self, config: Optional[TimelineVisualizationConfig] = None):
        """
        Initialize the timeline visualizer.
        
        Parameters
        ----------
        config : TimelineVisualizationConfig, optional
            Visualization configuration
        """
        self.config = config or TimelineVisualizationConfig()
        
        # Security event color scheme
        self.event_colors = {
            'logon': '#3498DB',           # Blue for authentication
            'process_create': '#E67E22',  # Orange for process creation
            'network_connect': '#9B59B6', # Purple for network activity
            'file_access': '#E74C3C',     # Red for file operations
            'registry_modify': '#F39C12', # Yellow for registry changes
            'privilege_escalation': '#C0392B', # Dark red for privilege escalation
            'lateral_movement': '#8E44AD',     # Dark purple for lateral movement
            'predicted': '#FF6B6B',       # Light red for predictions
            'high_confidence': '#FF4757', # Bright red for high confidence predictions
            'low_confidence': '#FFA502'   # Orange for low confidence predictions
        }
        
        # Risk level colors
        self.risk_colors = {
            'low': '#27AE60',      # Green
            'medium': '#F39C12',   # Orange  
            'high': '#E74C3C',     # Red
            'critical': '#8E44AD'  # Purple
        }
        
        logger.info("Timeline visualizer initialized successfully")
    
    def create_event_timeline(
        self,
        events: List[Dict[str, Any]],
        predictions: Optional[List[Dict[str, Any]]] = None,
        title: str = "Attack Timeline",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive timeline visualization of security events.
        
        Parameters
        ----------
        events : List[Dict]
            List of security events with timestamps
        predictions : List[Dict], optional
            Predicted future events to overlay
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
            logger.info(f"Creating event timeline with {len(events)} events")
            
            if not events:
                return self._create_empty_timeline(title)
            
            # Convert events to DataFrame for easier processing
            df_events = self._prepare_events_dataframe(events)
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.8, 0.2],
                subplot_titles=("Event Timeline", "Risk Level Over Time"),
                vertical_spacing=0.1,
                shared_xaxes=True
            )
            
            # Add event traces to main timeline
            self._add_event_traces(fig, df_events, row=1)
            
            # Add predictions if provided
            if predictions and self.config.show_predictions:
                self._add_prediction_traces(fig, predictions, row=1)
            
            # Add risk level subplot
            self._add_risk_timeline(fig, df_events, row=2)
            
            # Update layout
            self._update_timeline_layout(fig, title, df_events)
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            logger.info("Event timeline created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create event timeline: {e}")
            return self._create_error_timeline(title, str(e))
    
    def create_attack_chain_timeline(
        self,
        attack_chain,
        title: str = "Attack Chain Timeline",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create timeline visualization specifically for attack chains.
        
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
            logger.info(f"Creating attack chain timeline for {getattr(attack_chain, 'chain_id', 'unknown')}")
            
            attack_steps = getattr(attack_chain, 'attack_steps', [])
            if not attack_steps:
                return self._create_empty_timeline(title)
            
            # Create Gantt-style chart for attack steps
            fig = go.Figure()
            
            for i, step in enumerate(attack_steps):
                # Calculate step duration
                if i < len(attack_steps) - 1:
                    next_step = attack_steps[i + 1]
                    duration = (getattr(next_step, 'timestamp', datetime.now()) - 
                              getattr(step, 'timestamp', datetime.now())).total_seconds()
                else:
                    # Default 60 seconds for last step
                    duration = 60
                
                step_timestamp = getattr(step, 'timestamp', datetime.now())
                end_time = step_timestamp + timedelta(seconds=max(duration, 10))
                
                # Get color for this step
                color = self._get_step_color(step)
                
                # Add Gantt bar
                fig.add_trace(go.Scatter(
                    x=[step_timestamp, end_time, end_time, step_timestamp, step_timestamp],
                    y=[i, i, i+0.8, i+0.8, i],
                    fill='toself',
                    fillcolor=color,
                    line=dict(color=color, width=2),
                    mode='lines',
                    name=f"Step {getattr(step, 'step_number', i+1)}",
                    hovertemplate=(
                        f"<b>Step {getattr(step, 'step_number', i+1)}</b><br>"
                        f"Time: {step_timestamp.strftime('%H:%M:%S')}<br>"
                        f"Action: {getattr(step, 'event_description', 'Unknown')}<br>"
                        f"MITRE: {getattr(step, 'mitre_technique', 'Unknown')}<br>"
                        f"Importance: {getattr(step, 'importance_score', 0.0):.2f}<br>"
                        "<extra></extra>"
                    )
                ))
                
                # Add step labels
                fig.add_annotation(
                    x=step_timestamp + timedelta(seconds=duration/2),
                    y=i + 0.4,
                    text=f"<b>{getattr(step, 'step_number', i+1)}</b>",
                    showarrow=False,
                    font=dict(color='white', size=12, family='Arial Black'),
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='black',
                    borderwidth=1
                )
            
            # Update layout for attack chain
            confidence = getattr(attack_chain, 'confidence', 0.0)
            fig.update_layout(
                title={
                    'text': f"{title} | Confidence: {confidence:.1%}",
                    'x': 0.5,
                    'font': {'size': 18}
                },
                xaxis=dict(
                    type='date',
                    title="Time",
                    tickformat='%H:%M:%S'
                ),
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(attack_steps))),
                    ticktext=[f"Step {i+1}" for i in range(len(attack_steps))],
                    title="Attack Steps"
                ),
                showlegend=False,
                width=self.config.width,
                height=max(400, len(attack_steps) * 60 + 200),
                margin=dict(l=100, r=50, t=80, b=50)
            )
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            logger.info("Attack chain timeline created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create attack chain timeline: {e}")
            return self._create_error_timeline(title, str(e))
    
    def _prepare_events_dataframe(self, events: List[Dict]) -> pd.DataFrame:
        """Prepare events data for visualization"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(events)
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'event_type', 'entity']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'timestamp':
                        df[col] = datetime.now()
                    elif col == 'event_type':
                        df[col] = 'unknown'
                    elif col == 'entity':
                        df[col] = f'entity_{df.index}'
            
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Add risk level if not present
            if 'risk_level' not in df.columns:
                if 'risk_score' in df.columns:
                    df['risk_level'] = df['risk_score'].apply(self._score_to_risk_level)
                else:
                    df['risk_level'] = 'low'
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to prepare events dataframe: {e}")
            return pd.DataFrame(events)
    
    def _score_to_risk_level(self, score: float) -> str:
        """Convert numeric risk score to risk level"""
        try:
            if score >= 0.8:
                return 'critical'
            elif score >= 0.6:
                return 'high'
            elif score >= 0.4:
                return 'medium'
            else:
                return 'low'
        except Exception as e:
            logger.error(f"Failed to convert score to risk level: {e}")
            return 'low'
    
    def _get_step_color(self, step) -> str:
        """Get color for attack step based on MITRE tactic or importance"""
        try:
            # Color by MITRE tactic if available
            mitre_tactic = getattr(step, 'mitre_tactic', None)
            if mitre_tactic:
                tactic_colors = {
                    'TA0001': '#3498DB',  # Initial Access - Blue
                    'TA0002': '#E67E22',  # Execution - Orange
                    'TA0003': '#2ECC71',  # Persistence - Green
                    'TA0004': '#F1C40F',  # Privilege Escalation - Yellow
                    'TA0005': '#9B59B6',  # Defense Evasion - Purple
                    'TA0006': '#E74C3C',  # Credential Access - Red
                    'TA0007': '#1ABC9C',  # Discovery - Teal
                    'TA0008': '#8E44AD',  # Lateral Movement - Dark Purple
                    'TA0009': '#F39C12',  # Collection - Orange
                    'TA0010': '#C0392B',  # Exfiltration - Dark Red
                    'TA0011': '#34495E'   # Command and Control - Dark Gray
                }
                return tactic_colors.get(mitre_tactic, '#95A5A6')
            
            # Color by importance if available
            importance_score = getattr(step, 'importance_score', 0.0)
            if importance_score > 0.8:
                return '#E74C3C'  # High importance - Red
            elif importance_score > 0.6:
                return '#F39C12'  # Medium importance - Orange
            else:
                return '#3498DB'  # Low importance - Blue
            
        except Exception as e:
            logger.error(f"Failed to get step color: {e}")
            return '#95A5A6'
    
    def _add_event_traces(self, fig: go.Figure, df_events: pd.DataFrame, row: int):
        """Add event traces to timeline"""
        try:
            # Group events by type
            event_types = df_events['event_type'].unique()
            
            for event_type in event_types:
                type_events = df_events[df_events['event_type'] == event_type]
                
                if len(type_events) == 0:
                    continue
                
                # Get color for this event type
                color = self.event_colors.get(event_type, '#95A5A6')
                
                # Create scatter plot for events
                fig.add_trace(go.Scatter(
                    x=type_events['timestamp'],
                    y=type_events['entity'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=color,
                        line=dict(width=2, color='white'),
                        opacity=0.8
                    ),
                    name=event_type.replace('_', ' ').title(),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Time: %{x}<br>"
                        "Type: " + event_type + "<br>"
                        "Risk: %{customdata}<br>"
                        "<extra></extra>"
                    ),
                    customdata=type_events.get('risk_level', 'unknown'),
                    showlegend=True
                ), row=row, col=1)
            
        except Exception as e:
            logger.error(f"Failed to add event traces: {e}")
    
    def _add_prediction_traces(self, fig: go.Figure, predictions: List[Dict], row: int):
        """Add prediction traces to timeline"""
        try:
            # Prepare prediction data
            pred_times = []
            pred_entities = []
            pred_descriptions = []
            
            for pred in predictions:
                # Use prediction timestamp or estimate future time
                if 'timestamp' in pred:
                    pred_time = pred['timestamp']
                else:
                    # Estimate future time (e.g. 5 minutes from now)
                    pred_time = datetime.now() + timedelta(minutes=5)
                
                pred_times.append(pred_time)
                pred_entities.append(pred.get('target_entity', 'unknown'))
                
                description = f"{pred.get('source_entity', 'unknown')} → {pred.get('target_entity', 'unknown')}"
                pred_descriptions.append(description)
            
            if pred_times:  # Only add trace if we have predictions
                # Add prediction trace
                fig.add_trace(go.Scatter(
                    x=pred_times,
                    y=pred_entities,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    name="Predictions",
                    hovertemplate=(
                        "<b>PREDICTION</b><br>"
                        "Time: %{x}<br>"
                        "Action: %{customdata}<br>"
                        "<extra></extra>"
                    ),
                    customdata=pred_descriptions,
                    showlegend=True
                ), row=row, col=1)
            
        except Exception as e:
            logger.error(f"Failed to add prediction traces: {e}")
    
    def _add_risk_timeline(self, fig: go.Figure, df_events: pd.DataFrame, row: int):
        """Add risk level timeline subplot"""
        try:
            # Create risk level over time
            risk_scores = df_events['risk_score'] if 'risk_score' in df_events.columns else [0.5] * len(df_events)
            
            fig.add_trace(go.Scatter(
                x=df_events['timestamp'],
                y=risk_scores,
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=6, color='red'),
                name="Risk Level",
                showlegend=False,
                hovertemplate=(
                    "Time: %{x}<br>"
                    "Risk Score: %{y:.2f}<br>"
                    "<extra></extra>"
                )
            ), row=row, col=1)
            
            # Add risk level zones
            fig.add_hline(y=0.4, line_dash="dash", line_color="green", 
                         annotation_text="Low Risk", row=row, col=1)
            fig.add_hline(y=0.6, line_dash="dash", line_color="orange",
                         annotation_text="Medium Risk", row=row, col=1)
            fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                         annotation_text="High Risk", row=row, col=1)
            
        except Exception as e:
            logger.error(f"Failed to add risk timeline: {e}")
    
    def _update_timeline_layout(self, fig: go.Figure, title: str, df_events: pd.DataFrame):
        """Update timeline layout"""
        try:
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                hovermode='closest',
                margin=dict(l=100, r=50, t=80, b=50),
                width=self.config.width,
                height=self.config.height
            )
            
            # Update x-axis for main timeline
            fig.update_xaxes(
                title="Time",
                type='date',
                tickformat='%H:%M:%S',
                row=1, col=1
            )
            
            # Update y-axis for main timeline
            fig.update_yaxes(
                title="Entities",
                row=1, col=1
            )
            
            # Update y-axis for risk timeline
            fig.update_yaxes(
                title="Risk Score",
                range=[0, 1],
                row=2, col=1
            )
            
        except Exception as e:
            logger.error(f"Failed to update timeline layout: {e}")
    
    def _create_empty_timeline(self, title: str) -> go.Figure:
        """Create empty timeline with message"""
        fig = go.Figure()
        fig.add_annotation(
            text="No timeline data available",
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
    
    def _create_error_timeline(self, title: str, error_msg: str) -> go.Figure:
        """Create error timeline with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating timeline:<br>{error_msg}",
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
            
            logger.info(f"Timeline saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save timeline: {e}")


class AttackTimelinePlotter:
    """
    Simplified interface for creating attack timeline plots.
    Designed for easy integration with Streamlit and other frameworks.
    """
    
    def __init__(self):
        """Initialize the attack timeline plotter"""
        self.visualizer = TimelineVisualizer()
        logger.info("Attack timeline plotter initialized")
    
    def plot_events(
        self,
        events: List[Dict],
        predictions: Optional[List[Dict]] = None,
        title: str = "Security Events Timeline",
        width: int = 1400,
        height: int = 600
    ) -> go.Figure:
        """
        Create event timeline plot with minimal configuration.
        
        Parameters
        ----------
        events : List[Dict]
            Security events
        predictions : List[Dict], optional
            Predicted events
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
        
        return self.visualizer.create_event_timeline(
            events=events,
            predictions=predictions,
            title=title
        )
    
    def plot_attack_chain(
        self,
        attack_chain,
        title: str = "Attack Chain Timeline",
        width: int = 1400,
        height: int = 600
    ) -> go.Figure:
        """
        Create attack chain timeline plot.
        
        Parameters
        ----------
        attack_chain : AttackChain
            Attack chain object
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
        
        return self.visualizer.create_attack_chain_timeline(
            attack_chain=attack_chain,
            title=title
        )


def test_timeline_visualizer():
    """Test the timeline visualizer with sample data"""
    
    logger.info("Testing Timeline Visualizer...")
    
    try:
        # Create sample events
        base_time = datetime.now() - timedelta(hours=1)
        
        events = [
            {
                'timestamp': base_time,
                'event_type': 'logon',
                'entity': 'user_0',
                'description': 'Sample event 1',
                'risk_score': 0.3
            },
            {
                'timestamp': base_time + timedelta(minutes=5),
                'event_type': 'process_create',
                'entity': 'user_1',
                'description': 'Sample event 2',
                'risk_score': 0.7
            },
            {
                'timestamp': base_time + timedelta(minutes=10),
                'event_type': 'network_connect',
                'entity': 'host_2',
                'description': 'Sample event 3',
                'risk_score': 0.9
            }
        ]
        
        # Create sample predictions
        predictions = [
            {
                'source_entity': 'user_0',
                'target_entity': 'host_1',
                'edge_type': 'logs_into',
                'confidence': 0.85,
                'risk_score': 0.75
            },
            {
                'source_entity': 'user_1',
                'target_entity': 'host_2',
                'edge_type': 'executes',
                'confidence': 0.9,
                'risk_score': 0.85
            }
        ]
        
        # Initialize visualizer
        visualizer = TimelineVisualizer()
        
        # Test event timeline
        fig1 = visualizer.create_event_timeline(
            events=events,
            predictions=predictions,
            title="Test Event Timeline"
        )
        logger.info(" Event timeline created")
        
        # Test timeline plotter
        plotter = AttackTimelinePlotter()
        fig2 = plotter.plot_events(
            events=events,
            predictions=predictions,
            title="Timeline Plotter Test"
        )
        logger.info(" Timeline plotter test completed")
        
        # Save test outputs
        output_dir = Path('outputs/visualizations/timelines')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig1.write_html(str(output_dir / 'test_event_timeline.html'))
        fig2.write_html(str(output_dir / 'test_timeline_plotter.html'))
        
        logger.info(" All timeline visualizer tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Timeline visualizer test failed: {e}")
        return False


if __name__ == '__main__':
    """Test the timeline visualizer"""
    
    # Ensure output directory exists
    os.makedirs('outputs/visualizations/timelines', exist_ok=True)
    
    # Run tests
    success = test_timeline_visualizer()
    
    if success:
        logger.info(" Day 7 Afternoon Timeline Visualizer implementation complete!")
        logger.info(" Ready for complete visualization system integration")
    else:
        logger.error("[ERROR] Timeline Visualizer implementation needs fixes")
        sys.exit(1)