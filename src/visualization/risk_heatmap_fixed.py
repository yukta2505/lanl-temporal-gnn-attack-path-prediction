#!/usr/bin/env python3
"""
Day 7 Afternoon: Risk Heatmap Visualization for THGNN - FIXED VERSION
Production-ready risk heatmap visualization with configuration management.

Author: THGNN Expert
Date: January 9, 2026
Version: 2.0 - CONFIGURATION-DRIVEN RISK HEATMAP
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
from scipy.interpolate import griddata

# Import configuration and validation
sys.path.append(str(Path(__file__).parent.parent))
from utils.config_manager import get_config, get_risk_color, get_visualization_config
from utils.input_validator import InputValidator, ValidationResult

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
    """Type-safe configuration for risk heatmap visualization"""
    width: int
    height: int
    colorscale: str
    show_colorbar: bool
    interactive: bool
    export_format: str
    cell_opacity: float
    border_width: int
    border_color: str
    show_text: bool
    text_color: str
    text_size: int
    histogram_bins: int
    histogram_opacity: float
    grid_size: int
    interpolation: str

class RiskHeatmapVisualizer:
    """
    Production-ready risk heatmap visualizer with configuration management.
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
        # Load configuration from centralized config manager
        viz_config = get_visualization_config()
        heatmap_config = get_config('risk_heatmap', default={})
        
        if config is None:
            self.config = RiskHeatmapConfig(
                width=viz_config.default_width,
                height=viz_config.default_height,
                colorscale=heatmap_config.get('colorscale', 'Reds'),
                show_colorbar=heatmap_config.get('show_colorbar', True),
                interactive=True,
                export_format=viz_config.ui_theme_colors.get('export_format', 'html'),
                cell_opacity=heatmap_config.get('cell_opacity', 0.8),
                border_width=heatmap_config.get('border_width', 1),
                border_color=heatmap_config.get('border_color', 'white'),
                show_text=heatmap_config.get('show_text', True),
                text_color=heatmap_config.get('text_color', 'white'),
                text_size=heatmap_config.get('text_size', 10),
                histogram_bins=heatmap_config.get('histogram', {}).get('bins', 20),
                histogram_opacity=heatmap_config.get('histogram', {}).get('opacity', 0.7),
                grid_size=heatmap_config.get('threat_landscape', {}).get('grid_size', 20),
                interpolation=heatmap_config.get('threat_landscape', {}).get('interpolation', 'bilinear')
            )
        else:
            self.config = config
        
        # Initialize input validator
        self.validator = InputValidator()
        
        # Load risk thresholds from configuration
        risk_timeline_config = get_config('timeline_visualization', {}).get('risk_timeline', {})
        self.risk_thresholds = {
            'low': risk_timeline_config.get('low_threshold', 0.4),
            'medium': risk_timeline_config.get('medium_threshold', 0.6),
            'high': risk_timeline_config.get('high_threshold', 0.8),
            'critical': 1.0
        }
        
        # Load color scales from configuration
        self.color_scales = {
            'risk': self.config.colorscale,
            'activity': 'Blues',
            'anomaly': 'YlOrRd',
            'confidence': 'Greens'
        }
        
        logger.info("Risk heatmap visualizer initialized with configuration management")
    
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
            logger.info(f"Creating risk heatmap for {len(entities)} entities across {len(time_windows)} time windows")
            
            # Validate inputs
            validation_result = self._validate_heatmap_inputs(
                entities, risk_scores_over_time, time_windows
            )
            if not validation_result.is_valid:
                logger.error(f"Input validation failed: {validation_result.errors}")
                return self._create_error_heatmap(title, "Invalid input data")
            
            # Prepare data matrix
            risk_matrix = self._prepare_risk_matrix(
                entities, risk_scores_over_time, time_windows
            )
            
            if risk_matrix is None:
                return self._create_empty_heatmap(title)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=risk_matrix,
                x=[t.strftime('%H:%M:%S') for t in time_windows],
                y=entities,
                colorscale=self.config.colorscale,
                showscale=self.config.show_colorbar,
                zmin=0,
                zmax=1,
                hoverongaps=False,
                hovertemplate=(
                    "<b>Entity:</b> %{y}<br>"
                    "<b>Time:</b> %{x}<br>"
                    "<b>Risk Score:</b> %{z:.3f}<br>"
                    "<b>Risk Level:</b> %{customdata}<br>"
                    "<extra></extra>"
                ),
                customdata=self._get_risk_levels_matrix(risk_matrix),
                colorbar=dict(
                    title="Risk Score",
                    titleside="right",
                    tickmode="linear",
                    tick0=0,
                    dtick=0.2,
                    thickness=15,
                    len=0.8
                ) if self.config.show_colorbar else None
            ))
            
            # Add text annotations if configured
            if self.config.show_text:
                self._add_text_annotations(fig, risk_matrix, entities, time_windows)
            
            # Update layout
            self._update_heatmap_layout(fig, title, len(entities), len(time_windows))
            
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
        entities: List[str],
        risk_scores: List[float],
        title: str = "Entity Risk Distribution",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create histogram showing distribution of risk scores across entities.
        
        Parameters
        ----------
        entities : List[str]
            List of entity names
        risk_scores : List[float]
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
            logger.info(f"Creating risk distribution for {len(entities)} entities")
            
            # Validate inputs
            if len(entities) != len(risk_scores):
                logger.error("Entities and risk_scores length mismatch")
                return self._create_error_heatmap(title, "Data length mismatch")
            
            # Validate risk scores
            for i, score in enumerate(risk_scores):
                result = self.validator.validate_numeric(score, 'risk_score', required=True)
                if not result.is_valid:
                    logger.warning(f"Invalid risk score for entity {entities[i]}: {score}")
                    risk_scores[i] = 0.0
            
            # Create histogram
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=risk_scores,
                nbinsx=self.config.histogram_bins,
                opacity=self.config.histogram_opacity,
                marker=dict(
                    color=risk_scores,
                    colorscale=self.config.colorscale,
                    line=dict(color='white', width=1)
                ),
                hovertemplate=(
                    "<b>Risk Range:</b> %{x}<br>"
                    "<b>Entity Count:</b> %{y}<br>"
                    "<extra></extra>"
                ),
                name="Risk Distribution"
            ))
            
            # Add risk threshold lines
            for level, threshold in self.risk_thresholds.items():
                if level != 'critical':  # Skip critical as it's at 1.0
                    color = get_risk_color(level)
                    fig.add_vline(
                        x=threshold,
                        line_dash="dash",
                        line_color=color,
                        annotation_text=f"{level.title()} Risk",
                        annotation_position="top"
                    )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                xaxis=dict(
                    title="Risk Score",
                    range=[0, 1],
                    tickformat='.2f'
                ),
                yaxis=dict(
                    title="Number of Entities"
                ),
                showlegend=False,
                width=self.config.width,
                height=self.config.height,
                margin=dict(l=60, r=60, t=80, b=60)
            )
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            logger.info("Risk distribution histogram created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create risk distribution: {e}")
            return self._create_error_heatmap(title, str(e))
    
    def create_threat_landscape_heatmap(
        self,
        entity_positions: Dict[str, Tuple[float, float]],
        risk_scores: Dict[str, float],
        title: str = "Threat Landscape Heatmap",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create 2D threat landscape heatmap with spatial interpolation.
        
        Parameters
        ----------
        entity_positions : Dict[str, Tuple[float, float]]
            Entity positions in 2D space (x, y coordinates)
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
            logger.info(f"Creating threat landscape heatmap for {len(entity_positions)} entities")
            
            # Validate inputs
            if not entity_positions or not risk_scores:
                return self._create_empty_heatmap(title)
            
            # Extract coordinates and scores
            entities = list(entity_positions.keys())
            x_coords = [pos[0] for pos in entity_positions.values()]
            y_coords = [pos[1] for pos in entity_positions.values()]
            scores = [risk_scores.get(entity, 0.0) for entity in entities]
            
            # Validate coordinates and scores
            for i, (x, y, score) in enumerate(zip(x_coords, y_coords, scores)):
                # Validate coordinates
                x_result = self.validator.validate_numeric(x, required=True)
                y_result = self.validator.validate_numeric(y, required=True)
                score_result = self.validator.validate_numeric(score, 'risk_score', required=True)
                
                if not (x_result.is_valid and y_result.is_valid and score_result.is_valid):
                    logger.warning(f"Invalid data for entity {entities[i]}")
                    continue
            
            # Create interpolation grid
            grid_x, grid_y, grid_z = self._create_interpolation_grid(
                x_coords, y_coords, scores
            )
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=grid_z,
                x=grid_x[0, :],
                y=grid_y[:, 0],
                colorscale=self.config.colorscale,
                showscale=self.config.show_colorbar,
                zmin=0,
                zmax=1,
                hovertemplate=(
                    "<b>X:</b> %{x:.2f}<br>"
                    "<b>Y:</b> %{y:.2f}<br>"
                    "<b>Risk Score:</b> %{z:.3f}<br>"
                    "<extra></extra>"
                ),
                colorbar=dict(
                    title="Risk Score",
                    titleside="right"
                ) if self.config.show_colorbar else None
            ))
            
            # Add entity markers
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=scores,
                    colorscale=self.config.colorscale,
                    line=dict(width=2, color='white'),
                    cmin=0,
                    cmax=1
                ),
                text=entities,
                textposition="top center",
                textfont=dict(size=8, color='black'),
                hovertemplate=(
                    "<b>Entity:</b> %{text}<br>"
                    "<b>Position:</b> (%{x:.2f}, %{y:.2f})<br>"
                    "<b>Risk Score:</b> %{marker.color:.3f}<br>"
                    "<extra></extra>"
                ),
                name="Entities",
                showlegend=False
            ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                xaxis=dict(
                    title="X Coordinate",
                    scaleanchor="y",
                    scaleratio=1
                ),
                yaxis=dict(
                    title="Y Coordinate"
                ),
                width=self.config.width,
                height=self.config.height,
                margin=dict(l=60, r=60, t=80, b=60)
            )
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            logger.info("Threat landscape heatmap created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create threat landscape heatmap: {e}")
            return self._create_error_heatmap(title, str(e))
    
    def _validate_heatmap_inputs(
        self,
        entities: List[str],
        risk_scores_over_time: Dict[str, List[float]],
        time_windows: List[datetime]
    ) -> ValidationResult:
        """Validate heatmap input data"""
        try:
            errors = []
            warnings = []
            
            # Validate entities
            entities_result = self.validator.validate_array(entities, 'string', 1000, True)
            if not entities_result.is_valid:
                errors.extend(entities_result.errors)
            
            # Validate risk scores dictionary
            if not isinstance(risk_scores_over_time, dict):
                errors.append("risk_scores_over_time must be a dictionary")
            else:
                for entity, scores in risk_scores_over_time.items():
                    # Validate entity name
                    entity_result = self.validator.validate_string(entity, 'entity_name', 200, True)
                    if not entity_result.is_valid:
                        errors.extend([f"Entity '{entity}': {error}" for error in entity_result.errors])
                    
                    # Validate scores array
                    scores_result = self.validator.validate_array(scores, 'numeric', 10000, True)
                    if not scores_result.is_valid:
                        errors.extend([f"Scores for '{entity}': {error}" for error in scores_result.errors])
            
            # Validate time windows
            if not isinstance(time_windows, list):
                errors.append("time_windows must be a list")
            else:
                for i, tw in enumerate(time_windows):
                    if not isinstance(tw, datetime):
                        errors.append(f"Time window {i} is not a datetime object")
            
            # Check data consistency
            if len(time_windows) > 0 and len(risk_scores_over_time) > 0:
                expected_length = len(time_windows)
                for entity, scores in risk_scores_over_time.items():
                    if len(scores) != expected_length:
                        warnings.append(f"Entity '{entity}' has {len(scores)} scores but {expected_length} time windows")
            
            return ValidationResult(len(errors) == 0, errors, warnings)
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return ValidationResult(False, [f"Validation error: {e}"], [])
    
    def _prepare_risk_matrix(
        self,
        entities: List[str],
        risk_scores_over_time: Dict[str, List[float]],
        time_windows: List[datetime]
    ) -> Optional[np.ndarray]:
        """Prepare risk score matrix for heatmap"""
        try:
            if not entities or not time_windows:
                return None
            
            # Initialize matrix
            matrix = np.zeros((len(entities), len(time_windows)))
            
            # Fill matrix with risk scores
            for i, entity in enumerate(entities):
                if entity in risk_scores_over_time:
                    scores = risk_scores_over_time[entity]
                    # Ensure we don't exceed matrix bounds
                    max_len = min(len(scores), len(time_windows))
                    matrix[i, :max_len] = scores[:max_len]
            
            return matrix
            
        except Exception as e:
            logger.error(f"Failed to prepare risk matrix: {e}")
            return None
    
    def _get_risk_levels_matrix(self, risk_matrix: np.ndarray) -> np.ndarray:
        """Convert risk scores to risk level labels"""
        try:
            risk_levels = np.full(risk_matrix.shape, 'low', dtype=object)
            
            risk_levels[risk_matrix >= self.risk_thresholds['medium']] = 'medium'
            risk_levels[risk_matrix >= self.risk_thresholds['high']] = 'high'
            risk_levels[risk_matrix >= self.risk_thresholds['critical']] = 'critical'
            
            return risk_levels
            
        except Exception as e:
            logger.error(f"Failed to get risk levels matrix: {e}")
            return np.full(risk_matrix.shape, 'unknown', dtype=object)
    
    def _add_text_annotations(
        self,
        fig: go.Figure,
        risk_matrix: np.ndarray,
        entities: List[str],
        time_windows: List[datetime]
    ) -> None:
        """Add text annotations to heatmap cells"""
        try:
            if not self.config.show_text:
                return
            
            annotations = []
            for i, entity in enumerate(entities):
                for j, time_window in enumerate(time_windows):
                    if i < risk_matrix.shape[0] and j < risk_matrix.shape[1]:
                        value = risk_matrix[i, j]
                        annotations.append(
                            dict(
                                x=time_window.strftime('%H:%M:%S'),
                                y=entity,
                                text=f"{value:.2f}",
                                showarrow=False,
                                font=dict(
                                    color=self.config.text_color,
                                    size=self.config.text_size
                                )
                            )
                        )
            
            fig.update_layout(annotations=annotations)
            
        except Exception as e:
            logger.error(f"Failed to add text annotations: {e}")
    
    def _create_interpolation_grid(
        self,
        x_coords: List[float],
        y_coords: List[float],
        scores: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create interpolation grid for threat landscape"""
        try:
            # Create regular grid
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add padding
            x_range = x_max - x_min
            y_range = y_max - y_min
            padding = 0.1
            
            x_min -= x_range * padding
            x_max += x_range * padding
            y_min -= y_range * padding
            y_max += y_range * padding
            
            # Create grid
            grid_x, grid_y = np.meshgrid(
                np.linspace(x_min, x_max, self.config.grid_size),
                np.linspace(y_min, y_max, self.config.grid_size)
            )
            
            # Interpolate values
            points = np.column_stack((x_coords, y_coords))
            grid_z = griddata(
                points, scores, (grid_x, grid_y),
                method='linear', fill_value=0.0
            )
            
            return grid_x, grid_y, grid_z
            
        except Exception as e:
            logger.error(f"Failed to create interpolation grid: {e}")
            # Return simple grid
            grid_x, grid_y = np.meshgrid(
                np.linspace(0, 1, self.config.grid_size),
                np.linspace(0, 1, self.config.grid_size)
            )
            grid_z = np.zeros_like(grid_x)
            return grid_x, grid_y, grid_z
    
    def _update_heatmap_layout(
        self,
        fig: go.Figure,
        title: str,
        num_entities: int,
        num_time_windows: int
    ) -> None:
        """Update heatmap layout with configuration"""
        try:
     