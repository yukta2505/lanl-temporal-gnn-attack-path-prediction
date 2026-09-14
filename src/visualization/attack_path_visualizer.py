#!/usr/bin/env python3
"""
Attack Path Visualizer

Professional attack path visualization for cybersecurity threat detection.
Creates animated attack sequences with arrows, risk-based coloring, and
MITRE ATT&CK technique mapping for client demonstrations.

Features:
- Animated attack paths with directional arrows
- Risk-based color coding (red=critical, orange=high, yellow=medium, green=low)
- Interactive hover information with attack details
- MITRE ATT&CK technique annotations and badges
- Timeline-based attack progression visualization
- Export capabilities (PNG, HTML) for reports
- Professional styling for SOC operations and client demos
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class AttackPathVisualizer:
    """
    Professional attack path visualization for cybersecurity demonstrations.
    
    Creates compelling visual representations of predicted attack sequences
    with animated arrows, risk-based coloring, and MITRE technique mapping.
    Designed for security analysts and client presentations.
    """
    
    def __init__(self):
        """Initialize attack path visualizer with professional styling."""
        
        # Professional color scheme for cybersecurity
        self.colors = {
            'critical': '#E74C3C',    # Red - Critical threats
            'high': '#E67E22',        # Orange - High risk
            'medium': '#F39C12',      # Yellow - Medium risk
            'low': '#27AE60',         # Green - Low risk
            'normal': '#95A5A6',      # Gray - Normal activity
            'background': '#0E1117',  # Dark background
            'text': '#FAFAFA',        # Light text
            'grid': '#2C3E50',        # Grid lines
            'predicted': '#E74C3C',   # Red arrows for predictions
            'historical': '#3498DB',  # Blue arrows for historical
            'potential': '#F39C12'    # Orange arrows for potential
        }
        
        # MITRE ATT&CK tactic colors
        self.mitre_colors = {
            'TA0001': '#3498DB',  # Initial Access
            'TA0002': '#E67E22',  # Execution
            'TA0003': '#9B59B6',  # Persistence
            'TA0004': '#1ABC9C',  # Privilege Escalation
            'TA0005': '#F1C40F',  # Defense Evasion
            'TA0006': '#E74C3C',  # Credential Access
            'TA0007': '#34495E',  # Discovery
            'TA0008': '#16A085',  # Lateral Movement
            'TA0009': '#8E44AD',  # Collection
            'TA0010': '#D35400',  # Exfiltration
            'TA0011': '#2C3E50'   # Command and Control
        }
        
        # Node type styling
        self.node_styles = {
            'user': {'symbol': 'circle', 'size': 20, 'color': '#3498DB'},
            'host': {'symbol': 'square', 'size': 25, 'color': '#2ECC71'},
            'process': {'symbol': 'diamond', 'size': 18, 'color': '#F39C12'},
            'file': {'symbol': 'triangle-up', 'size': 16, 'color': '#9B59B6'},
            'registry': {'symbol': 'hexagon', 'size': 20, 'color': '#E67E22'}
        }
        
        # Edge type styling
        self.edge_styles = {
            'logs_into': {'color': '#3498DB', 'width': 3, 'dash': 'solid'},
            'executes': {'color': '#E74C3C', 'width': 4, 'dash': 'solid'},
            'accesses': {'color': '#F39C12', 'width': 2, 'dash': 'dot'},
            'modifies': {'color': '#E67E22', 'width': 3, 'dash': 'dash'},
            'connects_to': {'color': '#27AE60', 'width': 2, 'dash': 'solid'},
            'creates': {'color': '#9B59B6', 'width': 3, 'dash': 'solid'}
        }
        
        logger.info("Attack path visualizer initialized with professional styling")
    
    def plot_attack_sequence(
        self,
        attack_steps: List[Dict[str, Any]],
        graph_data: Optional[Dict[str, Any]] = None,
        title: str = "🚨 Predicted Attack Sequence",
        animate: bool = True,
        width: int = 1200,
        height: int = 800,
        show_confidence: bool = True,
        show_mitre: bool = True
    ) -> go.Figure:
        """
        Create animated attack path visualization with arrows and risk coloring.
        
        Args:
            attack_steps: List of attack steps with source, target, and metadata
            graph_data: Optional graph structure data
            title: Plot title
            animate: Enable animation effects
            width: Plot width in pixels
            height: Plot height in pixels
            show_confidence: Show confidence scores
            show_mitre: Show MITRE technique annotations
            
        Returns:
            Plotly figure with attack path visualization
        """
        try:
            if not attack_steps:
                return self._create_empty_plot(title, width, height)
            
            # Create network graph from attack steps
            G = self._build_attack_graph(attack_steps)
            
            # Generate layout
            pos = self._generate_layout(G, attack_steps)
            
            # Create figure
            fig = go.Figure()
            
            # Add edges (attack paths)
            self._add_attack_edges(fig, attack_steps, pos, animate)
            
            # Add nodes (entities)
            self._add_attack_nodes(fig, G, pos, attack_steps, show_confidence)
            
            # Add MITRE technique annotations
            if show_mitre:
                self._add_mitre_annotations(fig, attack_steps, pos)
            
            # Update layout with professional styling
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24, 'color': self.colors['critical']}
                },
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(0,0,0,0.8)',
                    bordercolor=self.colors['grid'],
                    borderwidth=1
                ),
                width=width,
                height=height,
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background'],
                font={'color': self.colors['text']},
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    range=[-1.2, 1.2]
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    range=[-1.2, 1.2]
                ),
                hovermode='closest',
                annotations=[]
            )
            
            # Add animation frames if enabled
            if animate:
                self._add_animation_frames(fig, attack_steps, pos)
            
            # Add risk level indicator
            self._add_risk_indicator(fig, attack_steps)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating attack sequence plot: {e}")
            return self._create_error_plot(str(e), width, height)
    
    def create_proactive_timeline(
        self,
        timeline_data: Dict[str, Any],
        title: str = "⏰ Predicted Attack Timeline",
        width: int = 1200,
        height: int = 600
    ) -> go.Figure:
        """
        Create timeline visualization for predicted attack progression.
        
        Args:
            timeline_data: Timeline data with steps and timestamps
            title: Plot title
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure with timeline visualization
        """
        try:
            timeline = timeline_data.get('timeline', [])
            
            if not timeline:
                return self._create_empty_plot(title, width, height)
            
            # Prepare timeline data
            df_data = []
            current_time = datetime.now()
            
            for step in timeline:
                step_time = current_time + timedelta(minutes=step['time_from_now_minutes'])
                
                df_data.append({
                    'step': f"Step {step['step_number']}",
                    'time': step_time,
                    'minutes_from_now': step['time_from_now_minutes'],
                    'action': f"{step['edge_type']}: {step['source_type']} → {step['target_type']}",
                    'confidence': step['confidence'],
                    'risk_level': self._get_risk_level(step['confidence']),
                    'mitre_techniques': ', '.join(step.get('mitre_techniques', []))
                })
            
            df = pd.DataFrame(df_data)
            
            # Create timeline plot
            fig = go.Figure()
            
            # Add timeline bars
            colors = [self._get_risk_color(row['confidence']) for _, row in df.iterrows()]
            
            fig.add_trace(go.Bar(
                x=df['minutes_from_now'],
                y=df['step'],
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=1)
                ),
                text=[f"{row['confidence']:.1%}" for _, row in df.iterrows()],
                textposition='inside',
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Time: %{x} minutes from now<br>"
                    "Action: %{customdata[0]}<br>"
                    "Confidence: %{customdata[1]:.1%}<br>"
                    "MITRE: %{customdata[2]}<br>"
                    "<extra></extra>"
                ),
                customdata=df[['action', 'confidence', 'mitre_techniques']].values,
                name="Attack Steps"
            ))
            
            # Add current time indicator
            fig.add_vline(
                x=0,
                line=dict(color=self.colors['critical'], width=3, dash='dash'),
                annotation_text="NOW",
                annotation_position="top"
            )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': self.colors['text']}
                },
                xaxis_title="Minutes from Now",
                yaxis_title="Attack Steps",
                width=width,
                height=height,
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background'],
                font={'color': self.colors['text']},
                showlegend=False,
                margin=dict(l=150, r=50, t=80, b=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating timeline visualization: {e}")
            return self._create_error_plot(str(e), width, height)
    
    def generate_mitre_mapping_viz(
        self,
        attack_steps: List[Dict[str, Any]],
        title: str = "🎯 MITRE ATT&CK Technique Mapping",
        width: int = 1000,
        height: int = 600
    ) -> go.Figure:
        """
        Create MITRE ATT&CK technique mapping visualization.
        
        Args:
            attack_steps: Attack steps with MITRE techniques
            title: Plot title
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure with MITRE mapping
        """
        try:
            # Extract MITRE techniques
            techniques = []
            for step in attack_steps:
                step_techniques = step.get('mitre_techniques', [])
                for technique in step_techniques:
                    techniques.append({
                        'technique': technique,
                        'step': step['step_number'],
                        'confidence': step['confidence'],
                        'edge_type': step['edge_type']
                    })
            
            if not techniques:
                return self._create_empty_plot(title, width, height)
            
            df = pd.DataFrame(techniques)
            
            # Create sunburst chart for MITRE techniques
            fig = go.Figure(go.Sunburst(
                labels=df['technique'].tolist(),
                parents=[''] * len(df),
                values=df['confidence'].tolist(),
                branchvalues="total",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Confidence: %{value:.1%}<br>"
                    "Step: %{customdata[0]}<br>"
                    "Action: %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
                customdata=df[['step', 'edge_type']].values,
                maxdepth=2
            ))
            
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': self.colors['text']}
                },
                width=width,
                height=height,
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background'],
                font={'color': self.colors['text']}
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating MITRE mapping visualization: {e}")
            return self._create_error_plot(str(e), width, height)
    
    def _build_attack_graph(self, attack_steps: List[Dict[str, Any]]) -> nx.DiGraph:
        """Build NetworkX graph from attack steps."""
        try:
            G = nx.DiGraph()
            
            # Add nodes and edges from attack steps
            for i, step in enumerate(attack_steps):
                source = f"{step['source_type']}_{step.get('source_node', i)}"
                target = f"{step['target_type']}_{step.get('target_node', i+1)}"
                
                # Add nodes with attributes
                G.add_node(source, 
                          node_type=step['source_type'],
                          step_number=step['step_number'],
                          confidence=step['confidence'])
                
                G.add_node(target,
                          node_type=step['target_type'], 
                          step_number=step['step_number'],
                          confidence=step['confidence'])
                
                # Add edge with attributes
                G.add_edge(source, target,
                          edge_type=step['edge_type'],
                          confidence=step['confidence'],
                          step_number=step['step_number'],
                          mitre_techniques=step.get('mitre_techniques', []))
            
            return G
            
        except Exception as e:
            logger.error(f"Error building attack graph: {e}")
            return nx.DiGraph()
    
    def _generate_layout(
        self, 
        G: nx.DiGraph, 
        attack_steps: List[Dict[str, Any]]
    ) -> Dict[str, Tuple[float, float]]:
        """Generate layout positions for nodes."""
        try:
            if len(G.nodes()) == 0:
                return {}
            
            # Use spring layout with custom parameters for attack visualization
            pos = nx.spring_layout(
                G, 
                k=2.0,  # Optimal distance between nodes
                iterations=50,
                seed=42  # Reproducible layout
            )
            
            # Adjust positions for better attack flow visualization
            # Arrange nodes in attack sequence order
            sorted_steps = sorted(attack_steps, key=lambda x: x['step_number'])
            
            if len(sorted_steps) > 1:
                # Create a flow from left to right
                for i, step in enumerate(sorted_steps):
                    source = f"{step['source_type']}_{step.get('source_node', i)}"
                    target = f"{step['target_type']}_{step.get('target_node', i+1)}"
                    
                    if source in pos:
                        pos[source] = (pos[source][0] - 0.3, pos[source][1])
                    if target in pos:
                        pos[target] = (pos[target][0] + 0.3, pos[target][1])
            
            return pos
            
        except Exception as e:
            logger.error(f"Error generating layout: {e}")
            return {}
    
    def _add_attack_edges(
        self,
        fig: go.Figure,
        attack_steps: List[Dict[str, Any]],
        pos: Dict[str, Tuple[float, float]],
        animate: bool = True
    ) -> None:
        """Add attack edges with arrows to the figure."""
        try:
            for i, step in enumerate(attack_steps):
                source = f"{step['source_type']}_{step.get('source_node', i)}"
                target = f"{step['target_type']}_{step.get('target_node', i+1)}"
                
                if source not in pos or target not in pos:
                    continue
                
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                
                # Get edge style
                edge_type = step['edge_type']
                style = self.edge_styles.get(edge_type, self.edge_styles['executes'])
                
                # Adjust color based on confidence
                confidence = step['confidence']
                edge_color = self._get_risk_color(confidence)
                
                # Add edge line
                fig.add_trace(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(
                        color=edge_color,
                        width=style['width'],
                        dash=style['dash']
                    ),
                    hovertemplate=(
                        f"<b>Step {step['step_number']}: {edge_type}</b><br>"
                        f"From: {step['source_type']}<br>"
                        f"To: {step['target_type']}<br>"
                        f"Confidence: {confidence:.1%}<br>"
                        f"MITRE: {', '.join(step.get('mitre_techniques', []))}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    name=f"Step {step['step_number']}"
                ))
                
                # Add arrow annotation
                fig.add_annotation(
                    x=x1,
                    y=y1,
                    ax=x0,
                    ay=y0,
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor=edge_color,
                    showarrow=True,
                    text=""
                )
                
        except Exception as e:
            logger.error(f"Error adding attack edges: {e}")
    
    def _add_attack_nodes(
        self,
        fig: go.Figure,
        G: nx.DiGraph,
        pos: Dict[str, Tuple[float, float]],
        attack_steps: List[Dict[str, Any]],
        show_confidence: bool = True
    ) -> None:
        """Add attack nodes to the figure."""
        try:
            # Group nodes by type
            node_groups = {}
            
            for node, data in G.nodes(data=True):
                node_type = data.get('node_type', 'unknown')
                if node_type not in node_groups:
                    node_groups[node_type] = []
                node_groups[node_type].append((node, data))
            
            # Add nodes by type
            for node_type, nodes in node_groups.items():
                if not nodes:
                    continue
                
                style = self.node_styles.get(node_type, self.node_styles['process'])
                
                x_coords = []
                y_coords = []
                hover_texts = []
                node_colors = []
                
                for node, data in nodes:
                    if node not in pos:
                        continue
                    
                    x, y = pos[node]
                    x_coords.append(x)
                    y_coords.append(y)
                    
                    # Color based on confidence/risk
                    confidence = data.get('confidence', 0.5)
                    node_colors.append(self._get_risk_color(confidence))
                    
                    # Hover text
                    hover_text = (
                        f"<b>{node_type.title()}</b><br>"
                        f"Node: {node}<br>"
                        f"Step: {data.get('step_number', 'N/A')}<br>"
                    )
                    
                    if show_confidence:
                        hover_text += f"Confidence: {confidence:.1%}<br>"
                    
                    hover_texts.append(hover_text)
                
                if x_coords:
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='markers',
                        marker=dict(
                            symbol=style['symbol'],
                            size=style['size'],
                            color=node_colors,
                            line=dict(color='white', width=2)
                        ),
                        text=hover_texts,
                        hovertemplate="%{text}<extra></extra>",
                        name=node_type.title(),
                        showlegend=True
                    ))
                    
        except Exception as e:
            logger.error(f"Error adding attack nodes: {e}")
    
    def _add_mitre_annotations(
        self,
        fig: go.Figure,
        attack_steps: List[Dict[str, Any]],
        pos: Dict[str, Tuple[float, float]]
    ) -> None:
        """Add MITRE technique annotations."""
        try:
            for i, step in enumerate(attack_steps):
                techniques = step.get('mitre_techniques', [])
                if not techniques:
                    continue
                
                # Position annotation near the step
                x_pos = 0.8 + (i * 0.1) % 0.4
                y_pos = 0.9 - (i * 0.1) % 0.4
                
                technique_text = "<br>".join(techniques[:3])  # Show max 3 techniques
                
                fig.add_annotation(
                    x=x_pos,
                    y=y_pos,
                    xref='paper',
                    yref='paper',
                    text=f"<b>Step {step['step_number']}</b><br>{technique_text}",
                    showarrow=False,
                    bgcolor='rgba(52, 152, 219, 0.8)',
                    bordercolor='white',
                    borderwidth=1,
                    font=dict(color='white', size=10)
                )
                
        except Exception as e:
            logger.error(f"Error adding MITRE annotations: {e}")
    
    def _add_animation_frames(
        self,
        fig: go.Figure,
        attack_steps: List[Dict[str, Any]],
        pos: Dict[str, Tuple[float, float]]
    ) -> None:
        """Add animation frames for attack progression."""
        try:
            # For now, we'll add a simple animation effect
            # In a full implementation, this would create frame-by-frame animation
            
            # Add play button
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=list([
                            dict(
                                args=[{"visible": [True] * len(fig.data)}],
                                label="Play Attack",
                                method="restyle"
                            ),
                            dict(
                                args=[{"visible": [False] * len(fig.data)}],
                                label="Reset",
                                method="restyle"
                            )
                        ]),
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.01,
                        xanchor="left",
                        y=1.02,
                        yanchor="top"
                    ),
                ]
            )
            
        except Exception as e:
            logger.error(f"Error adding animation frames: {e}")
    
    def _add_risk_indicator(
        self,
        fig: go.Figure,
        attack_steps: List[Dict[str, Any]]
    ) -> None:
        """Add overall risk level indicator."""
        try:
            if not attack_steps:
                return
            
            # Calculate overall risk
            avg_confidence = np.mean([step['confidence'] for step in attack_steps])
            risk_level = self._get_risk_level(avg_confidence)
            risk_color = self._get_risk_color(avg_confidence)
            
            # Add risk indicator
            fig.add_annotation(
                x=0.02,
                y=0.02,
                xref='paper',
                yref='paper',
                text=f"<b>RISK LEVEL: {risk_level}</b><br>Confidence: {avg_confidence:.1%}",
                showarrow=False,
                bgcolor=risk_color,
                bordercolor='white',
                borderwidth=2,
                font=dict(color='white', size=12, family='Arial Black')
            )
            
        except Exception as e:
            logger.error(f"Error adding risk indicator: {e}")
    
    def _get_risk_level(self, confidence: float) -> str:
        """Get risk level based on confidence score."""
        if confidence >= 0.8:
            return "CRITICAL"
        elif confidence >= 0.6:
            return "HIGH"
        elif confidence >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_risk_color(self, confidence: float) -> str:
        """Get color based on risk level."""
        if confidence >= 0.8:
            return self.colors['critical']
        elif confidence >= 0.6:
            return self.colors['high']
        elif confidence >= 0.4:
            return self.colors['medium']
        else:
            return self.colors['low']
    
    def _create_empty_plot(self, title: str, width: int, height: int) -> go.Figure:
        """Create empty plot with message."""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref='paper',
            yref='paper',
            text="No attack data available",
            showarrow=False,
            font=dict(size=20, color=self.colors['text'])
        )
        
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font={'color': self.colors['text']},
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _create_error_plot(self, error_msg: str, width: int, height: int) -> go.Figure:
        """Create error plot with message."""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref='paper',
            yref='paper',
            text=f"Error: {error_msg}",
            showarrow=False,
            font=dict(size=16, color=self.colors['critical'])
        )
        
        fig.update_layout(
            title="Visualization Error",
            width=width,
            height=height,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font={'color': self.colors['text']},
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig