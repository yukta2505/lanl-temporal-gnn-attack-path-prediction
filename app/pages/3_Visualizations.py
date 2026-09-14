#!/usr/bin/env python3
"""
Visualizations Page

Advanced interactive visualizations for THGNN threat analysis.
Professional enterprise-grade visualizations for cybersecurity threat detection.

Features:
- Interactive network graphs
- Temporal analysis charts
- Risk heatmaps
- MITRE ATT&CK coverage
- Export capabilities
"""

import streamlit as st
import sys
import os
import json
import logging
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Visualizations",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E74C3C;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #BDC3C7;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #333;
        text-align: center;
    }
    
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-info {
        background-color: rgba(52, 152, 219, 0.1);
        border-left-color: #3498DB;
        color: #3498DB;
    }
    
    .alert-warning {
        background-color: rgba(243, 156, 18, 0.1);
        border-left-color: #F39C12;
        color: #F39C12;
    }
    
    .viz-section {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #333;
    }
    
    .viz-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #E74C3C;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)
def render_header():
    """Render page header."""
    st.markdown('<div class="main-header">Threat Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced interactive visualizations for threat analysis</div>', unsafe_allow_html=True)

def check_data_availability():
    """Check if visualization data is available."""
    # Check for analysis data
    if st.session_state.get('analysis_complete', False) and 'analysis_data' in st.session_state:
        analysis_data = st.session_state['analysis_data']
        predictions = analysis_data.get('predictions', [])
        return predictions, 'analysis'
    
    # Check for prediction data
    elif st.session_state.get('predictions_generated', False) and st.session_state.get('predictions'):
        predictions = st.session_state['predictions']
        return predictions, 'predictions'
    
    return None, None

def render_no_data_state():
    """Render the no data available state."""
    st.markdown("""
    <div class="alert-box alert-info">
        <h4 style="margin: 0;">No Visualization Data Available</h4>
        <p style="margin: 8px 0 0 0;">
            Upload security logs and generate threat predictions to access advanced visualizations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Upload Security Logs", width='stretch'):
            st.switch_page("pages/Upload_Logs.py")
    
    with col2:
        if st.button("Generate Predictions", width='stretch'):
            st.switch_page("pages/6_Proactive_Prediction.py")

def create_network_visualization(predictions):
    """Create network visualization from predictions."""
    try:
        # Create a simple network graph
        fig = go.Figure()
        
        # Sample network data based on predictions
        if isinstance(predictions, list) and predictions:
            # Extract entities from predictions
            nodes = []
            edges = []
            
            for i, pred in enumerate(predictions[:20]):  # Limit to 20 for performance
                source = f"Entity_{i}"
                target = f"Target_{i}"
                
                nodes.extend([source, target])
                edges.append((source, target))
            
            # Remove duplicates
            unique_nodes = list(set(nodes))
            
            # Create positions for nodes
            import math
            positions = {}
            for i, node in enumerate(unique_nodes):
                angle = 2 * math.pi * i / len(unique_nodes)
                positions[node] = (math.cos(angle), math.sin(angle))
            
            # Add edges
            for source, target in edges:
                if source in positions and target in positions:
                    x0, y0 = positions[source]
                    x1, y1 = positions[target]
                    
                    fig.add_trace(go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(color='#E74C3C', width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            # Add nodes
            for node, (x, y) in positions.items():
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers+text',
                    marker=dict(size=20, color='#3498DB'),
                    text=node.split('_')[1],
                    textposition="middle center",
                    showlegend=False,
                    hovertemplate=f'<b>{node}</b><extra></extra>'
                ))
        
        fig.update_layout(
            title="Threat Network Graph",
            showlegend=False,
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating network visualization: {e}")
        return go.Figure()

def create_risk_timeline(predictions):
    """Create risk timeline visualization."""
    try:
        fig = go.Figure()
        
        if isinstance(predictions, list) and predictions:
            # Generate sample timeline data
            times = list(range(len(predictions[:50])))
            risks = [pred.get('risk_score', pred.get('confidence', 0.5)) if isinstance(pred, dict) else 0.5 
                    for pred in predictions[:50]]
            
            fig.add_trace(go.Scatter(
                x=times,
                y=risks,
                mode='lines+markers',
                line=dict(color='#E74C3C', width=3),
                marker=dict(size=8, color='#E74C3C'),
                name='Risk Score'
            ))
        
        fig.update_layout(
            title="Risk Score Timeline",
            xaxis_title="Time Sequence",
            yaxis_title="Risk Score",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating risk timeline: {e}")
        return go.Figure()

def create_threat_heatmap(predictions):
    """Create threat heatmap visualization."""
    try:
        fig = go.Figure()
        
        if isinstance(predictions, list) and predictions:
            # Create sample heatmap data
            categories = ['Critical', 'High', 'Medium', 'Low']
            times = ['Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5']
            
            # Generate sample data
            z_data = []
            for i in range(len(categories)):
                row = []
                for j in range(len(times)):
                    # Sample risk values
                    value = max(0, min(1, 0.8 - (i * 0.2) + (j * 0.05)))
                    row.append(value)
                z_data.append(row)
            
            fig.add_trace(go.Heatmap(
                z=z_data,
                x=times,
                y=categories,
                colorscale='Reds',
                showscale=True
            ))
        
        fig.update_layout(
            title="Threat Risk Heatmap",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating threat heatmap: {e}")
        return go.Figure()

def render_visualizations(predictions, data_type):
    """Render the main visualizations."""
    st.markdown("### Interactive Threat Visualizations")
    
    # Display data info
    if data_type == 'analysis':
        st.markdown("""
        <div class="alert-box alert-info">
            <strong>Analysis Data:</strong> Visualizing comprehensive threat analysis results
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-box alert-info">
            <strong>Prediction Data:</strong> Visualizing threat prediction results
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_predictions = len(predictions) if isinstance(predictions, list) else 1
        st.metric("Total Predictions", total_predictions)
    
    with col2:
        high_risk = len([p for p in predictions if isinstance(p, dict) and p.get('risk_score', 0) > 0.7]) if isinstance(predictions, list) else 0
        st.metric("High Risk Events", high_risk)
    
    with col3:
        avg_confidence = np.mean([p.get('confidence', 0.5) for p in predictions if isinstance(p, dict)]) if isinstance(predictions, list) else 0.5
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with col4:
        st.metric("Analysis Status", "Complete", delta="Ready")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Network Graph", 
        "Risk Timeline", 
        "Threat Heatmap",
        "Export Data"
    ])
    
    with tab1:
        st.markdown("#### Network Threat Visualization")
        with st.spinner("Generating network graph..."):
            network_fig = create_network_visualization(predictions)
            st.plotly_chart(network_fig, width='stretch')
    
    with tab2:
        st.markdown("#### Temporal Risk Analysis")
        with st.spinner("Generating timeline..."):
            timeline_fig = create_risk_timeline(predictions)
            st.plotly_chart(timeline_fig, width='stretch')
    
    with tab3:
        st.markdown("#### Risk Intelligence Heatmap")
        with st.spinner("Generating heatmap..."):
            heatmap_fig = create_threat_heatmap(predictions)
            st.plotly_chart(heatmap_fig, width='stretch')
    
    with tab4:
        st.markdown("#### Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Charts", width='stretch'):
                st.info("Chart export functionality - would generate high-resolution images")
        
        with col2:
            if st.button("Export Data", width='stretch'):
                if predictions:
                    csv_data = pd.DataFrame(predictions).to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="threat_data.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("Generate Report", width='stretch'):
                st.switch_page("pages/5_Reports.py")



def main():
    """Main function for the visualizations page."""
    try:
        # Render header
        render_header()
        
        # Check data availability
        predictions, data_type = check_data_availability()
        
        if predictions is None:
            # No data available
            render_no_data_state()
        else:
            # Render visualizations
            render_visualizations(predictions, data_type)
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        st.error(f"Application error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()