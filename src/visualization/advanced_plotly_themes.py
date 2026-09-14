#!/usr/bin/env python3
"""
Day 8 Afternoon: Advanced Plotly Themes for THGNN Visualizations
God-level visualization themes with perfect color coordination.

Author: THGNN Expert
Date: January 10, 2026
Version: 1.0 - ADVANCED THEMES
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional

class THGNNPlotlyThemes:
    """Advanced Plotly themes for THGNN visualizations"""
    
    @staticmethod
    def get_theme_config(theme_name: str) -> Dict[str, Any]:
        """Get complete theme configuration for Plotly"""
        
        themes = {
            "dark_cyber": {
                "layout": {
                    "paper_bgcolor": "#0E1117",
                    "plot_bgcolor": "#1A1A1A",
                    "font": {"color": "#FAFAFA", "family": "Consolas, monospace"},
                    "colorway": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"],
                    "grid": {"color": "#3D4551"},
                    "title": {"font": {"size": 20, "color": "#FF6B6B"}},
                    "legend": {"bgcolor": "rgba(0,0,0,0.5)", "bordercolor": "#3D4551", "borderwidth": 1}
                },
                "node_colors": {
                    "user": "#FF6B6B",
                    "host": "#4ECDC4", 
                    "process": "#45B7D1",
                    "file": "#96CEB4",
                    "registry": "#FFEAA7"
                },
                "edge_colors": {
                    "logs_into": "#FF6B6B",
                    "executes": "#4ECDC4",
                    "connects_to": "#45B7D1",
                    "accesses": "#96CEB4",
                    "modifies": "#FFEAA7",
                    "predicted": "#FF3333"
                },
                "risk_colors": {
                    "critical": "#FF4757",
                    "high": "#FF6B6B", 
                    "medium": "#FFA502",
                    "low": "#26de81"
                }
            },
            
            "light_professional": {
                "layout": {
                    "paper_bgcolor": "#FFFFFF",
                    "plot_bgcolor": "#F8F9FA",
                    "font": {"color": "#2C3E50", "family": "Arial, sans-serif"},
                    "colorway": ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22"],
                    "grid": {"color": "#ECF0F1"},
                    "title": {"font": {"size": 20, "color": "#2C3E50"}},
                    "legend": {"bgcolor": "rgba(255,255,255,0.9)", "bordercolor": "#BDC3C7", "borderwidth": 1}
                },
                "node_colors": {
                    "user": "#3498DB",
                    "host": "#E74C3C",
                    "process": "#2ECC71", 
                    "file": "#F39C12",
                    "registry": "#9B59B6"
                },
                "edge_colors": {
                    "logs_into": "#3498DB",
                    "executes": "#E74C3C",
                    "connects_to": "#2ECC71",
                    "accesses": "#F39C12",
                    "modifies": "#9B59B6",
                    "predicted": "#C0392B"
                },
                "risk_colors": {
                    "critical": "#C0392B",
                    "high": "#E74C3C",
                    "medium": "#F39C12", 
                    "low": "#27AE60"
                }
            },
            
            "cyber_red": {
                "layout": {
                    "paper_bgcolor": "#0A0A0A",
                    "plot_bgcolor": "#1A0000",
                    "font": {"color": "#FF0000", "family": "Courier New, monospace"},
                    "colorway": ["#FF0000", "#FF3333", "#FF6666", "#FF9999", "#FFCCCC", "#CC0000", "#990000"],
                    "grid": {"color": "#330000"},
                    "title": {"font": {"size": 22, "color": "#FF0000"}},
                    "legend": {"bgcolor": "rgba(26,0,0,0.8)", "bordercolor": "#FF0000", "borderwidth": 2}
                },
                "node_colors": {
                    "user": "#FF0000",
                    "host": "#FF3333",
                    "process": "#FF6666",
                    "file": "#FF9999", 
                    "registry": "#CC0000"
                },
                "edge_colors": {
                    "logs_into": "#FF0000",
                    "executes": "#FF3333",
                    "connects_to": "#FF6666",
                    "accesses": "#FF9999",
                    "modifies": "#CC0000",
                    "predicted": "#FFFF00"
                },
                "risk_colors": {
                    "critical": "#FFFF00",
                    "high": "#FF0000",
                    "medium": "#FF6666",
                    "low": "#FF9999"
                }
            },
            
            "security_blue": {
                "layout": {
                    "paper_bgcolor": "#001122",
                    "plot_bgcolor": "#002244",
                    "font": {"color": "#00CCFF", "family": "Roboto, sans-serif"},
                    "colorway": ["#00CCFF", "#0099CC", "#0066FF", "#3399FF", "#66CCFF", "#0088AA", "#004488"],
                    "grid": {"color": "#003366"},
                    "title": {"font": {"size": 20, "color": "#00CCFF"}},
                    "legend": {"bgcolor": "rgba(0,34,68,0.8)", "bordercolor": "#00CCFF", "borderwidth": 1}
                },
                "node_colors": {
                    "user": "#00CCFF",
                    "host": "#0099CC",
                    "process": "#0066FF",
                    "file": "#3399FF",
                    "registry": "#66CCFF"
                },
                "edge_colors": {
                    "logs_into": "#00CCFF",
                    "executes": "#0099CC", 
                    "connects_to": "#0066FF",
                    "accesses": "#3399FF",
                    "modifies": "#66CCFF",
                    "predicted": "#FFAA00"
                },
                "risk_colors": {
                    "critical": "#FF4444",
                    "high": "#FFAA00",
                    "medium": "#00CCFF",
                    "low": "#00FF88"
                }
            },
            
            "matrix_green": {
                "layout": {
                    "paper_bgcolor": "#000000",
                    "plot_bgcolor": "#001100",
                    "font": {"color": "#00FF00", "family": "Courier New, monospace"},
                    "colorway": ["#00FF00", "#33FF33", "#66FF66", "#99FF99", "#CCFFCC", "#00CC00", "#009900"],
                    "grid": {"color": "#002200"},
                    "title": {"font": {"size": 20, "color": "#00FF00"}},
                    "legend": {"bgcolor": "rgba(0,17,0,0.8)", "bordercolor": "#00FF00", "borderwidth": 1}
                },
                "node_colors": {
                    "user": "#00FF00",
                    "host": "#33FF33",
                    "process": "#66FF66",
                    "file": "#99FF99",
                    "registry": "#CCFFCC"
                },
                "edge_colors": {
                    "logs_into": "#00FF00",
                    "executes": "#33FF33",
                    "connects_to": "#66FF66", 
                    "accesses": "#99FF99",
                    "modifies": "#CCFFCC",
                    "predicted": "#FFFF00"
                },
                "risk_colors": {
                    "critical": "#FF0000",
                    "high": "#FFFF00",
                    "medium": "#00FF00",
                    "low": "#99FF99"
                }
            }
        }
        
        return themes.get(theme_name, themes["dark_cyber"])
    
    @staticmethod
    def apply_theme_to_figure(fig: go.Figure, theme_name: str) -> go.Figure:
        """Apply theme to existing Plotly figure"""
        
        theme_config = THGNNPlotlyThemes.get_theme_config(theme_name)
        layout_config = theme_config["layout"]
        
        fig.update_layout(
            paper_bgcolor=layout_config["paper_bgcolor"],
            plot_bgcolor=layout_config["plot_bgcolor"],
            font=layout_config["font"],
            colorway=layout_config["colorway"],
            title_font=layout_config["title"]["font"],
            legend=layout_config["legend"],
            xaxis=dict(
                gridcolor=layout_config["grid"]["color"],
                zerolinecolor=layout_config["grid"]["color"]
            ),
            yaxis=dict(
                gridcolor=layout_config["grid"]["color"],
                zerolinecolor=layout_config["grid"]["color"]
            )
        )
        
        return fig
    
    @staticmethod
    def create_themed_scatter(
        x, y, 
        theme_name: str = "dark_cyber",
        title: str = "THGNN Visualization",
        hover_data: Optional[Dict] = None,
        color_column: Optional[str] = None,
        size_column: Optional[str] = None
    ) -> go.Figure:
        """Create themed scatter plot"""
        
        theme_config = THGNNPlotlyThemes.get_theme_config(theme_name)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=12,
                color=theme_config["layout"]["colorway"][0],
                line=dict(width=2, color=theme_config["layout"]["font"]["color"])
            ),
            hovertemplate="<b>%{text}</b><br>X: %{x}<br>Y: %{y}<extra></extra>"
        ))
        
        fig = THGNNPlotlyThemes.apply_theme_to_figure(fig, theme_name)
        fig.update_layout(title=title)
        
        return fig
    
    @staticmethod
    def create_themed_heatmap(
        z_data,
        x_labels,
        y_labels,
        theme_name: str = "dark_cyber",
        title: str = "Risk Heatmap"
    ) -> go.Figure:
        """Create themed heatmap"""
        
        theme_config = THGNNPlotlyThemes.get_theme_config(theme_name)
        
        # Choose colorscale based on theme
        colorscales = {
            "dark_cyber": "Reds",
            "light_professional": "Blues", 
            "cyber_red": "Reds",
            "security_blue": "Blues",
            "matrix_green": "Greens"
        }
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale=colorscales.get(theme_name, "Reds"),
            hoverongaps=False,
            hovertemplate="<b>%{y} → %{x}</b><br>Risk: %{z}<extra></extra>"
        ))
        
        fig = THGNNPlotlyThemes.apply_theme_to_figure(fig, theme_name)
        fig.update_layout(title=title)
        
        return fig
    
    @staticmethod
    def create_themed_timeline(
        timestamps,
        values,
        theme_name: str = "dark_cyber",
        title: str = "Threat Timeline",
        line_name: str = "Risk Score"
    ) -> go.Figure:
        """Create themed timeline plot"""
        
        theme_config = THGNNPlotlyThemes.get_theme_config(theme_name)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines+markers',
            name=line_name,
            line=dict(
                color=theme_config["layout"]["colorway"][0],
                width=3
            ),
            marker=dict(
                size=8,
                color=theme_config["layout"]["colorway"][1],
                line=dict(width=2, color=theme_config["layout"]["font"]["color"])
            ),
            hovertemplate="<b>%{fullData.name}</b><br>Time: %{x}<br>Value: %{y}<extra></extra>"
        ))
        
        fig = THGNNPlotlyThemes.apply_theme_to_figure(fig, theme_name)
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=line_name
        )
        
        return fig
    
    @staticmethod
    def create_themed_network_graph(
        node_trace,
        edge_trace,
        theme_name: str = "dark_cyber",
        title: str = "Network Graph"
    ) -> go.Figure:
        """Create themed network graph"""
        
        theme_config = THGNNPlotlyThemes.get_theme_config(theme_name)
        
        fig = go.Figure()
        
        # Add edge traces
        if isinstance(edge_trace, list):
            for trace in edge_trace:
                fig.add_trace(trace)
        else:
            fig.add_trace(edge_trace)
        
        # Add node traces
        if isinstance(node_trace, list):
            for trace in node_trace:
                fig.add_trace(trace)
        else:
            fig.add_trace(node_trace)
        
        fig = THGNNPlotlyThemes.apply_theme_to_figure(fig, theme_name)
        
        fig.update_layout(
            title=title,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Interactive network visualization - hover and click to explore",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color=theme_config["layout"]["font"]["color"], size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig

def test_advanced_themes():
    """Test all advanced themes"""
    
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    print("Testing Advanced THGNN Plotly Themes...")
    
    # Sample data
    x_data = np.random.randn(50)
    y_data = np.random.randn(50)
    timestamps = pd.date_range(start=datetime.now(), periods=24, freq='H')
    values = np.random.randint(20, 100, 24)
    
    heatmap_data = np.random.randint(0, 100, (5, 5))
    x_labels = ['Web', 'DB', 'DC', 'File', 'Mail']
    y_labels = ['Zone A', 'Zone B', 'Zone C', 'Zone D', 'Zone E']
    
    themes = ["dark_cyber", "light_professional", "cyber_red", "security_blue", "matrix_green"]
    
    for theme in themes:
        print(f"Testing theme: {theme}")
        
        # Test scatter plot
        fig1 = THGNNPlotlyThemes.create_themed_scatter(
            x_data, y_data, 
            theme_name=theme,
            title=f"Scatter Plot - {theme}"
        )
        
        # Test heatmap
        fig2 = THGNNPlotlyThemes.create_themed_heatmap(
            heatmap_data, x_labels, y_labels,
            theme_name=theme,
            title=f"Risk Heatmap - {theme}"
        )
        
        # Test timeline
        fig3 = THGNNPlotlyThemes.create_themed_timeline(
            timestamps, values,
            theme_name=theme,
            title=f"Threat Timeline - {theme}"
        )
        
        # Save test outputs
        from pathlib import Path
        output_dir = Path('outputs/visualizations/themes')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig1.write_html(str(output_dir / f'scatter_{theme}.html'))
        fig2.write_html(str(output_dir / f'heatmap_{theme}.html'))
        fig3.write_html(str(output_dir / f'timeline_{theme}.html'))
        
        print(f" {theme} theme tested successfully")
    
    print(" All advanced themes tested successfully!")

if __name__ == '__main__':
    test_advanced_themes()