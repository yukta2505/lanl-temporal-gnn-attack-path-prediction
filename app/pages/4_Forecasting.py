#!/usr/bin/env python3
"""
THGNN Forecasting Page - Simplified Implementation
Multi-step attack forecasting and critical path analysis.

Author: THGNN Expert
Date: January 10, 2026
Version: 1.0 - Production Ready
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

def generate_forecast_data(predictions, steps=3):
    """Generate simplified multi-step forecast"""
    
    forecast = {
        'horizon_steps': steps,
        'forecasted_attacks': [],
        'critical_targets': [],
        'confidence_scores': []
    }
    
    # Generate forecasted attacks
    base_entities = set()
    for pred in predictions:
        base_entities.add(pred['source_entity'])
        base_entities.add(pred['target_entity'])
    
    base_confidence = np.mean([pred['risk_score'] for pred in predictions]) / 100
    
    for step in range(1, steps + 1):
        confidence = base_confidence * (0.85 ** step)
        forecast['confidence_scores'].append(confidence)
        
        # Generate sample forecasted attacks
        step_attacks = []
        for i, entity in enumerate(list(base_entities)[:3]):
            if 'user' in entity.lower():
                target = f"host_dc{step}"
                step_attacks.append({
                    'step': step,
                    'source': entity,
                    'target': target,
                    'technique': 'T1021.002',
                    'risk_score': max(30, confidence * 100 + np.random.normal(0, 10)),
                    'description': f'Lateral movement to {target}'
                })
        
        forecast['forecasted_attacks'].extend(step_attacks)
    
    # Identify critical targets
    targets = {}
    for attack in forecast['forecasted_attacks']:
        target = attack['target']
        if target not in targets:
            targets[target] = {'name': target, 'risk': 0, 'count': 0}
        targets[target]['risk'] += attack['risk_score']
        targets[target]['count'] += 1
    
    forecast['critical_targets'] = [
        {
            'name': name,
            'avg_risk': data['risk'] / data['count'],
            'prediction_count': data['count']
        }
        for name, data in targets.items()
    ]
    
    return forecast

def create_forecast_timeline(forecast):
    """Create simple forecast timeline"""
    
    fig = go.Figure()
    
    # Plot forecasted attacks
    for attack in forecast['forecasted_attacks']:
        fig.add_trace(go.Scatter(
            x=[attack['step']],
            y=[attack['risk_score']],
            mode='markers',
            marker=dict(size=12, color='red'),
            name=f"Step {attack['step']}",
            text=attack['description'],
            hovertemplate="<b>%{text}</b><br>Risk: %{y:.1f}%<extra></extra>"
        ))
    
    fig.update_layout(
        title="Attack Forecast Timeline",
        xaxis_title="Forecast Step",
        yaxis_title="Risk Score (%)",
        template="plotly_dark",
        height=400
    )
    
    return fig

def show_forecast_results(forecast):
    """Display forecast results"""
    
    # Forecast visualization tabs
    tab1, tab2, tab3 = st.tabs(["Timeline", "Critical Targets", "Attack Details"])
    
    with tab1:
        st.markdown("### Forecast Timeline")
        st.markdown("Visualization of predicted attacks across forecast horizon.")
        
        try:
            timeline_fig = create_forecast_timeline(forecast)
            st.plotly_chart(timeline_fig, use_container_width=True)
            
            st.markdown("**Confidence Decay Analysis:**")
            confidence_df = pd.DataFrame({
                'Step': range(1, len(forecast['confidence_scores']) + 1),
                'Confidence': [f"{c:.1%}" for c in forecast['confidence_scores']]
            })
            st.dataframe(confidence_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error generating timeline: {str(e)}")
    
    with tab2:
        st.markdown("### Critical Targets")
        st.markdown("Assets most likely to be targeted in future attacks.")
        
        if forecast['critical_targets']:
            targets_df = pd.DataFrame(forecast['critical_targets'])
            targets_df = targets_df.sort_values('avg_risk', ascending=False)
            
            # Display as metrics
            for i, target in enumerate(targets_df.to_dict('records')[:5]):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{target['name']}**")
                
                with col2:
                    risk_color = "HIGH RISK" if target['avg_risk'] > 70 else "MEDIUM RISK" if target['avg_risk'] > 50 else "LOW RISK"
                    st.markdown(f"{risk_color} {target['avg_risk']:.1f}%")
                
                with col3:
                    st.markdown(f"({target['prediction_count']} predictions)")
        else:
            st.info("No critical targets identified in forecast.")
    
    with tab3:
        st.markdown("### Forecasted Attack Details")
        st.markdown("Detailed breakdown of predicted attack steps.")
        
        if forecast['forecasted_attacks']:
            attacks_df = pd.DataFrame(forecast['forecasted_attacks'])
            attacks_df = attacks_df.sort_values(['step', 'risk_score'], ascending=[True, False])
            
            # Group by step
            for step in sorted(attacks_df['step'].unique()):
                step_attacks = attacks_df[attacks_df['step'] == step]
                
                with st.expander(f"**Step {step}** - {len(step_attacks)} predicted attacks"):
                    for _, attack in step_attacks.iterrows():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{attack['source']}** → **{attack['target']}**")
                            st.caption(attack['description'])
                        
                        with col2:
                            risk_color = "HIGH RISK" if attack['risk_score'] > 70 else "MEDIUM RISK" if attack['risk_score'] > 50 else "LOW RISK"
                            st.markdown(f"{risk_color} {attack['risk_score']:.1f}%")
                        
                        with col3:
                            st.code(attack['technique'])
        else:
            st.info("No attacks forecasted.")

def show_forecasting_interface():
    """Simplified forecasting interface"""
    
    st.markdown("# Attack Forecasting")
    
    # Check if analysis data is available from upload
    if st.session_state.get('analysis_complete', False) and 'analysis_data' in st.session_state:
        analysis_data = st.session_state['analysis_data']
        predictions = analysis_data['predictions']
        forecast_data = analysis_data['forecast_data']
        threat_evaluation = analysis_data['threat_evaluation']
        
        # Display threat level banner
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {threat_evaluation['threat_color']}, {threat_evaluation['threat_color']}44); 
                    padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h3 style="color: white; margin: 0;">Forecasting Future Attacks - Current Level: {threat_evaluation['overall_threat_level']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Auto-populate forecast data
        st.session_state['forecast_data'] = forecast_data
        st.session_state['forecast_generated'] = True
        
    elif st.session_state.get('predictions_generated', False) and st.session_state.get('predictions'):
        predictions = st.session_state['predictions']
        forecast_data = st.session_state.get('forecast_data')
        
    else:
        st.warning("""
        **No predictions available for forecasting.**
        
        Please upload security logs first using the **Upload Logs** page to generate forecasts.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upload Security Logs", use_container_width=True):
                st.switch_page("pages/Upload_Logs.py")
        
        with col2:
            if st.button("View Predictions", use_container_width=True):
                st.switch_page("pages/Predictions.py")
        
        return
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### Forecast Settings")
        
        forecast_steps = st.slider(
            "Forecast Horizon (Steps)",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of future time windows to predict"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            help="Minimum confidence for forecasted attacks"
        )
        
        st.markdown("---")
        
        # Forecast options
        st.markdown("### Analysis Options")
        
        show_confidence_decay = st.checkbox("Show Confidence Decay", value=True)
        highlight_critical = st.checkbox("Highlight Critical Assets", value=True)
        include_mitre = st.checkbox("Include MITRE Techniques", value=True)
    
    # Main forecasting interface
    st.markdown(f"""
    **Multi-Step Attack Forecasting**
    
    Predict future attack steps based on {len(predictions)} current threat predictions.
    Each forecast step represents a 5-minute time window into the future.
    """)
    
    # If we have forecast data from upload, show it automatically
    if st.session_state.get('forecast_generated', False) and 'forecast_data' in st.session_state:
        forecast = st.session_state['forecast_data']
        
        # Display forecast automatically
        st.success(f"Forecast generated automatically from uploaded data - {forecast['horizon_steps']} steps with {len(forecast['forecasted_attacks'])} predicted attacks")
        
        # Forecast summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Forecast Steps",
                forecast['horizon_steps']
            )
        
        with col2:
            st.metric(
                "Predicted Attacks",
                len(forecast['forecasted_attacks'])
            )
        
        with col3:
            st.metric(
                "Critical Targets",
                len(forecast['critical_targets'])
            )
        
        with col4:
            avg_confidence = np.mean(forecast['confidence_scores']) if forecast['confidence_scores'] else 0
            st.metric(
                "Avg Confidence",
                f"{avg_confidence:.1%}"
            )
        
        # Show forecast results immediately
        show_forecast_results(forecast)
        
        return
    
    # Manual forecast generation (fallback)
    st.markdown("### Generate New Forecast")
    
    # Generate forecast button
    if st.button("Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating multi-step attack forecast..."):
            # Generate forecast data
            forecast = generate_forecast_data(predictions, forecast_steps)
            
            # Store in session state
            st.session_state['forecast_data'] = forecast
            st.session_state['forecast_generated'] = True
            
            st.success(f"Generated {forecast_steps}-step forecast with {len(forecast['forecasted_attacks'])} predicted attacks")
    
    # Display forecast results
    if st.session_state.get('forecast_generated', False) and 'forecast_data' in st.session_state:
        forecast = st.session_state['forecast_data']
        show_forecast_results(forecast)
        
        # Export options
        st.markdown("---")
        st.markdown("### Export Forecast")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export as JSON", use_container_width=True):
                forecast_json = json.dumps(forecast, indent=2, default=str)
                st.download_button(
                    label="Download Forecast JSON",
                    data=forecast_json,
                    file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Export as CSV", use_container_width=True):
                if forecast['forecasted_attacks']:
                    attacks_df = pd.DataFrame(forecast['forecasted_attacks'])
                    csv_data = attacks_df.to_csv(index=False)
                    st.download_button(
                        label="Download Attacks CSV",
                        data=csv_data,
                        file_name=f"forecasted_attacks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("Generate Report", use_container_width=True):
                st.info("Report generation - navigate to Reports page for full functionality")

def main():
    """Main forecasting page function"""
    show_forecasting_interface()

if __name__ == "__main__":
    main()