#!/usr/bin/env python3
"""
THGNN Predictions Page - Integrated with Upload Data
Displays threat predictions from uploaded security logs.

Author: THGNN Expert
Date: January 10, 2026
Version: 2.0 - Upload Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json

def show_predictions_interface():
    """Professional predictions interface with uploaded data integration"""
    
    st.markdown("# Threat Predictions")
    
    # Check if analysis data is available from upload
    if st.session_state.get('analysis_complete', False) and 'analysis_data' in st.session_state:
        analysis_data = st.session_state['analysis_data']
        predictions = analysis_data['predictions']
        threat_evaluation = analysis_data['threat_evaluation']
        
        # Display threat level banner
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {threat_evaluation['threat_color']}, {threat_evaluation['threat_color']}44); 
                    padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h3 style="color: white; margin: 0;">THREAT LEVEL: {threat_evaluation['overall_threat_level']} ({threat_evaluation['threat_score']}/100)</h3>
        </div>
        """, unsafe_allow_html=True)
        
    elif st.session_state.get('predictions_generated', False) and st.session_state.get('predictions'):
        predictions = st.session_state['predictions']
        threat_evaluation = None
        
    else:
        st.warning("""
        **No threat predictions available.**
        
        Please upload security logs first using the **Upload Logs** page to generate predictions.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upload Security Logs", width='stretch'):
                st.switch_page("pages/Upload_Logs.py")
        
        with col2:
            if st.button("Back to Dashboard", width='stretch'):
                st.switch_page("streamlit_app.py")
        
        return
    
    # Main predictions interface
    st.markdown(f"""
    **Advanced Threat Prediction Analysis**
    
    Displaying {len(predictions)} threat predictions generated from your uploaded security logs.
    Each prediction represents a potential future attack step based on current network activity.
    """)
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("### Prediction Filters")
        
        # Priority filter
        priority_options = list(set([p['priority'] for p in predictions]))
        priority_filter = st.multiselect(
            "Priority Levels",
            options=priority_options,
            default=priority_options,
            help="Filter predictions by priority level"
        )
        
        # Risk score filter
        risk_range = st.slider(
            "Risk Score Range",
            min_value=0,
            max_value=100,
            value=(30, 100),
            help="Filter by risk score range"
        )
        
        # Attack stage filter
        stages = list(set([p['attack_stage'] for p in predictions]))
        stage_filter = st.multiselect(
            "Attack Stages",
            options=stages,
            default=stages,
            help="Filter by attack stage"
        )
        
        # Results limit
        max_results = st.selectbox(
            "Max Results",
            options=[10, 20, 50, 100],
            index=1,
            help="Maximum number of predictions to display"
        )
        
        st.markdown("---")
        
        # Display options
        st.markdown("### Display Options")
        show_details = st.checkbox("Show Detailed Info", value=True)
        show_mitre = st.checkbox("Show MITRE Techniques", value=True)
        group_by_priority = st.checkbox("Group by Priority", value=False)
    
    # Filter predictions
    filtered_predictions = []
    for pred in predictions:
        if (pred['priority'] in priority_filter and
            risk_range[0] <= pred['risk_score'] <= risk_range[1] and
            pred['attack_stage'] in stage_filter):
            filtered_predictions.append(pred)
    
    # Sort by risk score
    filtered_predictions.sort(key=lambda x: x['risk_score'], reverse=True)
    filtered_predictions = filtered_predictions[:max_results]
    
    if not filtered_predictions:
        st.warning("No predictions match the current filter criteria.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Filtered Results", len(filtered_predictions))
    
    with col2:
        critical_count = len([p for p in filtered_predictions if p['priority'] == 'CRITICAL'])
        st.metric("Critical Threats", critical_count)
    
    with col3:
        avg_risk = np.mean([p['risk_score'] for p in filtered_predictions])
        st.metric("Average Risk", f"{avg_risk:.1f}%")
    
    with col4:
        high_confidence = len([p for p in filtered_predictions if p['confidence'] > 0.8])
        st.metric("High Confidence", high_confidence)
    
    # Display predictions
    if group_by_priority:
        # Group by priority
        priority_groups = {}
        for pred in filtered_predictions:
            priority = pred['priority']
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(pred)
        
        # Display each priority group
        for priority in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if priority in priority_groups:
                st.markdown(f"### {priority} Priority ({len(priority_groups[priority])} threats)")
                
                for pred in priority_groups[priority]:
                    display_prediction_card(pred, show_details, show_mitre)
    
    else:
        # Display all predictions in order
        st.markdown(f"### Top {len(filtered_predictions)} Threat Predictions")
        
        for i, pred in enumerate(filtered_predictions, 1):
            display_prediction_card(pred, show_details, show_mitre, index=i)
    
    # Export options
    st.markdown("---")
    st.markdown("### Export Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export as JSON", width='stretch'):
            json_data = json.dumps(filtered_predictions, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"threat_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Export as CSV", width='stretch'):
            df = pd.DataFrame(filtered_predictions)
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"threat_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("View Visualizations", width='stretch'):
            st.switch_page("pages/3_Visualizations.py")

def get_priority_emoji(priority):
    """Get emoji for priority level"""
    return {
        'CRITICAL': 'CRITICAL',
        'HIGH': 'HIGH', 
        'MEDIUM': 'MEDIUM',
        'LOW': 'LOW'
    }.get(priority, 'UNKNOWN')

def display_prediction_card(pred, show_details=True, show_mitre=True, index=None):
    """Display individual prediction card"""
    
    # Priority color
    priority_colors = {
        'CRITICAL': '#d32f2f',
        'HIGH': '#f57c00',
        'MEDIUM': '#1976d2', 
        'LOW': '#388e3c'
    }
    
    color = priority_colors.get(pred['priority'], '#666666')
    emoji = get_priority_emoji(pred['priority'])
    
    # Create expandable card
    title = f"[{pred['priority']}] {pred['source_entity']} → {pred['target_entity']}"
    if index:
        title = f"{index}. {title}"
    
    with st.expander(f"{title} (Risk: {pred['risk_score']:.1f}%)", expanded=False):
        
        # Main info row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Risk Score**: {pred['risk_score']:.1f}%")
            st.markdown(f"**Confidence**: {pred['confidence']:.1%}")
            st.markdown(f"**Priority**: {pred['priority']}")
        
        with col2:
            st.markdown(f"**Edge Type**: {pred['edge_type']}")
            st.markdown(f"**Timestamp**: {pred['timestamp'][:19]}")
            st.markdown(f"**Attack Stage**: {pred['attack_stage']}")
        
        with col3:
            if show_mitre:
                st.markdown(f"**MITRE Technique**: {pred['mitre_technique']}")
            st.markdown(f"**Process**: {pred.get('process_involved', 'N/A')}")
            st.markdown(f"**Prediction ID**: {pred['prediction_id']}")
        
        if show_details:
            st.markdown("---")
            st.markdown(f"**Description**: {pred['description']}")
            
            # Risk assessment
            if pred['risk_score'] >= 80:
                st.error("**IMMEDIATE ACTION REQUIRED** - Critical threat detected")
            elif pred['risk_score'] >= 60:
                st.warning("**HIGH PRIORITY** - Investigate promptly")
            elif pred['risk_score'] >= 40:
                st.info("**MEDIUM PRIORITY** - Monitor closely")
            else:
                st.success("**LOW PRIORITY** - Routine monitoring")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(f"Investigate", key=f"investigate_{pred['prediction_id']}"):
                st.info("Investigation workflow would be triggered here")
        
        with col2:
            if st.button(f"Visualize", key=f"visualize_{pred['prediction_id']}"):
                st.info("Individual prediction visualization would be shown here")
        
        with col3:
            if st.button(f"Generate Report", key=f"report_{pred['prediction_id']}"):
                st.info("Detailed report would be generated here")

def main():
    """Main predictions page function"""
    show_predictions_interface()

if __name__ == "__main__":
    main()