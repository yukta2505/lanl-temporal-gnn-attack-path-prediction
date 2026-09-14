#!/usr/bin/env python3
"""
THGNN Upload & Analysis Pipeline - Complete End-to-End Processing
Upload logs → Automatic processing → Comprehensive threat analysis dashboard

Author: THGNN Expert
Date: January 10, 2026
Version: 2.0 - Complete Pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_uploaded_file(uploaded_file, file_format):
    """Process uploaded log file with REAL ML - NO MOCK DATA!"""
    
    try:
        # Validate file size
        file_content = uploaded_file.getvalue()
        if len(file_content) == 0:
            return {
                'success': False,
                'error': 'File is empty. Please upload a file with security log data.'
            }
        
        # Decode file content
        try:
            content_str = file_content.decode("utf-8")
        except UnicodeDecodeError:
            return {
                'success': False,
                'error': 'File encoding error. Please ensure the file is UTF-8 encoded.'
            }
        
        # Parse file content based on format
        events = []
        
        if file_format == "JSON":
            try:
                content = json.loads(content_str)
                if isinstance(content, list):
                    events = content
                elif isinstance(content, dict):
                    events = [content]
                else:
                    return {
                        'success': False,
                        'error': 'JSON file must contain an object or array of security events.'
                    }
            except json.JSONDecodeError as e:
                return {
                    'success': False,
                    'error': f'Invalid JSON format: {str(e)}. Please check your JSON syntax.'
                }
        
        elif file_format == "CSV":
            try:
                df = pd.read_csv(uploaded_file)
                if df.empty:
                    return {
                        'success': False,
                        'error': 'CSV file is empty. Please upload a file with security log data.'
                    }
                events = df.to_dict('records')
            except Exception as e:
                return {
                    'success': False,
                    'error': f'CSV parsing error: {str(e)}. Please check your CSV format.'
                }
        
        else:  # JSONL
            try:
                lines = content_str.strip().split('\n')
                for line_num, line in enumerate(lines, 1):
                    if line.strip():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            return {
                                'success': False,
                                'error': f'Invalid JSON on line {line_num}: {str(e)}'
                            }
            except Exception as e:
                return {
                    'success': False,
                    'error': f'JSONL parsing error: {str(e)}'
                }
        
        # Validate events
        if not events:
            return {
                'success': False,
                'error': 'No valid security events found in the file. Please check the file format and content.'
            }
        
        # Validate event structure
        required_fields = ['timestamp', 'event_type']
        
        valid_events = []
        for i, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            
            # Check for required fields
            missing_required = [field for field in required_fields if field not in event]
            if missing_required:
                # Try alternative field names
                if 'timestamp' not in event:
                    for alt in ['time', 'datetime', 'event_time', '@timestamp']:
                        if alt in event:
                            event['timestamp'] = event[alt]
                            break
                
                if 'event_type' not in event:
                    for alt in ['type', 'event_name', 'EventID', 'event_id']:
                        if alt in event:
                            event['event_type'] = event[alt]
                            break
            
            # Add default values for missing fields
            if 'timestamp' not in event:
                event['timestamp'] = datetime.now().isoformat()
            
            if 'event_type' not in event:
                event['event_type'] = 'unknown'
            
            valid_events.append(event)
        
        if not valid_events:
            return {
                'success': False,
                'error': 'No valid security events found. Events should contain at least timestamp and event_type fields.'
            }
        
        events = valid_events
        
        # 🚀 REAL ML PROCESSING - NO MOCK DATA!
        logger.info(f"🔥 Processing {len(events)} events with REAL ML PIPELINE...")
        
        try:
            # Import and use REAL ML processor
            import sys
            from pathlib import Path
            
            # Add parent directory to path
            app_dir = Path(__file__).parent.parent
            if str(app_dir) not in sys.path:
                sys.path.insert(0, str(app_dir))
            
            from real_ml_processor import get_real_ml_processor
            
            ml_processor = get_real_ml_processor()
            result = ml_processor.process_events_real_ml(events)
            
            logger.info("✅ REAL ML processing completed successfully!")
            return result
            
        except Exception as ml_error:
            logger.error(f"REAL ML processing failed: {ml_error}")
            import traceback
            logger.error(traceback.format_exc())
            
            # FALLBACK: If ML fails, return error (NO MOCK DATA!)
            return {
                'success': False,
                'error': f'ML processing failed: {str(ml_error)}. Please check your data format and try again.'
            }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Processing error: {str(e)}. Please check your file format and try again.'
        }

# REMOVED ALL FAKE FUNCTIONS - NOW USING REAL ML ONLY!
# All threat analysis is now powered by actual THGNN model

def show_comprehensive_analysis(analysis_data):
    """Display comprehensive threat analysis dashboard"""
    
    st.markdown("# Comprehensive Threat Analysis Dashboard")
    
    # Overall threat assessment
    evaluation = analysis_data.get('threat_evaluation', {})
    predictions = analysis_data.get('predictions', [])
    stats = analysis_data.get('stats', {})
    
    # Threat level banner
    threat_level = evaluation.get('overall_threat_level', 'UNKNOWN')
    threat_color = evaluation.get('threat_color', '#666666')
    threat_score = evaluation.get('threat_score', 0)
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {threat_color}, {threat_color}44); 
                padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h2 style="color: white; margin: 0;">THREAT LEVEL: {threat_level}</h2>
        <h3 style="color: white; margin: 5px 0;">Overall Risk Score: {threat_score}/100</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics - USE REAL DATA
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_predictions = stats.get('total_predictions', len(predictions))
        high_risk = stats.get('high_risk_count', len([p for p in predictions if p.get('risk_score', 0) > 70]))
        st.metric(
            "Total Threats",
            total_predictions,
            delta=f"+{high_risk} high-risk"
        )
    
    with col2:
        critical_count = stats.get('critical_count', len([p for p in predictions if p.get('priority') == 'CRITICAL']))
        st.metric(
            "Critical Alerts",
            critical_count,
            delta="Immediate action required" if critical_count > 0 else "No critical threats"
        )
    
    with col3:
        forecast_data = analysis_data.get('forecast_data', {})
        forecasted_attacks = len(forecast_data.get('forecasted_attacks', []))
        st.metric(
            "Forecasted Attacks",
            forecasted_attacks,
            delta="Next 3 time windows"
        )
    
    with col4:
        attack_paths = analysis_data.get('attack_paths', [])
        st.metric(
            "Attack Paths",
            len(attack_paths),
            delta="Multi-step sequences"
        )
    
    # Main analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Threat Overview", 
        "Detailed Predictions", 
        "Attack Forecast", 
        "Visualizations",
        "Attack Paths"
    ])
    
    with tab1:
        st.markdown("### Threat Assessment Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Risk Distribution")
            risk_dist = evaluation.get('risk_distribution', {})
            for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                count = risk_dist.get(level, 0)
                st.markdown(f"**{level}**: {count} threats")
        
        with col2:
            st.markdown("#### Top MITRE Techniques")
            techniques = evaluation.get('mitre_techniques', {})
            sorted_techniques = sorted(techniques.items(), key=lambda x: x[1], reverse=True)[:5]
            for tech, count in sorted_techniques:
                st.markdown(f"• **{tech}**: {count} predictions")
        
        st.markdown("#### Immediate Recommendations")
        recommendations = evaluation.get('recommendations', [])
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    with tab2:
        st.markdown("### Detailed Threat Predictions")
        
        # Get predictions safely
        predictions = analysis_data.get('predictions', [])
        
        if not predictions:
            st.info("No predictions available")
            return
        
        # Create DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            # Get available priorities from data
            available_priorities = list(predictions_df['priority'].unique()) if 'priority' in predictions_df.columns else ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            priority_filter = st.multiselect(
                "Priority Filter",
                options=available_priorities,
                default=available_priorities[:2] if len(available_priorities) >= 2 else available_priorities
            )
        with col2:
            min_risk = st.slider("Minimum Risk Score", 0, 100, 50)
        with col3:
            max_results = st.selectbox("Max Results", [10, 20, 50, 100], index=1)
        
        # Filter and display predictions
        if 'priority' in predictions_df.columns:
            filtered_df = predictions_df[
                (predictions_df['priority'].isin(priority_filter)) &
                (predictions_df['risk_score'] >= min_risk)
            ].head(max_results)
        else:
            # Fallback if priority column missing
            filtered_df = predictions_df[
                predictions_df['risk_score'] >= min_risk
            ].head(max_results)
        
        if len(filtered_df) == 0:
            st.info("No predictions match the current filters")
        else:
            for _, pred in filtered_df.iterrows():
                # Safely get values with defaults
                priority = pred.get('priority', 'MEDIUM')
                risk_score = float(pred.get('risk_score', 0))
                confidence = float(pred.get('confidence', 0))
                source = pred.get('source_entity', 'unknown')
                target = pred.get('target_entity', 'unknown')
                
                with st.expander(f"THREAT: {source} → {target} (Risk: {risk_score:.1f}%)"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"**Priority**: {priority}")
                        st.markdown(f"**Risk Score**: {risk_score:.1f}%")
                        st.markdown(f"**Confidence**: {confidence:.1%}")
                    
                    with col2:
                        st.markdown(f"**Attack Stage**: {pred.get('attack_stage', 'Unknown')}")
                        st.markdown(f"**MITRE Technique**: {pred.get('mitre_technique', 'Unknown')}")
                        st.markdown(f"**Edge Type**: {pred.get('edge_type', 'unknown')}")
                    
                    with col3:
                        st.markdown(f"**Process**: {pred.get('process_involved', 'N/A')}")
                        timestamp = pred.get('timestamp', '')
                        if timestamp and len(timestamp) > 19:
                            timestamp = timestamp[:19]
                        st.markdown(f"**Timestamp**: {timestamp}")
                        st.markdown(f"**Description**: {pred.get('description', 'No description')}")
    
    with tab3:
        st.markdown("### Multi-Step Attack Forecast")
        
        forecast = analysis_data.get('forecast_data', {})
        
        if not forecast or not forecast.get('forecasted_attacks'):
            st.info("No forecast data available")
        else:
            horizon = forecast.get('horizon_steps', 3)
            st.markdown(f"**Forecast Horizon**: {horizon} time windows (15 minutes)")
            
            # Forecasted attacks by step
            forecasted_attacks = forecast.get('forecasted_attacks', [])
            for step in range(1, horizon + 1):
                step_attacks = [a for a in forecasted_attacks if a.get('step') == step]
                
                if step_attacks:
                    st.markdown(f"#### Step {step} - Predicted Attacks")
                    
                    for attack in step_attacks:
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            source = attack.get('source', 'unknown')
                            target = attack.get('target', 'unknown')
                            st.markdown(f"**{source} → {target}**")
                            st.caption(attack.get('description', 'No description'))
                        
                        with col2:
                            risk = attack.get('risk_score', 0)
                            risk_color = "🔴 HIGH RISK" if risk > 70 else "🟠 MEDIUM RISK" if risk > 50 else "🟢 LOW RISK"
                            st.markdown(f"{risk_color} {risk:.1f}%")
                        
                        with col3:
                            st.code(attack.get('technique', 'Unknown'))
            
            # Critical targets
            st.markdown("#### Critical Targets")
            critical_targets = forecast.get('critical_targets', [])
            if critical_targets:
                targets_df = pd.DataFrame(critical_targets)
                st.dataframe(targets_df, use_container_width=True)
            else:
                st.info("No critical targets identified")
    
    with tab4:
        st.markdown("### Threat Visualizations")
        
        visualizations = analysis_data.get('visualizations', {})
        
        if not visualizations:
            st.info("No visualizations available")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                risk_fig = visualizations.get('risk_distribution')
                if risk_fig:
                    st.plotly_chart(risk_fig, use_container_width=True)
                else:
                    st.info("Risk distribution chart not available")
            
            with col2:
                timeline_fig = visualizations.get('threat_timeline')
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)
                else:
                    st.info("Threat timeline not available")
    
    with tab5:
        st.markdown("### Attack Path Analysis")
        
        attack_paths = analysis_data.get('attack_paths', [])
        
        if not attack_paths:
            st.info("No multi-step attack paths detected in current analysis.")
        else:
            for path in attack_paths:
                path_id = path.get('path_id', 'unknown')
                description = path.get('description', 'No description')
                total_risk = path.get('total_risk', 0)
                
                with st.expander(f"ATTACK PATH: {description} (Total Risk: {total_risk:.1f})"):
                    st.markdown(f"**Attack Type**: {path.get('attack_type', 'Unknown')}")
                    
                    steps = path.get('steps', [])
                    for step in steps:
                        step_num = step.get('step', 0)
                        from_entity = step.get('from', 'unknown')
                        to_entity = step.get('to', 'unknown')
                        technique = step.get('technique', 'Unknown')
                        risk = step.get('risk', 0)
                        desc = step.get('description', 'No description')
                        
                        st.markdown(f"**Step {step_num}**: {from_entity} → {to_entity}")
                        st.markdown(f"   • Technique: {technique}")
                        st.markdown(f"   • Risk: {risk:.1f}%")
                        st.markdown(f"   • Description: {desc}")

def main():
    """Main upload and analysis page"""
    
    st.set_page_config(
        page_title="THGNN - Upload & Analysis",
        page_icon="⚙",
        layout="wide"
    )
    
    # Check if analysis is already complete
    if st.session_state.get('analysis_complete', False):
        show_comprehensive_analysis(st.session_state['analysis_data'])
        
        # Option to upload new files
        if st.sidebar.button("Upload New Files", width='stretch'):
            # Clear session state
            for key in ['analysis_complete', 'analysis_data']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        return
    
    # Upload interface
    st.markdown("# Upload Security Logs")
    st.markdown("Upload your security logs for comprehensive threat analysis and attack prediction.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a security log file",
        type=['json', 'jsonl', 'csv'],
        help="Upload security logs in JSON, JSONL, or CSV format"
    )
    
    if uploaded_file is not None:
        # Show file info
        st.success(f"File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        # Process button
        if st.button("Analyze Threats", type="primary", width='stretch'):
            
            # Processing pipeline
            with st.spinner("Processing logs and analyzing threats..."):
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: File processing
                status_text.text("Processing uploaded file...")
                progress_bar.progress(20)
                time.sleep(1)
                
                # Step 2: Threat detection
                status_text.text("Detecting threats and anomalies...")
                progress_bar.progress(40)
                time.sleep(1)
                
                # Step 3: Attack prediction
                status_text.text("Predicting future attack steps...")
                progress_bar.progress(60)
                time.sleep(1)
                
                # Step 4: Path analysis
                status_text.text("Analyzing attack paths...")
                progress_bar.progress(80)
                time.sleep(1)
                
                # Step 5: Generate analysis
                status_text.text("Generating comprehensive analysis...")
                progress_bar.progress(100)
                
                # Determine file format
                file_format = uploaded_file.name.split('.')[-1].upper()
                
                # Process file
                result = process_uploaded_file(uploaded_file, file_format)
                
                if result['success']:
                    # Store analysis data
                    st.session_state['analysis_data'] = result
                    st.session_state['analysis_complete'] = True
                    
                    status_text.text("Analysis complete!")
                    st.success("Threat analysis completed successfully!")
                    st.balloons()
                    
                    # Auto-refresh to show analysis
                    time.sleep(1)
                    st.rerun()
                    
                else:
                    st.error(f"Processing failed: {result['error']}")

if __name__ == "__main__":
    main()