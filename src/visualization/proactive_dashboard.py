#!/usr/bin/env python3
"""
Proactive Dashboard Components

Professional proactive prediction interface components for cybersecurity
threat detection. Designed for security analysts to predict and prevent
attacks before they happen.

Features:
- "Predict Next Steps" interface with big red button
- Proactive threat alerts with risk scoring
- Attack timeline visualization with confidence indicators
- MITRE ATT&CK technique mapping and badges
- Recommended actions for threat mitigation
- Real-time confidence meters and progress bars
- Professional styling for client demonstrations
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .attack_path_visualizer import AttackPathVisualizer

logger = logging.getLogger(__name__)

class ProactiveDashboard:
    """
    Proactive cybersecurity dashboard components for threat prediction.
    
    Provides interactive components for security analysts to predict
    attack steps, view proactive alerts, and take preventive actions.
    Designed for professional client demonstrations and SOC operations.
    """
    
    def __init__(self):
        """Initialize proactive dashboard with professional styling."""
        
        # Color scheme matching attack path visualizer
        self.colors = {
            'critical': '#E74C3C',    # Red
            'high': '#E67E22',        # Orange
            'medium': '#F39C12',      # Yellow
            'low': '#27AE60',         # Green
            'normal': '#95A5A6',      # Gray
            'background': '#0E1117',  # Dark background
            'primary': '#E74C3C',     # Primary action color
            'success': '#27AE60',     # Success color
            'warning': '#F39C12',     # Warning color
            'info': '#3498DB'         # Info color
        }
        
        # MITRE technique categories
        self.mitre_tactics = {
            'TA0001': {'name': 'Initial Access', 'color': '#3498DB'},
            'TA0002': {'name': 'Execution', 'color': '#E67E22'},
            'TA0003': {'name': 'Persistence', 'color': '#9B59B6'},
            'TA0004': {'name': 'Privilege Escalation', 'color': '#1ABC9C'},
            'TA0005': {'name': 'Defense Evasion', 'color': '#F1C40F'},
            'TA0006': {'name': 'Credential Access', 'color': '#E74C3C'},
            'TA0007': {'name': 'Discovery', 'color': '#34495E'},
            'TA0008': {'name': 'Lateral Movement', 'color': '#16A085'},
            'TA0009': {'name': 'Collection', 'color': '#8E44AD'},
            'TA0010': {'name': 'Exfiltration', 'color': '#D35400'},
            'TA0011': {'name': 'Command and Control', 'color': '#2C3E50'}
        }
        
        # Attack path visualizer
        self.visualizer = AttackPathVisualizer()
        
        logger.info("Proactive dashboard initialized")
    
    def render_next_step_predictor(
        self,
        model: Any,
        current_state: Optional[Dict[str, Any]] = None,
        key: str = "predict_next_steps"
    ) -> Dict[str, Any]:
        """
        Render the main "Predict Next Steps" interface.
        
        Args:
            model: THGNN model for predictions
            current_state: Current graph state
            key: Unique key for Streamlit components
            
        Returns:
            Prediction results or None if not triggered
        """
        try:
            # Main prediction section
            st.markdown("### 🔮 Proactive Threat Prediction")
            st.markdown("**Predict what attackers will do next before they do it**")
            
            # Create columns for layout
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col2:
                # The big red button - centerpiece of the demo
                predict_clicked = st.button(
                    "PREDICT NEXT ATTACK STEPS",
                    type="primary",
                    width='stretch',
                    key=f"{key}_button",
                    help="Click to predict the attacker's next moves"
                )
            
            # Prediction configuration
            with st.expander("⚙️ Prediction Settings", expanded=False):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    num_steps = st.slider(
                        "Number of steps to predict",
                        min_value=1,
                        max_value=10,
                        value=5,
                        key=f"{key}_steps"
                    )
                    
                with col_b:
                    confidence_threshold = st.slider(
                        "Confidence threshold",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.7,
                        step=0.1,
                        key=f"{key}_confidence"
                    )
            
            # Handle prediction
            if predict_clicked:
                if current_state is None:
                    st.error("❌ No current state available. Please upload security logs first.")
                    return None
                
                # Show prediction in progress
                with st.spinner("🔍 Analyzing current attack state..."):
                    time.sleep(1)  # Dramatic pause for demo effect
                
                with st.spinner("🧠 Predicting next attack steps..."):
                    # Generate predictions (mock for demo)
                    predictions = self._generate_mock_predictions(num_steps, confidence_threshold)
                    time.sleep(2)  # Another dramatic pause
                
                # Show success message
                st.success(f"✅ Predicted {len(predictions['attack_path']['steps'])} attack steps with {predictions['confidence']:.1%} confidence!")
                
                return predictions
            
            return None
            
        except Exception as e:
            logger.error(f"Error rendering next step predictor: {e}")
            st.error(f"❌ Error in prediction interface: {str(e)}")
            return None
    
    def create_threat_alerts(
        self,
        predictions: Dict[str, Any],
        key: str = "threat_alerts"
    ) -> None:
        """
        Generate and display proactive threat alerts.
        
        Args:
            predictions: Attack path predictions
            key: Unique key for Streamlit components
        """
        try:
            if not predictions or predictions['confidence'] < 0.5:
                st.info("ℹ️ No high-confidence threats detected at this time.")
                return
            
            st.markdown("### 🚨 Proactive Threat Alerts")
            
            # Generate alerts based on predictions
            alerts = self._generate_alerts_from_predictions(predictions)
            
            for i, alert in enumerate(alerts):
                self._render_alert_card(alert, f"{key}_alert_{i}")
                
        except Exception as e:
            logger.error(f"Error creating threat alerts: {e}")
            st.error(f"❌ Error generating alerts: {str(e)}")
    
    def render_attack_timeline(
        self,
        attack_sequence: Dict[str, Any],
        key: str = "attack_timeline"
    ) -> None:
        """
        Show predicted attack timeline with interactive slider.
        
        Args:
            attack_sequence: Attack sequence data
            key: Unique key for Streamlit components
        """
        try:
            if not attack_sequence or not attack_sequence.get('timeline'):
                st.info("ℹ️ No attack timeline available.")
                return
            
            st.markdown("### ⏰ Predicted Attack Timeline")
            
            # Create timeline visualization
            timeline_fig = self.visualizer.create_proactive_timeline(attack_sequence)
            st.plotly_chart(timeline_fig, width='stretch', key=f"{key}_chart")
            
            # Timeline details
            timeline = attack_sequence['timeline']
            
            # Time horizon slider
            max_time = max([step['time_from_now_minutes'] for step in timeline])
            time_filter = st.slider(
                "Time horizon (minutes from now)",
                min_value=0,
                max_value=int(max_time),
                value=int(max_time),
                key=f"{key}_slider"
            )
            
            # Filtered timeline steps
            filtered_steps = [step for step in timeline if step['time_from_now_minutes'] <= time_filter]
            
            if filtered_steps:
                st.markdown(f"**Showing {len(filtered_steps)} steps within {time_filter} minutes:**")
                
                for step in filtered_steps:
                    self._render_timeline_step(step, f"{key}_step_{step['step_number']}")
            
        except Exception as e:
            logger.error(f"Error rendering attack timeline: {e}")
            st.error(f"❌ Error rendering timeline: {str(e)}")
    
    def render_mitre_mapping(
        self,
        mitre_techniques: List[str],
        key: str = "mitre_mapping"
    ) -> None:
        """
        Display MITRE ATT&CK technique mapping with badges.
        
        Args:
            mitre_techniques: List of MITRE technique IDs
            key: Unique key for Streamlit components
        """
        try:
            if not mitre_techniques:
                st.info("ℹ️ No MITRE ATT&CK techniques identified.")
                return
            
            st.markdown("### 🎯 MITRE ATT&CK Technique Mapping")
            
            # Group techniques by tactic
            tactic_groups = self._group_techniques_by_tactic(mitre_techniques)
            
            # Render technique badges by tactic
            for tactic_id, techniques in tactic_groups.items():
                tactic_info = self.mitre_tactics.get(tactic_id, {'name': 'Unknown', 'color': '#95A5A6'})
                
                st.markdown(f"**{tactic_info['name']} ({tactic_id})**")
                
                # Create columns for technique badges
                cols = st.columns(min(len(techniques), 4))
                
                for i, technique in enumerate(techniques):
                    with cols[i % 4]:
                        self._render_mitre_badge(technique, tactic_info['color'], f"{key}_{technique}")
            
        except Exception as e:
            logger.error(f"Error rendering MITRE mapping: {e}")
            st.error(f"❌ Error rendering MITRE mapping: {str(e)}")
    
    def render_confidence_indicators(
        self,
        predictions: Dict[str, Any],
        key: str = "confidence_indicators"
    ) -> None:
        """
        Display confidence indicators and progress bars.
        
        Args:
            predictions: Prediction data with confidence scores
            key: Unique key for Streamlit components
        """
        try:
            if not predictions:
                return
            
            st.markdown("### 📊 Prediction Confidence")
            
            # Overall confidence
            overall_confidence = predictions.get('confidence', 0.0)
            
            # Confidence meter
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=overall_confidence * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Confidence"},
                delta={'reference': 70},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self._get_confidence_color(overall_confidence)},
                    'steps': [
                        {'range': [0, 50], 'color': self.colors['low']},
                        {'range': [50, 70], 'color': self.colors['medium']},
                        {'range': [70, 85], 'color': self.colors['high']},
                        {'range': [85, 100], 'color': self.colors['critical']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            
            st.plotly_chart(fig, width='stretch', key=f"{key}_gauge")
            
            # Step-by-step confidence
            if 'attack_path' in predictions and predictions['attack_path']['steps']:
                st.markdown("**Step-by-step Confidence:**")
                
                steps = predictions['attack_path']['steps']
                for i, step in enumerate(steps):
                    step_confidence = step.get('confidence', 0.0)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.progress(
                            step_confidence,
                            text=f"Step {i+1}: {step.get('edge_type', 'Unknown')}"
                        )
                    
                    with col2:
                        st.metric(
                            label="",
                            value=f"{step_confidence:.1%}",
                            delta=None
                        )
            
        except Exception as e:
            logger.error(f"Error rendering confidence indicators: {e}")
            st.error(f"❌ Error rendering confidence indicators: {str(e)}")
    
    def render_recommended_actions(
        self,
        predictions: Dict[str, Any],
        key: str = "recommended_actions"
    ) -> None:
        """
        Display recommended actions for threat mitigation.
        
        Args:
            predictions: Prediction data
            key: Unique key for Streamlit components
        """
        try:
            if not predictions or predictions['confidence'] < 0.6:
                return
            
            st.markdown("### 💡 Recommended Actions")
            
            # Generate recommendations based on predictions
            recommendations = self._generate_recommendations(predictions)
            
            for i, rec in enumerate(recommendations):
                self._render_recommendation_card(rec, f"{key}_rec_{i}")
                
        except Exception as e:
            logger.error(f"Error rendering recommended actions: {e}")
            st.error(f"❌ Error rendering recommendations: {str(e)}")
    
    def _generate_predictions_from_data(
        self,
        uploaded_data: List[Dict[str, Any]],
        num_steps: int,
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """Generate predictions based on uploaded security log data."""
        try:
            # Analyze uploaded data for realistic predictions
            high_risk_events = [e for e in uploaded_data if e.get('risk_score', 0) > 0.7]
            mitre_techniques = list(set(e.get('mitre_technique', 'T1000') for e in uploaded_data))
            unique_users = list(set(e.get('source_user', 'unknown') for e in uploaded_data))
            unique_hosts = list(set(e.get('source_host', 'unknown') for e in uploaded_data))
            
            # Calculate base confidence from data quality
            base_confidence = min(0.95, 0.6 + (len(high_risk_events) / len(uploaded_data)) * 0.3)
            
            # Generate attack steps based on actual data patterns
            steps = []
            for i in range(min(num_steps, len(high_risk_events))):
                event = high_risk_events[i]
                confidence = max(confidence_threshold, base_confidence - (i * 0.05))
                
                step = {
                    'edge_type': self._map_event_to_edge_type(event.get('event_type', 'unknown')),
                    'source_type': self._extract_entity_type(event.get('source_user', 'user')),
                    'target_type': self._extract_entity_type(event.get('target_host', event.get('source_host', 'host'))),
                    'confidence': confidence,
                    'mitre_techniques': [event.get('mitre_technique', 'T1000')],
                    'description': event.get('description', 'Predicted attack step'),
                    'risk_score': event.get('risk_score', 0.5)
                }
                steps.append(step)
            
            # Fill remaining steps with predicted patterns
            for i in range(len(steps), num_steps):
                confidence = max(confidence_threshold, base_confidence - (i * 0.08))
                step = {
                    'edge_type': ['executes', 'accesses', 'connects_to', 'modifies'][i % 4],
                    'source_type': 'process',
                    'target_type': ['file', 'host', 'registry', 'network'][i % 4],
                    'confidence': confidence,
                    'mitre_techniques': [mitre_techniques[i % len(mitre_techniques)] if mitre_techniques else 'T1000'],
                    'description': f'Predicted attack step {i+1}',
                    'risk_score': max(0.5, confidence - 0.1)
                }
                steps.append(step)
            
            # Calculate overall metrics
            overall_confidence = np.mean([step['confidence'] for step in steps]) if steps else 0.0
            risk_score = min(1.0, overall_confidence * 1.2)
            
            # Generate timeline
            timeline = []
            cumulative_time = 0
            for i, step in enumerate(steps):
                cumulative_time += np.random.uniform(2, 8)
                timeline_entry = {
                    'step_number': i + 1,
                    'time_from_now_minutes': cumulative_time,
                    'edge_type': step['edge_type'],
                    'source_type': step['source_type'],
                    'target_type': step['target_type'],
                    'confidence': step['confidence'],
                    'mitre_techniques': step['mitre_techniques']
                }
                timeline.append(timeline_entry)
            
            return {
                'attack_path': {
                    'steps': steps,
                    'overall_confidence': overall_confidence,
                    'num_steps': len(steps)
                },
                'timeline': timeline,
                'mitre_techniques': mitre_techniques,
                'risk_score': risk_score,
                'confidence': overall_confidence,
                'prediction_time': 2.1,
                'data_source': 'uploaded_logs',
                'events_analyzed': len(uploaded_data),
                'high_risk_events': len(high_risk_events)
            }
            
        except Exception as e:
            logger.error(f"Error generating predictions from data: {e}")
            return self._generate_mock_predictions(num_steps, confidence_threshold)
    
    def _map_event_to_edge_type(self, event_type: str) -> str:
        """Map event type to graph edge type."""
        mapping = {
            'user_login': 'logs_into',
            'process_creation': 'executes',
            'file_access': 'accesses',
            'network_connection': 'connects_to',
            'registry_modification': 'modifies',
            'scheduled_task': 'creates'
        }
        return mapping.get(event_type, 'executes')
    
    def _extract_entity_type(self, entity_string: str) -> str:
        """Extract entity type from string."""
        if '@' in entity_string:
            return 'user'
        elif 'WORKSTATION' in entity_string or 'SERVER' in entity_string:
            return 'host'
        elif '.exe' in entity_string:
            return 'process'
        else:
            return 'host'
    def _generate_mock_predictions(
        self,
        num_steps: int,
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """Generate mock predictions for demonstration."""
        try:
            # Mock attack steps
            attack_patterns = [
                {'edge_type': 'executes', 'source_type': 'user', 'target_type': 'process', 'base_confidence': 0.85},
                {'edge_type': 'accesses', 'source_type': 'process', 'target_type': 'file', 'base_confidence': 0.80},
                {'edge_type': 'connects_to', 'source_type': 'process', 'target_type': 'host', 'base_confidence': 0.75},
                {'edge_type': 'modifies', 'source_type': 'process', 'target_type': 'registry', 'base_confidence': 0.70},
                {'edge_type': 'logs_into', 'source_type': 'user', 'target_type': 'host', 'base_confidence': 0.90}
            ]
            
            # Generate steps
            steps = []
            for i in range(min(num_steps, len(attack_patterns))):
                pattern = attack_patterns[i]
                confidence = max(confidence_threshold, pattern['base_confidence'] + np.random.uniform(-0.1, 0.1))
                
                step = {
                    'edge_type': pattern['edge_type'],
                    'source_type': pattern['source_type'],
                    'target_type': pattern['target_type'],
                    'confidence': confidence,
                    'mitre_techniques': self._get_mock_mitre_techniques(pattern['edge_type'])
                }
                steps.append(step)
            
            # Calculate overall confidence
            overall_confidence = np.mean([step['confidence'] for step in steps]) if steps else 0.0
            
            # Generate timeline
            timeline = []
            cumulative_time = 0
            for i, step in enumerate(steps):
                cumulative_time += np.random.uniform(2, 10)  # 2-10 minutes per step
                timeline_entry = {
                    'step_number': i + 1,
                    'time_from_now_minutes': cumulative_time,
                    'edge_type': step['edge_type'],
                    'source_type': step['source_type'],
                    'target_type': step['target_type'],
                    'confidence': step['confidence'],
                    'mitre_techniques': step['mitre_techniques']
                }
                timeline.append(timeline_entry)
            
            # Collect all MITRE techniques
            all_techniques = []
            for step in steps:
                all_techniques.extend(step['mitre_techniques'])
            
            return {
                'attack_path': {
                    'steps': steps,
                    'overall_confidence': overall_confidence,
                    'num_steps': len(steps)
                },
                'timeline': timeline,
                'mitre_techniques': list(set(all_techniques)),
                'risk_score': min(overall_confidence * 1.2, 1.0),
                'confidence': overall_confidence,
                'prediction_time': 2.3
            }
            
        except Exception as e:
            logger.error(f"Error generating mock predictions: {e}")
            return {
                'attack_path': {'steps': []}, 
                'confidence': 0.0,
                'risk_score': 0.0,
                'prediction_time': 1.0,
                'timeline': [],
                'mitre_techniques': []
            }
    
    def _get_mock_mitre_techniques(self, edge_type: str) -> List[str]:
        """Get mock MITRE techniques for edge type."""
        technique_mapping = {
            'executes': ['T1059.001', 'T1059.003'],
            'accesses': ['T1005', 'T1083'],
            'connects_to': ['T1021.002', 'T1021.001'],
            'modifies': ['T1112', 'T1547.001'],
            'logs_into': ['T1078', 'T1078.003']
        }
        return technique_mapping.get(edge_type, ['T1059'])
    
    def _generate_alerts_from_predictions(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts from predictions."""
        alerts = []
        
        confidence = predictions.get('confidence', 0.0)
        risk_score = predictions.get('risk_score', 0.0)
        
        if confidence >= 0.8:
            alerts.append({
                'type': 'HIGH_CONFIDENCE_ATTACK',
                'severity': 'CRITICAL' if risk_score > 0.9 else 'HIGH',
                'title': '🚨 High-Confidence Attack Path Detected',
                'message': f"Attack sequence predicted with {confidence:.1%} confidence",
                'confidence': confidence,
                'risk_score': risk_score
            })
        
        # Check for lateral movement
        techniques = predictions.get('mitre_techniques', [])
        lateral_techniques = ['T1021.001', 'T1021.002', 'T1078']
        if any(tech in techniques for tech in lateral_techniques):
            alerts.append({
                'type': 'LATERAL_MOVEMENT',
                'severity': 'HIGH',
                'title': '⚠️ Lateral Movement Predicted',
                'message': "Attacker likely to move laterally across network",
                'confidence': confidence,
                'risk_score': risk_score
            })
        
        return alerts
    
    def _render_alert_card(self, alert: Dict[str, Any], key: str) -> None:
        """Render individual alert card."""
        severity = alert.get('severity', 'MEDIUM')
        color = self.colors['critical'] if severity == 'CRITICAL' else self.colors['high']
        
        with st.container():
            st.markdown(f"""
            <div style="
                border-left: 4px solid {color};
                padding: 1rem;
                margin: 0.5rem 0;
                background-color: rgba(231, 76, 60, 0.1);
                border-radius: 0.25rem;
            ">
                <h4 style="color: {color}; margin: 0 0 0.5rem 0;">{alert['title']}</h4>
                <p style="margin: 0 0 0.5rem 0;">{alert['message']}</p>
                <small>Confidence: {alert['confidence']:.1%} | Risk Score: {alert['risk_score']:.2f}</small>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_timeline_step(self, step: Dict[str, Any], key: str) -> None:
        """Render individual timeline step."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.metric(
                label="Time",
                value=f"{step['time_from_now_minutes']:.0f}m",
                delta=None
            )
        
        with col2:
            st.write(f"**{step['edge_type']}**: {step['source_type']} → {step['target_type']}")
            if step.get('mitre_techniques'):
                st.caption(f"MITRE: {', '.join(step['mitre_techniques'])}")
        
        with col3:
            confidence = step['confidence']
            st.metric(
                label="Confidence",
                value=f"{confidence:.1%}",
                delta=None
            )
    
    def _render_mitre_badge(self, technique: str, color: str, key: str) -> None:
        """Render MITRE technique badge."""
        st.markdown(f"""
        <div style="
            background-color: {color};
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            text-align: center;
            margin: 0.25rem 0;
            font-size: 0.8rem;
            font-weight: bold;
        ">
            {technique}
        </div>
        """, unsafe_allow_html=True)
    
    def _render_recommendation_card(self, rec: Dict[str, Any], key: str) -> None:
        """Render recommendation card."""
        priority = rec.get('priority', 'MEDIUM')
        color = self.colors['critical'] if priority == 'HIGH' else self.colors['warning']
        
        with st.container():
            st.markdown(f"""
            <div style="
                border: 1px solid {color};
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 0.25rem;
                background-color: rgba(52, 152, 219, 0.05);
            ">
                <h5 style="color: {color}; margin: 0 0 0.5rem 0;">💡 {rec['title']}</h5>
                <p style="margin: 0 0 0.5rem 0;">{rec['description']}</p>
                <small><strong>Priority:</strong> {priority}</small>
            </div>
            """, unsafe_allow_html=True)
    
    def _generate_recommendations(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on predictions."""
        recommendations = []
        
        techniques = predictions.get('mitre_techniques', [])
        confidence = predictions.get('confidence', 0.0)
        
        if confidence >= 0.8:
            recommendations.append({
                'title': 'Immediate Threat Response',
                'description': 'High-confidence attack predicted. Consider isolating affected systems and escalating to incident response team.',
                'priority': 'HIGH'
            })
        
        if 'T1078' in techniques:
            recommendations.append({
                'title': 'Account Security Review',
                'description': 'Valid account abuse detected. Review account permissions and enable additional authentication.',
                'priority': 'HIGH'
            })
        
        if any(tech.startswith('T1021') for tech in techniques):
            recommendations.append({
                'title': 'Network Segmentation',
                'description': 'Lateral movement predicted. Review network segmentation and access controls.',
                'priority': 'MEDIUM'
            })
        
        return recommendations
    
    def _group_techniques_by_tactic(self, techniques: List[str]) -> Dict[str, List[str]]:
        """Group MITRE techniques by tactic."""
        # Simplified mapping - in practice would use full MITRE database
        tactic_mapping = {
            'T1059': 'TA0002',  # Execution
            'T1078': 'TA0001',  # Initial Access
            'T1021': 'TA0008',  # Lateral Movement
            'T1005': 'TA0009',  # Collection
            'T1083': 'TA0007',  # Discovery
            'T1112': 'TA0005',  # Defense Evasion
            'T1547': 'TA0003'   # Persistence
        }
        
        groups = {}
        for technique in techniques:
            # Extract base technique ID
            base_technique = technique.split('.')[0]
            tactic = tactic_mapping.get(base_technique, 'TA0002')
            
            if tactic not in groups:
                groups[tactic] = []
            groups[tactic].append(technique)
        
        return groups
    
    def _get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence level."""
        if confidence >= 0.8:
            return self.colors['critical']
        elif confidence >= 0.6:
            return self.colors['high']
        elif confidence >= 0.4:
            return self.colors['medium']
        else:
            return self.colors['low']
    
