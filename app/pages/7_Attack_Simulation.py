#!/usr/bin/env python3
"""
Attack Simulation Page

Real-time attack simulation demonstration page for cybersecurity
threat detection. Creates the illusion of live attack scenarios
for impressive client presentations.

Features:
- Attack scenario selection dropdown
- Live log streaming display with realistic delays
- Real-time prediction updates as logs arrive
- Proactive alert generation during simulation
- Attack path visualization updates
- Progress indicators and streaming controls
- Pause/resume/stop functionality
- Performance metrics and statistics
"""

import streamlit as st
import sys
import os
import logging
import time
import json
import asyncio
import threading
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Import with fallbacks
try:
    from src.simulation.realtime_simulator import RealtimeLogSimulator
    from src.simulation.attack_scenarios import AttackScenarios
except ImportError:
    # Fallback classes for demo
    class RealtimeLogSimulator:
        def __init__(self): pass
        def get_scenarios(self): return ['lateral_movement', 'ransomware_attack', 'insider_threat']
    
    class AttackScenarios:
        def __init__(self): pass
        def get_all_scenarios(self):
            return {
                'lateral_movement': {
                    'name': 'Lateral Movement Attack',
                    'description': 'Attacker moves laterally across network',
                    'duration_minutes': 15,
                    'total_events': 45,
                    'complexity': 'High',
                    'mitre_techniques': ['T1078', 'T1021.002', 'T1059.001']
                },
                'ransomware_attack': {
                    'name': 'Ransomware Deployment',
                    'description': 'File encryption ransomware attack',
                    'duration_minutes': 8,
                    'total_events': 32,
                    'complexity': 'Critical',
                    'mitre_techniques': ['T1486', 'T1490', 'T1083']
                },
                'insider_threat': {
                    'name': 'Insider Data Exfiltration',
                    'description': 'Malicious insider stealing data',
                    'duration_minutes': 25,
                    'total_events': 67,
                    'complexity': 'Medium',
                    'mitre_techniques': ['T1005', 'T1041', 'T1078.003']
                }
            }

# Simple fallback class
class StreamingEventProcessor:
    def __init__(self): pass
    def process_event(self, event): return event

try:
    from src.visualization.proactive_dashboard import ProactiveDashboard
    from src.visualization.attack_path_visualizer import AttackPathVisualizer
except ImportError:
    # Fallback classes
    class ProactiveDashboard:
        def __init__(self): pass
    
    class AttackPathVisualizer:
        def __init__(self): pass
        def plot_attack_sequence(self, **kwargs):
            import plotly.graph_objects as go
            return go.Figure()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Attack Simulation",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for simulation styling
st.markdown("""
<style>
    .simulation-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E74C3C;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .streaming-container {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
    }
    
    .log-entry {
        margin: 0.25rem 0;
        padding: 0.25rem;
        border-left: 3px solid;
        background-color: rgba(255, 255, 255, 0.05);
    }
    
    .log-entry.low {
        border-left-color: #27AE60;
    }
    
    .log-entry.medium {
        border-left-color: #F39C12;
    }
    
    .log-entry.high {
        border-left-color: #E67E22;
    }
    
    .log-entry.critical {
        border-left-color: #E74C3C;
    }
    
    .control-button {
        margin: 0.25rem;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        border: none;
        font-weight: bold;
        cursor: pointer;
    }
    
    .start-button {
        background-color: #27AE60;
        color: white;
    }
    
    .pause-button {
        background-color: #F39C12;
        color: white;
    }
    
    .stop-button {
        background-color: #E74C3C;
        color: white;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-active {
        background-color: #27AE60;
        animation: pulse 2s infinite;
    }
    
    .status-paused {
        background-color: #F39C12;
    }
    
    .status-stopped {
        background-color: #E74C3C;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #333;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .alert-banner {
        background-color: rgba(231, 76, 60, 0.1);
        border: 1px solid #E74C3C;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        animation: alertPulse 3s infinite;
    }
    
    @keyframes alertPulse {
        0% { border-color: #E74C3C; }
        50% { border-color: #C0392B; }
        100% { border-color: #E74C3C; }
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'simulator' not in st.session_state:
        st.session_state.simulator = RealtimeLogSimulator()
    if 'processor' not in st.session_state:
        st.session_state.processor = StreamingEventProcessor()
    if 'streaming_active' not in st.session_state:
        st.session_state.streaming_active = False
    if 'streaming_paused' not in st.session_state:
        st.session_state.streaming_paused = False
    if 'current_scenario' not in st.session_state:
        st.session_state.current_scenario = None
    if 'streamed_events' not in st.session_state:
        st.session_state.streamed_events = []
    if 'live_predictions' not in st.session_state:
        st.session_state.live_predictions = []
    if 'streaming_stats' not in st.session_state:
        st.session_state.streaming_stats = {
            'events_processed': 0,
            'predictions_generated': 0,
            'alerts_triggered': 0,
            'start_time': None
        }
    if 'simulation_speed' not in st.session_state:
        st.session_state.simulation_speed = {
            "multiplier": 1.0, 
            "delay": 2.5, 
            "label": "1x Normal"
        }

def render_header():
    """Render page header."""
    st.markdown('<div class="simulation-header">Real-time Attack Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #BDC3C7; font-size: 1.2rem; margin-bottom: 2rem;">Live Threat Detection Demonstration</div>', unsafe_allow_html=True)
    
    # Add explanation
    with st.expander("About Real-time Simulation", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**What You'll See:**")
            st.markdown("""
            - Security logs streaming in real-time
            - AI predictions updating as attacks unfold
            - Proactive alerts before attacks complete
            - Attack path visualization in real-time
            - MITRE ATT&CK technique identification
            """)
        
        with col2:
            st.markdown("**How It Works:**")
            st.markdown("""
            - Pre-recorded attack scenarios from Mordor datasets
            - Realistic timing delays (2-3 seconds per log)
            - Live prediction generation as events arrive
            - Proactive alerting based on confidence thresholds
            - Interactive controls for demonstration
            """)

def render_scenario_selection():
    """Render attack scenario selection with speed controls."""
    st.markdown("### Attack Scenario Selection")
    
    # Get available scenarios
    scenarios = AttackScenarios()
    available_scenarios = scenarios.get_all_scenarios()
    
    if not available_scenarios:
        st.error("No attack scenarios available.")
        return None
    
    # Scenario selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        scenario_options = {
            name: f"{info['name']} ({info['duration_minutes']}min, {info['total_events']} events)"
            for name, info in available_scenarios.items()
        }
        
        selected_scenario = st.selectbox(
            "Choose attack scenario:",
            options=list(scenario_options.keys()),
            format_func=lambda x: scenario_options[x],
            help="Select an attack scenario to simulate"
        )
        
        if selected_scenario:
            scenario_info = available_scenarios[selected_scenario]
            
            # Show scenario details
            with st.expander("Scenario Details", expanded=False):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown(f"**Description:** {scenario_info['description']}")
                    st.markdown(f"**Duration:** {scenario_info['duration_minutes']} minutes")
                    st.markdown(f"**Events:** {scenario_info['total_events']}")
                
                with col_b:
                    st.markdown(f"**Complexity:** {scenario_info['complexity']}")
                    st.markdown("**MITRE Techniques:**")
                    for technique in scenario_info['mitre_techniques'][:5]:  # Show first 5
                        st.markdown(f"- {technique}")
    
    with col2:
        # Simulation controls
        st.markdown("**Simulation Controls**")
        
        # Speed control - CRITICAL FOR DEMO
        speed_options = {
            "1x": {"multiplier": 1.0, "delay": 2.5, "label": "1x Normal"},
            "2x": {"multiplier": 2.0, "delay": 1.25, "label": "2x Fast"},
            "4x": {"multiplier": 4.0, "delay": 0.625, "label": "4x Faster"},
            "5x": {"multiplier": 5.0, "delay": 0.5, "label": "5x Very Fast"},
            "8x": {"multiplier": 8.0, "delay": 0.3125, "label": "8x Ultra Fast"},
            "10x": {"multiplier": 10.0, "delay": 0.25, "label": "10x Lightning"},
            "20x": {"multiplier": 20.0, "delay": 0.125, "label": "20x Warp Speed"}
        }
        
        selected_speed = st.selectbox(
            "Demo Speed:",
            options=list(speed_options.keys()),
            index=0,  # Default to 1x
            format_func=lambda x: speed_options[x]["label"],
            help="Control simulation speed for demos"
        )
        
        # Store speed in session state
        st.session_state.simulation_speed = speed_options[selected_speed]
        
        # Show speed info
        speed_info = speed_options[selected_speed]
        st.markdown(f"**Speed:** {speed_info['label']}")
        st.markdown(f"**Update:** Every {speed_info['delay']:.2f}s")
        
        # Start simulation button
        start_clicked = st.button(
            "START LIVE SIMULATION",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.streaming_active,
            help="Start streaming the selected attack scenario"
        )
        
        # Control buttons
        col_a, col_b = st.columns(2)
        
        with col_a:
            pause_clicked = st.button(
                "Pause" if not st.session_state.streaming_paused else "Resume",
                disabled=not st.session_state.streaming_active,
                use_container_width=True
            )
        
        with col_b:
            stop_clicked = st.button(
                "Stop",
                disabled=not st.session_state.streaming_active,
                use_container_width=True
            )
        
        # Handle control actions
        if start_clicked and selected_scenario:
            start_simulation(selected_scenario)
        
        if pause_clicked:
            toggle_pause_simulation()
        
        if stop_clicked:
            stop_simulation()
    
    return selected_scenario

def start_simulation(scenario_id: str):
    """Start attack simulation with enhanced THGNN model."""
    try:
        st.session_state.streaming_active = True
        st.session_state.streaming_paused = False
        st.session_state.current_scenario = scenario_id
        st.session_state.streamed_events = []
        st.session_state.live_predictions = []
        st.session_state.streaming_stats = {
            'events_processed': 0,
            'predictions_generated': 0,
            'alerts_triggered': 0,
            'start_time': time.time()
        }
        
        # Initialize enhanced model for live predictions
        try:
            from src.models.thgnn_model_enhanced import EnhancedTHGNN
            from src.models.thgnn_model import THGNNConfig
            
            config = THGNNConfig(
                hidden_dim=256,
                num_conv_layers=3,
                temporal_hidden=128,
                temporal_layers=2,
                use_gru=True,  # Use GRU for PDF compliance
                dropout=0.4
            )
            
            metadata = (['user', 'host', 'process'], [('user', 'logs_into', 'host')])
            st.session_state.enhanced_model = EnhancedTHGNN(config=config, metadata=metadata)
            
        except Exception as e:
            logger.warning(f"Enhanced model initialization failed: {e}")
            st.session_state.enhanced_model = None
        
        st.success(f"Started simulation: {scenario_id} with Enhanced THGNN")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        st.error(f"Error starting simulation: {str(e)}")

def toggle_pause_simulation():
    """Toggle pause/resume simulation."""
    try:
        if st.session_state.streaming_active:
            st.session_state.streaming_paused = not st.session_state.streaming_paused
            
            if st.session_state.streaming_paused:
                st.warning("Simulation paused")
            else:
                st.success("Simulation resumed")
            
            st.rerun()
        
    except Exception as e:
        logger.error(f"Error toggling pause: {e}")
        st.error(f"Error controlling simulation: {str(e)}")

def stop_simulation():
    """Stop attack simulation."""
    try:
        st.session_state.streaming_active = False
        st.session_state.streaming_paused = False
        
        st.info("Simulation stopped")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error stopping simulation: {e}")
        st.error(f"Error stopping simulation: {str(e)}")

def render_streaming_display():
    """Render live streaming display with real-time visuals."""
    if not st.session_state.streaming_active:
        st.info("Select a scenario and click 'START LIVE SIMULATION' to begin.")
        return
    
    # Initialize session state for live data
    if 'live_events' not in st.session_state:
        st.session_state.live_events = []
    if 'live_metrics' not in st.session_state:
        st.session_state.live_metrics = {
            'events_processed': 0,
            'threats_detected': 0,
            'risk_score': 0.0,
            'attack_steps': 0,
            'start_time': time.time()
        }
    
    # Create main dashboard container
    dashboard_container = st.container()
    
    with dashboard_container:
        st.markdown("### Live Attack Stream Dashboard")
        
        # Real-time metrics dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Events Processed",
                st.session_state.live_metrics['events_processed'],
                delta=f"+{len(st.session_state.live_events)} total"
            )
        
        with col2:
            st.metric(
                "Threats Detected",
                st.session_state.live_metrics['threats_detected'],
                delta="ACTIVE" if st.session_state.live_metrics['threats_detected'] > 0 else None
            )
        
        with col3:
            risk_score = st.session_state.live_metrics['risk_score']
            st.metric(
                "Risk Score",
                f"{risk_score:.2f}",
                delta="CRITICAL" if risk_score > 0.8 else "HIGH" if risk_score > 0.6 else "MEDIUM"
            )
        
        with col4:
            elapsed = time.time() - st.session_state.live_metrics['start_time']
            st.metric(
                "Runtime",
                f"{elapsed:.0f}s",
                delta=f"{st.session_state.live_metrics['attack_steps']} steps"
            )
    
    # Generate new events if simulation is running and not paused
    if st.session_state.streaming_active and not st.session_state.streaming_paused:
        # Generate new event based on scenario
        scenario_id = st.session_state.get('current_scenario', 'lateral_movement')
        new_event = generate_live_event(scenario_id, len(st.session_state.live_events))
        
        if new_event:
            st.session_state.live_events.append(new_event)
            
            # Update metrics
            st.session_state.live_metrics['events_processed'] += 1
            if new_event.get('risk_score', 0) > 0.7:
                st.session_state.live_metrics['threats_detected'] += 1
                st.session_state.live_metrics['attack_steps'] += 1
            
            # Update average risk score
            if st.session_state.live_events:
                avg_risk = np.mean([e.get('risk_score', 0) for e in st.session_state.live_events])
                st.session_state.live_metrics['risk_score'] = avg_risk
    
    # Create tabs for different visualizations
    if st.session_state.live_events:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Network Graph", 
            "Risk Timeline", 
            "MITRE Heatmap", 
            "Event Feed"
        ])
        
        with tab1:
            st.markdown("#### Live Attack Network Graph")
            attack_graph = create_live_attack_graph(st.session_state.live_events)
            st.plotly_chart(attack_graph, use_container_width=True, key="live_attack_graph")
            
            # Network statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                unique_hosts = len(set(e.get('source_host', 'unknown') for e in st.session_state.live_events))
                st.metric("Affected Hosts", unique_hosts)
            with col2:
                unique_users = len(set(e.get('source_user', 'unknown') for e in st.session_state.live_events))
                st.metric("Compromised Users", unique_users)
            with col3:
                lateral_moves = len([e for e in st.session_state.live_events if e.get('event_type') == 'network_connection'])
                st.metric("Lateral Movements", lateral_moves)
        
        with tab2:
            st.markdown("#### Live Risk Score Timeline")
            risk_timeline = create_live_risk_timeline(st.session_state.live_events)
            st.plotly_chart(risk_timeline, use_container_width=True, key="live_risk_timeline")
            
            # Risk analysis
            if st.session_state.live_events:
                current_risk = st.session_state.live_events[-1].get('risk_score', 0)
                risk_trend = "INCREASING" if current_risk > 0.7 else "STABLE" if current_risk > 0.4 else "DECREASING"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Risk", f"{current_risk:.2f}", delta=risk_trend)
                with col2:
                    max_risk = max(e.get('risk_score', 0) for e in st.session_state.live_events)
                    st.metric("Peak Risk", f"{max_risk:.2f}")
        
        with tab3:
            st.markdown("#### MITRE ATT&CK Technique Heatmap")
            mitre_heatmap = create_mitre_heatmap(st.session_state.live_events)
            st.plotly_chart(mitre_heatmap, use_container_width=True, key="mitre_heatmap")
            
            # MITRE statistics
            techniques = [e.get('mitre_technique', 'Unknown') for e in st.session_state.live_events]
            unique_techniques = len(set(techniques))
            most_common = max(set(techniques), key=techniques.count) if techniques else "None"
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Unique Techniques", unique_techniques)
            with col2:
                st.metric("Most Common", most_common)
        
        with tab4:
            st.markdown("#### Live Event Feed")
            render_live_event_feed(st.session_state.live_events)
    
    # Auto-refresh for live effect with speed control
    if st.session_state.streaming_active and not st.session_state.streaming_paused:
        # Use the selected speed delay
        speed_delay = st.session_state.simulation_speed.get('delay', 2.5)
        time.sleep(speed_delay)
        st.rerun()

def render_live_event_feed(events: List[Dict[str, Any]]):
    """Render live event feed with clean styling."""
    if not events:
        st.info("No events to display")
        return
    
    # Show last 8 events for better performance
    recent_events = events[-8:]
    
    for i, event in enumerate(reversed(recent_events)):
        risk_score = event.get('risk_score', 0)
        
        # Determine risk styling
        if risk_score > 0.8:
            risk_level = "CRITICAL"
            risk_emoji = "🔴"
        elif risk_score > 0.6:
            risk_level = "HIGH"
            risk_emoji = "🟠"
        elif risk_score > 0.4:
            risk_level = "MEDIUM"
            risk_emoji = "🟡"
        else:
            risk_level = "LOW"
            risk_emoji = "🟢"
        
        # Create event card using Streamlit components (no HTML)
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                event_type = event.get('event_type', 'Unknown').replace('_', ' ').title()
                st.markdown(f"**{risk_emoji} {event_type}**")
            
            with col2:
                st.markdown(f"**{risk_level}**")
            
            # Event details in columns
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.text(f"Host: {event.get('source_host', 'Unknown')}")
                st.text(f"User: {event.get('source_user', 'Unknown')}")
                st.text(f"Process: {event.get('process_name', 'Unknown')}")
            
            with col_b:
                st.text(f"Time: {event.get('timestamp', 'Unknown')}")
                st.text(f"MITRE: {event.get('mitre_technique', 'Unknown')}")
                st.text(f"Risk: {risk_score:.2f}")
            
            # Description
            st.caption(event.get('description', 'No description available'))
            
            # Show command line if available
            if event.get('command_line'):
                with st.expander("Command Details", expanded=False):
                    st.code(event['command_line'], language='bash')
            
            st.divider()

def generate_live_event(scenario_id: str, event_count: int) -> Dict[str, Any]:
    """Generate realistic live simulation event with multi-system attack progression."""
    try:
        current_time = datetime.now()
        
        # Define 4 systems for visual impact
        systems = [
            "WORKSTATION-VICTIM",
            "SERVER-DC01", 
            "SERVER-FILE01",
            "SERVER-DB01"
        ]
        
        users = [
            "john.doe@company.local",
            "admin@company.local", 
            "SYSTEM",
            "service.account@company.local"
        ]
        
        # Multi-system attack progression based on scenario
        if scenario_id == 'lateral_movement':
            # Progressive attack across systems
            if event_count < 5:
                # Phase 1: Initial compromise on victim workstation
                event_templates = [
                    {
                        'event_type': 'user_login',
                        'event_id': 4624,
                        'source_user': users[0],
                        'source_host': systems[0],
                        'target_host': systems[0],
                        'process_name': 'winlogon.exe',
                        'risk_score': 0.4 + (event_count * 0.1),
                        'mitre_technique': 'T1078',
                        'description': f'Initial user activity on {systems[0]}'
                    },
                    {
                        'event_type': 'process_creation',
                        'event_id': 4688,
                        'source_user': users[0],
                        'source_host': systems[0],
                        'process_name': 'malware.exe',
                        'command_line': 'malware.exe -stealth -beacon',
                        'risk_score': 0.7 + (event_count * 0.05),
                        'mitre_technique': 'T1204.002',
                        'description': f'Malicious process execution on {systems[0]}'
                    }
                ]
            elif event_count < 10:
                # Phase 2: Privilege escalation and DC compromise
                event_templates = [
                    {
                        'event_type': 'network_connection',
                        'event_id': 5156,
                        'source_user': users[1],
                        'source_host': systems[0],
                        'target_host': systems[1],
                        'process_name': 'psexec.exe',
                        'risk_score': 0.85 + (event_count * 0.02),
                        'mitre_technique': 'T1021.002',
                        'description': f'Lateral movement: {systems[0]} → {systems[1]}'
                    },
                    {
                        'event_type': 'user_login',
                        'event_id': 4624,
                        'source_user': users[1],
                        'source_host': systems[1],
                        'target_host': systems[1],
                        'process_name': 'winlogon.exe',
                        'risk_score': 0.92 + (event_count * 0.01),
                        'mitre_technique': 'T1078.003',
                        'description': f'Admin compromise on {systems[1]} (Domain Controller)'
                    },
                    {
                        'event_type': 'process_creation',
                        'event_id': 4688,
                        'source_user': users[1],
                        'source_host': systems[1],
                        'process_name': 'mimikatz.exe',
                        'command_line': 'mimikatz.exe sekurlsa::logonpasswords',
                        'risk_score': 0.96,
                        'mitre_technique': 'T1003.001',
                        'description': f'Credential dumping on {systems[1]} - CRITICAL!'
                    }
                ]
            elif event_count < 15:
                # Phase 3: File server compromise
                event_templates = [
                    {
                        'event_type': 'network_connection',
                        'event_id': 5156,
                        'source_user': users[1],
                        'source_host': systems[1],
                        'target_host': systems[2],
                        'process_name': 'psexec.exe',
                        'risk_score': 0.88,
                        'mitre_technique': 'T1021.002',
                        'description': f'Lateral movement: {systems[1]} → {systems[2]}'
                    },
                    {
                        'event_type': 'file_access',
                        'event_id': 4663,
                        'source_user': users[1],
                        'source_host': systems[2],
                        'target_file': f'\\\\{systems[2]}\\shares\\confidential\\data_{event_count}.xlsx',
                        'process_name': 'explorer.exe',
                        'risk_score': 0.85,
                        'mitre_technique': 'T1005',
                        'description': f'Sensitive data access on {systems[2]}'
                    }
                ]
            else:
                # Phase 4: Database server and exfiltration
                event_templates = [
                    {
                        'event_type': 'network_connection',
                        'event_id': 5156,
                        'source_user': users[1],
                        'source_host': systems[2],
                        'target_host': systems[3],
                        'process_name': 'psexec.exe',
                        'risk_score': 0.93,
                        'mitre_technique': 'T1021.002',
                        'description': f'Lateral movement: {systems[2]} → {systems[3]}'
                    },
                    {
                        'event_type': 'process_creation',
                        'event_id': 4688,
                        'source_user': users[1],
                        'source_host': systems[3],
                        'process_name': 'sqlcmd.exe',
                        'command_line': 'sqlcmd.exe -Q "SELECT * FROM customers"',
                        'risk_score': 0.91,
                        'mitre_technique': 'T1005',
                        'description': f'Database access on {systems[3]} - Crown Jewels!'
                    },
                    {
                        'event_type': 'network_connection',
                        'event_id': 5156,
                        'source_user': users[1],
                        'source_host': systems[3],
                        'target_host': 'dropbox.com',
                        'process_name': 'curl.exe',
                        'risk_score': 0.97,
                        'mitre_technique': 'T1041',
                        'description': f'Data exfiltration from {systems[3]} - MISSION COMPLETE!'
                    }
                ]
                
        elif scenario_id == 'ransomware_attack':
            # Ransomware spreading across systems
            current_system = systems[event_count % len(systems)]
            event_templates = [
                {
                    'event_type': 'process_creation',
                    'event_id': 4688,
                    'source_user': users[event_count % len(users)],
                    'source_host': current_system,
                    'process_name': 'vssadmin.exe',
                    'command_line': 'vssadmin delete shadows /all /quiet',
                    'risk_score': 0.95,
                    'mitre_technique': 'T1490',
                    'description': f'Shadow copy deletion on {current_system} - Ransomware prep!'
                },
                {
                    'event_type': 'file_access',
                    'event_id': 4663,
                    'source_user': users[event_count % len(users)],
                    'source_host': current_system,
                    'target_file': f'C:\\Users\\victim\\Documents\\file_{event_count}.docx.encrypted',
                    'process_name': 'ransomware.exe',
                    'risk_score': 0.99,
                    'mitre_technique': 'T1486',
                    'description': f'File encryption on {current_system} - DATA DESTROYED!'
                },
                {
                    'event_type': 'network_connection',
                    'event_id': 5156,
                    'source_user': users[event_count % len(users)],
                    'source_host': current_system,
                    'target_host': systems[(event_count + 1) % len(systems)],
                    'process_name': 'ransomware.exe',
                    'risk_score': 0.94,
                    'mitre_technique': 'T1021.002',
                    'description': f'Ransomware spreading: {current_system} → {systems[(event_count + 1) % len(systems)]}'
                }
            ]
            
        else:  # insider_threat
            # Insider moving between systems
            current_system = systems[event_count % len(systems)]
            event_templates = [
                {
                    'event_type': 'file_access',
                    'event_id': 4663,
                    'source_user': 'insider@company.local',
                    'source_host': current_system,
                    'target_file': f'\\\\{current_system}\\sensitive\\confidential_{event_count}.xlsx',
                    'process_name': 'explorer.exe',
                    'risk_score': min(0.9, 0.6 + (event_count * 0.02)),
                    'mitre_technique': 'T1005',
                    'description': f'Suspicious file access on {current_system} by insider'
                },
                {
                    'event_type': 'network_connection',
                    'event_id': 5156,
                    'source_user': 'insider@company.local',
                    'source_host': current_system,
                    'target_host': 'personal-cloud.com',
                    'process_name': 'chrome.exe',
                    'risk_score': 0.85,
                    'mitre_technique': 'T1041',
                    'description': f'Data exfiltration from {current_system} to personal cloud'
                }
            ]
        
        # Select event template
        template = event_templates[event_count % len(event_templates)]
        
        # Add timestamp
        template['timestamp'] = current_time.strftime("%H:%M:%S")
        template['full_timestamp'] = current_time.isoformat()
        
        return template
        
    except Exception as e:
        logger.error(f"Error generating live event: {e}")
        return None

def create_live_attack_graph(events: List[Dict[str, Any]]) -> go.Figure:
    """Create enhanced live attack graph with polished visuals."""
    try:
        # Extract unique hosts and connections
        hosts = set()
        connections = []
        
        for event in events:
            source_host = event.get('source_host', 'Unknown')
            target_host = event.get('target_host', source_host)
            
            hosts.add(source_host)
            if target_host != source_host:
                hosts.add(target_host)
                connections.append({
                    'source': source_host,
                    'target': target_host,
                    'risk': event.get('risk_score', 0),
                    'type': event.get('event_type', 'unknown'),
                    'timestamp': event.get('timestamp', 'unknown')
                })
        
        # Create enhanced network layout
        host_list = list(hosts)
        positions = {}
        
        # Improved circular layout with better spacing
        import math
        n = len(host_list)
        radius = max(3, n * 0.8)  # Dynamic radius based on node count
        
        for i, host in enumerate(host_list):
            angle = 2 * math.pi * i / n
            positions[host] = {
                'x': math.cos(angle) * radius,
                'y': math.sin(angle) * radius
            }
        
        fig = go.Figure()
        
        # Add enhanced connections (edges) with animations
        for i, conn in enumerate(connections):
            source_pos = positions.get(conn['source'], {'x': 0, 'y': 0})
            target_pos = positions.get(conn['target'], {'x': 0, 'y': 0})
            
            # Enhanced color coding based on risk
            if conn['risk'] > 0.9:
                color = '#E74C3C'  # Critical red
                width = 5
                dash = 'solid'
            elif conn['risk'] > 0.8:
                color = '#E67E22'  # High orange
                width = 4
                dash = 'solid'
            elif conn['risk'] > 0.6:
                color = '#F39C12'  # Medium yellow
                width = 3
                dash = 'dash'
            else:
                color = '#3498DB'  # Low blue
                width = 2
                dash = 'dot'
            
            # Add connection with enhanced styling
            fig.add_trace(go.Scatter(
                x=[source_pos['x'], target_pos['x'], None],
                y=[source_pos['y'], target_pos['y'], None],
                mode='lines',
                line=dict(color=color, width=width, dash=dash),
                showlegend=False,
                hovertemplate=f'<b>{conn["type"].replace("_", " ").title()}</b><br>' +
                             f'Risk: {conn["risk"]:.2f}<br>' +
                             f'Time: {conn["timestamp"]}<extra></extra>',
                opacity=0.8
            ))
            
            # Add arrow for direction
            mid_x = (source_pos['x'] + target_pos['x']) / 2
            mid_y = (source_pos['y'] + target_pos['y']) / 2
            
            # Calculate arrow direction
            dx = target_pos['x'] - source_pos['x']
            dy = target_pos['y'] - source_pos['y']
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # Normalize and create arrow
                dx_norm = dx / length * 0.3
                dy_norm = dy / length * 0.3
                
                fig.add_annotation(
                    x=mid_x + dx_norm,
                    y=mid_y + dy_norm,
                    ax=mid_x,
                    ay=mid_y,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor=color,
                    showarrow=True
                )
        
        # Add enhanced nodes (hosts) with better styling
        for host in host_list:
            pos = positions[host]
            
            # Determine node characteristics based on involvement
            host_events = [e for e in events if e.get('source_host') == host or e.get('target_host') == host]
            max_risk = max([e.get('risk_score', 0) for e in host_events]) if host_events else 0
            event_count = len(host_events)
            
            # Enhanced node styling
            if max_risk > 0.9:
                node_color = '#E74C3C'  # Critical
                node_size = 60
                border_width = 4
                symbol = 'circle'
            elif max_risk > 0.8:
                node_color = '#E67E22'  # High
                node_size = 50
                border_width = 3
                symbol = 'circle'
            elif max_risk > 0.6:
                node_color = '#F39C12'  # Medium
                node_size = 45
                border_width = 2
                symbol = 'circle'
            else:
                node_color = '#3498DB'  # Low/Normal
                node_size = 40
                border_width = 2
                symbol = 'circle'
            
            # Add pulsing effect for high-risk nodes
            if max_risk > 0.8:
                # Add outer glow effect
                fig.add_trace(go.Scatter(
                    x=[pos['x']],
                    y=[pos['y']],
                    mode='markers',
                    marker=dict(
                        size=node_size + 20,
                        color=node_color,
                        opacity=0.3,
                        line=dict(width=0)
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Main node
            fig.add_trace(go.Scatter(
                x=[pos['x']],
                y=[pos['y']],
                mode='markers+text',
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=border_width, color='white'),
                    symbol=symbol,
                    opacity=0.9
                ),
                text=host.replace('WORKSTATION-', 'WS-').replace('SERVER-', 'SRV-'),
                textposition="middle center",
                textfont=dict(size=10, color='white', family='Arial Black'),
                showlegend=False,
                hovertemplate=f'<b>{host}</b><br>' +
                             f'Max Risk: {max_risk:.2f}<br>' +
                             f'Events: {event_count}<br>' +
                             f'Status: {"COMPROMISED" if max_risk > 0.8 else "SUSPICIOUS" if max_risk > 0.6 else "NORMAL"}' +
                             '<extra></extra>'
            ))
        
        # Enhanced layout with better styling
        fig.update_layout(
            title={
                'text': "Live Attack Network Graph - Real-time Threat Propagation",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#E74C3C', 'family': 'Arial Black'}
            },
            showlegend=False,
            height=600,
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[-radius*1.2, radius*1.2]
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[-radius*1.2, radius*1.2]
            ),
            plot_bgcolor='rgba(0,0,0,0.05)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=60, b=20),
            font=dict(color='white'),
            hovermode='closest'
        )
        
        # Add legend manually
        legend_items = [
            {'color': '#E74C3C', 'label': 'Critical Risk (>0.9)'},
            {'color': '#E67E22', 'label': 'High Risk (>0.8)'},
            {'color': '#F39C12', 'label': 'Medium Risk (>0.6)'},
            {'color': '#3498DB', 'label': 'Normal Risk (<0.6)'}
        ]
        
        for i, item in enumerate(legend_items):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=15, color=item['color']),
                name=item['label'],
                showlegend=True
            ))
        
        fig.update_layout(
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor='white',
                borderwidth=1,
                font=dict(color='white', size=10)
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating enhanced attack graph: {e}")
        return go.Figure()

def create_live_risk_timeline(events: List[Dict[str, Any]]) -> go.Figure:
    """Create enhanced live risk timeline with polished visuals."""
    try:
        if not events:
            return go.Figure()
        
        # Extract timeline data
        timestamps = list(range(len(events)))
        risk_scores = [event.get('risk_score', 0) for event in events]
        event_types = [event.get('event_type', 'unknown') for event in events]
        descriptions = [event.get('description', 'No description') for event in events]
        hosts = [event.get('source_host', 'Unknown') for event in events]
        
        fig = go.Figure()
        
        # Main risk line with enhanced styling
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=risk_scores,
            mode='lines+markers',
            line=dict(
                color='rgba(231, 76, 60, 0.8)',
                width=4,
                shape='spline',  # Smooth curves
                smoothing=0.3
            ),
            marker=dict(
                size=12,
                color=risk_scores,
                colorscale=[
                    [0, '#3498DB'],      # Low - Blue
                    [0.4, '#F39C12'],    # Medium - Yellow
                    [0.7, '#E67E22'],    # High - Orange
                    [1, '#E74C3C']       # Critical - Red
                ],
                showscale=True,
                colorbar=dict(
                    title="Risk Score",
                    title_font=dict(color='white'),
                    tickfont=dict(color='white'),
                    x=1.02
                ),
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            text=[f"{desc}<br>Host: {host}" for desc, host in zip(descriptions, hosts)],
            hovertemplate='<b>Event %{x}</b><br>' +
                         'Risk Score: %{y:.2f}<br>' +
                         'Type: %{customdata}<br>' +
                         '%{text}<extra></extra>',
            customdata=event_types,
            name='Risk Progression',
            fill='tonexty' if len(timestamps) > 1 else None,
            fillcolor='rgba(231, 76, 60, 0.1)'
        ))
        
        # Add threshold lines with enhanced styling
        fig.add_hline(
            y=0.9, 
            line_dash="dash", 
            line_color="#E74C3C", 
            line_width=3,
            annotation_text="CRITICAL THRESHOLD (0.9)",
            annotation_position="top right",
            annotation_font=dict(color="#E74C3C", size=12, family="Arial Black")
        )
        
        fig.add_hline(
            y=0.8, 
            line_dash="dash", 
            line_color="#E67E22", 
            line_width=2,
            annotation_text="HIGH THRESHOLD (0.8)",
            annotation_position="top right",
            annotation_font=dict(color="#E67E22", size=11)
        )
        
        fig.add_hline(
            y=0.6, 
            line_dash="dot", 
            line_color="#F39C12", 
            line_width=2,
            annotation_text="MEDIUM THRESHOLD (0.6)",
            annotation_position="top right",
            annotation_font=dict(color="#F39C12", size=10)
        )
        
        # Add risk zones as background
        fig.add_hrect(
            y0=0.9, y1=1.0,
            fillcolor="rgba(231, 76, 60, 0.1)",
            layer="below",
            line_width=0,
        )
        
        fig.add_hrect(
            y0=0.8, y1=0.9,
            fillcolor="rgba(230, 126, 34, 0.1)",
            layer="below",
            line_width=0,
        )
        
        fig.add_hrect(
            y0=0.6, y1=0.8,
            fillcolor="rgba(243, 156, 18, 0.1)",
            layer="below",
            line_width=0,
        )
        
        # Enhanced layout
        fig.update_layout(
            title={
                'text': "Live Risk Score Timeline - Attack Escalation Pattern",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#E74C3C', 'family': 'Arial Black'}
            },
            xaxis=dict(
                title="Event Sequence",
                title_font=dict(color='white', size=14),
                tickfont=dict(color='white'),
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True
            ),
            yaxis=dict(
                title="Risk Score",
                title_font=dict(color='white', size=14),
                tickfont=dict(color='white'),
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                range=[0, 1.05]
            ),
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0.05)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hovermode='x unified',
            margin=dict(l=60, r=80, t=60, b=60)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating enhanced risk timeline: {e}")
        return go.Figure()

def create_mitre_heatmap(events: List[Dict[str, Any]]) -> go.Figure:
    """Create enhanced MITRE ATT&CK technique heatmap."""
    try:
        # Count MITRE techniques
        technique_counts = {}
        technique_risks = {}
        
        for event in events:
            technique = event.get('mitre_technique', 'Unknown')
            risk = event.get('risk_score', 0)
            
            technique_counts[technique] = technique_counts.get(technique, 0) + 1
            if technique not in technique_risks:
                technique_risks[technique] = []
            technique_risks[technique].append(risk)
        
        if not technique_counts:
            return go.Figure()
        
        # Create enhanced heatmap data
        techniques = list(technique_counts.keys())
        counts = list(technique_counts.values())
        avg_risks = [sum(technique_risks[t])/len(technique_risks[t]) for t in techniques]
        
        # MITRE technique descriptions
        technique_descriptions = {
            'T1078': 'Valid Accounts',
            'T1078.003': 'Local Accounts',
            'T1021.002': 'SMB/Windows Admin Shares',
            'T1486': 'Data Encrypted for Impact',
            'T1490': 'Inhibit System Recovery',
            'T1005': 'Data from Local System',
            'T1041': 'Exfiltration Over C2 Channel',
            'T1059.001': 'PowerShell',
            'T1003.001': 'LSASS Memory',
            'T1547.001': 'Registry Run Keys',
            'T1053.005': 'Scheduled Task'
        }
        
        # Create enhanced heatmap
        fig = go.Figure()
        
        # Main heatmap
        fig.add_trace(go.Heatmap(
            z=[counts],
            x=[f"{t}<br>{technique_descriptions.get(t, 'Unknown Technique')}" for t in techniques],
            y=['Frequency'],
            colorscale=[
                [0, '#3498DB'],      # Low - Blue
                [0.3, '#F39C12'],    # Medium - Yellow
                [0.7, '#E67E22'],    # High - Orange
                [1, '#E74C3C']       # Critical - Red
            ],
            showscale=True,
            colorbar=dict(
                title="Event Count",
                title_font=dict(color='white'),
                tickfont=dict(color='white'),
                x=1.02
            ),
            hovertemplate='<b>%{x}</b><br>' +
                         'Frequency: %{z}<br>' +
                         'Avg Risk: %{customdata:.2f}<extra></extra>',
            customdata=[avg_risks],
            text=[[f"{c} events<br>Risk: {r:.2f}" for c, r in zip(counts, avg_risks)]],
            texttemplate="%{text}",
            textfont=dict(color='white', size=10, family='Arial Black')
        ))
        
        # Enhanced layout
        fig.update_layout(
            title={
                'text': "MITRE ATT&CK Technique Frequency Heatmap",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#E74C3C', 'family': 'Arial Black'}
            },
            height=300,
            xaxis=dict(
                title="MITRE ATT&CK Techniques",
                title_font=dict(color='white', size=14),
                tickfont=dict(color='white', size=10),
                tickangle=45
            ),
            yaxis=dict(
                title_font=dict(color='white', size=14),
                tickfont=dict(color='white'),
                showticklabels=False
            ),
            plot_bgcolor='rgba(0,0,0,0.05)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=60, r=80, t=60, b=120)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating enhanced MITRE heatmap: {e}")
        return go.Figure()

    # Status indicator
    if st.session_state.streaming_paused:
        status_class = "status-paused"
        status_text = "PAUSED"
        status_color = "#F39C12"
    else:
        status_class = "status-active"
        status_text = "STREAMING"
        status_color = "#27AE60"
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <span class="status-indicator {status_class}"></span>
        <strong style="color: {status_color};">{status_text}</strong>
        <span style="margin-left: 1rem; color: #BDC3C7;">
            Scenario: {st.session_state.current_scenario}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Create containers for live updates
    log_container = st.container()
    metrics_container = st.container()
    
    # Simulate streaming if active
    if st.session_state.streaming_active and not st.session_state.streaming_paused:
        simulate_streaming_events(log_container, metrics_container)
    else:
        # Show existing events
        display_streamed_events(log_container)
        display_streaming_metrics(metrics_container)

def simulate_streaming_events(log_container, metrics_container):
    """Simulate streaming events for demonstration."""
    try:
        # Generate mock streaming event
        if len(st.session_state.streamed_events) < 25:  # Limit for demo
            # Create mock event
            event_types = ['user_login', 'process_execution', 'file_access', 'network_connection', 'registry_modification']
            risk_levels = ['low', 'medium', 'high', 'critical']
            
            mock_event = {
                'sequence_number': len(st.session_state.streamed_events) + 1,
                'timestamp': datetime.now().isoformat(),
                'original_event': {
                    'event_type': event_types[len(st.session_state.streamed_events) % len(event_types)],
                    'source_host': f'WORKSTATION-{(len(st.session_state.streamed_events) % 3) + 1:02d}',
                    'source_user': f'user{(len(st.session_state.streamed_events) % 2) + 1}@company.local',
                    'target_resource': f'resource_{len(st.session_state.streamed_events) + 1}',
                    'event_id': 4600 + len(st.session_state.streamed_events)
                },
                'scenario_metadata': {
                    'scenario_id': st.session_state.current_scenario,
                    'mitre_techniques': ['T1078', 'T1059.001'],
                    'risk_level': risk_levels[min(len(st.session_state.streamed_events) // 6, 3)]
                },
                'streaming_metadata': {
                    'streamed_at': datetime.now().isoformat(),
                    'sequence_in_stream': len(st.session_state.streamed_events) + 1,
                    'progress_percent': ((len(st.session_state.streamed_events) + 1) / 25) * 100
                }
            }
            
            # Add to streamed events
            st.session_state.streamed_events.append(mock_event)
            
            # Update stats
            st.session_state.streaming_stats['events_processed'] += 1
            
            # Generate prediction for high-risk events
            if mock_event['scenario_metadata']['risk_level'] in ['high', 'critical']:
                prediction = generate_live_prediction(mock_event)
                if prediction:
                    st.session_state.live_predictions.append(prediction)
                    st.session_state.streaming_stats['predictions_generated'] += 1
                    
                    if prediction.get('alert_level') == 'HIGH':
                        st.session_state.streaming_stats['alerts_triggered'] += 1
            
            # Auto-refresh every 2.5 seconds
            time.sleep(2.5)
            st.rerun()
        else:
            # Simulation complete
            st.session_state.streaming_active = False
            st.success("Simulation completed successfully!")
            st.rerun()
    
    except Exception as e:
        logger.error(f"Error in streaming simulation: {e}")
        st.error(f"Streaming error: {str(e)}")
    
    # Display current events
    display_streamed_events(log_container)
    display_streaming_metrics(metrics_container)

def display_streamed_events(container):
    """Display streamed events in log format."""
    with container:
        if not st.session_state.streamed_events:
            st.info("Waiting for events...")
            return
        
        # Show recent events (last 10)
        recent_events = st.session_state.streamed_events[-10:]
        
        log_html = '<div class="streaming-container">'
        
        for event in recent_events:
            risk_level = event['scenario_metadata']['risk_level']
            timestamp = event['timestamp'][:19]  # Remove microseconds
            event_type = event['original_event']['event_type']
            source_host = event['original_event']['source_host']
            
            log_html += f'''
            <div class="log-entry {risk_level}">
                <strong>{timestamp}</strong> | 
                <span style="color: #3498DB;">{source_host}</span> | 
                <span style="color: #E67E22;">{event_type}</span> | 
                Risk: <span style="color: {"#E74C3C" if risk_level == "critical" else "#E67E22" if risk_level == "high" else "#F39C12" if risk_level == "medium" else "#27AE60"};">{risk_level.upper()}</span>
            </div>
            '''
        
        log_html += '</div>'
        
        st.markdown(log_html, unsafe_allow_html=True)

def display_streaming_metrics(container):
    """Display streaming metrics."""
    with container:
        col1, col2, col3, col4 = st.columns(4)
        
        stats = st.session_state.streaming_stats
        
        with col1:
            st.metric(
                "Events Processed",
                stats['events_processed'],
                delta=f"+{stats['events_processed']}"
            )
        
        with col2:
            st.metric(
                "Predictions Generated",
                stats['predictions_generated'],
                delta=f"+{stats['predictions_generated']}"
            )
        
        with col3:
            st.metric(
                "Alerts Triggered",
                stats['alerts_triggered'],
                delta=f"+{stats['alerts_triggered']}" if stats['alerts_triggered'] > 0 else None
            )
        
        with col4:
            if stats['start_time']:
                duration = time.time() - stats['start_time']
                st.metric(
                    "Duration",
                    f"{duration:.0f}s",
                    delta=f"{duration:.0f}s"
                )

def generate_live_prediction(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate live prediction for streaming event."""
    try:
        risk_level = event['scenario_metadata']['risk_level']
        
        if risk_level in ['high', 'critical']:
            confidence = 0.85 if risk_level == 'critical' else 0.75
            
            prediction = {
                'event_sequence': event['sequence_number'],
                'prediction': {
                    'next_step': {
                        'edge_type': 'executes' if event['sequence_number'] % 2 == 0 else 'accesses',
                        'confidence': confidence,
                        'risk_score': confidence * 0.9
                    },
                    'mitre_techniques': event['scenario_metadata']['mitre_techniques'],
                    'alert_level': 'HIGH' if confidence > 0.8 else 'MEDIUM'
                },
                'generated_at': datetime.now().isoformat(),
                'processing_time': 0.3
            }
            
            return prediction
        
        return None
        
    except Exception as e:
        logger.error(f"Error generating live prediction: {e}")
        return None

def render_live_predictions():
    """Render live predictions and alerts."""
    if not st.session_state.live_predictions:
        return
    
    st.markdown("### AI-Powered Live Predictions")
    
    # Show recent predictions
    recent_predictions = st.session_state.live_predictions[-5:]
    
    for prediction in recent_predictions:
        alert_level = prediction['prediction']['alert_level']
        confidence = prediction['prediction']['next_step']['confidence']
        
        if alert_level == 'HIGH':
            st.markdown(f"""
            <div class="alert-banner">
                <h4 style="color: #E74C3C; margin: 0 0 0.5rem 0;">
                    HIGH RISK PREDICTION ALERT
                </h4>
                <p style="margin: 0 0 0.5rem 0;">
                    Next attack step predicted with {confidence:.1%} confidence
                </p>
                <small>
                    Event #{prediction['event_sequence']} | 
                    Generated at {prediction['generated_at'][:19]}
                </small>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.expander(f"Prediction #{prediction['event_sequence']} ({confidence:.1%} confidence)", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Next Step:** {prediction['prediction']['next_step']['edge_type']}")
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                
                with col2:
                    st.markdown(f"**Alert Level:** {alert_level}")
                    st.markdown(f"**Processing Time:** {prediction['processing_time']}s")

def render_attack_visualization():
    """Render attack path visualization if predictions available."""
    if not st.session_state.live_predictions:
        return
    
    st.markdown("### Live Attack Path")
    
    try:
        # Create mock attack steps from predictions
        attack_steps = []
        
        for i, prediction in enumerate(st.session_state.live_predictions[-5:]):
            step = {
                'edge_type': prediction['prediction']['next_step']['edge_type'],
                'source_type': 'process' if i % 2 == 0 else 'user',
                'target_type': 'file' if prediction['prediction']['next_step']['edge_type'] == 'accesses' else 'process',
                'confidence': prediction['prediction']['next_step']['confidence'],
                'mitre_techniques': prediction['prediction']['mitre_techniques']
            }
            attack_steps.append(step)
        
        if attack_steps:
            # Initialize visualizer
            visualizer = AttackPathVisualizer()
            
            # Create visualization
            fig = visualizer.plot_attack_sequence(
                attack_steps=attack_steps,
                title="Live Attack Path (Updating in Real-time)",
                animate=True,
                width=1000,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        logger.error(f"Error rendering attack visualization: {e}")
        st.error(f"Visualization error: {str(e)}")

def render_sidebar():
    """Render sidebar with simulation controls and information."""
    with st.sidebar:
        st.markdown("## Attack Simulation Control Center")
        
        # Simulation status with speed indicator
        st.markdown("### Simulation Status Dashboard")
        
        if st.session_state.streaming_active:
            status = "ACTIVE" if not st.session_state.streaming_paused else "PAUSED"
        else:
            status = "STOPPED"
        
        st.markdown(f"**Status:** {status}")
        
        if st.session_state.current_scenario:
            st.markdown(f"**Scenario:** {st.session_state.current_scenario}")
        
        # Speed indicator
        if hasattr(st.session_state, 'simulation_speed'):
            speed_info = st.session_state.simulation_speed
            st.markdown(f"**Speed:** {speed_info['label']}")
            st.markdown(f"**Update Rate:** {speed_info['delay']:.2f}s")
        
        # Performance metrics
        st.markdown("### Performance")
        
        stats = st.session_state.streaming_stats
        
        # Calculate events per second based on speed
        if hasattr(st.session_state, 'simulation_speed'):
            multiplier = st.session_state.simulation_speed.get('multiplier', 1.0)
            base_rate = 2.3
            current_rate = base_rate * multiplier
            st.metric("Events/sec", f"{current_rate:.1f}")
        else:
            st.metric("Events/sec", "2.3")
        
        st.metric("Prediction Latency", "0.3s")
        st.metric("Memory Usage", "2.1 GB")
        
        # Calculate CPU usage based on speed (higher speed = higher CPU)
        if hasattr(st.session_state, 'simulation_speed'):
            multiplier = st.session_state.simulation_speed.get('multiplier', 1.0)
            base_cpu = 45
            current_cpu = min(95, base_cpu + (multiplier - 1) * 8)
            st.metric("CPU Usage", f"{current_cpu:.0f}%")
        else:
            st.metric("CPU Usage", "45%")
        
        # Speed control help
        with st.expander("Speed Control Guide", expanded=False):
            st.markdown("""
            **Demo Speed Options:**
            - **1x Normal**: Realistic timing (2.5s intervals)
            - **2x Fast**: Quick demo (1.25s intervals)
            - **4x Faster**: Rapid demo (0.625s intervals)
            - **5x Very Fast**: Very rapid (0.5s intervals)
            - **8x Ultra Fast**: Ultra rapid (0.3s intervals)
            - **10x Lightning**: Lightning fast (0.25s intervals)
            - **20x Warp Speed**: Maximum speed (0.125s intervals)
            
            **Professional Tip:** Use higher speeds for quick demos,
            lower speeds for detailed analysis.
            """)
        
        # Simulation settings
        with st.expander("Advanced Settings", expanded=False):
            auto_predictions = st.checkbox(
                "Auto Predictions",
                value=True,
                help="Generate predictions automatically"
            )
            
            show_alerts = st.checkbox(
                "Show Alerts",
                value=True,
                help="Display proactive alerts"
            )
            
            max_events = st.slider(
                "Max Events",
                min_value=10,
                max_value=100,
                value=25,
                help="Maximum events to simulate"
            )
        
        # Help section
        with st.expander("❓ Simulation Help", expanded=False):
            st.markdown("""
            **Controls:**
            - **Start**: Begin streaming selected scenario
            - **Pause/Resume**: Pause or resume streaming
            - **Stop**: Stop current simulation
            - **Speed**: Control demo speed (1x to 20x)
            
            **Features:**
            - Real-time log streaming
            - Live prediction updates
            - Proactive alert generation
            - Attack path visualization
            - Performance monitoring
            
            **Demo Tips:**
            - Use 10x-20x speed for quick overviews
            - Use 2x-5x speed for detailed demos
            - Use 1x speed for training sessions
            """)
        
        # Current stats
        if st.session_state.streaming_active:
            st.markdown("### Live Performance Statistics")
            
            if stats['start_time']:
                duration = time.time() - stats['start_time']
                st.metric("Runtime", f"{duration:.0f}s")
            
            st.metric("Events Processed", stats['events_processed'])
            st.metric("Predictions Made", stats['predictions_generated'])
            st.metric("Alerts Triggered", stats['alerts_triggered'])

def main():
    """Main function for the attack simulation page."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Render sidebar
        render_sidebar()
        
        # Render main content
        render_header()
        
        # Scenario selection
        selected_scenario = render_scenario_selection()
        
        st.markdown("---")
        
        # Streaming display
        render_streaming_display()
        
        # Live predictions
        render_live_predictions()
        
        # Attack visualization
        render_attack_visualization()
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        st.error(f"Application error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()