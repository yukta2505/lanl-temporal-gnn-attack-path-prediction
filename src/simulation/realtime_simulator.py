#!/usr/bin/env python3
"""
Real-time Log Simulator

Real-time log streaming simulation for offline cybersecurity demonstrations.
Creates the illusion of live threat detection by streaming pre-recorded
attack scenarios with realistic delays and live prediction updates.

Features:
- Pre-loaded attack scenarios from Mordor datasets
- Realistic streaming delays (2-3 seconds per log)
- Live prediction updates as logs arrive
- Progress indicators and streaming controls
- Pause/resume functionality for demonstrations
- Multiple scenario options (lateral movement, privilege escalation, etc.)
- Professional presentation mode for client demos
"""

import logging
import time
import asyncio
import json
import random
from typing import Dict, List, Tuple, Optional, Any, Generator, AsyncGenerator
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
import threading
from queue import Queue, Empty

from .attack_scenarios import AttackScenarios

logger = logging.getLogger(__name__)

@dataclass
class StreamingEvent:
    """Represents a streaming log event."""
    timestamp: datetime
    event_type: str
    source_host: str
    source_user: str
    target_host: Optional[str] = None
    target_user: Optional[str] = None
    process_name: Optional[str] = None
    command_line: Optional[str] = None
    file_path: Optional[str] = None
    network_connection: Optional[str] = None
    mitre_technique: Optional[str] = None
    risk_score: float = 0.5
    confidence: float = 0.7
    raw_data: Optional[Dict[str, Any]] = None

@dataclass
class StreamingState:
    """Represents the current streaming state."""
    is_running: bool = False
    is_paused: bool = False
    current_scenario: Optional[str] = None
    events_streamed: int = 0
    total_events: int = 0
    start_time: Optional[datetime] = None
    current_event: Optional[StreamingEvent] = None
    predictions: List[Dict[str, Any]] = None

class RealtimeLogSimulator:
    """
    Real-time log streaming simulator for cybersecurity demonstrations.
    
    Simulates live security log ingestion by streaming pre-recorded attack
    scenarios with realistic timing. Designed for offline client demonstrations
    where actual live data is not available.
    """
    
    def __init__(self, scenario_path: Optional[str] = None):
        """
        Initialize real-time log simulator.
        
        Args:
            scenario_path: Path to attack scenario data
        """
        self.scenario_path = scenario_path or "data/demo_scenarios/"
        self.scenarios = AttackScenarios()
        
        # Streaming configuration
        self.config = {
            'default_delay': 2.5,  # Seconds between events
            'delay_variance': 1.0,  # Random variance in delay
            'batch_size': 1,  # Events per batch
            'buffer_size': 100,  # Event buffer size
            'prediction_frequency': 5,  # Update predictions every N events
            'max_concurrent_streams': 3  # Maximum concurrent scenarios
        }
        
        # Streaming state
        self.state = StreamingState()
        self.event_queue = Queue(maxsize=self.config['buffer_size'])
        self.prediction_queue = Queue(maxsize=50)
        
        # Event callbacks
        self.event_callbacks = []
        self.prediction_callbacks = []
        
        # Threading
        self.streaming_thread = None
        self.prediction_thread = None
        self._stop_event = threading.Event()
        
        logger.info(f"Real-time log simulator initialized with scenario path: {self.scenario_path}")
    
    def add_event_callback(self, callback) -> None:
        """Add callback for streaming events."""
        self.event_callbacks.append(callback)
    
    def add_prediction_callback(self, callback) -> None:
        """Add callback for prediction updates."""
        self.prediction_callbacks.append(callback)
    
    def get_available_scenarios(self) -> List[Dict[str, Any]]:
        """Get list of available attack scenarios."""
        try:
            return [
                {
                    'id': 'lateral_movement',
                    'name': 'Advanced Persistent Threat - Lateral Movement',
                    'description': 'Multi-stage attack with credential dumping and lateral movement',
                    'duration_minutes': 15,
                    'total_events': 25,
                    'mitre_techniques': ['T1078', 'T1021.002', 'T1003', 'T1087'],
                    'risk_level': 'HIGH',
                    'complexity': 'ADVANCED'
                },
                {
                    'id': 'insider_threat',
                    'name': 'Insider Threat - Data Exfiltration',
                    'description': 'Malicious insider accessing and exfiltrating sensitive data',
                    'duration_minutes': 10,
                    'total_events': 18,
                    'mitre_techniques': ['T1087', 'T1083', 'T1005', 'T1041'],
                    'risk_level': 'MEDIUM',
                    'complexity': 'MODERATE'
                },
                {
                    'id': 'ransomware_attack',
                    'name': 'Ransomware Deployment',
                    'description': 'Ransomware attack with file encryption and persistence',
                    'duration_minutes': 12,
                    'total_events': 22,
                    'mitre_techniques': ['T1486', 'T1547.001', 'T1112', 'T1059.001'],
                    'risk_level': 'CRITICAL',
                    'complexity': 'ADVANCED'
                },
                {
                    'id': 'privilege_escalation',
                    'name': 'Privilege Escalation Attack',
                    'description': 'UAC bypass and privilege escalation techniques',
                    'duration_minutes': 8,
                    'total_events': 15,
                    'mitre_techniques': ['T1548.002', 'T1134', 'T1055', 'T1078.003'],
                    'risk_level': 'HIGH',
                    'complexity': 'MODERATE'
                }
            ]
            
        except Exception as e:
            logger.error(f"Error getting available scenarios: {e}")
            return []
    
    def start_scenario_stream(
        self,
        scenario_id: str,
        delay: Optional[float] = None,
        enable_predictions: bool = True
    ) -> bool:
        """
        Start streaming a specific attack scenario.
        
        Args:
            scenario_id: ID of scenario to stream
            delay: Custom delay between events
            enable_predictions: Enable live predictions
            
        Returns:
            True if streaming started successfully
        """
        try:
            if self.state.is_running:
                logger.warning("Streaming already in progress")
                return False
            
            # Get scenario data
            scenario_data = self._load_scenario_data(scenario_id)
            if not scenario_data:
                logger.error(f"Could not load scenario: {scenario_id}")
                return False
            
            # Update state
            self.state.is_running = True
            self.state.is_paused = False
            self.state.current_scenario = scenario_id
            self.state.events_streamed = 0
            self.state.total_events = len(scenario_data['events'])
            self.state.start_time = datetime.now()
            self.state.predictions = []
            
            # Clear queues
            self._clear_queues()
            
            # Start streaming thread
            self._stop_event.clear()
            self.streaming_thread = threading.Thread(
                target=self._stream_scenario,
                args=(scenario_data, delay or self.config['default_delay'])
            )
            self.streaming_thread.daemon = True
            self.streaming_thread.start()
            
            # Start prediction thread if enabled
            if enable_predictions:
                self.prediction_thread = threading.Thread(
                    target=self._generate_predictions
                )
                self.prediction_thread.daemon = True
                self.prediction_thread.start()
            
            logger.info(f"Started streaming scenario: {scenario_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting scenario stream: {e}")
            self.state.is_running = False
            return False
    
    def pause_stream(self) -> bool:
        """Pause the current stream."""
        try:
            if not self.state.is_running:
                return False
            
            self.state.is_paused = True
            logger.info("Stream paused")
            return True
            
        except Exception as e:
            logger.error(f"Error pausing stream: {e}")
            return False
    
    def resume_stream(self) -> bool:
        """Resume the paused stream."""
        try:
            if not self.state.is_running or not self.state.is_paused:
                return False
            
            self.state.is_paused = False
            logger.info("Stream resumed")
            return True
            
        except Exception as e:
            logger.error(f"Error resuming stream: {e}")
            return False
    
    def stop_stream(self) -> bool:
        """Stop the current stream."""
        try:
            if not self.state.is_running:
                return False
            
            # Signal threads to stop
            self._stop_event.set()
            self.state.is_running = False
            self.state.is_paused = False
            
            # Wait for threads to finish
            if self.streaming_thread and self.streaming_thread.is_alive():
                self.streaming_thread.join(timeout=5.0)
            
            if self.prediction_thread and self.prediction_thread.is_alive():
                self.prediction_thread.join(timeout=5.0)
            
            # Clear state
            self.state.current_scenario = None
            self.state.events_streamed = 0
            self.state.total_events = 0
            self.state.start_time = None
            self.state.current_event = None
            
            logger.info("Stream stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
            return False
    
    def get_next_event(self, timeout: float = 1.0) -> Optional[StreamingEvent]:
        """
        Get the next streaming event.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Next streaming event or None if timeout
        """
        try:
            event = self.event_queue.get(timeout=timeout)
            return event
            
        except Empty:
            return None
        except Exception as e:
            logger.error(f"Error getting next event: {e}")
            return None
    
    def get_next_prediction(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Get the next prediction update.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Next prediction or None if timeout
        """
        try:
            prediction = self.prediction_queue.get(timeout=timeout)
            return prediction
            
        except Empty:
            return None
        except Exception as e:
            logger.error(f"Error getting next prediction: {e}")
            return None
    
    def get_streaming_state(self) -> StreamingState:
        """Get current streaming state."""
        return self.state
    
    def get_progress(self) -> Dict[str, Any]:
        """Get streaming progress information."""
        try:
            if not self.state.is_running:
                return {
                    'is_running': False,
                    'progress_percent': 0.0,
                    'events_streamed': 0,
                    'total_events': 0,
                    'elapsed_time': 0.0,
                    'estimated_remaining': 0.0
                }
            
            progress_percent = (
                (self.state.events_streamed / self.state.total_events * 100)
                if self.state.total_events > 0 else 0.0
            )
            
            elapsed_time = (
                (datetime.now() - self.state.start_time).total_seconds()
                if self.state.start_time else 0.0
            )
            
            estimated_remaining = (
                (elapsed_time / max(self.state.events_streamed, 1)) * 
                (self.state.total_events - self.state.events_streamed)
                if self.state.events_streamed > 0 else 0.0
            )
            
            return {
                'is_running': self.state.is_running,
                'is_paused': self.state.is_paused,
                'scenario': self.state.current_scenario,
                'progress_percent': progress_percent,
                'events_streamed': self.state.events_streamed,
                'total_events': self.state.total_events,
                'elapsed_time': elapsed_time,
                'estimated_remaining': estimated_remaining,
                'current_event': self.state.current_event
            }
            
        except Exception as e:
            logger.error(f"Error getting progress: {e}")
            return {'is_running': False, 'progress_percent': 0.0}
    
    def _load_scenario_data(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Load scenario data from file or generate mock data."""
        try:
            # Try to load from file first
            scenario_file = Path(self.scenario_path) / f"{scenario_id}.json"
            
            if scenario_file.exists():
                with open(scenario_file, 'r') as f:
                    return json.load(f)
            
            # Generate mock scenario data
            return self._generate_mock_scenario(scenario_id)
            
        except Exception as e:
            logger.error(f"Error loading scenario data: {e}")
            return None
    
    def _generate_mock_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """Generate mock scenario data for demonstration."""
        try:
            scenarios = self.get_available_scenarios()
            scenario_info = next((s for s in scenarios if s['id'] == scenario_id), None)
            
            if not scenario_info:
                return None
            
            # Generate mock events based on scenario type
            events = []
            base_time = datetime.now()
            
            if scenario_id == 'lateral_movement':
                events = self._generate_lateral_movement_events(base_time, scenario_info['total_events'])
            elif scenario_id == 'insider_threat':
                events = self._generate_insider_threat_events(base_time, scenario_info['total_events'])
            elif scenario_id == 'ransomware_attack':
                events = self._generate_ransomware_events(base_time, scenario_info['total_events'])
            elif scenario_id == 'privilege_escalation':
                events = self._generate_privilege_escalation_events(base_time, scenario_info['total_events'])
            else:
                events = self._generate_generic_events(base_time, scenario_info['total_events'])
            
            return {
                'metadata': scenario_info,
                'events': events
            }
            
        except Exception as e:
            logger.error(f"Error generating mock scenario: {e}")
            return None
    
    def _generate_lateral_movement_events(self, base_time: datetime, num_events: int) -> List[Dict[str, Any]]:
        """Generate lateral movement attack events."""
        events = []
        current_time = base_time
        
        # Phase 1: Initial access
        events.append({
            'timestamp': current_time.isoformat(),
            'event_type': 'user_login',
            'source_host': 'WORKSTATION-01',
            'source_user': 'john.doe@company.local',
            'target_host': 'DC-01',
            'mitre_technique': 'T1078',
            'risk_score': 0.65,
            'confidence': 0.75,
            'description': 'User login from workstation to domain controller'
        })
        
        current_time += timedelta(minutes=2)
        
        # Phase 2: Credential dumping
        events.append({
            'timestamp': current_time.isoformat(),
            'event_type': 'process_execution',
            'source_host': 'DC-01',
            'source_user': 'john.doe@company.local',
            'process_name': 'mimikatz.exe',
            'command_line': 'mimikatz.exe sekurlsa::logonpasswords',
            'mitre_technique': 'T1003',
            'risk_score': 0.95,
            'confidence': 0.90,
            'description': 'Credential dumping with Mimikatz'
        })
        
        current_time += timedelta(minutes=3)
        
        # Phase 3: Lateral movement
        events.append({
            'timestamp': current_time.isoformat(),
            'event_type': 'network_connection',
            'source_host': 'DC-01',
            'target_host': 'FILE-SERVER-01',
            'source_user': 'admin@company.local',
            'network_connection': 'SMB',
            'mitre_technique': 'T1021.002',
            'risk_score': 0.85,
            'confidence': 0.80,
            'description': 'SMB connection to file server with elevated privileges'
        })
        
        # Generate additional events to reach target count
        for i in range(3, num_events):
            current_time += timedelta(minutes=random.uniform(1, 4))
            
            event_types = ['file_access', 'registry_modification', 'process_execution', 'network_connection']
            event_type = random.choice(event_types)
            
            events.append({
                'timestamp': current_time.isoformat(),
                'event_type': event_type,
                'source_host': random.choice(['DC-01', 'FILE-SERVER-01', 'WORKSTATION-02']),
                'source_user': random.choice(['admin@company.local', 'service@company.local']),
                'mitre_technique': random.choice(['T1078', 'T1021.002', 'T1003', 'T1087']),
                'risk_score': random.uniform(0.6, 0.9),
                'confidence': random.uniform(0.7, 0.85),
                'description': f'Continued lateral movement activity - {event_type}'
            })
        
        return events
    
    def _generate_insider_threat_events(self, base_time: datetime, num_events: int) -> List[Dict[str, Any]]:
        """Generate insider threat events."""
        events = []
        current_time = base_time
        
        # Generate insider threat pattern
        for i in range(num_events):
            current_time += timedelta(minutes=random.uniform(0.5, 2))
            
            if i < 5:
                # Discovery phase
                event_type = 'file_access'
                mitre_technique = 'T1083'
                risk_score = random.uniform(0.4, 0.6)
            elif i < 12:
                # Collection phase
                event_type = 'data_collection'
                mitre_technique = 'T1005'
                risk_score = random.uniform(0.6, 0.8)
            else:
                # Exfiltration phase
                event_type = 'network_connection'
                mitre_technique = 'T1041'
                risk_score = random.uniform(0.8, 0.95)
            
            events.append({
                'timestamp': current_time.isoformat(),
                'event_type': event_type,
                'source_host': 'WORKSTATION-05',
                'source_user': 'insider@company.local',
                'mitre_technique': mitre_technique,
                'risk_score': risk_score,
                'confidence': random.uniform(0.65, 0.80),
                'description': f'Insider threat activity - {event_type}'
            })
        
        return events
    
    def _generate_ransomware_events(self, base_time: datetime, num_events: int) -> List[Dict[str, Any]]:
        """Generate ransomware attack events."""
        events = []
        current_time = base_time
        
        # Generate ransomware pattern
        for i in range(num_events):
            current_time += timedelta(minutes=random.uniform(0.3, 1.5))
            
            if i < 8:
                # Initial execution and persistence
                event_type = random.choice(['process_execution', 'registry_modification'])
                mitre_technique = random.choice(['T1059.001', 'T1547.001'])
                risk_score = random.uniform(0.7, 0.85)
            else:
                # File encryption phase
                event_type = 'file_modification'
                mitre_technique = 'T1486'
                risk_score = random.uniform(0.9, 0.98)
            
            events.append({
                'timestamp': current_time.isoformat(),
                'event_type': event_type,
                'source_host': f'WORKSTATION-{random.randint(1, 10):02d}',
                'source_user': 'system@company.local',
                'mitre_technique': mitre_technique,
                'risk_score': risk_score,
                'confidence': random.uniform(0.80, 0.95),
                'description': f'Ransomware activity - {event_type}'
            })
        
        return events
    
    def _generate_privilege_escalation_events(self, base_time: datetime, num_events: int) -> List[Dict[str, Any]]:
        """Generate privilege escalation events."""
        events = []
        current_time = base_time
        
        # Generate privilege escalation pattern
        for i in range(num_events):
            current_time += timedelta(minutes=random.uniform(0.5, 2))
            
            event_types = ['process_execution', 'token_manipulation', 'uac_bypass']
            mitre_techniques = ['T1548.002', 'T1134', 'T1055', 'T1078.003']
            
            events.append({
                'timestamp': current_time.isoformat(),
                'event_type': random.choice(event_types),
                'source_host': 'WORKSTATION-03',
                'source_user': 'user@company.local',
                'mitre_technique': random.choice(mitre_techniques),
                'risk_score': random.uniform(0.6, 0.9),
                'confidence': random.uniform(0.70, 0.85),
                'description': f'Privilege escalation attempt - step {i+1}'
            })
        
        return events
    
    def _generate_generic_events(self, base_time: datetime, num_events: int) -> List[Dict[str, Any]]:
        """Generate generic attack events."""
        events = []
        current_time = base_time
        
        for i in range(num_events):
            current_time += timedelta(minutes=random.uniform(1, 3))
            
            events.append({
                'timestamp': current_time.isoformat(),
                'event_type': 'security_event',
                'source_host': f'HOST-{random.randint(1, 5):02d}',
                'source_user': f'user{random.randint(1, 10)}@company.local',
                'mitre_technique': 'T1059',
                'risk_score': random.uniform(0.5, 0.8),
                'confidence': random.uniform(0.60, 0.80),
                'description': f'Generic security event {i+1}'
            })
        
        return events
    
    def _stream_scenario(self, scenario_data: Dict[str, Any], delay: float) -> None:
        """Stream scenario events in separate thread."""
        try:
            events = scenario_data['events']
            
            for i, event_data in enumerate(events):
                if self._stop_event.is_set():
                    break
                
                # Wait while paused
                while self.state.is_paused and not self._stop_event.is_set():
                    time.sleep(0.1)
                
                if self._stop_event.is_set():
                    break
                
                # Create streaming event
                event = StreamingEvent(
                    timestamp=datetime.fromisoformat(event_data['timestamp'].replace('Z', '+00:00')) if 'Z' in event_data['timestamp'] else datetime.fromisoformat(event_data['timestamp']),
                    event_type=event_data['event_type'],
                    source_host=event_data['source_host'],
                    source_user=event_data['source_user'],
                    target_host=event_data.get('target_host'),
                    target_user=event_data.get('target_user'),
                    process_name=event_data.get('process_name'),
                    command_line=event_data.get('command_line'),
                    file_path=event_data.get('file_path'),
                    network_connection=event_data.get('network_connection'),
                    mitre_technique=event_data.get('mitre_technique'),
                    risk_score=event_data.get('risk_score', 0.5),
                    confidence=event_data.get('confidence', 0.7),
                    raw_data=event_data
                )
                
                # Update state
                self.state.current_event = event
                self.state.events_streamed = i + 1
                
                # Add to queue
                try:
                    self.event_queue.put(event, timeout=1.0)
                except:
                    logger.warning("Event queue full, dropping event")
                
                # Notify callbacks
                for callback in self.event_callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in event callback: {e}")
                
                # Wait with variance
                actual_delay = delay + random.uniform(-self.config['delay_variance'], self.config['delay_variance'])
                time.sleep(max(0.1, actual_delay))
            
            # Mark streaming as complete
            self.state.is_running = False
            logger.info("Scenario streaming completed")
            
        except Exception as e:
            logger.error(f"Error in streaming thread: {e}")
            self.state.is_running = False
    
    def _generate_predictions(self) -> None:
        """Generate live predictions in separate thread."""
        try:
            prediction_counter = 0
            
            while self.state.is_running and not self._stop_event.is_set():
                if self.state.is_paused:
                    time.sleep(0.5)
                    continue
                
                # Generate prediction every N events
                if (self.state.events_streamed > 0 and 
                    self.state.events_streamed % self.config['prediction_frequency'] == 0 and
                    prediction_counter < self.state.events_streamed):
                    
                    prediction = self._create_mock_prediction()
                    
                    try:
                        self.prediction_queue.put(prediction, timeout=1.0)
                        prediction_counter = self.state.events_streamed
                        
                        # Notify callbacks
                        for callback in self.prediction_callbacks:
                            try:
                                callback(prediction)
                            except Exception as e:
                                logger.error(f"Error in prediction callback: {e}")
                                
                    except:
                        logger.warning("Prediction queue full, dropping prediction")
                
                time.sleep(1.0)
                
        except Exception as e:
            logger.error(f"Error in prediction thread: {e}")
    
    def _create_mock_prediction(self) -> Dict[str, Any]:
        """Create mock prediction for demonstration."""
        try:
            # Generate realistic prediction based on current scenario
            scenario_id = self.state.current_scenario
            
            if scenario_id == 'lateral_movement':
                next_steps = [
                    {'action': 'credential_access', 'confidence': 0.85, 'mitre': 'T1003'},
                    {'action': 'lateral_movement', 'confidence': 0.78, 'mitre': 'T1021.002'},
                    {'action': 'persistence', 'confidence': 0.72, 'mitre': 'T1547.001'}
                ]
            elif scenario_id == 'ransomware_attack':
                next_steps = [
                    {'action': 'file_encryption', 'confidence': 0.92, 'mitre': 'T1486'},
                    {'action': 'persistence', 'confidence': 0.88, 'mitre': 'T1547.001'},
                    {'action': 'defense_evasion', 'confidence': 0.75, 'mitre': 'T1112'}
                ]
            else:
                next_steps = [
                    {'action': 'discovery', 'confidence': 0.70, 'mitre': 'T1083'},
                    {'action': 'collection', 'confidence': 0.65, 'mitre': 'T1005'},
                    {'action': 'exfiltration', 'confidence': 0.60, 'mitre': 'T1041'}
                ]
            
            # Add some randomness
            for step in next_steps:
                step['confidence'] += random.uniform(-0.1, 0.1)
                step['confidence'] = max(0.1, min(0.99, step['confidence']))
            
            return {
                'timestamp': datetime.now().isoformat(),
                'scenario': scenario_id,
                'events_processed': self.state.events_streamed,
                'predicted_steps': next_steps,
                'overall_confidence': np.mean([step['confidence'] for step in next_steps]),
                'risk_level': 'HIGH' if any(step['confidence'] > 0.8 for step in next_steps) else 'MEDIUM',
                'alert_generated': any(step['confidence'] > 0.85 for step in next_steps)
            }
            
        except Exception as e:
            logger.error(f"Error creating mock prediction: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _clear_queues(self) -> None:
        """Clear all queues."""
        try:
            while not self.event_queue.empty():
                self.event_queue.get_nowait()
        except:
            pass
        
        try:
            while not self.prediction_queue.empty():
                self.prediction_queue.get_nowait()
        except:
            pass