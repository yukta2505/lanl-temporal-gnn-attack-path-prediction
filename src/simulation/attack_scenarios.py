#!/usr/bin/env python3
"""
Attack Scenarios

Pre-built attack scenarios for cybersecurity demonstrations and testing.
Contains realistic attack patterns based on real-world threats and
MITRE ATT&CK framework techniques.

Features:
- Pre-built attack scenarios (lateral movement, insider threat, ransomware, etc.)
- MITRE ATT&CK technique mapping for each scenario
- Realistic timing and progression patterns
- Customizable scenario parameters
- Export capabilities for different formats
- Integration with real-time simulator
"""

import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import random

logger = logging.getLogger(__name__)

@dataclass
class AttackStep:
    """Represents a single step in an attack scenario."""
    step_number: int
    timestamp_offset_minutes: float
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
    mitre_tactic: Optional[str] = None
    risk_score: float = 0.5
    confidence: float = 0.7
    description: str = ""
    indicators: List[str] = None

    def __post_init__(self):
        if self.indicators is None:
            self.indicators = []

@dataclass
class AttackScenario:
    """Represents a complete attack scenario."""
    id: str
    name: str
    description: str
    duration_minutes: int
    total_events: int
    mitre_techniques: List[str]
    mitre_tactics: List[str]
    risk_level: str
    complexity: str
    attack_type: str
    steps: List[AttackStep]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class AttackScenarios:
    """
    Pre-built attack scenarios for cybersecurity demonstrations.
    
    Provides realistic attack patterns based on real-world threats
    and MITRE ATT&CK framework for training and demonstration purposes.
    """
    
    def __init__(self):
        """Initialize attack scenarios."""
        self.scenarios = {}
        self._initialize_scenarios()
        
        # Only log once during initialization
        if not hasattr(self.__class__, '_initialized'):
            logger.info(f"Attack scenarios initialized with {len(self.scenarios)} scenarios")
            self.__class__._initialized = True
    
    def _initialize_scenarios(self) -> None:
        """Initialize all pre-built scenarios."""
        try:
            # Lateral Movement Scenario
            self.scenarios['lateral_movement'] = {
                'id': 'lateral_movement',
                'name': 'APT Lateral Movement Attack',
                'description': 'Multi-stage lateral movement attack using PsExec and credential dumping',
                'duration_minutes': 15,
                'total_events': 25,
                'complexity': 'HIGH',
                'mitre_techniques': ['T1078', 'T1021.002', 'T1003', 'T1055'],
                'events': []
            }
            
            # Insider Threat Scenario
            self.scenarios['insider_threat'] = {
                'id': 'insider_threat',
                'name': 'Insider Data Exfiltration',
                'description': 'Malicious insider accessing and exfiltrating sensitive data',
                'duration_minutes': 10,
                'total_events': 18,
                'complexity': 'MEDIUM',
                'mitre_techniques': ['T1087', 'T1083', 'T1005', 'T1041'],
                'events': []
            }
            
            # Ransomware Attack Scenario
            self.scenarios['ransomware_attack'] = {
                'id': 'ransomware_attack',
                'name': 'Ransomware Attack Chain',
                'description': 'Complete ransomware attack from initial access to encryption',
                'duration_minutes': 12,
                'total_events': 22,
                'complexity': 'HIGH',
                'mitre_techniques': ['T1486', 'T1490', 'T1083', 'T1059.001'],
                'events': []
            }
            
            # Only log once during initialization
            if not hasattr(self.__class__, '_scenarios_initialized'):
                logger.info("All attack scenarios initialized successfully")
                self.__class__._scenarios_initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing scenarios: {e}")
            # Fallback scenarios
            self.scenarios = {
                'demo_attack': {
                    'id': 'demo_attack',
                    'name': 'Demo Attack Scenario',
                    'description': 'Basic attack demonstration',
                    'duration_minutes': 5,
                    'total_events': 10,
                    'complexity': 'LOW',
                    'mitre_techniques': ['T1078'],
                    'events': []
                }
            }
    
    def get_scenario(self, scenario_id: str) -> Optional[AttackScenario]:
        """
        Get specific attack scenario by ID.
        
        Args:
            scenario_id: Scenario identifier
            
        Returns:
            Attack scenario or None if not found
        """
        return self.scenarios.get(scenario_id)
    
    def get_all_scenarios(self) -> Dict[str, AttackScenario]:
        """Get all available scenarios."""
        return self.scenarios.copy()
    
    def get_scenario_list(self) -> List[Dict[str, Any]]:
        """Get list of scenarios with metadata."""
        scenario_list = []
        
        for scenario_id, scenario in self.scenarios.items():
            scenario_list.append({
                'id': scenario.id,
                'name': scenario.name,
                'description': scenario.description,
                'duration_minutes': scenario.duration_minutes,
                'total_events': scenario.total_events,
                'mitre_techniques': scenario.mitre_techniques,
                'mitre_tactics': scenario.mitre_tactics,
                'risk_level': scenario.risk_level,
                'complexity': scenario.complexity,
                'attack_type': scenario.attack_type
            })
        
        return scenario_list
    
    def customize_scenario(
        self,
        base_scenario_id: str,
        modifications: Dict[str, Any]
    ) -> Optional[AttackScenario]:
        """
        Customize existing scenario with modifications.
        
        Args:
            base_scenario_id: Base scenario to modify
            modifications: Dictionary of modifications
            
        Returns:
            Customized scenario or None if base not found
        """
        try:
            base_scenario = self.get_scenario(base_scenario_id)
            if not base_scenario:
                return None
            
            # Create copy of base scenario
            custom_scenario = AttackScenario(
                id=modifications.get('id', f"{base_scenario.id}_custom"),
                name=modifications.get('name', f"{base_scenario.name} (Custom)"),
                description=modifications.get('description', base_scenario.description),
                duration_minutes=modifications.get('duration_minutes', base_scenario.duration_minutes),
                total_events=modifications.get('total_events', base_scenario.total_events),
                mitre_techniques=modifications.get('mitre_techniques', base_scenario.mitre_techniques),
                complexity=modifications.get('complexity', base_scenario.complexity),
                events=modifications.get('events', base_scenario.events)
            )
            
        except Exception as e:
            logger.error(f"Error getting all scenarios: {e}")
            return {}
    
    def customize_scenario(
        self,
        base_scenario: str,
        modifications: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Customize scenario for specific demo needs.
        
        Args:
            base_scenario: Base scenario name
            modifications: Dictionary of modifications to apply
            
        Returns:
            Customized scenario or None if base scenario not found
        """
        try:
            scenario = self.get_scenario(base_scenario)
            if not scenario:
                return None
            
            # Apply modifications
            for key, value in modifications.items():
                if key in scenario:
                    scenario[key] = value
                elif key == 'add_techniques':
                    scenario['mitre_techniques'].extend(value)
                elif key == 'remove_techniques':
                    scenario['mitre_techniques'] = [
                        t for t in scenario['mitre_techniques'] if t not in value
                    ]
                elif key == 'scale_duration':
                    scenario['duration_minutes'] = int(scenario['duration_minutes'] * value)
                    scenario['total_events'] = int(scenario['total_events'] * value)
            
            # Regenerate derived data
            scenario['attack_steps'] = self._generate_attack_steps(scenario)
            scenario['timing'] = self._generate_timing_info(scenario)
            
            return scenario
            
        except Exception as e:
            logger.error(f"Error customizing scenario: {e}")
            return None
    
    def _generate_attack_steps(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed attack steps for scenario."""
        try:
            steps = []
            techniques = scenario['mitre_techniques']
            phases = scenario['attack_phases']
            total_events = scenario['total_events']
            
            # Distribute events across phases
            events_per_phase = total_events // len(phases)
            
            step_id = 1
            for phase_idx, phase in enumerate(phases):
                phase_events = events_per_phase
                if phase_idx == len(phases) - 1:  # Last phase gets remaining events
                    phase_events = total_events - (events_per_phase * (len(phases) - 1))
                
                # Generate steps for this phase
                phase_steps = self._generate_phase_steps(
                    phase, phase_events, techniques, step_id
                )
                steps.extend(phase_steps)
                step_id += len(phase_steps)
            
            return steps
            
        except Exception as e:
            logger.error(f"Error generating attack steps: {e}")
            return []
    
    def _generate_phase_steps(
        self,
        phase: str,
        num_events: int,
        techniques: List[str],
        start_id: int
    ) -> List[Dict[str, Any]]:
        """Generate steps for a specific attack phase."""
        try:
            steps = []
            
            # Define phase-specific patterns
            phase_patterns = {
                'initial_access': [
                    {'edge_type': 'logs_into', 'source_type': 'user', 'target_type': 'host'},
                    {'edge_type': 'executes', 'source_type': 'user', 'target_type': 'process'}
                ],
                'credential_access': [
                    {'edge_type': 'accesses', 'source_type': 'process', 'target_type': 'file'},
                    {'edge_type': 'executes', 'source_type': 'process', 'target_type': 'process'},
                    {'edge_type': 'modifies', 'source_type': 'process', 'target_type': 'registry'}
                ],
                'lateral_movement': [
                    {'edge_type': 'connects_to', 'source_type': 'process', 'target_type': 'host'},
                    {'edge_type': 'logs_into', 'source_type': 'user', 'target_type': 'host'},
                    {'edge_type': 'executes', 'source_type': 'user', 'target_type': 'process'}
                ],
                'persistence': [
                    {'edge_type': 'modifies', 'source_type': 'process', 'target_type': 'registry'},
                    {'edge_type': 'creates', 'source_type': 'process', 'target_type': 'file'}
                ],
                'discovery': [
                    {'edge_type': 'accesses', 'source_type': 'process', 'target_type': 'file'},
                    {'edge_type': 'connects_to', 'source_type': 'process', 'target_type': 'host'}
                ],
                'collection': [
                    {'edge_type': 'accesses', 'source_type': 'process', 'target_type': 'file'},
                    {'edge_type': 'creates', 'source_type': 'process', 'target_type': 'file'}
                ],
                'exfiltration': [
                    {'edge_type': 'connects_to', 'source_type': 'process', 'target_type': 'network'},
                    {'edge_type': 'accesses', 'source_type': 'process', 'target_type': 'file'}
                ],
                'impact': [
                    {'edge_type': 'modifies', 'source_type': 'process', 'target_type': 'file'},
                    {'edge_type': 'executes', 'source_type': 'process', 'target_type': 'process'}
                ],
                'defense_evasion': [
                    {'edge_type': 'modifies', 'source_type': 'process', 'target_type': 'registry'},
                    {'edge_type': 'executes', 'source_type': 'process', 'target_type': 'process'}
                ],
                'privilege_escalation': [
                    {'edge_type': 'executes', 'source_type': 'user', 'target_type': 'process'},
                    {'edge_type': 'modifies', 'source_type': 'process', 'target_type': 'registry'}
                ]
            }
            
            patterns = phase_patterns.get(phase, phase_patterns['discovery'])
            
            # Generate steps
            for i in range(num_events):
                pattern = patterns[i % len(patterns)]
                
                # Select relevant technique for this step
                step_techniques = [t for t in techniques if self._technique_matches_phase(t, phase)]
                if not step_techniques:
                    step_techniques = techniques[:1]  # Fallback to first technique
                
                step = {
                    'step_id': start_id + i,
                    'phase': phase,
                    'edge_type': pattern['edge_type'],
                    'source_type': pattern['source_type'],
                    'target_type': pattern['target_type'],
                    'mitre_techniques': step_techniques[:2],  # Max 2 techniques per step
                    'confidence': self._calculate_step_confidence(phase, i, num_events),
                    'risk_level': self._calculate_step_risk(phase, i, num_events),
                    'description': self._generate_step_description(pattern, step_techniques)
                }
                
                steps.append(step)
            
            return steps
            
        except Exception as e:
            logger.error(f"Error generating phase steps: {e}")
            return []
    
    def _technique_matches_phase(self, technique: str, phase: str) -> bool:
        """Check if MITRE technique matches attack phase."""
        try:
            technique_info = self.mitre_techniques.get(technique, {})
            tactic = technique_info.get('tactic', '')
            
            # Map tactics to phases
            tactic_phase_mapping = {
                'TA0001': ['initial_access'],
                'TA0002': ['initial_access', 'lateral_movement'],
                'TA0003': ['persistence'],
                'TA0004': ['privilege_escalation'],
                'TA0005': ['defense_evasion'],
                'TA0006': ['credential_access'],
                'TA0007': ['discovery'],
                'TA0008': ['lateral_movement'],
                'TA0009': ['collection'],
                'TA0010': ['exfiltration'],
                'TA0040': ['impact']
            }
            
            return phase in tactic_phase_mapping.get(tactic, [])
            
        except Exception as e:
            logger.error(f"Error matching technique to phase: {e}")
            return False
    
    def _calculate_step_confidence(self, phase: str, step_index: int, total_steps: int) -> float:
        """Calculate confidence for attack step."""
        try:
            # Base confidence varies by phase
            base_confidence = {
                'initial_access': 0.85,
                'credential_access': 0.80,
                'lateral_movement': 0.75,
                'persistence': 0.70,
                'discovery': 0.90,
                'collection': 0.85,
                'exfiltration': 0.70,
                'impact': 0.95,
                'defense_evasion': 0.65,
                'privilege_escalation': 0.75
            }.get(phase, 0.75)
            
            # Add some variation based on step position
            progress = step_index / total_steps
            variation = np.random.uniform(-0.1, 0.1)
            
            # Later steps in phase might have lower confidence
            position_factor = 1.0 - (progress * 0.2)
            
            confidence = base_confidence * position_factor + variation
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating step confidence: {e}")
            return 0.75
    
    def _calculate_step_risk(self, phase: str, step_index: int, total_steps: int) -> str:
        """Calculate risk level for attack step."""
        try:
            # Risk generally increases through attack progression
            progress = step_index / total_steps
            
            # Phase-specific risk levels
            phase_risk = {
                'initial_access': 'medium',
                'credential_access': 'high',
                'lateral_movement': 'high',
                'persistence': 'critical',
                'discovery': 'low',
                'collection': 'medium',
                'exfiltration': 'critical',
                'impact': 'critical',
                'defense_evasion': 'medium',
                'privilege_escalation': 'high'
            }.get(phase, 'medium')
            
            # Escalate risk based on progress
            if progress > 0.7 and phase_risk in ['medium', 'high']:
                return 'critical'
            elif progress > 0.4 and phase_risk == 'low':
                return 'medium'
            
            return phase_risk
            
        except Exception as e:
            logger.error(f"Error calculating step risk: {e}")
            return 'medium'
    
    def _generate_step_description(
        self,
        pattern: Dict[str, str],
        techniques: List[str]
    ) -> str:
        """Generate human-readable description for attack step."""
        try:
            edge_type = pattern['edge_type']
            source_type = pattern['source_type']
            target_type = pattern['target_type']
            
            # Generate description based on pattern
            descriptions = {
                ('logs_into', 'user', 'host'): "User authentication to remote host",
                ('executes', 'user', 'process'): "User executes malicious process",
                ('executes', 'process', 'process'): "Process spawns child process",
                ('accesses', 'process', 'file'): "Process accesses sensitive file",
                ('connects_to', 'process', 'host'): "Process establishes network connection",
                ('connects_to', 'process', 'network'): "Process connects to external network",
                ('modifies', 'process', 'registry'): "Process modifies registry keys",
                ('modifies', 'process', 'file'): "Process modifies system files",
                ('creates', 'process', 'file'): "Process creates new file"
            }
            
            key = (edge_type, source_type, target_type)
            base_description = descriptions.get(key, f"{source_type} {edge_type} {target_type}")
            
            # Add technique context if available
            if techniques:
                technique_names = [
                    self.mitre_techniques.get(t, {}).get('name', t) 
                    for t in techniques[:1]  # Just first technique
                ]
                if technique_names:
                    base_description += f" ({technique_names[0]})"
            
            return base_description
            
        except Exception as e:
            logger.error(f"Error generating step description: {e}")
            return f"{pattern['source_type']} {pattern['edge_type']} {pattern['target_type']}"
    
    def _generate_timing_info(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Generate timing information for scenario."""
        try:
            duration_minutes = scenario['duration_minutes']
            total_events = scenario['total_events']
            risk_progression = scenario['risk_progression']
            
            # Generate timing based on progression pattern
            if risk_progression == 'escalating':
                # Slow start, fast finish
                intervals = self._generate_escalating_intervals(total_events, duration_minutes)
            elif risk_progression == 'explosive':
                # Fast start, maintain pace
                intervals = self._generate_explosive_intervals(total_events, duration_minutes)
            elif risk_progression == 'gradual':
                # Steady progression
                intervals = self._generate_steady_intervals(total_events, duration_minutes)
            elif risk_progression == 'rapid':
                # Fast throughout
                intervals = self._generate_rapid_intervals(total_events, duration_minutes)
            else:
                # Default steady
                intervals = self._generate_steady_intervals(total_events, duration_minutes)
            
            return {
                'total_duration_minutes': duration_minutes,
                'total_events': total_events,
                'event_intervals_seconds': intervals,
                'progression_pattern': risk_progression,
                'average_interval_seconds': sum(intervals) / len(intervals) if intervals else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating timing info: {e}")
            return {}
    
    def _generate_escalating_intervals(self, num_events: int, duration_minutes: int) -> List[float]:
        """Generate escalating timing intervals."""
        total_seconds = duration_minutes * 60
        intervals = []
        
        # Start slow, get faster
        for i in range(num_events):
            progress = i / num_events
            # Longer intervals at start, shorter at end
            base_interval = total_seconds / num_events
            factor = 2.0 - progress  # 2.0 to 1.0
            interval = base_interval * factor
            intervals.append(max(5.0, interval))  # Minimum 5 seconds
        
        # Scale to fit total duration
        current_total = sum(intervals)
        if current_total > 0:
            scale_factor = total_seconds / current_total
            intervals = [i * scale_factor for i in intervals]
        
        return intervals
    
    def _generate_explosive_intervals(self, num_events: int, duration_minutes: int) -> List[float]:
        """Generate explosive timing intervals."""
        total_seconds = duration_minutes * 60
        base_interval = total_seconds / num_events
        
        # Short intervals throughout with some variation
        intervals = []
        for i in range(num_events):
            variation = np.random.uniform(0.5, 1.5)
            interval = base_interval * 0.7 * variation  # Generally faster
            intervals.append(max(2.0, interval))  # Minimum 2 seconds
        
        return intervals
    
    def _generate_steady_intervals(self, num_events: int, duration_minutes: int) -> List[float]:
        """Generate steady timing intervals."""
        total_seconds = duration_minutes * 60
        base_interval = total_seconds / num_events
        
        intervals = []
        for i in range(num_events):
            variation = np.random.uniform(0.8, 1.2)
            interval = base_interval * variation
            intervals.append(max(10.0, interval))  # Minimum 10 seconds
        
        return intervals
    
    def _generate_rapid_intervals(self, num_events: int, duration_minutes: int) -> List[float]:
        """Generate rapid timing intervals."""
        total_seconds = duration_minutes * 60
        base_interval = total_seconds / num_events
        
        intervals = []
        for i in range(num_events):
            variation = np.random.uniform(0.3, 0.8)
            interval = base_interval * variation  # Generally faster
            intervals.append(max(3.0, interval))  # Minimum 3 seconds
        
        return intervals
    
    def _generate_entity_relationships(self, scenario: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate entity relationships for scenario."""
        try:
            target_entities = scenario['target_entities']
            
            # Define typical relationships
            relationships = {
                'user': ['host', 'process'],
                'host': ['user', 'process', 'file', 'network'],
                'process': ['user', 'host', 'file', 'registry', 'network'],
                'file': ['process', 'host'],
                'registry': ['process'],
                'network': ['process', 'host']
            }
            
            # Filter to only target entities
            filtered_relationships = {}
            for entity in target_entities:
                if entity in relationships:
                    filtered_relationships[entity] = [
                        rel for rel in relationships[entity] 
                        if rel in target_entities
                    ]
            
            return filtered_relationships
            
        except Exception as e:
            logger.error(f"Error generating entity relationships: {e}")
            return {}
    
    def export_scenario_to_json(self, scenario_name: str, output_path: str) -> bool:
        """Export scenario to JSON file."""
        try:
            scenario = self.get_scenario(scenario_name)
            if not scenario:
                return False
            
            with open(output_path, 'w') as f:
                json.dump(scenario, f, indent=2, default=str)
            
            logger.info(f"Scenario exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting scenario: {e}")
            return False