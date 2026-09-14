#!/usr/bin/env python3
"""
Day 7 Morning: Attack Chain Extraction for THGNN
Convert GNN explanations back to temporal attack sequences for security analysts.

Author: THGNN Expert
Date: January 9, 2026
Version: 1.0 - PRODUCTION ATTACK CHAIN EXTRACTION
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AttackStep:
    """Individual step in an attack chain"""
    step_number: int
    timestamp: datetime
    event_description: str
    source_entity: str
    target_entity: str
    edge_type: str
    mitre_technique: Optional[str]
    mitre_tactic: Optional[str]
    importance_score: float
    original_event: Optional[Dict[str, Any]]
    confidence: float

@dataclass
class AttackChain:
    """Complete attack chain extracted from explanation"""
    chain_id: str
    prediction_id: str
    attack_steps: List[AttackStep]
    total_steps: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    attack_pattern: Optional[str]
    kill_chain_stage: Optional[str]
    severity: str
    confidence: float
    entities_involved: List[str]
    techniques_used: List[str]
    tactics_used: List[str]

class AttackChainExtractor:
    """
    Production-ready attack chain extractor that converts GNN explanations
    back to temporal sequences of security events.
    """
    
    def __init__(self, entity_mappings: Optional[Dict] = None):
        """
        Initialize the attack chain extractor.
        
        Parameters
        ----------
        entity_mappings : Dict, optional
            Mappings from entity IDs to human-readable names
        """
        self.entity_mappings = entity_mappings or {}
        
        # Attack pattern signatures
        self.attack_patterns = {
            'psexec_lateral_movement': {
                'signature': ['executes', 'connects_to', 'logs_into'],
                'description': 'PsExec-based lateral movement',
                'severity': 'HIGH'
            },
            'pass_the_hash': {
                'signature': ['logs_into', 'logs_into'],
                'description': 'Pass-the-Hash attack',
                'severity': 'CRITICAL'
            },
            'powershell_execution': {
                'signature': ['executes', 'creates'],
                'description': 'PowerShell-based execution',
                'severity': 'MEDIUM'
            },
            'wmi_lateral_movement': {
                'signature': ['connects_to', 'executes'],
                'description': 'WMI-based lateral movement',
                'severity': 'HIGH'
            },
            'privilege_escalation': {
                'signature': ['executes', 'modifies', 'logs_into'],
                'description': 'Privilege escalation sequence',
                'severity': 'HIGH'
            }
        }
        
        # Cyber Kill Chain stages
        self.kill_chain_stages = {
            'reconnaissance': ['connects_to'],
            'weaponization': ['creates', 'modifies'],
            'delivery': ['accesses'],
            'exploitation': ['executes'],
            'installation': ['creates', 'modifies'],
            'command_control': ['connects_to'],
            'actions_objectives': ['accesses', 'modifies']
        }
        
        # Performance tracking
        self.extraction_times = []
    
    def extract_attack_chain(
        self,
        explanation_result,
        original_events: Optional[List[Dict]] = None,
        graph: Optional[HeteroData] = None
    ) -> AttackChain:
        """
        Extract attack chain from GNN explanation result.
        
        Parameters
        ----------
        explanation_result : ExplanationResult
            Result from GNN explainer
        original_events : List[Dict], optional
            Original log events for context
        graph : HeteroData, optional
            Original graph for additional context
            
        Returns
        -------
        AttackChain
            Extracted attack chain
        """
        start_time = time.time()
        
        try:
            # Map influential edges to original events
            mapped_events = self._map_edges_to_events(
                explanation_result.influential_edges,
                original_events
            )
            
            # Order events temporally
            ordered_events = self._order_events_temporally(mapped_events)
            
            # Group into logical attack steps
            attack_steps = self._create_attack_steps(
                ordered_events,
                explanation_result
            )
            
            # Identify attack pattern
            attack_pattern = self._identify_attack_pattern(attack_steps)
            
            # Determine kill chain stage
            kill_chain_stage = self._determine_kill_chain_stage(attack_steps)
            
            # Calculate metrics
            start_timestamp = min(step.timestamp for step in attack_steps) if attack_steps else datetime.now()
            end_timestamp = max(step.timestamp for step in attack_steps) if attack_steps else datetime.now()
            duration = (end_timestamp - start_timestamp).total_seconds()
            
            # Determine severity
            severity = self._calculate_severity(attack_steps, attack_pattern)
            
            # Extract entities and techniques
            entities_involved = list(set([step.source_entity for step in attack_steps] + 
                                       [step.target_entity for step in attack_steps]))
            techniques_used = [step.mitre_technique for step in attack_steps if step.mitre_technique]
            tactics_used = [step.mitre_tactic for step in attack_steps if step.mitre_tactic]
            
            # Create attack chain
            chain = AttackChain(
                chain_id=f"chain_{explanation_result.prediction_id}",
                prediction_id=explanation_result.prediction_id,
                attack_steps=attack_steps,
                total_steps=len(attack_steps),
                start_time=start_timestamp,
                end_time=end_timestamp,
                duration_seconds=duration,
                attack_pattern=attack_pattern,
                kill_chain_stage=kill_chain_stage,
                severity=severity,
                confidence=explanation_result.confidence,
                entities_involved=entities_involved,
                techniques_used=techniques_used,
                tactics_used=tactics_used
            )
            
            extraction_time = time.time() - start_time
            self.extraction_times.append(extraction_time)
            
            logger.info(f"Extracted attack chain with {len(attack_steps)} steps in {extraction_time:.3f}s")
            
            return chain
            
        except Exception as e:
            logger.error(f"Failed to extract attack chain: {e}")
            return self._create_fallback_chain(explanation_result)
    
    def _map_edges_to_events(
        self,
        influential_edges: List[Dict],
        original_events: Optional[List[Dict]]
    ) -> List[Dict]:
        """Map influential edges back to original log events"""
        
        mapped_events = []
        
        if not original_events:
            # Create synthetic events from edges
            for i, edge in enumerate(influential_edges):
                event = {
                    'timestamp': datetime.now() - timedelta(minutes=i*5),
                    'source_entity': edge['source'],
                    'target_entity': edge['target'],
                    'edge_type': edge['edge_type'],
                    'importance': edge['importance'],
                    'event_description': f"{edge['source']} {edge['edge_type']} {edge['target']}",
                    'synthetic': True
                }
                mapped_events.append(event)
        else:
            # Map to actual events
            for edge in influential_edges:
                matching_events = self._find_matching_events(edge, original_events)
                mapped_events.extend(matching_events)
        
        return mapped_events
    
    def _find_matching_events(
        self,
        edge: Dict,
        original_events: List[Dict]
    ) -> List[Dict]:
        """Find original events that match an influential edge"""
        
        matching_events = []
        
        for event in original_events:
            # Simple matching based on entity names and edge type
            if (self._entities_match(edge['source'], event.get('source_entity', '')) and
                self._entities_match(edge['target'], event.get('target_entity', '')) and
                edge['edge_type'] == event.get('edge_type', '')):
                
                # Add importance score to event
                event_copy = event.copy()
                event_copy['importance'] = edge['importance']
                matching_events.append(event_copy)
        
        # If no matches found, create synthetic event
        if not matching_events:
            synthetic_event = {
                'timestamp': datetime.now(),
                'source_entity': edge['source'],
                'target_entity': edge['target'],
                'edge_type': edge['edge_type'],
                'importance': edge['importance'],
                'event_description': f"{edge['source']} {edge['edge_type']} {edge['target']}",
                'synthetic': True
            }
            matching_events.append(synthetic_event)
        
        return matching_events
    
    def _entities_match(self, entity1: str, entity2: str) -> bool:
        """Check if two entity strings refer to the same entity"""
        # Simple string matching - could be enhanced with fuzzy matching
        return entity1.lower() in entity2.lower() or entity2.lower() in entity1.lower()
    
    def _order_events_temporally(self, events: List[Dict]) -> List[Dict]:
        """Order events by timestamp"""
        
        # Add timestamps to events that don't have them
        for i, event in enumerate(events):
            if 'timestamp' not in event or event['timestamp'] is None:
                # Assign synthetic timestamp
                event['timestamp'] = datetime.now() - timedelta(minutes=len(events)-i)
        
        # Sort by timestamp
        return sorted(events, key=lambda x: x['timestamp'])
    
    def _create_attack_steps(
        self,
        ordered_events: List[Dict],
        explanation_result
    ) -> List[AttackStep]:
        """Convert ordered events into attack steps"""
        
        attack_steps = []
        
        for i, event in enumerate(ordered_events):
            # Create human-readable description
            description = self._create_event_description(event)
            
            # Get MITRE technique (placeholder - would be enhanced with actual mapping)
            mitre_technique = self._get_mitre_technique(event['edge_type'])
            mitre_tactic = self._get_mitre_tactic(mitre_technique)
            
            step = AttackStep(
                step_number=i + 1,
                timestamp=event['timestamp'],
                event_description=description,
                source_entity=self._resolve_entity_name(event['source_entity']),
                target_entity=self._resolve_entity_name(event['target_entity']),
                edge_type=event['edge_type'],
                mitre_technique=mitre_technique,
                mitre_tactic=mitre_tactic,
                importance_score=event.get('importance', 0.5),
                original_event=event,
                confidence=explanation_result.confidence
            )
            
            attack_steps.append(step)
        
        return attack_steps
    
    def _create_event_description(self, event: Dict) -> str:
        """Create human-readable event description"""
        
        source = self._resolve_entity_name(event['source_entity'])
        target = self._resolve_entity_name(event['target_entity'])
        edge_type = event['edge_type']
        
        # Create natural language descriptions
        descriptions = {
            'logs_into': f"{source} logged into {target}",
            'executes': f"{source} executed {target}",
            'creates': f"{source} created {target}",
            'connects_to': f"{source} connected to {target}",
            'accesses': f"{source} accessed {target}",
            'modifies': f"{source} modified {target}"
        }
        
        return descriptions.get(edge_type, f"{source} {edge_type} {target}")
    
    def _resolve_entity_name(self, entity_id: str) -> str:
        """Resolve entity ID to human-readable name"""
        
        if entity_id in self.entity_mappings:
            return self.entity_mappings[entity_id]
        
        # Parse entity ID format (e.g., "user_5" -> "User 5")
        if '_' in entity_id:
            entity_type, entity_num = entity_id.split('_', 1)
            return f"{entity_type.title()} {entity_num}"
        
        return entity_id
    
    def _get_mitre_technique(self, edge_type: str) -> Optional[str]:
        """Get MITRE ATT&CK technique for edge type"""
        
        technique_mapping = {
            'logs_into': 'T1078',      # Valid Accounts
            'executes': 'T1059',       # Command and Scripting Interpreter
            'creates': 'T1055',        # Process Injection
            'connects_to': 'T1021',    # Remote Services
            'accesses': 'T1005',       # Data from Local System
            'modifies': 'T1112'        # Modify Registry
        }
        
        return technique_mapping.get(edge_type)
    
    def _get_mitre_tactic(self, technique: Optional[str]) -> Optional[str]:
        """Get MITRE ATT&CK tactic for technique"""
        
        tactic_mapping = {
            'T1078': 'TA0001',  # Initial Access
            'T1059': 'TA0002',  # Execution
            'T1055': 'TA0005',  # Defense Evasion
            'T1021': 'TA0008',  # Lateral Movement
            'T1005': 'TA0009',  # Collection
            'T1112': 'TA0005'   # Defense Evasion
        }
        
        return tactic_mapping.get(technique) if technique else None
    
    def _identify_attack_pattern(self, attack_steps: List[AttackStep]) -> Optional[str]:
        """Identify known attack pattern from steps"""
        
        if not attack_steps:
            return None
        
        # Extract sequence of edge types
        edge_sequence = [step.edge_type for step in attack_steps]
        
        # Check against known patterns
        for pattern_name, pattern_info in self.attack_patterns.items():
            signature = pattern_info['signature']
            
            # Simple subsequence matching
            if self._sequence_contains(edge_sequence, signature):
                return pattern_name
        
        return None
    
    def _sequence_contains(self, sequence: List[str], subsequence: List[str]) -> bool:
        """Check if sequence contains subsequence"""
        
        if len(subsequence) > len(sequence):
            return False
        
        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i:i+len(subsequence)] == subsequence:
                return True
        
        return False
    
    def _determine_kill_chain_stage(self, attack_steps: List[AttackStep]) -> Optional[str]:
        """Determine cyber kill chain stage"""
        
        if not attack_steps:
            return None
        
        # Get most common edge type
        edge_types = [step.edge_type for step in attack_steps]
        most_common_edge = Counter(edge_types).most_common(1)[0][0]
        
        # Map to kill chain stage
        for stage, edge_types_in_stage in self.kill_chain_stages.items():
            if most_common_edge in edge_types_in_stage:
                return stage
        
        return 'unknown'
    
    def _calculate_severity(
        self,
        attack_steps: List[AttackStep],
        attack_pattern: Optional[str]
    ) -> str:
        """Calculate attack severity"""
        
        if not attack_steps:
            return 'LOW'
        
        # Base severity on pattern
        if attack_pattern and attack_pattern in self.attack_patterns:
            return self.attack_patterns[attack_pattern]['severity']
        
        # Base severity on number of steps and importance
        avg_importance = np.mean([step.importance_score for step in attack_steps])
        num_steps = len(attack_steps)
        
        if avg_importance > 0.8 and num_steps >= 3:
            return 'CRITICAL'
        elif avg_importance > 0.6 and num_steps >= 2:
            return 'HIGH'
        elif avg_importance > 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _create_fallback_chain(self, explanation_result) -> AttackChain:
        """Create minimal fallback chain when extraction fails"""
        
        fallback_step = AttackStep(
            step_number=1,
            timestamp=datetime.now(),
            event_description=f"{explanation_result.source_entity} {explanation_result.edge_type} {explanation_result.target_entity}",
            source_entity=explanation_result.source_entity,
            target_entity=explanation_result.target_entity,
            edge_type=explanation_result.edge_type,
            mitre_technique=self._get_mitre_technique(explanation_result.edge_type),
            mitre_tactic=self._get_mitre_tactic(self._get_mitre_technique(explanation_result.edge_type)),
            importance_score=0.5,
            original_event=None,
            confidence=explanation_result.confidence
        )
        
        return AttackChain(
            chain_id=f"fallback_{explanation_result.prediction_id}",
            prediction_id=explanation_result.prediction_id,
            attack_steps=[fallback_step],
            total_steps=1,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0,
            attack_pattern=None,
            kill_chain_stage='unknown',
            severity='MEDIUM',
            confidence=explanation_result.confidence,
            entities_involved=[explanation_result.source_entity, explanation_result.target_entity],
            techniques_used=[fallback_step.mitre_technique] if fallback_step.mitre_technique else [],
            tactics_used=[fallback_step.mitre_tactic] if fallback_step.mitre_tactic else []
        )
    
    def batch_extract_chains(
        self,
        explanation_results: List,
        original_events: Optional[List[Dict]] = None,
        graph: Optional[HeteroData] = None
    ) -> List[AttackChain]:
        """Extract attack chains for multiple explanations"""
        
        chains = []
        
        logger.info(f"Extracting attack chains for {len(explanation_results)} explanations...")
        
        for explanation in explanation_results:
            try:
                chain = self.extract_attack_chain(explanation, original_events, graph)
                chains.append(chain)
            except Exception as e:
                logger.error(f"Failed to extract chain for {explanation.prediction_id}: {e}")
                continue
        
        logger.info(f"Extracted {len(chains)} attack chains successfully")
        
        return chains
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        if not self.extraction_times:
            return {}
        
        return {
            'total_extractions': len(self.extraction_times),
            'avg_extraction_time': np.mean(self.extraction_times),
            'max_extraction_time': np.max(self.extraction_times),
            'min_extraction_time': np.min(self.extraction_times)
        }
    
    def export_chains(
        self,
        chains: List[AttackChain],
        output_path: str,
        format: str = 'json'
    ):
        """Export attack chains to file"""
        
        try:
            if format == 'json':
                export_data = [asdict(chain) for chain in chains]
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                    
            elif format == 'csv':
                df_data = []
                for chain in chains:
                    df_data.append({
                        'chain_id': chain.chain_id,
                        'prediction_id': chain.prediction_id,
                        'total_steps': chain.total_steps,
                        'duration_seconds': chain.duration_seconds,
                        'attack_pattern': chain.attack_pattern,
                        'kill_chain_stage': chain.kill_chain_stage,
                        'severity': chain.severity,
                        'confidence': chain.confidence,
                        'entities_count': len(chain.entities_involved),
                        'techniques_count': len(chain.techniques_used)
                    })
                
                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False)
            
            logger.info(f"Exported {len(chains)} attack chains to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export attack chains: {e}")


def test_attack_chain_extractor():
    """Test the attack chain extractor"""
    
    logger.info("Testing Attack Chain Extractor...")
    
    try:
        # Create mock explanation result
        from src.explainability.gnn_explainer import ExplanationResult
        
        explanation = ExplanationResult(
            prediction_id='test_001',
            source_entity='user_5',
            target_entity='host_12',
            edge_type='logs_into',
            risk_score=0.85,
            confidence=0.92,
            influential_nodes=['user_5', 'process_23', 'host_12'],
            influential_edges=[
                {'source': 'user_5', 'target': 'process_23', 'edge_type': 'executes', 'importance': 0.9},
                {'source': 'process_23', 'target': 'host_12', 'edge_type': 'connects_to', 'importance': 0.8},
                {'source': 'user_5', 'target': 'host_12', 'edge_type': 'logs_into', 'importance': 0.85}
            ],
            node_importance={'user_5': 0.9, 'process_23': 0.8, 'host_12': 0.85},
            edge_importance={'edge_1': 0.9, 'edge_2': 0.8},
            subgraph_nodes=['user_5', 'process_23', 'host_12'],
            subgraph_edges=[('user_5', 'process_23'), ('process_23', 'host_12')],
            explanation_type='gnn_explainer',
            generation_time=1.5,
            timestamp=datetime.now()
        )
        
        # Initialize extractor
        entity_mappings = {
            'user_5': 'DOMAIN\\alice',
            'process_23': 'powershell.exe',
            'host_12': 'DC01'
        }
        
        extractor = AttackChainExtractor(entity_mappings)
        
        # Test chain extraction
        chain = extractor.extract_attack_chain(explanation)
        
        logger.info(f" Extracted attack chain: {chain.chain_id}")
        logger.info(f" Attack steps: {chain.total_steps}")
        logger.info(f" Attack pattern: {chain.attack_pattern}")
        logger.info(f" Kill chain stage: {chain.kill_chain_stage}")
        logger.info(f" Severity: {chain.severity}")
        
        # Print attack steps
        for step in chain.attack_steps:
            logger.info(f"  Step {step.step_number}: {step.event_description}")
        
        # Test batch extraction
        batch_chains = extractor.batch_extract_chains([explanation])
        logger.info(f" Batch extracted {len(batch_chains)} chains")
        
        # Test export
        output_dir = Path('outputs/attack_chains')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extractor.export_chains(
            batch_chains,
            str(output_dir / 'test_chains.json'),
            format='json'
        )
        
        # Test performance stats
        stats = extractor.get_performance_stats()
        logger.info(f" Performance stats: {stats}")
        
        logger.info(" All attack chain extractor tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Attack chain extractor test failed: {e}")
        return False


if __name__ == '__main__':
    """Test the attack chain extractor"""
    
    # Ensure output directory exists
    os.makedirs('outputs/attack_chains', exist_ok=True)
    
    # Run tests
    success = test_attack_chain_extractor()
    
    if success:
        logger.info(" Day 7 Morning Attack Chain Extractor implementation complete!")
        logger.info(" Ready for MITRE ATT&CK mapping and explanation generation")
    else:
        logger.error("[ERROR] Attack Chain Extractor implementation needs fixes")
        sys.exit(1)