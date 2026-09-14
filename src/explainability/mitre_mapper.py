#!/usr/bin/env python3
"""
Day 7 Morning: MITRE ATT&CK Framework Mapping for THGNN
Map attack chains to MITRE ATT&CK tactics, techniques, and procedures.

Author: THGNN Expert
Date: January 9, 2026
Version: 1.0 - PRODUCTION MITRE MAPPING
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MITRETechnique:
    """MITRE ATT&CK technique information"""
    technique_id: str
    name: str
    description: str
    tactic_id: str
    tactic_name: str
    platforms: List[str]
    data_sources: List[str]
    url: str

@dataclass
class MITREMapping:
    """Complete MITRE mapping for an attack chain"""
    chain_id: str
    techniques: List[MITRETechnique]
    tactics: List[str]
    kill_chain_phases: List[str]
    attack_pattern: Optional[str]
    confidence: float
    coverage_score: float  # How well MITRE covers this attack

class MITREMapper:
    """
    Production-ready MITRE ATT&CK framework mapper.
    Maps cybersecurity attack chains to MITRE tactics and techniques.
    """
    
    def __init__(self):
        """Initialize the MITRE mapper with comprehensive technique database"""
        
        # Comprehensive MITRE ATT&CK technique mappings
        self.techniques = {
            # Initial Access (TA0001)
            'T1078': MITRETechnique(
                technique_id='T1078',
                name='Valid Accounts',
                description='Adversaries may obtain and abuse credentials of existing accounts',
                tactic_id='TA0001',
                tactic_name='Initial Access',
                platforms=['Windows', 'Linux', 'macOS'],
                data_sources=['Authentication logs', 'Account usage'],
                url='https://attack.mitre.org/techniques/T1078/'
            ),
            
            # Execution (TA0002)
            'T1059': MITRETechnique(
                technique_id='T1059',
                name='Command and Scripting Interpreter',
                description='Adversaries may abuse command and script interpreters',
                tactic_id='TA0002',
                tactic_name='Execution',
                platforms=['Windows', 'Linux', 'macOS'],
                data_sources=['Process monitoring', 'Command line'],
                url='https://attack.mitre.org/techniques/T1059/'
            ),
            'T1059.001': MITRETechnique(
                technique_id='T1059.001',
                name='PowerShell',
                description='Adversaries may abuse PowerShell commands and scripts',
                tactic_id='TA0002',
                tactic_name='Execution',
                platforms=['Windows'],
                data_sources=['PowerShell logs', 'Process monitoring'],
                url='https://attack.mitre.org/techniques/T1059/001/'
            ),
            'T1059.003': MITRETechnique(
                technique_id='T1059.003',
                name='Windows Command Shell',
                description='Adversaries may abuse the Windows command shell',
                tactic_id='TA0002',
                tactic_name='Execution',
                platforms=['Windows'],
                data_sources=['Process monitoring', 'Command line'],
                url='https://attack.mitre.org/techniques/T1059/003/'
            ),
            
            # Persistence (TA0003)
            'T1053': MITRETechnique(
                technique_id='T1053',
                name='Scheduled Task/Job',
                description='Adversaries may abuse task scheduling functionality',
                tactic_id='TA0003',
                tactic_name='Persistence',
                platforms=['Windows', 'Linux', 'macOS'],
                data_sources=['Process monitoring', 'File monitoring'],
                url='https://attack.mitre.org/techniques/T1053/'
            ),
            
            # Privilege Escalation (TA0004)
            'T1055': MITRETechnique(
                technique_id='T1055',
                name='Process Injection',
                description='Adversaries may inject code into processes',
                tactic_id='TA0004',
                tactic_name='Privilege Escalation',
                platforms=['Windows', 'Linux', 'macOS'],
                data_sources=['Process monitoring', 'API monitoring'],
                url='https://attack.mitre.org/techniques/T1055/'
            ),
            
            # Defense Evasion (TA0005)
            'T1112': MITRETechnique(
                technique_id='T1112',
                name='Modify Registry',
                description='Adversaries may interact with the Windows Registry',
                tactic_id='TA0005',
                tactic_name='Defense Evasion',
                platforms=['Windows'],
                data_sources=['Windows Registry', 'Process monitoring'],
                url='https://attack.mitre.org/techniques/T1112/'
            ),
            'T1070': MITRETechnique(
                technique_id='T1070',
                name='Indicator Removal on Host',
                description='Adversaries may delete or alter generated artifacts',
                tactic_id='TA0005',
                tactic_name='Defense Evasion',
                platforms=['Windows', 'Linux', 'macOS'],
                data_sources=['File monitoring', 'Process monitoring'],
                url='https://attack.mitre.org/techniques/T1070/'
            ),
            
            # Credential Access (TA0006)
            'T1003': MITRETechnique(
                technique_id='T1003',
                name='OS Credential Dumping',
                description='Adversaries may attempt to dump credentials',
                tactic_id='TA0006',
                tactic_name='Credential Access',
                platforms=['Windows', 'Linux', 'macOS'],
                data_sources=['Process monitoring', 'API monitoring'],
                url='https://attack.mitre.org/techniques/T1003/'
            ),
            
            # Discovery (TA0007)
            'T1018': MITRETechnique(
                technique_id='T1018',
                name='Remote System Discovery',
                description='Adversaries may attempt to get a listing of other systems',
                tactic_id='TA0007',
                tactic_name='Discovery',
                platforms=['Windows', 'Linux', 'macOS'],
                data_sources=['Network protocol analysis', 'Process monitoring'],
                url='https://attack.mitre.org/techniques/T1018/'
            ),
            'T1082': MITRETechnique(
                technique_id='T1082',
                name='System Information Discovery',
                description='Adversaries may attempt to get detailed information about the operating system',
                tactic_id='TA0007',
                tactic_name='Discovery',
                platforms=['Windows', 'Linux', 'macOS'],
                data_sources=['Process monitoring', 'Command line'],
                url='https://attack.mitre.org/techniques/T1082/'
            ),
            
            # Lateral Movement (TA0008)
            'T1021': MITRETechnique(
                technique_id='T1021',
                name='Remote Services',
                description='Adversaries may use valid accounts to log into a service',
                tactic_id='TA0008',
                tactic_name='Lateral Movement',
                platforms=['Windows', 'Linux', 'macOS'],
                data_sources=['Authentication logs', 'Network traffic'],
                url='https://attack.mitre.org/techniques/T1021/'
            ),
            'T1021.002': MITRETechnique(
                technique_id='T1021.002',
                name='SMB/Windows Admin Shares',
                description='Adversaries may use SMB to interact with file shares',
                tactic_id='TA0008',
                tactic_name='Lateral Movement',
                platforms=['Windows'],
                data_sources=['Authentication logs', 'Network traffic'],
                url='https://attack.mitre.org/techniques/T1021/002/'
            ),
            'T1021.003': MITRETechnique(
                technique_id='T1021.003',
                name='Distributed Component Object Model',
                description='Adversaries may use DCOM for lateral movement',
                tactic_id='TA0008',
                tactic_name='Lateral Movement',
                platforms=['Windows'],
                data_sources=['Authentication logs', 'Network traffic'],
                url='https://attack.mitre.org/techniques/T1021/003/'
            ),
            'T1021.006': MITRETechnique(
                technique_id='T1021.006',
                name='Windows Remote Management',
                description='Adversaries may use WinRM for lateral movement',
                tactic_id='TA0008',
                tactic_name='Lateral Movement',
                platforms=['Windows'],
                data_sources=['Authentication logs', 'Network traffic'],
                url='https://attack.mitre.org/techniques/T1021/006/'
            ),
            
            # Collection (TA0009)
            'T1005': MITRETechnique(
                technique_id='T1005',
                name='Data from Local System',
                description='Adversaries may search local system sources',
                tactic_id='TA0009',
                tactic_name='Collection',
                platforms=['Windows', 'Linux', 'macOS'],
                data_sources=['File monitoring', 'Process monitoring'],
                url='https://attack.mitre.org/techniques/T1005/'
            ),
            
            # Command and Control (TA0011)
            'T1071': MITRETechnique(
                technique_id='T1071',
                name='Application Layer Protocol',
                description='Adversaries may communicate using application layer protocols',
                tactic_id='TA0011',
                tactic_name='Command and Control',
                platforms=['Windows', 'Linux', 'macOS'],
                data_sources=['Network traffic', 'Packet capture'],
                url='https://attack.mitre.org/techniques/T1071/'
            ),
            
            # Exfiltration (TA0010)
            'T1041': MITRETechnique(
                technique_id='T1041',
                name='Exfiltration Over C2 Channel',
                description='Adversaries may steal data by exfiltrating it over an existing C2 channel',
                tactic_id='TA0010',
                tactic_name='Exfiltration',
                platforms=['Windows', 'Linux', 'macOS'],
                data_sources=['Network traffic', 'Process monitoring'],
                url='https://attack.mitre.org/techniques/T1041/'
            )
        }
        
        # Edge type to technique mappings
        self.edge_to_technique = {
            'logs_into': ['T1078'],                    # Valid Accounts
            'executes': ['T1059', 'T1059.001'],       # Command Interpreter, PowerShell
            'creates': ['T1055', 'T1053'],            # Process Injection, Scheduled Task
            'connects_to': ['T1021', 'T1021.002'],    # Remote Services, SMB
            'accesses': ['T1005'],                     # Data from Local System
            'modifies': ['T1112', 'T1070']            # Modify Registry, Indicator Removal
        }
        
        # Process name to technique mappings
        self.process_to_technique = {
            'powershell.exe': ['T1059.001'],
            'cmd.exe': ['T1059.003'],
            'psexec.exe': ['T1021.002'],
            'wmic.exe': ['T1021.003'],
            'winrm.exe': ['T1021.006'],
            'mimikatz.exe': ['T1003'],
            'net.exe': ['T1018', 'T1082']
        }
        
        # Attack pattern to technique mappings
        self.pattern_to_techniques = {
            'psexec_lateral_movement': ['T1021.002', 'T1078', 'T1059'],
            'pass_the_hash': ['T1003', 'T1078', 'T1021'],
            'powershell_execution': ['T1059.001', 'T1055'],
            'wmi_lateral_movement': ['T1021.003', 'T1047'],
            'privilege_escalation': ['T1055', 'T1078', 'T1112']
        }
        
        # Tactic information
        self.tactics = {
            'TA0001': 'Initial Access',
            'TA0002': 'Execution',
            'TA0003': 'Persistence',
            'TA0004': 'Privilege Escalation',
            'TA0005': 'Defense Evasion',
            'TA0006': 'Credential Access',
            'TA0007': 'Discovery',
            'TA0008': 'Lateral Movement',
            'TA0009': 'Collection',
            'TA0010': 'Exfiltration',
            'TA0011': 'Command and Control'
        }
        
        # Kill chain phase mappings
        self.kill_chain_phases = {
            'reconnaissance': ['T1018', 'T1082'],
            'weaponization': ['T1059', 'T1055'],
            'delivery': ['T1078'],
            'exploitation': ['T1059', 'T1055'],
            'installation': ['T1053', 'T1112'],
            'command_control': ['T1071'],
            'actions_objectives': ['T1005', 'T1041']
        }
    
    def map_attack_chain(self, attack_chain) -> MITREMapping:
        """
        Map an attack chain to MITRE ATT&CK framework.
        
        Parameters
        ----------
        attack_chain : AttackChain
            Attack chain to map
            
        Returns
        -------
        MITREMapping
            Complete MITRE mapping
        """
        try:
            # Collect all techniques from attack steps
            all_techniques = []
            all_tactics = set()
            
            for step in attack_chain.attack_steps:
                # Get techniques from edge type
                edge_techniques = self._get_techniques_for_edge(step.edge_type)
                
                # Get techniques from entities (if process names are available)
                entity_techniques = self._get_techniques_for_entities(
                    step.source_entity, step.target_entity
                )
                
                # Combine techniques
                step_techniques = list(set(edge_techniques + entity_techniques))
                
                # Add to collection
                for tech_id in step_techniques:
                    if tech_id in self.techniques:
                        technique = self.techniques[tech_id]
                        all_techniques.append(technique)
                        all_tactics.add(technique.tactic_id)
            
            # Get additional techniques from attack pattern
            if attack_chain.attack_pattern:
                pattern_techniques = self._get_techniques_for_pattern(attack_chain.attack_pattern)
                for tech_id in pattern_techniques:
                    if tech_id in self.techniques:
                        technique = self.techniques[tech_id]
                        if technique not in all_techniques:
                            all_techniques.append(technique)
                            all_tactics.add(technique.tactic_id)
            
            # Remove duplicates while preserving order
            unique_techniques = []
            seen_ids = set()
            for tech in all_techniques:
                if tech.technique_id not in seen_ids:
                    unique_techniques.append(tech)
                    seen_ids.add(tech.technique_id)
            
            # Get kill chain phases
            kill_chain_phases = self._get_kill_chain_phases(unique_techniques)
            
            # Calculate coverage score
            coverage_score = self._calculate_coverage_score(
                attack_chain, unique_techniques
            )
            
            # Create mapping
            mapping = MITREMapping(
                chain_id=attack_chain.chain_id,
                techniques=unique_techniques,
                tactics=list(all_tactics),
                kill_chain_phases=kill_chain_phases,
                attack_pattern=attack_chain.attack_pattern,
                confidence=attack_chain.confidence,
                coverage_score=coverage_score
            )
            
            logger.info(f"Mapped {len(unique_techniques)} MITRE techniques for chain {attack_chain.chain_id}")
            
            return mapping
            
        except Exception as e:
            logger.error(f"Failed to map attack chain {attack_chain.chain_id}: {e}")
            return self._create_fallback_mapping(attack_chain)
    
    def _get_techniques_for_edge(self, edge_type: str) -> List[str]:
        """Get MITRE techniques for an edge type"""
        return self.edge_to_technique.get(edge_type, [])
    
    def _get_techniques_for_entities(
        self, 
        source_entity: str, 
        target_entity: str
    ) -> List[str]:
        """Get MITRE techniques based on entity names"""
        techniques = []
        
        # Check if entities contain process names
        for entity in [source_entity, target_entity]:
            entity_lower = entity.lower()
            for process_name, process_techniques in self.process_to_technique.items():
                if process_name in entity_lower:
                    techniques.extend(process_techniques)
        
        return techniques
    
    def _get_techniques_for_pattern(self, attack_pattern: str) -> List[str]:
        """Get MITRE techniques for an attack pattern"""
        return self.pattern_to_techniques.get(attack_pattern, [])
    
    def _get_kill_chain_phases(self, techniques: List[MITRETechnique]) -> List[str]:
        """Determine kill chain phases from techniques"""
        phases = []
        technique_ids = [t.technique_id for t in techniques]
        
        for phase, phase_techniques in self.kill_chain_phases.items():
            if any(tech_id in technique_ids for tech_id in phase_techniques):
                phases.append(phase)
        
        return phases
    
    def _calculate_coverage_score(
        self, 
        attack_chain, 
        techniques: List[MITRETechnique]
    ) -> float:
        """Calculate how well MITRE covers this attack"""
        
        if not attack_chain.attack_steps:
            return 0.0
        
        # Simple coverage: ratio of steps with techniques to total steps
        steps_with_techniques = 0
        
        for step in attack_chain.attack_steps:
            edge_techniques = self._get_techniques_for_edge(step.edge_type)
            entity_techniques = self._get_techniques_for_entities(
                step.source_entity, step.target_entity
            )
            
            if edge_techniques or entity_techniques:
                steps_with_techniques += 1
        
        return steps_with_techniques / len(attack_chain.attack_steps)
    
    def _create_fallback_mapping(self, attack_chain) -> MITREMapping:
        """Create minimal fallback mapping"""
        
        # Use generic technique based on first step
        fallback_technique = self.techniques.get('T1059', self.techniques['T1078'])
        
        return MITREMapping(
            chain_id=attack_chain.chain_id,
            techniques=[fallback_technique],
            tactics=[fallback_technique.tactic_id],
            kill_chain_phases=['exploitation'],
            attack_pattern=attack_chain.attack_pattern,
            confidence=attack_chain.confidence * 0.5,  # Lower confidence for fallback
            coverage_score=0.5
        )
    
    def get_technique_details(self, technique_id: str) -> Optional[MITRETechnique]:
        """Get detailed information about a MITRE technique"""
        return self.techniques.get(technique_id)
    
    def get_tactic_name(self, tactic_id: str) -> Optional[str]:
        """Get tactic name from tactic ID"""
        return self.tactics.get(tactic_id)
    
    def search_techniques(self, query: str) -> List[MITRETechnique]:
        """Search techniques by name or description"""
        query_lower = query.lower()
        matching_techniques = []
        
        for technique in self.techniques.values():
            if (query_lower in technique.name.lower() or 
                query_lower in technique.description.lower()):
                matching_techniques.append(technique)
        
        return matching_techniques
    
    def get_techniques_by_tactic(self, tactic_id: str) -> List[MITRETechnique]:
        """Get all techniques for a specific tactic"""
        return [t for t in self.techniques.values() if t.tactic_id == tactic_id]
    
    def export_mapping(
        self, 
        mapping: MITREMapping, 
        output_path: str,
        format: str = 'json'
    ):
        """Export MITRE mapping to file"""
        
        try:
            if format == 'json':
                export_data = asdict(mapping)
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                    
            elif format == 'csv':
                import pandas as pd
                
                # Create DataFrame from techniques
                df_data = []
                for technique in mapping.techniques:
                    df_data.append({
                        'chain_id': mapping.chain_id,
                        'technique_id': technique.technique_id,
                        'technique_name': technique.name,
                        'tactic_id': technique.tactic_id,
                        'tactic_name': technique.tactic_name,
                        'description': technique.description,
                        'platforms': ', '.join(technique.platforms),
                        'url': technique.url
                    })
                
                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False)
            
            logger.info(f"Exported MITRE mapping to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export MITRE mapping: {e}")
    
    def batch_map_chains(self, attack_chains: List) -> List[MITREMapping]:
        """Map multiple attack chains to MITRE"""
        
        mappings = []
        
        logger.info(f"Mapping {len(attack_chains)} attack chains to MITRE...")
        
        for chain in attack_chains:
            try:
                mapping = self.map_attack_chain(chain)
                mappings.append(mapping)
            except Exception as e:
                logger.error(f"Failed to map chain {chain.chain_id}: {e}")
                continue
        
        logger.info(f"Successfully mapped {len(mappings)} chains to MITRE")
        
        return mappings
    
    def generate_mitre_report(self, mappings: List[MITREMapping]) -> Dict[str, Any]:
        """Generate comprehensive MITRE analysis report"""
        
        if not mappings:
            return {}
        
        # Collect statistics
        all_techniques = []
        all_tactics = []
        all_patterns = []
        coverage_scores = []
        
        for mapping in mappings:
            all_techniques.extend([t.technique_id for t in mapping.techniques])
            all_tactics.extend(mapping.tactics)
            if mapping.attack_pattern:
                all_patterns.append(mapping.attack_pattern)
            coverage_scores.append(mapping.coverage_score)
        
        # Count occurrences
        from collections import Counter
        technique_counts = Counter(all_techniques)
        tactic_counts = Counter(all_tactics)
        pattern_counts = Counter(all_patterns)
        
        # Generate report
        report = {
            'summary': {
                'total_chains': len(mappings),
                'unique_techniques': len(set(all_techniques)),
                'unique_tactics': len(set(all_tactics)),
                'avg_coverage_score': sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0
            },
            'top_techniques': dict(technique_counts.most_common(10)),
            'top_tactics': dict(tactic_counts.most_common()),
            'top_patterns': dict(pattern_counts.most_common()),
            'coverage_distribution': {
                'high_coverage': len([s for s in coverage_scores if s > 0.8]),
                'medium_coverage': len([s for s in coverage_scores if 0.5 <= s <= 0.8]),
                'low_coverage': len([s for s in coverage_scores if s < 0.5])
            }
        }
        
        return report


def test_mitre_mapper():
    """Test the MITRE mapper"""
    
    logger.info("Testing MITRE Mapper...")
    
    try:
        # Create mock attack chain
        from src.explainability.attack_chain_extractor import AttackChain, AttackStep
        from datetime import datetime
        
        # Create attack steps
        steps = [
            AttackStep(
                step_number=1,
                timestamp=datetime.now(),
                event_description="User alice executed PowerShell",
                source_entity="DOMAIN\\alice",
                target_entity="powershell.exe",
                edge_type="executes",
                mitre_technique="T1059.001",
                mitre_tactic="TA0002",
                importance_score=0.9,
                original_event=None,
                confidence=0.92
            ),
            AttackStep(
                step_number=2,
                timestamp=datetime.now(),
                event_description="PowerShell connected to DC01",
                source_entity="powershell.exe",
                target_entity="DC01",
                edge_type="connects_to",
                mitre_technique="T1021",
                mitre_tactic="TA0008",
                importance_score=0.8,
                original_event=None,
                confidence=0.88
            )
        ]
        
        # Create attack chain
        chain = AttackChain(
            chain_id="test_chain_001",
            prediction_id="pred_001",
            attack_steps=steps,
            total_steps=2,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=300.0,
            attack_pattern="psexec_lateral_movement",
            kill_chain_stage="lateral_movement",
            severity="HIGH",
            confidence=0.90,
            entities_involved=["DOMAIN\\alice", "powershell.exe", "DC01"],
            techniques_used=["T1059.001", "T1021"],
            tactics_used=["TA0002", "TA0008"]
        )
        
        # Initialize mapper
        mapper = MITREMapper()
        
        # Test mapping
        mapping = mapper.map_attack_chain(chain)
        
        logger.info(f" Mapped chain: {mapping.chain_id}")
        logger.info(f" Techniques found: {len(mapping.techniques)}")
        logger.info(f" Tactics: {mapping.tactics}")
        logger.info(f" Kill chain phases: {mapping.kill_chain_phases}")
        logger.info(f" Coverage score: {mapping.coverage_score:.2f}")
        
        # Print techniques
        for technique in mapping.techniques:
            logger.info(f"  {technique.technique_id}: {technique.name}")
        
        # Test batch mapping
        batch_mappings = mapper.batch_map_chains([chain])
        logger.info(f" Batch mapped {len(batch_mappings)} chains")
        
        # Test report generation
        report = mapper.generate_mitre_report(batch_mappings)
        logger.info(f" Generated MITRE report: {report['summary']}")
        
        # Test technique search
        powershell_techniques = mapper.search_techniques("powershell")
        logger.info(f" Found {len(powershell_techniques)} PowerShell techniques")
        
        # Test export
        output_dir = Path('outputs/mitre_mappings')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mapper.export_mapping(
            mapping,
            str(output_dir / 'test_mapping.json'),
            format='json'
        )
        
        logger.info(" All MITRE mapper tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] MITRE mapper test failed: {e}")
        return False


if __name__ == '__main__':
    """Test the MITRE mapper"""
    
    # Ensure output directory exists
    os.makedirs('outputs/mitre_mappings', exist_ok=True)
    
    # Run tests
    success = test_mitre_mapper()
    
    if success:
        logger.info(" Day 7 Morning MITRE Mapper implementation complete!")
        logger.info(" Ready for explanation text generation")
    else:
        logger.error("[ERROR] MITRE Mapper implementation needs fixes")
        sys.exit(1)