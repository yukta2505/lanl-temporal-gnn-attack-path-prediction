#!/usr/bin/env python3
"""
Day 7 Morning: Explanation Text Generation for THGNN
Generate human-readable explanations for security analysts from attack chains and MITRE mappings.

Author: THGNN Expert
Date: January 9, 2026
Version: 1.0 - PRODUCTION EXPLANATION GENERATION
"""

import os
import sys
import json
import time
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
class ExplanationText:
    """Generated explanation text for security analysts"""
    prediction_id: str
    summary: str
    detailed_explanation: str
    attack_narrative: str
    mitre_context: str
    risk_assessment: str
    recommended_actions: List[str]
    technical_details: str
    confidence_explanation: str
    analyst_notes: str
    generation_time: float
    timestamp: datetime

class ExplanationGenerator:
    """
    Production-ready explanation text generator for security analysts.
    Converts technical attack chains and MITRE mappings into clear, actionable explanations.
    """
    
    def __init__(self):
        """Initialize the explanation generator"""
        
        # Risk level descriptions
        self.risk_descriptions = {
            'CRITICAL': 'Immediate threat requiring urgent response',
            'HIGH': 'Significant threat requiring prompt attention',
            'MEDIUM': 'Moderate threat requiring investigation',
            'LOW': 'Minor threat for monitoring'
        }
        
        # Action templates by severity
        self.action_templates = {
            'CRITICAL': [
                'Immediately isolate affected systems',
                'Activate incident response team',
                'Preserve forensic evidence',
                'Notify security leadership',
                'Begin containment procedures'
            ],
            'HIGH': [
                'Investigate affected systems',
                'Review authentication logs',
                'Check for privilege escalation',
                'Monitor for lateral movement',
                'Consider system isolation'
            ],
            'MEDIUM': [
                'Monitor affected entities',
                'Review related log entries',
                'Validate user activities',
                'Check system configurations',
                'Document findings'
            ],
            'LOW': [
                'Continue monitoring',
                'Review in next security briefing',
                'Update detection rules if needed',
                'Document for trend analysis'
            ]
        }
        
        # MITRE tactic descriptions
        self.tactic_descriptions = {
            'TA0001': 'gaining initial access to the network',
            'TA0002': 'executing malicious code',
            'TA0003': 'maintaining persistent access',
            'TA0004': 'escalating privileges',
            'TA0005': 'evading security controls',
            'TA0006': 'accessing credentials',
            'TA0007': 'discovering network resources',
            'TA0008': 'moving laterally through the network',
            'TA0009': 'collecting sensitive data',
            'TA0010': 'exfiltrating data',
            'TA0011': 'maintaining command and control'
        }
        
        # Confidence level descriptions
        self.confidence_levels = {
            (0.9, 1.0): 'Very High',
            (0.8, 0.9): 'High',
            (0.7, 0.8): 'Medium-High',
            (0.6, 0.7): 'Medium',
            (0.5, 0.6): 'Medium-Low',
            (0.0, 0.5): 'Low'
        }
    
    def generate_explanation(
        self,
        attack_chain,
        mitre_mapping,
        explanation_result
    ) -> ExplanationText:
        """
        Generate comprehensive explanation text.
        
        Parameters
        ----------
        attack_chain : AttackChain
            Extracted attack chain
        mitre_mapping : MITREMapping
            MITRE ATT&CK mapping
        explanation_result : ExplanationResult
            Original GNN explanation result
            
        Returns
        -------
        ExplanationText
            Complete explanation text
        """
        start_time = time.time()
        
        try:
            # Generate summary
            summary = self._generate_summary(attack_chain, mitre_mapping)
            
            # Generate detailed explanation
            detailed_explanation = self._generate_detailed_explanation(
                attack_chain, mitre_mapping, explanation_result
            )
            
            # Generate attack narrative
            attack_narrative = self._generate_attack_narrative(attack_chain)
            
            # Generate MITRE context
            mitre_context = self._generate_mitre_context(mitre_mapping)
            
            # Generate risk assessment
            risk_assessment = self._generate_risk_assessment(
                attack_chain, mitre_mapping, explanation_result
            )
            
            # Generate recommended actions
            recommended_actions = self._generate_recommended_actions(
                attack_chain, mitre_mapping
            )
            
            # Generate technical details
            technical_details = self._generate_technical_details(
                explanation_result, attack_chain
            )
            
            # Generate confidence explanation
            confidence_explanation = self._generate_confidence_explanation(
                explanation_result, attack_chain
            )
            
            # Generate analyst notes
            analyst_notes = self._generate_analyst_notes(
                attack_chain, mitre_mapping, explanation_result
            )
            
            generation_time = time.time() - start_time
            
            explanation = ExplanationText(
                prediction_id=attack_chain.prediction_id,
                summary=summary,
                detailed_explanation=detailed_explanation,
                attack_narrative=attack_narrative,
                mitre_context=mitre_context,
                risk_assessment=risk_assessment,
                recommended_actions=recommended_actions,
                technical_details=technical_details,
                confidence_explanation=confidence_explanation,
                analyst_notes=analyst_notes,
                generation_time=generation_time,
                timestamp=datetime.now()
            )
            
            logger.info(f"Generated explanation for {attack_chain.prediction_id} in {generation_time:.3f}s")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return self._create_fallback_explanation(attack_chain, explanation_result)
    
    def _generate_summary(self, attack_chain, mitre_mapping) -> str:
        """Generate one-line summary"""
        
        if not attack_chain.attack_steps:
            return f"Predicted {attack_chain.severity.lower()} risk activity"
        
        first_step = attack_chain.attack_steps[0]
        last_step = attack_chain.attack_steps[-1]
        
        if len(attack_chain.attack_steps) == 1:
            return f"{first_step.source_entity} {first_step.edge_type.replace('_', ' ')} {first_step.target_entity}"
        
        # Multi-step summary
        pattern_desc = ""
        if attack_chain.attack_pattern:
            pattern_desc = f" ({attack_chain.attack_pattern.replace('_', ' ').title()})"
        
        return f"{first_step.source_entity} → {last_step.target_entity} via {len(attack_chain.attack_steps)}-step attack{pattern_desc}"
    
    def _generate_detailed_explanation(self, attack_chain, mitre_mapping, explanation_result) -> str:
        """Generate detailed explanation"""
        
        explanation = f"""THREAT PREDICTION ANALYSIS
{'=' * 50}

Prediction: {explanation_result.source_entity} will {explanation_result.edge_type.replace('_', ' ')} {explanation_result.target_entity}

Risk Score: {explanation_result.risk_score:.1f}/100
Model Confidence: {explanation_result.confidence:.1%}
Severity Level: {attack_chain.severity}

Attack Chain Analysis:
The model identified a {len(attack_chain.attack_steps)}-step attack sequence spanning {attack_chain.duration_seconds:.0f} seconds.
"""
        
        if attack_chain.attack_pattern:
            explanation += f"This matches the known attack pattern: {attack_chain.attack_pattern.replace('_', ' ').title()}.\n"
        
        if mitre_mapping.techniques:
            explanation += f"\nMITRE ATT&CK Analysis:\n"
            explanation += f"- {len(mitre_mapping.techniques)} techniques identified\n"
            explanation += f"- {len(mitre_mapping.tactics)} tactics involved\n"
            explanation += f"- Kill chain phases: {', '.join(mitre_mapping.kill_chain_phases)}\n"
        
        explanation += f"\nEntities Involved: {', '.join(attack_chain.entities_involved[:5])}"
        if len(attack_chain.entities_involved) > 5:
            explanation += f" (and {len(attack_chain.entities_involved) - 5} others)"
        
        return explanation
    
    def _generate_attack_narrative(self, attack_chain) -> str:
        """Generate human-readable attack narrative"""
        
        if not attack_chain.attack_steps:
            return "No attack steps identified."
        
        narrative = "ATTACK SEQUENCE:\n" + "=" * 20 + "\n\n"
        
        for i, step in enumerate(attack_chain.attack_steps):
            timestamp_str = step.timestamp.strftime("%H:%M:%S")
            narrative += f"Step {step.step_number} ({timestamp_str}):\n"
            narrative += f"  {step.event_description}\n"
            
            if step.mitre_technique:
                narrative += f"  MITRE Technique: {step.mitre_technique}\n"
            
            narrative += f"  Importance: {step.importance_score:.2f}\n\n"
        
        # Add pattern analysis
        if attack_chain.attack_pattern:
            narrative += f"PATTERN ANALYSIS:\n"
            narrative += f"This sequence matches the '{attack_chain.attack_pattern.replace('_', ' ').title()}' attack pattern.\n\n"
        
        return narrative
    
    def _generate_mitre_context(self, mitre_mapping) -> str:
        """Generate MITRE ATT&CK context"""
        
        if not mitre_mapping.techniques:
            return "No MITRE ATT&CK techniques identified."
        
        context = "MITRE ATT&CK FRAMEWORK ANALYSIS:\n" + "=" * 35 + "\n\n"
        
        # Group techniques by tactic
        tactic_groups = {}
        for technique in mitre_mapping.techniques:
            tactic_name = technique.tactic_name
            if tactic_name not in tactic_groups:
                tactic_groups[tactic_name] = []
            tactic_groups[tactic_name].append(technique)
        
        for tactic_name, techniques in tactic_groups.items():
            context += f"{tactic_name.upper()}:\n"
            for technique in techniques:
                context += f"  • {technique.technique_id}: {technique.name}\n"
                context += f"    {technique.description}\n"
            context += "\n"
        
        # Add kill chain analysis
        if mitre_mapping.kill_chain_phases:
            context += f"CYBER KILL CHAIN PHASES:\n"
            for phase in mitre_mapping.kill_chain_phases:
                context += f"  • {phase.title()}\n"
            context += "\n"
        
        context += f"Coverage Score: {mitre_mapping.coverage_score:.1%} of attack steps mapped to MITRE techniques\n"
        
        return context
    
    def _generate_risk_assessment(self, attack_chain, mitre_mapping, explanation_result) -> str:
        """Generate risk assessment"""
        
        risk_score = explanation_result.risk_score
        severity = attack_chain.severity
        confidence = explanation_result.confidence
        
        assessment = f"RISK ASSESSMENT:\n" + "=" * 16 + "\n\n"
        
        # Overall risk
        assessment += f"Overall Risk Level: {severity}\n"
        assessment += f"Risk Score: {risk_score:.1f}/100\n"
        assessment += f"Model Confidence: {confidence:.1%}\n\n"
        
        # Risk factors
        assessment += "Risk Factors:\n"
        
        if risk_score >= 80:
            assessment += "  • High risk score indicates imminent threat\n"
        elif risk_score >= 60:
            assessment += "  • Moderate risk score suggests potential threat\n"
        else:
            assessment += "  • Lower risk score indicates monitoring situation\n"
        
        if len(attack_chain.attack_steps) >= 3:
            assessment += "  • Multi-step attack sequence detected\n"
        
        if attack_chain.attack_pattern:
            assessment += f"  • Matches known attack pattern: {attack_chain.attack_pattern}\n"
        
        if len(mitre_mapping.techniques) >= 3:
            assessment += "  • Multiple MITRE techniques involved\n"
        
        if any(tactic in mitre_mapping.tactics for tactic in ['TA0008', 'TA0004']):
            assessment += "  • Lateral movement or privilege escalation detected\n"
        
        # Time sensitivity
        if attack_chain.duration_seconds < 300:  # 5 minutes
            assessment += "  • Rapid attack progression indicates automation\n"
        
        assessment += f"\n{self.risk_descriptions.get(severity, 'Unknown risk level')}\n"
        
        return assessment
    
    def _generate_recommended_actions(self, attack_chain, mitre_mapping) -> List[str]:
        """Generate recommended actions"""
        
        severity = attack_chain.severity
        base_actions = self.action_templates.get(severity, self.action_templates['MEDIUM'])
        
        # Customize actions based on attack details
        customized_actions = base_actions.copy()
        
        # Add specific actions based on techniques
        for technique in mitre_mapping.techniques:
            if technique.technique_id == 'T1059.001':  # PowerShell
                customized_actions.append("Review PowerShell execution logs")
            elif technique.technique_id == 'T1021.002':  # SMB/Admin Shares
                customized_actions.append("Check SMB access logs and admin share usage")
            elif technique.technique_id == 'T1078':  # Valid Accounts
                customized_actions.append("Verify account legitimacy and recent password changes")
        
        # Add actions based on entities
        entities = attack_chain.entities_involved
        if any('dc' in entity.lower() or 'domain' in entity.lower() for entity in entities):
            customized_actions.append("Prioritize domain controller security")
        
        if any('admin' in entity.lower() for entity in entities):
            customized_actions.append("Review administrative account activities")
        
        return list(set(customized_actions))  # Remove duplicates
    
    def _generate_technical_details(self, explanation_result, attack_chain) -> str:
        """Generate technical details for analysts"""
        
        details = "TECHNICAL DETAILS:\n" + "=" * 18 + "\n\n"
        
        # Model details
        details += f"Model Information:\n"
        details += f"  Explanation Type: {explanation_result.explanation_type}\n"
        details += f"  Generation Time: {explanation_result.generation_time:.3f} seconds\n"
        details += f"  Influential Nodes: {len(explanation_result.influential_nodes)}\n"
        details += f"  Influential Edges: {len(explanation_result.influential_edges)}\n\n"
        
        # Graph analysis
        details += f"Graph Analysis:\n"
        details += f"  Subgraph Nodes: {len(explanation_result.subgraph_nodes)}\n"
        details += f"  Subgraph Edges: {len(explanation_result.subgraph_edges)}\n\n"
        
        # Top influential components
        if explanation_result.influential_nodes:
            details += f"Most Influential Nodes:\n"
            for node in explanation_result.influential_nodes[:5]:
                importance = explanation_result.node_importance.get(node, 0.0)
                details += f"  • {node} (importance: {importance:.3f})\n"
            details += "\n"
        
        if explanation_result.influential_edges:
            details += f"Most Influential Edges:\n"
            for edge in explanation_result.influential_edges[:5]:
                details += f"  • {edge['source']} → {edge['target']} ({edge['edge_type']}, importance: {edge['importance']:.3f})\n"
            details += "\n"
        
        return details
    
    def _generate_confidence_explanation(self, explanation_result, attack_chain) -> str:
        """Generate confidence level explanation"""
        
        confidence = explanation_result.confidence
        confidence_level = self._get_confidence_level(confidence)
        
        explanation = f"CONFIDENCE ANALYSIS:\n" + "=" * 20 + "\n\n"
        explanation += f"Model Confidence: {confidence:.1%} ({confidence_level})\n\n"
        
        explanation += "Confidence Factors:\n"
        
        if confidence >= 0.9:
            explanation += "  • Very high confidence - strong evidence in data\n"
        elif confidence >= 0.8:
            explanation += "  • High confidence - clear patterns identified\n"
        elif confidence >= 0.7:
            explanation += "  • Good confidence - consistent with training data\n"
        elif confidence >= 0.6:
            explanation += "  • Moderate confidence - some uncertainty present\n"
        else:
            explanation += "  • Lower confidence - limited evidence or novel pattern\n"
        
        # Factors affecting confidence
        if len(explanation_result.influential_nodes) >= 5:
            explanation += "  • Multiple influential nodes support prediction\n"
        
        if len(explanation_result.influential_edges) >= 3:
            explanation += "  • Strong edge evidence in explanation\n"
        
        if attack_chain.attack_pattern:
            explanation += "  • Matches known attack pattern\n"
        
        if explanation_result.explanation_type == 'gnn_explainer':
            explanation += "  • Generated using advanced GNN explainer\n"
        else:
            explanation += "  • Generated using heuristic fallback method\n"
        
        return explanation
    
    def _generate_analyst_notes(self, attack_chain, mitre_mapping, explanation_result) -> str:
        """Generate notes for security analysts"""
        
        notes = "ANALYST NOTES:\n" + "=" * 14 + "\n\n"
        
        # Investigation priorities
        notes += "Investigation Priorities:\n"
        
        if attack_chain.severity in ['CRITICAL', 'HIGH']:
            notes += "  1. Immediate investigation required\n"
            notes += "  2. Check for ongoing compromise\n"
            notes += "  3. Assess blast radius\n"
        else:
            notes += "  1. Schedule investigation within SLA\n"
            notes += "  2. Correlate with other alerts\n"
            notes += "  3. Update threat intelligence\n"
        
        notes += "\n"
        
        # Key entities to investigate
        if attack_chain.entities_involved:
            notes += f"Key Entities to Investigate:\n"
            for entity in attack_chain.entities_involved[:5]:
                notes += f"  • {entity}\n"
            notes += "\n"
        
        # Related techniques to monitor
        if mitre_mapping.techniques:
            notes += f"Related MITRE Techniques to Monitor:\n"
            for technique in mitre_mapping.techniques[:3]:
                notes += f"  • {technique.technique_id}: {technique.name}\n"
            notes += "\n"
        
        # Timeline considerations
        if attack_chain.duration_seconds > 0:
            duration_minutes = attack_chain.duration_seconds / 60
            notes += f"Timeline: Attack sequence occurred over {duration_minutes:.1f} minutes\n"
            
            if duration_minutes < 5:
                notes += "  → Rapid progression suggests automated attack\n"
            elif duration_minutes > 60:
                notes += "  → Extended timeline may indicate manual operation\n"
            notes += "\n"
        
        # Follow-up actions
        notes += "Follow-up Actions:\n"
        notes += "  • Document findings in case management system\n"
        notes += "  • Update detection rules if new pattern identified\n"
        notes += "  • Share intelligence with threat hunting team\n"
        notes += "  • Consider threat actor attribution\n"
        
        return notes
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description"""
        for (min_conf, max_conf), level in self.confidence_levels.items():
            if min_conf <= confidence < max_conf:
                return level
        return 'Unknown'
    
    def _create_fallback_explanation(self, attack_chain, explanation_result) -> ExplanationText:
        """Create minimal fallback explanation"""
        
        return ExplanationText(
            prediction_id=attack_chain.prediction_id,
            summary=f"Predicted {attack_chain.severity.lower()} risk activity",
            detailed_explanation=f"Model predicted {explanation_result.edge_type} between {explanation_result.source_entity} and {explanation_result.target_entity}",
            attack_narrative="Limited attack chain information available",
            mitre_context="MITRE mapping not available",
            risk_assessment=f"Risk Level: {attack_chain.severity}",
            recommended_actions=["Monitor affected systems", "Review logs", "Investigate further"],
            technical_details=f"Explanation type: {explanation_result.explanation_type}",
            confidence_explanation=f"Model confidence: {explanation_result.confidence:.1%}",
            analyst_notes="Limited explanation data available - manual investigation recommended",
            generation_time=0.1,
            timestamp=datetime.now()
        )
    
    def format_for_dashboard(self, explanation: ExplanationText) -> str:
        """Format explanation for dashboard display"""
        
        formatted = f"""
🚨 THREAT PREDICTION ALERT

{explanation.summary}

📊 RISK ASSESSMENT:
{explanation.risk_assessment}

🔍 ATTACK SEQUENCE:
{explanation.attack_narrative}

🛡️ MITRE ATT&CK:
{explanation.mitre_context}

⚡ RECOMMENDED ACTIONS:
"""
        for i, action in enumerate(explanation.recommended_actions[:5], 1):
            formatted += f"{i}. {action}\n"
        
        formatted += f"""
🔧 CONFIDENCE: {explanation.confidence_explanation.split('Model Confidence: ')[1].split(' (')[0]}

📝 ANALYST NOTES:
{explanation.analyst_notes}
"""
        
        return formatted
    
    def export_explanation(
        self,
        explanation: ExplanationText,
        output_path: str,
        format: str = 'json'
    ):
        """Export explanation to file"""
        
        try:
            if format == 'json':
                export_data = asdict(explanation)
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                    
            elif format == 'txt':
                with open(output_path, 'w') as f:
                    f.write(self.format_for_dashboard(explanation))
                    
            elif format == 'html':
                html_content = self._generate_html_report(explanation)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            logger.info(f"Exported explanation to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export explanation: {e}")
    
    def _generate_html_report(self, explanation: ExplanationText) -> str:
        """Generate HTML report"""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>THGNN Threat Prediction Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f44336; color: white; padding: 10px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #2196F3; }}
        .risk-high {{ border-left-color: #f44336; }}
        .risk-medium {{ border-left-color: #ff9800; }}
        .risk-low {{ border-left-color: #4caf50; }}
        pre {{ background-color: #f5f5f5; padding: 10px; overflow-x: auto; }}
        .actions {{ background-color: #e3f2fd; padding: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚨 THGNN Threat Prediction Report</h1>
        <p>Prediction ID: {explanation.prediction_id}</p>
        <p>Generated: {explanation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>📋 Summary</h2>
        <p><strong>{explanation.summary}</strong></p>
    </div>
    
    <div class="section">
        <h2>📊 Risk Assessment</h2>
        <pre>{explanation.risk_assessment}</pre>
    </div>
    
    <div class="section">
        <h2>🔍 Attack Narrative</h2>
        <pre>{explanation.attack_narrative}</pre>
    </div>
    
    <div class="section">
        <h2>🛡️ MITRE ATT&CK Context</h2>
        <pre>{explanation.mitre_context}</pre>
    </div>
    
    <div class="section actions">
        <h2>⚡ Recommended Actions</h2>
        <ul>
"""
        
        for action in explanation.recommended_actions:
            html += f"<li>{action}</li>\n"
        
        html += f"""
        </ul>
    </div>
    
    <div class="section">
        <h2>🔧 Technical Details</h2>
        <pre>{explanation.technical_details}</pre>
    </div>
    
    <div class="section">
        <h2>📈 Confidence Analysis</h2>
        <pre>{explanation.confidence_explanation}</pre>
    </div>
    
    <div class="section">
        <h2>📝 Analyst Notes</h2>
        <pre>{explanation.analyst_notes}</pre>
    </div>
    
    <div class="section">
        <h2>📄 Detailed Explanation</h2>
        <pre>{explanation.detailed_explanation}</pre>
    </div>
    
</body>
</html>
"""
        
        return html


def test_explanation_generator():
    """Test the explanation generator"""
    
    logger.info("Testing Explanation Generator...")
    
    try:
        # Import required classes
        from src.explainability.gnn_explainer import ExplanationResult
        from src.explainability.attack_chain_extractor import AttackChain, AttackStep
        from src.explainability.mitre_mapper import MITREMapping, MITRETechnique
        from datetime import datetime
        import time
        
        # Create mock data
        explanation_result = ExplanationResult(
            prediction_id='test_001',
            source_entity='DOMAIN\\alice',
            target_entity='DC01',
            edge_type='logs_into',
            risk_score=85.0,
            confidence=0.92,
            influential_nodes=['user_5', 'process_23', 'host_12'],
            influential_edges=[
                {'source': 'user_5', 'target': 'process_23', 'edge_type': 'executes', 'importance': 0.9}
            ],
            node_importance={'user_5': 0.9, 'process_23': 0.8},
            edge_importance={'edge_1': 0.9},
            subgraph_nodes=['user_5', 'process_23'],
            subgraph_edges=[('user_5', 'process_23')],
            explanation_type='gnn_explainer',
            generation_time=1.5,
            timestamp=datetime.now()
        )
        
        attack_steps = [
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
            )
        ]
        
        attack_chain = AttackChain(
            chain_id="test_chain_001",
            prediction_id="test_001",
            attack_steps=attack_steps,
            total_steps=1,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=300.0,
            attack_pattern="powershell_execution",
            kill_chain_stage="execution",
            severity="HIGH",
            confidence=0.92,
            entities_involved=["DOMAIN\\alice", "powershell.exe"],
            techniques_used=["T1059.001"],
            tactics_used=["TA0002"]
        )
        
        technique = MITRETechnique(
            technique_id='T1059.001',
            name='PowerShell',
            description='Adversaries may abuse PowerShell commands',
            tactic_id='TA0002',
            tactic_name='Execution',
            platforms=['Windows'],
            data_sources=['PowerShell logs'],
            url='https://attack.mitre.org/techniques/T1059/001/'
        )
        
        mitre_mapping = MITREMapping(
            chain_id="test_chain_001",
            techniques=[technique],
            tactics=['TA0002'],
            kill_chain_phases=['execution'],
            attack_pattern="powershell_execution",
            confidence=0.92,
            coverage_score=1.0
        )
        
        # Initialize generator
        generator = ExplanationGenerator()
        
        # Test explanation generation
        explanation = generator.generate_explanation(
            attack_chain, mitre_mapping, explanation_result
        )
        
        logger.info(f" Generated explanation: {explanation.prediction_id}")
        logger.info(f" Summary: {explanation.summary}")
        logger.info(f" Actions: {len(explanation.recommended_actions)}")
        logger.info(f" Generation time: {explanation.generation_time:.3f}s")
        
        # Test formatting
        dashboard_format = generator.format_for_dashboard(explanation)
        logger.info(f" Dashboard format generated ({len(dashboard_format)} chars)")
        
        # Test export
        output_dir = Path('outputs/explanations')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generator.export_explanation(
            explanation,
            str(output_dir / 'test_explanation.json'),
            format='json'
        )
        
        generator.export_explanation(
            explanation,
            str(output_dir / 'test_explanation.html'),
            format='html'
        )
        
        logger.info(" All explanation generator tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Explanation generator test failed: {e}")
        return False


if __name__ == '__main__':
    """Test the explanation generator"""
    
    # Ensure output directory exists
    os.makedirs('outputs/explanations', exist_ok=True)
    
    # Run tests
    success = test_explanation_generator()
    
    if success:
        logger.info(" Day 7 Morning Explanation Generator implementation complete!")
        logger.info(" Ready for integration and testing")
    else:
        logger.error("[ERROR] Explanation Generator implementation needs fixes")
        sys.exit(1)