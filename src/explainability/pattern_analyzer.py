#!/usr/bin/env python3
"""
Day 7 Afternoon: Pattern Analysis for THGNN Explainability
Production-ready pattern clustering and attack technique analysis.

Author: THGNN Expert
Date: January 9, 2026
Version: 1.0 - PATTERN ANALYSIS SYSTEM
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import networkx as nx

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PatternAnalysisConfig:
    """Configuration for pattern analysis"""
    min_cluster_size: int = 3
    max_clusters: int = 20
    similarity_threshold: float = 0.7
    min_pattern_frequency: int = 2
    output_dir: str = 'outputs'
    save_patterns: bool = True

class PatternAnalyzer:
    """
    Production-ready pattern analyzer for THGNN explainability.
    Clusters similar explanations and identifies recurring attack techniques.
    """
    
    def __init__(self, config: Optional[PatternAnalysisConfig] = None):
        """
        Initialize the pattern analyzer.
        
        Parameters
        ----------
        config : PatternAnalysisConfig, optional
            Analysis configuration
        """
        self.config = config or PatternAnalysisConfig()
        
        # MITRE ATT&CK technique categories
        self.technique_categories = {
            'T1059': 'Command and Scripting Interpreter',
            'T1021': 'Remote Services',
            'T1078': 'Valid Accounts',
            'T1003': 'OS Credential Dumping',
            'T1087': 'Account Discovery',
            'T1135': 'Network Share Discovery',
            'T1018': 'Remote System Discovery',
            'T1083': 'File and Directory Discovery',
            'T1082': 'System Information Discovery',
            'T1057': 'Process Discovery',
            'T1049': 'System Network Connections Discovery',
            'T1033': 'System Owner/User Discovery',
            'T1007': 'System Service Discovery',
            'T1124': 'System Time Discovery',
            'T1016': 'System Network Configuration Discovery'
        }
        
        # Attack pattern templates
        self.pattern_templates = {
            'lateral_movement': {
                'techniques': ['T1021.002', 'T1021.001', 'T1078'],
                'entities': ['user', 'host'],
                'description': 'Lateral movement using remote services'
            },
            'credential_access': {
                'techniques': ['T1003.001', 'T1003.002', 'T1003.003'],
                'entities': ['process', 'user'],
                'description': 'Credential dumping and access'
            },
            'discovery': {
                'techniques': ['T1087.002', 'T1018', 'T1135'],
                'entities': ['user', 'host', 'process'],
                'description': 'Network and system discovery'
            },
            'execution': {
                'techniques': ['T1059.001', 'T1059.003', 'T1059.005'],
                'entities': ['process', 'user'],
                'description': 'Command execution and scripting'
            }
        }
        
        # Initialize clustering models
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        logger.info("Pattern analyzer initialized successfully")
    
    def cluster_explanations(
        self,
        explanations: List[Dict[str, Any]],
        method: str = 'kmeans'
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Cluster similar explanations based on attack patterns.
        
        Parameters
        ----------
        explanations : List[Dict]
            List of explanation dictionaries
        method : str
            Clustering method ('kmeans' or 'dbscan')
            
        Returns
        -------
        Dict[str, List[Dict]]
            Clustered explanations by pattern
        """
        try:
            logger.info(f"Clustering {len(explanations)} explanations using {method}")
            
            if not explanations:
                return {}
            
            # Extract features for clustering
            features = self._extract_explanation_features(explanations)
            
            if len(features) == 0:
                logger.warning("No features extracted from explanations")
                return {'unclustered': explanations}
            
            # Vectorize explanation text
            explanation_texts = [
                self._explanation_to_text(exp) for exp in explanations
            ]
            
            try:
                text_vectors = self.vectorizer.fit_transform(explanation_texts)
            except ValueError as e:
                logger.error(f"Vectorization failed: {e}")
                return {'unclustered': explanations}
            
            # Combine text and structural features
            if text_vectors.shape[0] != len(features):
                logger.warning("Feature dimension mismatch, using text features only")
                combined_features = text_vectors.toarray()
            else:
                combined_features = np.hstack([
                    text_vectors.toarray(),
                    np.array(features)
                ])
            
            # Perform clustering
            if method == 'kmeans':
                clusters = self._kmeans_clustering(combined_features)
            elif method == 'dbscan':
                clusters = self._dbscan_clustering(combined_features)
            else:
                logger.error(f"Unknown clustering method: {method}")
                return {'unclustered': explanations}
            
            # Group explanations by cluster
            clustered_explanations = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                cluster_name = f"pattern_{cluster_id}" if cluster_id != -1 else "noise"
                clustered_explanations[cluster_name].append(explanations[i])
            
            # Filter small clusters
            filtered_clusters = {}
            for cluster_name, cluster_explanations in clustered_explanations.items():
                if len(cluster_explanations) >= self.config.min_cluster_size or cluster_name == "noise":
                    filtered_clusters[cluster_name] = cluster_explanations
            
            logger.info(f"Created {len(filtered_clusters)} explanation clusters")
            return dict(filtered_clusters)
            
        except Exception as e:
            logger.error(f"Failed to cluster explanations: {e}")
            return {'unclustered': explanations}
    
    def identify_recurring_techniques(
        self,
        attack_chains: List[Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Identify recurring MITRE ATT&CK techniques across attack chains.
        
        Parameters
        ----------
        attack_chains : List[AttackChain]
            List of attack chain objects
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Recurring techniques with frequency and context
        """
        try:
            logger.info(f"Analyzing {len(attack_chains)} attack chains for recurring techniques")
            
            if not attack_chains:
                return {}
            
            # Extract techniques from attack chains
            technique_occurrences = defaultdict(list)
            technique_contexts = defaultdict(lambda: defaultdict(int))
            
            for chain in attack_chains:
                chain_techniques = set()
                attack_steps = getattr(chain, 'attack_steps', [])
                
                for step in attack_steps:
                    technique = getattr(step, 'mitre_technique', None)
                    if technique:
                        chain_techniques.add(technique)
                        
                        # Collect context information
                        context = {
                            'chain_id': getattr(chain, 'chain_id', 'unknown'),
                            'step_number': getattr(step, 'step_number', 0),
                            'importance_score': getattr(step, 'importance_score', 0.0),
                            'event_description': getattr(step, 'event_description', ''),
                            'source_entity': getattr(step, 'source_entity', ''),
                            'target_entity': getattr(step, 'target_entity', '')
                        }
                        technique_occurrences[technique].append(context)
                        
                        # Track entity type combinations
                        source_type = self._extract_entity_type(context['source_entity'])
                        target_type = self._extract_entity_type(context['target_entity'])
                        technique_contexts[technique][f"{source_type}->{target_type}"] += 1
            
            # Filter recurring techniques
            recurring_techniques = {}
            for technique, occurrences in technique_occurrences.items():
                if len(occurrences) >= self.config.min_pattern_frequency:
                    # Calculate statistics
                    importance_scores = [occ['importance_score'] for occ in occurrences]
                    
                    recurring_techniques[technique] = {
                        'frequency': len(occurrences),
                        'avg_importance': np.mean(importance_scores) if importance_scores else 0.0,
                        'max_importance': np.max(importance_scores) if importance_scores else 0.0,
                        'technique_name': self.technique_categories.get(
                            technique.split('.')[0], 'Unknown Technique'
                        ),
                        'entity_patterns': dict(technique_contexts[technique]),
                        'example_chains': [occ['chain_id'] for occ in occurrences[:3]],
                        'common_descriptions': self._get_common_descriptions(occurrences)
                    }
            
            logger.info(f"Identified {len(recurring_techniques)} recurring techniques")
            return recurring_techniques
            
        except Exception as e:
            logger.error(f"Failed to identify recurring techniques: {e}")
            return {}
    
    def generate_pattern_summaries(
        self,
        clustered_explanations: Dict[str, List[Dict[str, Any]]],
        recurring_techniques: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate human-readable summaries of identified patterns.
        
        Parameters
        ----------
        clustered_explanations : Dict[str, List[Dict]]
            Clustered explanations
        recurring_techniques : Dict[str, Dict[str, Any]]
            Recurring techniques analysis
            
        Returns
        -------
        List[Dict[str, Any]]
            Pattern summaries with descriptions and statistics
        """
        try:
            logger.info("Generating pattern summaries")
            
            pattern_summaries = []
            
            # Summarize explanation clusters
            for cluster_name, explanations in clustered_explanations.items():
                if cluster_name == "noise" or len(explanations) < self.config.min_cluster_size:
                    continue
                
                # Extract common elements
                common_techniques = self._extract_common_techniques(explanations)
                common_entities = self._extract_common_entities(explanations)
                avg_confidence = np.mean([
                    exp.get('confidence', 0.0) for exp in explanations
                ])
                
                # Generate description
                description = self._generate_cluster_description(
                    explanations, common_techniques, common_entities
                )
                
                pattern_summary = {
                    'pattern_id': cluster_name,
                    'pattern_type': 'explanation_cluster',
                    'frequency': len(explanations),
                    'avg_confidence': avg_confidence,
                    'common_techniques': common_techniques,
                    'common_entities': common_entities,
                    'description': description,
                    'example_explanations': explanations[:2]  # First 2 as examples
                }
                
                pattern_summaries.append(pattern_summary)
            
            # Summarize recurring techniques
            for technique, details in recurring_techniques.items():
                pattern_summary = {
                    'pattern_id': f"technique_{technique}",
                    'pattern_type': 'recurring_technique',
                    'frequency': details['frequency'],
                    'avg_importance': details['avg_importance'],
                    'technique': technique,
                    'technique_name': details['technique_name'],
                    'entity_patterns': details['entity_patterns'],
                    'description': f"Recurring use of {details['technique_name']} "
                                 f"({technique}) across {details['frequency']} attack chains",
                    'example_chains': details['example_chains']
                }
                
                pattern_summaries.append(pattern_summary)
            
            # Sort by frequency/importance
            pattern_summaries.sort(
                key=lambda x: (x.get('frequency', 0), x.get('avg_importance', 0)),
                reverse=True
            )
            
            logger.info(f"Generated {len(pattern_summaries)} pattern summaries")
            return pattern_summaries
            
        except Exception as e:
            logger.error(f"Failed to generate pattern summaries: {e}")
            return []
    
    def analyze_attack_patterns(
        self,
        explanations: List[Dict[str, Any]],
        attack_chains: List[Any]
    ) -> Dict[str, Any]:
        """
        Complete pattern analysis workflow.
        
        Parameters
        ----------
        explanations : List[Dict]
            List of explanations
        attack_chains : List[AttackChain]
            List of attack chains
            
        Returns
        -------
        Dict[str, Any]
            Complete pattern analysis results
        """
        try:
            logger.info("Starting complete pattern analysis")
            
            # Cluster explanations
            clustered_explanations = self.cluster_explanations(explanations)
            
            # Identify recurring techniques
            recurring_techniques = self.identify_recurring_techniques(attack_chains)
            
            # Generate pattern summaries
            pattern_summaries = self.generate_pattern_summaries(
                clustered_explanations, recurring_techniques
            )
            
            # Create comprehensive analysis result
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'total_explanations': len(explanations),
                'total_attack_chains': len(attack_chains),
                'clustered_explanations': clustered_explanations,
                'recurring_techniques': recurring_techniques,
                'pattern_summaries': pattern_summaries,
                'statistics': {
                    'num_clusters': len([k for k in clustered_explanations.keys() if k != 'noise']),
                    'num_recurring_techniques': len(recurring_techniques),
                    'avg_cluster_size': np.mean([
                        len(v) for k, v in clustered_explanations.items() if k != 'noise'
                    ]) if clustered_explanations else 0,
                    'most_common_technique': max(
                        recurring_techniques.items(),
                        key=lambda x: x[1]['frequency']
                    )[0] if recurring_techniques else None
                }
            }
            
            # Save results if configured
            if self.config.save_patterns:
                self._save_pattern_analysis(analysis_result)
            
            logger.info("Pattern analysis completed successfully")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to complete pattern analysis: {e}")
            return {}
    
    def _extract_explanation_features(self, explanations: List[Dict]) -> List[List[float]]:
        """Extract numerical features from explanations"""
        try:
            features = []
            
            for exp in explanations:
                feature_vector = [
                    exp.get('confidence', 0.0),
                    exp.get('risk_score', 0.0),
                    len(exp.get('influential_nodes', [])),
                    len(exp.get('influential_edges', [])),
                    len(exp.get('attack_chain', [])),
                    len(exp.get('mitre_tactics', []))
                ]
                features.append(feature_vector)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract explanation features: {e}")
            return []
    
    def _explanation_to_text(self, explanation: Dict) -> str:
        """Convert explanation to text for vectorization"""
        try:
            text_parts = []
            
            # Add attack chain descriptions
            attack_chain = explanation.get('attack_chain', [])
            if isinstance(attack_chain, list):
                text_parts.extend(attack_chain)
            
            # Add MITRE tactics
            mitre_tactics = explanation.get('mitre_tactics', [])
            if isinstance(mitre_tactics, list):
                text_parts.extend(mitre_tactics)
            
            # Add entity information
            predicted_edge = explanation.get('predicted_edge', {})
            if isinstance(predicted_edge, dict):
                text_parts.append(predicted_edge.get('type', ''))
                text_parts.append(predicted_edge.get('source', ''))
                text_parts.append(predicted_edge.get('target', ''))
            
            return ' '.join(str(part) for part in text_parts if part)
            
        except Exception as e:
            logger.error(f"Failed to convert explanation to text: {e}")
            return ""
    
    def _kmeans_clustering(self, features: np.ndarray) -> List[int]:
        """Perform K-means clustering"""
        try:
            n_samples = features.shape[0]
            n_clusters = min(self.config.max_clusters, max(2, n_samples // 3))
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            
            clusters = kmeans.fit_predict(features)
            return clusters.tolist()
            
        except Exception as e:
            logger.error(f"K-means clustering failed: {e}")
            return [0] * features.shape[0]
    
    def _dbscan_clustering(self, features: np.ndarray) -> List[int]:
        """Perform DBSCAN clustering"""
        try:
            dbscan = DBSCAN(
                eps=0.5,
                min_samples=self.config.min_cluster_size
            )
            
            clusters = dbscan.fit_predict(features)
            return clusters.tolist()
            
        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {e}")
            return [0] * features.shape[0]
    
    def _extract_entity_type(self, entity_name: str) -> str:
        """Extract entity type from entity name"""
        try:
            if not entity_name or not isinstance(entity_name, str):
                return 'unknown'
            
            entity_lower = entity_name.lower()
            
            if 'user_' in entity_lower or '\\' in entity_name:
                return 'user'
            elif 'host_' in entity_lower or '.' in entity_name:
                return 'host'
            elif 'process_' in entity_lower or '.exe' in entity_lower:
                return 'process'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Failed to extract entity type: {e}")
            return 'unknown'
    
    def _get_common_descriptions(self, occurrences: List[Dict]) -> List[str]:
        """Get most common descriptions from occurrences"""
        try:
            descriptions = [occ.get('event_description', '') for occ in occurrences]
            description_counts = Counter(descriptions)
            
            # Return top 3 most common descriptions
            return [desc for desc, count in description_counts.most_common(3) if desc]
            
        except Exception as e:
            logger.error(f"Failed to get common descriptions: {e}")
            return []
    
    def _extract_common_techniques(self, explanations: List[Dict]) -> List[str]:
        """Extract common MITRE techniques from explanations"""
        try:
            all_techniques = []
            
            for exp in explanations:
                mitre_tactics = exp.get('mitre_tactics', [])
                if isinstance(mitre_tactics, list):
                    all_techniques.extend(mitre_tactics)
            
            technique_counts = Counter(all_techniques)
            threshold = max(1, len(explanations) * 0.3)  # At least 30% of explanations
            
            return [tech for tech, count in technique_counts.items() if count >= threshold]
            
        except Exception as e:
            logger.error(f"Failed to extract common techniques: {e}")
            return []
    
    def _extract_common_entities(self, explanations: List[Dict]) -> List[str]:
        """Extract common entities from explanations"""
        try:
            all_entities = []
            
            for exp in explanations:
                predicted_edge = exp.get('predicted_edge', {})
                if isinstance(predicted_edge, dict):
                    source = predicted_edge.get('source', '')
                    target = predicted_edge.get('target', '')
                    if source:
                        all_entities.append(self._extract_entity_type(source))
                    if target:
                        all_entities.append(self._extract_entity_type(target))
            
            entity_counts = Counter(all_entities)
            threshold = max(1, len(explanations) * 0.3)
            
            return [entity for entity, count in entity_counts.items() if count >= threshold]
            
        except Exception as e:
            logger.error(f"Failed to extract common entities: {e}")
            return []
    
    def _generate_cluster_description(
        self,
        explanations: List[Dict],
        common_techniques: List[str],
        common_entities: List[str]
    ) -> str:
        """Generate human-readable cluster description"""
        try:
            description_parts = []
            
            # Add technique information
            if common_techniques:
                tech_str = ', '.join(common_techniques[:3])
                description_parts.append(f"Common techniques: {tech_str}")
            
            # Add entity information
            if common_entities:
                entity_str = ', '.join(common_entities)
                description_parts.append(f"Involves: {entity_str}")
            
            # Add frequency information
            description_parts.append(f"Observed in {len(explanations)} cases")
            
            # Add confidence information
            confidences = [exp.get('confidence', 0.0) for exp in explanations]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            description_parts.append(f"Average confidence: {avg_confidence:.2f}")
            
            return '. '.join(description_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate cluster description: {e}")
            return f"Pattern cluster with {len(explanations)} explanations"
    
    def _save_pattern_analysis(self, analysis_result: Dict[str, Any]) -> None:
        """Save pattern analysis results to file"""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save complete analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = output_dir / f"pattern_analysis_{timestamp}.json"
            
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, default=str)
            
            # Save attack patterns library
            patterns_file = output_dir / "attack_patterns.json"
            patterns_library = {
                'timestamp': analysis_result['timestamp'],
                'pattern_summaries': analysis_result['pattern_summaries'],
                'recurring_techniques': analysis_result['recurring_techniques'],
                'statistics': analysis_result['statistics']
            }
            
            with open(patterns_file, 'w', encoding='utf-8') as f:
                json.dump(patterns_library, f, indent=2, default=str)
            
            logger.info(f"Pattern analysis saved to {analysis_file}")
            logger.info(f"Attack patterns library saved to {patterns_file}")
            
        except Exception as e:
            logger.error(f"Failed to save pattern analysis: {e}")


def test_pattern_analyzer():
    """Test the pattern analyzer with sample data"""
    
    logger.info("Testing Pattern Analyzer...")
    
    try:
        # Create sample explanations
        explanations = [
            {
                'confidence': 0.85,
                'risk_score': 0.9,
                'predicted_edge': {'source': 'user_1', 'target': 'host_2', 'type': 'logs_into'},
                'influential_nodes': ['user_1', 'process_3'],
                'influential_edges': [{'from': 'user_1', 'to': 'process_3'}],
                'attack_chain': ['User executed PowerShell', 'Lateral movement attempt'],
                'mitre_tactics': ['T1059.001', 'T1021.002']
            },
            {
                'confidence': 0.78,
                'risk_score': 0.85,
                'predicted_edge': {'source': 'user_2', 'target': 'host_3', 'type': 'logs_into'},
                'influential_nodes': ['user_2', 'process_4'],
                'influential_edges': [{'from': 'user_2', 'to': 'process_4'}],
                'attack_chain': ['User executed PowerShell', 'Remote access'],
                'mitre_tactics': ['T1059.001', 'T1078']
            },
            {
                'confidence': 0.92,
                'risk_score': 0.95,
                'predicted_edge': {'source': 'process_1', 'target': 'host_1', 'type': 'connects_to'},
                'influential_nodes': ['process_1', 'user_3'],
                'influential_edges': [{'from': 'process_1', 'to': 'host_1'}],
                'attack_chain': ['Process created', 'Network connection established'],
                'mitre_tactics': ['T1003.001', 'T1087.002']
            }
        ]
        
        # Create sample attack chains
        class MockAttackStep:
            def __init__(self, step_num, technique, description, importance, source, target):
                self.step_number = step_num
                self.mitre_technique = technique
                self.event_description = description
                self.importance_score = importance
                self.source_entity = source
                self.target_entity = target
        
        class MockAttackChain:
            def __init__(self, chain_id, steps):
                self.chain_id = chain_id
                self.attack_steps = steps
        
        attack_chains = [
            MockAttackChain("chain_1", [
                MockAttackStep(1, "T1059.001", "PowerShell execution", 0.9, "user_1", "process_1"),
                MockAttackStep(2, "T1021.002", "PsExec lateral movement", 0.85, "process_1", "host_2")
            ]),
            MockAttackChain("chain_2", [
                MockAttackStep(1, "T1059.001", "PowerShell execution", 0.88, "user_2", "process_2"),
                MockAttackStep(2, "T1078", "Valid account usage", 0.75, "user_2", "host_3")
            ]),
            MockAttackChain("chain_3", [
                MockAttackStep(1, "T1003.001", "Credential dumping", 0.92, "process_3", "user_3"),
                MockAttackStep(2, "T1087.002", "Domain account discovery", 0.80, "user_3", "host_1")
            ])
        ]
        
        # Initialize analyzer
        config = PatternAnalysisConfig(
            min_cluster_size=2,
            max_clusters=5,
            min_pattern_frequency=2,
            save_patterns=True
        )
        
        analyzer = PatternAnalyzer(config)
        logger.info("Pattern analyzer initialized")
        
        # Test clustering
        clustered_explanations = analyzer.cluster_explanations(explanations)
        logger.info(f"Clustered explanations into {len(clustered_explanations)} groups")
        
        # Test recurring techniques
        recurring_techniques = analyzer.identify_recurring_techniques(attack_chains)
        logger.info(f"Identified {len(recurring_techniques)} recurring techniques")
        
        # Test pattern summaries
        pattern_summaries = analyzer.generate_pattern_summaries(
            clustered_explanations, recurring_techniques
        )
        logger.info(f"Generated {len(pattern_summaries)} pattern summaries")
        
        # Test complete analysis
        analysis_result = analyzer.analyze_attack_patterns(explanations, attack_chains)
        logger.info("Complete pattern analysis completed")
        
        # Verify results
        assert len(clustered_explanations) > 0, "No clusters created"
        assert len(recurring_techniques) > 0, "No recurring techniques found"
        assert len(pattern_summaries) > 0, "No pattern summaries generated"
        assert 'statistics' in analysis_result, "Missing statistics in analysis result"
        
        logger.info("All pattern analyzer tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Pattern analyzer test failed: {e}")
        return False


if __name__ == '__main__':
    """Test the pattern analyzer"""
    
    # Ensure output directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Run tests
    success = test_pattern_analyzer()
    
    if success:
        logger.info("Day 7 Afternoon Pattern Analyzer implementation complete!")
        logger.info("Ready for integration with explainability system")
    else:
        logger.error("Pattern Analyzer implementation needs fixes")
        sys.exit(1)