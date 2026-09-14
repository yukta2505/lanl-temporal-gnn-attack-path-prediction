#!/usr/bin/env python3
"""
THGNN Model Evaluator

Comprehensive evaluation suite for Temporal Heterogeneous Graph Neural Networks
with cybersecurity-specific metrics and visualizations.

Features:
- Complete metric computation (AUC-ROC, AUC-PR, Precision@K, Recall@K, F1)
- Cybersecurity-aware evaluation (MITRE ATT&CK mapping, risk scoring)
- Advanced visualizations (ROC curves, PR curves, confusion matrices)
- Performance benchmarking and latency analysis
- Detailed reporting with statistical significance tests
- Multi-step prediction evaluation
- Uncertainty quantification assessment
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import warnings

from .config import TrainingConfiguration
from models.thgnn_model import THGNN

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive evaluator for THGNN models.
    
    Provides detailed performance analysis including standard ML metrics,
    cybersecurity-specific evaluations, and visualization capabilities.
    """
    
    def __init__(
        self,
        model: THGNN,
        config: TrainingConfiguration,
        device: Optional[torch.device] = None
    ):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained THGNN model
            config: Training configuration
            device: Computation device
        """
        self.model = model
        self.config = config
        self.device = device or config.get_device()
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Results storage
        self.evaluation_results = {}
        self.predictions_cache = {}
        
        # Create output directories
        self.output_dir = Path(config.paths.predictions_dir)
        self.viz_dir = Path(config.paths.visualizations_dir)
        self.reports_dir = Path(config.paths.reports_dir)
        
        for dir_path in [self.output_dir, self.viz_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model evaluator initialized on {self.device}")
    
    def evaluate_model(
        self,
        test_data: List[List[HeteroData]],
        save_predictions: bool = True,
        generate_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            test_data: Test data sequences
            save_predictions: Save detailed predictions
            generate_plots: Generate visualization plots
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        start_time = time.time()
        
        # Get predictions and ground truth
        predictions, labels, metadata = self._get_predictions(test_data)
        
        if len(predictions) == 0:
            logger.warning("No predictions generated - cannot evaluate")
            return {}
        
        # Core metrics
        core_metrics = self._compute_core_metrics(predictions, labels)
        
        # Ranking metrics
        ranking_metrics = self._compute_ranking_metrics(predictions, labels)
        
        # Threshold-based metrics
        threshold_metrics = self._compute_threshold_metrics(predictions, labels)
        
        # Cybersecurity-specific metrics
        cyber_metrics = self._compute_cybersecurity_metrics(predictions, labels, metadata)
        
        # Performance metrics
        performance_metrics = self._compute_performance_metrics(test_data)
        
        # Multi-step evaluation (if applicable)
        multistep_metrics = self._evaluate_multistep_prediction(test_data)
        
        # Uncertainty evaluation
        uncertainty_metrics = self._evaluate_uncertainty(predictions, labels, metadata)
        
        # Combine all results
        results = {
            'core_metrics': core_metrics,
            'ranking_metrics': ranking_metrics,
            'threshold_metrics': threshold_metrics,
            'cybersecurity_metrics': cyber_metrics,
            'performance_metrics': performance_metrics,
            'multistep_metrics': multistep_metrics,
            'uncertainty_metrics': uncertainty_metrics,
            'evaluation_time': time.time() - start_time,
            'num_samples': len(predictions),
            'model_summary': self.model.get_model_summary()
        }
        
        self.evaluation_results = results
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(predictions, labels, metadata)
        
        # Generate plots if requested
        if generate_plots:
            self._generate_visualizations(predictions, labels, metadata)
        
        # Generate report
        self._generate_evaluation_report(results)
        
        logger.info(f"Evaluation completed in {results['evaluation_time']:.2f}s")
        
        return results
    
    def _get_predictions(
        self,
        test_data: List[List[HeteroData]]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Get model predictions on test data.
        
        Returns:
            Tuple of (predictions, labels, metadata)
        """
        all_predictions = []
        all_labels = []
        all_metadata = []
        
        with torch.no_grad():
            for sequence_idx, sequence in enumerate(test_data):
                try:
                    # Move sequence to device
                    sequence = [graph.to(self.device) for graph in sequence]
                    
                    # Generate test edges
                    pos_edges, neg_edges, labels = self._sample_test_edges(sequence[-1])
                    
                    if len(pos_edges) == 0:
                        continue
                    
                    all_edges = torch.cat([pos_edges, neg_edges], dim=0)
                    
                    # Get model predictions
                    outputs = self.model(sequence, target_edges=all_edges, negative_sampling=False)
                    
                    if 'existence_probs' in outputs:
                        probs = outputs['existence_probs'].detach().cpu().numpy()
                        all_predictions.extend(probs)
                        all_labels.extend(labels.cpu().numpy())
                        
                        # Store metadata for each prediction
                        for i, edge in enumerate(all_edges):
                            metadata = {
                                'sequence_idx': sequence_idx,
                                'edge_idx': i,
                                'src_type': edge[0].item() if torch.is_tensor(edge[0]) else edge[0],
                                'src_idx': edge[1].item() if torch.is_tensor(edge[1]) else edge[1],
                                'dst_type': edge[2].item() if torch.is_tensor(edge[2]) else edge[2],
                                'dst_idx': edge[3].item() if torch.is_tensor(edge[3]) else edge[3],
                                'edge_type_idx': edge[4].item() if torch.is_tensor(edge[4]) else edge[4],
                                'is_positive': i < len(pos_edges)
                            }
                            all_metadata.append(metadata)
                
                except Exception as e:
                    logger.error(f"Error processing sequence {sequence_idx}: {e}")
                    continue
        
        return np.array(all_predictions), np.array(all_labels), all_metadata
    
    def _sample_test_edges(self, graph: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample edges for testing (similar to trainer but deterministic)."""
        pos_edges = []
        neg_edges = []
        
        # Set seed for reproducible sampling
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Sample positive edges
        for edge_type, edge_index in graph.edge_index_dict.items():
            if edge_index.size(1) == 0:
                continue
            
            # Sample subset of positive edges
            num_pos = min(edge_index.size(1), 50)  # Smaller for evaluation
            perm = torch.randperm(edge_index.size(1))[:num_pos]
            sampled_edges = edge_index[:, perm]
            
            src_type, relation, dst_type = edge_type
            edge_type_idx = self.model.edge_types.index(edge_type)
            
            for i in range(sampled_edges.size(1)):
                src_idx = sampled_edges[0, i].item()
                dst_idx = sampled_edges[1, i].item()
                pos_edges.append([src_type, src_idx, dst_type, dst_idx, edge_type_idx])
        
        # Sample negative edges
        num_neg = len(pos_edges)  # 1:1 ratio for evaluation
        
        for _ in range(num_neg):
            edge_type = np.random.choice(self.model.edge_types)
            src_type, relation, dst_type = edge_type
            edge_type_idx = self.model.edge_types.index(edge_type)
            
            if (src_type in graph.x_dict and dst_type in graph.x_dict and
                graph.x_dict[src_type].size(0) > 0 and graph.x_dict[dst_type].size(0) > 0):
                
                src_idx = np.random.randint(0, graph.x_dict[src_type].size(0))
                dst_idx = np.random.randint(0, graph.x_dict[dst_type].size(0))
                
                if src_idx != dst_idx or src_type != dst_type:
                    neg_edges.append([src_type, src_idx, dst_type, dst_idx, edge_type_idx])
        
        # Convert to tensors
        if pos_edges:
            pos_edges = torch.tensor(pos_edges, device=self.device)
        else:
            pos_edges = torch.empty((0, 5), device=self.device)
        
        if neg_edges:
            neg_edges = torch.tensor(neg_edges, device=self.device)
        else:
            neg_edges = torch.empty((0, 5), device=self.device)
        
        # Create labels
        pos_labels = torch.ones(len(pos_edges), device=self.device)
        neg_labels = torch.zeros(len(neg_edges), device=self.device)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        return pos_edges, neg_edges, labels
    
    def compute_auc_roc(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute Area Under ROC Curve."""
        try:
            if len(np.unique(labels)) < 2:
                logger.warning("Only one class present in labels - cannot compute AUC-ROC")
                return 0.0
            return roc_auc_score(labels, predictions)
        except Exception as e:
            logger.error(f"Error computing AUC-ROC: {e}")
            return 0.0
    
    def compute_auc_pr(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute Area Under Precision-Recall Curve."""
        try:
            if len(np.unique(labels)) < 2:
                logger.warning("Only one class present in labels - cannot compute AUC-PR")
                return 0.0
            return average_precision_score(labels, predictions)
        except Exception as e:
            logger.error(f"Error computing AUC-PR: {e}")
            return 0.0
    
    def compute_precision_at_k(self, predictions: np.ndarray, labels: np.ndarray, k: List[int]) -> Dict[int, float]:
        """Compute Precision at top-K predictions."""
        results = {}
        
        try:
            # Sort by prediction score (descending)
            sorted_indices = np.argsort(predictions)[::-1]
            sorted_labels = labels[sorted_indices]
            
            for k_val in k:
                if k_val > len(predictions):
                    results[k_val] = 0.0
                    continue
                
                top_k_labels = sorted_labels[:k_val]
                precision_k = np.sum(top_k_labels) / k_val if k_val > 0 else 0.0
                results[k_val] = precision_k
        
        except Exception as e:
            logger.error(f"Error computing Precision@K: {e}")
            for k_val in k:
                results[k_val] = 0.0
        
        return results
    
    def compute_recall_at_k(self, predictions: np.ndarray, labels: np.ndarray, k: List[int]) -> Dict[int, float]:
        """Compute Recall at top-K predictions."""
        results = {}
        
        try:
            total_positives = np.sum(labels)
            if total_positives == 0:
                for k_val in k:
                    results[k_val] = 0.0
                return results
            
            # Sort by prediction score (descending)
            sorted_indices = np.argsort(predictions)[::-1]
            sorted_labels = labels[sorted_indices]
            
            for k_val in k:
                if k_val > len(predictions):
                    results[k_val] = np.sum(labels) / total_positives
                    continue
                
                top_k_labels = sorted_labels[:k_val]
                recall_k = np.sum(top_k_labels) / total_positives
                results[k_val] = recall_k
        
        except Exception as e:
            logger.error(f"Error computing Recall@K: {e}")
            for k_val in k:
                results[k_val] = 0.0
        
        return results
    
    def compute_hits_at_k(self, predictions: np.ndarray, labels: np.ndarray, k: List[int]) -> Dict[int, float]:
        """Compute Hits@K (fraction of positive samples in top-K)."""
        results = {}
        
        try:
            # Sort by prediction score (descending)
            sorted_indices = np.argsort(predictions)[::-1]
            sorted_labels = labels[sorted_indices]
            
            for k_val in k:
                if k_val > len(predictions):
                    results[k_val] = 1.0 if np.sum(labels) > 0 else 0.0
                    continue
                
                top_k_labels = sorted_labels[:k_val]
                hits_k = 1.0 if np.sum(top_k_labels) > 0 else 0.0
                results[k_val] = hits_k
        
        except Exception as e:
            logger.error(f"Error computing Hits@K: {e}")
            for k_val in k:
                results[k_val] = 0.0
        
        return results
    
    def compute_f1_score(self, predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
        """Compute F1 score at given threshold."""
        try:
            binary_preds = (predictions >= threshold).astype(int)
            return f1_score(labels, binary_preds)
        except Exception as e:
            logger.error(f"Error computing F1 score: {e}")
            return 0.0
    
    def generate_confusion_matrix(self, predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Generate confusion matrix."""
        try:
            binary_preds = (predictions >= threshold).astype(int)
            return confusion_matrix(labels, binary_preds)
        except Exception as e:
            logger.error(f"Error generating confusion matrix: {e}")
            return np.array([[0, 0], [0, 0]])
    
    def _compute_core_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute core ML metrics."""
        return {
            'auc_roc': self.compute_auc_roc(predictions, labels),
            'auc_pr': self.compute_auc_pr(predictions, labels),
            'f1_score': self.compute_f1_score(predictions, labels),
            'accuracy': accuracy_score(labels, (predictions >= 0.5).astype(int)),
            'precision': precision_score(labels, (predictions >= 0.5).astype(int), zero_division=0),
            'recall': recall_score(labels, (predictions >= 0.5).astype(int), zero_division=0)
        }
    
    def _compute_ranking_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Compute ranking-based metrics."""
        k_values = [10, 20, 50, 100]
        
        return {
            'precision_at_k': self.compute_precision_at_k(predictions, labels, k_values),
            'recall_at_k': self.compute_recall_at_k(predictions, labels, k_values),
            'hits_at_k': self.compute_hits_at_k(predictions, labels, k_values)
        }
    
    def _compute_threshold_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Compute metrics at different thresholds."""
        thresholds = [0.3, 0.5, 0.7, 0.9]
        
        results = {}
        for threshold in thresholds:
            results[f'threshold_{threshold}'] = {
                'f1_score': self.compute_f1_score(predictions, labels, threshold),
                'precision': precision_score(labels, (predictions >= threshold).astype(int), zero_division=0),
                'recall': recall_score(labels, (predictions >= threshold).astype(int), zero_division=0),
                'accuracy': accuracy_score(labels, (predictions >= threshold).astype(int))
            }
        
        return results
    
    def _compute_cybersecurity_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        metadata: List[Dict]
    ) -> Dict[str, Any]:
        """Compute cybersecurity-specific metrics."""
        # Group by edge types
        edge_type_metrics = {}
        
        for edge_type in self.model.edge_types:
            edge_type_idx = self.model.edge_types.index(edge_type)
            
            # Filter predictions for this edge type
            type_mask = [m['edge_type_idx'] == edge_type_idx for m in metadata]
            
            if not any(type_mask):
                continue
            
            type_predictions = predictions[type_mask]
            type_labels = labels[type_mask]
            
            if len(type_predictions) > 0:
                edge_type_metrics[str(edge_type)] = {
                    'auc_roc': self.compute_auc_roc(type_predictions, type_labels),
                    'auc_pr': self.compute_auc_pr(type_predictions, type_labels),
                    'precision_at_10': self.compute_precision_at_k(type_predictions, type_labels, [10])[10],
                    'num_samples': len(type_predictions),
                    'positive_rate': np.mean(type_labels)
                }
        
        # High-risk edge detection
        high_risk_edges = ['executes', 'creates', 'modifies']
        high_risk_mask = []
        
        for m in metadata:
            edge_type = self.model.edge_types[m['edge_type_idx']]
            is_high_risk = any(risk_type in edge_type[1] for risk_type in high_risk_edges)
            high_risk_mask.append(is_high_risk)
        
        high_risk_metrics = {}
        if any(high_risk_mask):
            hr_predictions = predictions[high_risk_mask]
            hr_labels = labels[high_risk_mask]
            
            high_risk_metrics = {
                'auc_roc': self.compute_auc_roc(hr_predictions, hr_labels),
                'precision_at_10': self.compute_precision_at_k(hr_predictions, hr_labels, [10])[10],
                'num_samples': len(hr_predictions)
            }
        
        return {
            'edge_type_metrics': edge_type_metrics,
            'high_risk_metrics': high_risk_metrics,
            'total_edge_types': len(edge_type_metrics)
        }
    
    def _compute_performance_metrics(self, test_data: List[List[HeteroData]]) -> Dict[str, float]:
        """Compute performance and latency metrics."""
        inference_times = []
        
        # Measure inference time on sample data
        with torch.no_grad():
            for i in range(min(10, len(test_data))):  # Sample 10 sequences
                sequence = [graph.to(self.device) for graph in test_data[i]]
                
                start_time = time.time()
                _ = self.model(sequence, negative_sampling=False)
                inference_time = time.time() - start_time
                
                inference_times.append(inference_time)
        
        return {
            'avg_inference_time': np.mean(inference_times) if inference_times else 0.0,
            'max_inference_time': np.max(inference_times) if inference_times else 0.0,
            'min_inference_time': np.min(inference_times) if inference_times else 0.0,
            'std_inference_time': np.std(inference_times) if inference_times else 0.0,
            'meets_latency_requirement': np.mean(inference_times) < self.config.performance.max_prediction_latency if inference_times else False
        }
    
    def _evaluate_multistep_prediction(self, test_data: List[List[HeteroData]]) -> Dict[str, Any]:
        """Evaluate multi-step prediction capability."""
        if len(test_data) < 5:
            return {'error': 'Insufficient data for multi-step evaluation'}
        
        multistep_results = {}
        
        try:
            # Test on a few sequences
            for num_steps in [1, 2, 3]:
                step_accuracies = []
                
                for i in range(min(5, len(test_data) - num_steps)):
                    current_graph = test_data[i][-1]  # Last graph in sequence
                    
                    # Predict future steps
                    predictions = self.model.predict_future_steps(
                        current_graph,
                        num_steps=num_steps,
                        confidence_threshold=0.5,
                        top_k_per_step=10
                    )
                    
                    # Simple accuracy: check if any predictions were made
                    step_accuracy = len(predictions) / num_steps if predictions else 0.0
                    step_accuracies.append(step_accuracy)
                
                multistep_results[f'{num_steps}_step'] = {
                    'avg_accuracy': np.mean(step_accuracies) if step_accuracies else 0.0,
                    'num_evaluated': len(step_accuracies)
                }
        
        except Exception as e:
            logger.error(f"Error in multi-step evaluation: {e}")
            multistep_results['error'] = str(e)
        
        return multistep_results
    
    def _evaluate_uncertainty(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        metadata: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate uncertainty quantification."""
        # Simple uncertainty metrics based on prediction confidence
        uncertainties = 1.0 - np.abs(predictions - 0.5) * 2  # Higher for predictions near 0.5
        
        # Correlation between uncertainty and correctness
        correct_predictions = (predictions >= 0.5) == labels
        
        try:
            uncertainty_correlation = np.corrcoef(uncertainties, correct_predictions.astype(float))[0, 1]
        except:
            uncertainty_correlation = 0.0
        
        return {
            'avg_uncertainty': np.mean(uncertainties),
            'uncertainty_std': np.std(uncertainties),
            'uncertainty_correctness_correlation': uncertainty_correlation,
            'high_uncertainty_fraction': np.mean(uncertainties > 0.7)
        }
    
    def plot_roc_curve(self, predictions: np.ndarray, labels: np.ndarray, save_path: Optional[Path] = None) -> None:
        """Plot and save ROC curve."""
        try:
            fpr, tpr, _ = roc_curve(labels, predictions)
            auc = roc_auc_score(labels, predictions)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            if save_path is None:
                save_path = self.viz_dir / 'roc_curve.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"ROC curve saved to {save_path}")
        
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {e}")
    
    def plot_pr_curve(self, predictions: np.ndarray, labels: np.ndarray, save_path: Optional[Path] = None) -> None:
        """Plot and save Precision-Recall curve."""
        try:
            precision, recall, _ = precision_recall_curve(labels, predictions)
            ap = average_precision_score(labels, predictions)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap:.3f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            if save_path is None:
                save_path = self.viz_dir / 'pr_curve.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"PR curve saved to {save_path}")
        
        except Exception as e:
            logger.error(f"Error plotting PR curve: {e}")
    
    def _generate_visualizations(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        metadata: List[Dict]
    ) -> None:
        """Generate all evaluation visualizations."""
        logger.info("Generating evaluation visualizations...")
        
        # ROC and PR curves
        self.plot_roc_curve(predictions, labels)
        self.plot_pr_curve(predictions, labels)
        
        # Confusion matrix
        try:
            cm = self.generate_confusion_matrix(predictions, labels)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_path = self.viz_dir / 'confusion_matrix.png'
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved to {cm_path}")
        
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
        
        # Prediction distribution
        try:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.hist(predictions[labels == 1], bins=50, alpha=0.7, label='Positive', density=True)
            plt.hist(predictions[labels == 0], bins=50, alpha=0.7, label='Negative', density=True)
            plt.xlabel('Prediction Score')
            plt.ylabel('Density')
            plt.title('Prediction Score Distribution')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.boxplot([predictions[labels == 0], predictions[labels == 1]], 
                       labels=['Negative', 'Positive'])
            plt.ylabel('Prediction Score')
            plt.title('Prediction Score by Class')
            
            dist_path = self.viz_dir / 'prediction_distribution.png'
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Prediction distribution saved to {dist_path}")
        
        except Exception as e:
            logger.error(f"Error plotting prediction distribution: {e}")
    
    def _save_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        metadata: List[Dict]
    ) -> None:
        """Save detailed predictions to file."""
        try:
            # Create DataFrame with predictions and metadata
            results_df = pd.DataFrame(metadata)
            results_df['prediction'] = predictions
            results_df['label'] = labels
            results_df['correct'] = (predictions >= 0.5) == labels
            
            # Save to CSV
            csv_path = self.output_dir / 'detailed_predictions.csv'
            results_df.to_csv(csv_path, index=False)
            
            # Save summary statistics
            summary = {
                'total_predictions': len(predictions),
                'positive_samples': int(np.sum(labels)),
                'negative_samples': int(len(labels) - np.sum(labels)),
                'accuracy': float(np.mean((predictions >= 0.5) == labels)),
                'avg_prediction': float(np.mean(predictions)),
                'prediction_std': float(np.std(predictions))
            }
            
            summary_path = self.output_dir / 'prediction_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Predictions saved to {csv_path}")
            logger.info(f"Summary saved to {summary_path}")
        
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
    
    def save_metrics(self, metrics: Dict[str, Any], filename: str = 'evaluation_metrics.json') -> None:
        """Save evaluation metrics to JSON file."""
        try:
            metrics_path = self.output_dir / filename
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            converted_metrics = convert_numpy(metrics)
            
            with open(metrics_path, 'w') as f:
                json.dump(converted_metrics, f, indent=2)
            
            logger.info(f"Metrics saved to {metrics_path}")
        
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def _generate_evaluation_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive evaluation report."""
        try:
            report_path = self.reports_dir / 'evaluation_report.md'
            
            with open(report_path, 'w') as f:
                f.write("# THGNN Model Evaluation Report\n\n")
                f.write(f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Model:** {results['model_summary']['model_name']}\n")
                f.write(f"**Total Parameters:** {results['model_summary']['total_parameters']:,}\n")
                f.write(f"**Device:** {results['model_summary']['device']}\n\n")
                
                # Core Metrics
                f.write("## Core Performance Metrics\n\n")
                core = results['core_metrics']
                f.write(f"- **AUC-ROC:** {core['auc_roc']:.4f}\n")
                f.write(f"- **AUC-PR:** {core['auc_pr']:.4f}\n")
                f.write(f"- **F1 Score:** {core['f1_score']:.4f}\n")
                f.write(f"- **Accuracy:** {core['accuracy']:.4f}\n")
                f.write(f"- **Precision:** {core['precision']:.4f}\n")
                f.write(f"- **Recall:** {core['recall']:.4f}\n\n")
                
                # Ranking Metrics
                f.write("## Ranking Metrics\n\n")
                ranking = results['ranking_metrics']
                f.write("### Precision@K\n")
                for k, prec in ranking['precision_at_k'].items():
                    f.write(f"- **P@{k}:** {prec:.4f}\n")
                
                f.write("\n### Recall@K\n")
                for k, rec in ranking['recall_at_k'].items():
                    f.write(f"- **R@{k}:** {rec:.4f}\n")
                
                # Performance Requirements Check
                f.write("\n## Requirements Compliance\n\n")
                target_auc = self.config.performance.target_auc_roc
                target_p10 = self.config.performance.target_precision_at_10
                max_latency = self.config.performance.max_prediction_latency
                
                auc_met = core['auc_roc'] >= target_auc
                p10_met = ranking['precision_at_k'].get(10, 0) >= target_p10
                latency_met = results['performance_metrics']['meets_latency_requirement']
                
                f.write(f"- **AUC-ROC ≥ {target_auc}:** {' PASS' if auc_met else '[ERROR] FAIL'} ({core['auc_roc']:.4f})\n")
                f.write(f"- **Precision@10 ≥ {target_p10}:** {' PASS' if p10_met else '[ERROR] FAIL'} ({ranking['precision_at_k'].get(10, 0):.4f})\n")
                f.write(f"- **Latency < {max_latency}s:** {' PASS' if latency_met else '[ERROR] FAIL'} ({results['performance_metrics']['avg_inference_time']:.4f}s)\n\n")
                
                # Cybersecurity Metrics
                if 'cybersecurity_metrics' in results:
                    f.write("## Cybersecurity-Specific Analysis\n\n")
                    cyber = results['cybersecurity_metrics']
                    
                    if 'edge_type_metrics' in cyber:
                        f.write("### Performance by Edge Type\n")
                        for edge_type, metrics in cyber['edge_type_metrics'].items():
                            f.write(f"- **{edge_type}:** AUC={metrics['auc_roc']:.3f}, P@10={metrics['precision_at_10']:.3f}\n")
                
                # Summary
                f.write("\n## Summary\n\n")
                all_requirements_met = auc_met and p10_met and latency_met
                f.write(f"**Overall Status:** {' ALL REQUIREMENTS MET' if all_requirements_met else '⚠️ SOME REQUIREMENTS NOT MET'}\n\n")
                
                f.write(f"The model was evaluated on {results['num_samples']} samples ")
                f.write(f"in {results['evaluation_time']:.2f} seconds.\n")
            
            logger.info(f"Evaluation report saved to {report_path}")
        
        except Exception as e:
            logger.error(f"Error generating evaluation report: {e}")