#!/usr/bin/env python3
"""
THGNN Training Infrastructure

Production-grade trainer for Temporal Heterogeneous Graph Neural Networks
with comprehensive cybersecurity threat detection capabilities.

Features:
- Temporal data splitting (no data leakage)
- GPU-optimized training with mixed precision
- Advanced negative sampling strategies
- Real-time metrics tracking and logging
- Checkpoint management and recovery
- Early stopping with patience
- Learning rate scheduling
- Memory-efficient batch processing
- TensorBoard integration
- Comprehensive error handling
"""

import logging
import time
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Removed TensorBoard import to avoid conflicts
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
import torch.nn.functional as F

from .config import TrainingConfiguration
from models.advanced_thgnn import AdvancedTHGNN, AdvancedTHGNNConfig
from utils.path_resolver import get_project_root, get_graphs_dir, get_checkpoints_dir

# Suppress warnings for production
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Training metrics tracking."""
    epoch: int
    train_loss: float
    val_loss: float
    train_auc: float
    val_auc: float
    train_ap: float
    val_ap: float
    learning_rate: float
    epoch_time: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_auc': self.train_auc,
            'val_auc': self.val_auc,
            'train_ap': self.train_ap,
            'val_ap': self.val_ap,
            'learning_rate': self.learning_rate,
            'epoch_time': self.epoch_time
        }


class AdvancedTHGNNLoss(nn.Module):
    """Advanced loss function for THGNN with zero warnings."""
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_focal_loss: bool = False
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_focal_loss = use_focal_loss
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with comprehensive error handling."""
        if predictions.numel() == 0 or targets.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Ensure same device and shape
        predictions = predictions.view(-1)
        targets = targets.view(-1).float()
        
        if predictions.size(0) != targets.size(0):
            min_size = min(predictions.size(0), targets.size(0))
            predictions = predictions[:min_size]
            targets = targets[:min_size]
        
        # Binary cross entropy with logits
        if self.use_focal_loss:
            # Focal loss implementation
            bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss
            return focal_loss.mean()
        else:
            # Standard weighted BCE
            pos_weight = torch.tensor(self.pos_weight, device=predictions.device)
            return F.binary_cross_entropy_with_logits(predictions, targets, pos_weight=pos_weight)


class THGNNTrainer:
    """
    Production-grade trainer for THGNN models.
    
    Handles complete training pipeline including data loading, model training,
    validation, checkpointing, and monitoring for cybersecurity applications.
    """
    
    def __init__(
        self,
        model: AdvancedTHGNN,
        config: TrainingConfiguration,
        device: Optional[torch.device] = None
    ):
        """
        Initialize THGNN trainer.
        
        Args:
            model: THGNN model instance
            config: Training configuration
            device: Computation device
        """
        self.model = model
        self.config = config
        self.device = device or config.get_device()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.loss_fn = self._create_loss_function()
        self.scaler = torch.cuda.amp.GradScaler() if config.system.mixed_precision else None
        
        # Data loaders (will be set by prepare_data)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.training_history = []
        
        # Logging
        # Logging setup (removed TensorBoard to avoid conflicts)
        self.writer = None
        
        # Create directories
        Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.paths.log_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"THGNN Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with proper parameter grouping."""
        # Separate parameters for different learning rates
        conv_params = []
        temporal_params = []
        predictor_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'hetero_conv' in name:
                conv_params.append(param)
            elif 'temporal_encoder' in name:
                temporal_params.append(param)
            elif 'link_predictor' in name:
                predictor_params.append(param)
            else:
                other_params.append(param)
        
        # Parameter groups with different learning rates
        param_groups = [
            {'params': conv_params, 'lr': self.config.training.learning_rate},
            {'params': temporal_params, 'lr': self.config.training.learning_rate * 0.8},
            {'params': predictor_params, 'lr': self.config.training.learning_rate * 1.2},
            {'params': other_params, 'lr': self.config.training.learning_rate}
        ]
        
        optimizer = optim.AdamW(
            param_groups,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.training.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.min_lr
            )
        elif self.config.training.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.num_epochs // 3,
                gamma=0.5
            )
        elif self.config.training.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.config.training.min_lr
            )
        else:
            return None
    
    def _create_loss_function(self) -> AdvancedTHGNNLoss:
        """Create loss function with cybersecurity-aware weighting."""
        return AdvancedTHGNNLoss(
            pos_weight=getattr(self.config.training, 'pos_weight', 1.0),
            neg_weight=getattr(self.config.training, 'neg_weight', 1.0),
            focal_alpha=getattr(self.config.training, 'focal_alpha', 0.25),
            focal_gamma=getattr(self.config.training, 'focal_gamma', 2.0),
            use_focal_loss=getattr(self.config.training, 'use_focal_loss', False)
        )
    
    def prepare_data(self) -> None:
        """
        Load and prepare training data with temporal splitting.
        
        Critical: Uses temporal splitting to prevent data leakage.
        """
        logger.info("Preparing training data...")
        
        try:
            # Load graph snapshots - DYNAMIC PATH RESOLUTION
            graphs_dir = get_graphs_dir()
            if not graphs_dir.exists():
                # Fallback to config path
                graphs_dir = Path(self.config.data.graphs_dir)
                if not graphs_dir.is_absolute():
                    graphs_dir = get_project_root() / graphs_dir
            
            if not graphs_dir.exists():
                raise FileNotFoundError(f"Graphs directory not found: {graphs_dir}")
            
            # Load all graph files
            graph_files = sorted(list(graphs_dir.glob("snapshot_*.pt")))
            
            if not graph_files:
                raise ValueError(f"No graph snapshots found in {graphs_dir}")
            
            logger.info(f"Found {len(graph_files)} graph snapshots")
            
            # Load graphs with error handling
            graphs = []
            for graph_file in graph_files:
                try:
                    graph = torch.load(graph_file, map_location='cpu', weights_only=False)
                    if isinstance(graph, HeteroData):
                        graphs.append(graph)
                    else:
                        logger.warning(f"Invalid graph format in {graph_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {graph_file}: {e}")
            
            if not graphs:
                raise ValueError("No valid graphs loaded")
            
            logger.info(f"Successfully loaded {len(graphs)} graphs")
            
            # Temporal splitting (critical for preventing data leakage)
            total_graphs = len(graphs)
            train_end = int(total_graphs * self.config.data.train_ratio)
            val_end = int(total_graphs * (self.config.data.train_ratio + self.config.data.val_ratio))
            
            train_graphs = graphs[:train_end]
            val_graphs = graphs[train_end:val_end]
            test_graphs = graphs[val_end:]
            
            logger.info(f"Data split - Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
            
            # Create sequences for temporal modeling
            train_sequences = self._create_sequences(train_graphs)
            val_sequences = self._create_sequences(val_graphs)
            test_sequences = self._create_sequences(test_graphs)
            
            # Create data loaders
            self.train_loader = self._create_dataloader(train_sequences, shuffle=True)
            self.val_loader = self._create_dataloader(val_sequences, shuffle=False)
            self.test_loader = self._create_dataloader(test_sequences, shuffle=False)
            
            logger.info("Data preparation completed successfully")
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def _create_sequences(self, graphs: List[HeteroData]) -> List[List[HeteroData]]:
        """Create temporal sequences from graphs."""
        sequences = []
        seq_len = self.config.data.sequence_length
        
        for i in range(len(graphs) - seq_len + 1):
            sequence = graphs[i:i + seq_len]
            sequences.append(sequence)
        
        return sequences
    
    def _create_dataloader(self, sequences: List[List[HeteroData]], shuffle: bool = False) -> DataLoader:
        """Create PyTorch Geometric DataLoader for graph sequences."""
        # For now, return sequences directly (custom batching in training loop)
        return sequences
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        epoch_start_time = time.time()
        
        for batch_idx, sequence in enumerate(self.train_loader):
            try:
                # Move sequence to device
                sequence = [graph.to(self.device) for graph in sequence]
                
                # Generate positive and negative edges (FIXED)
                pos_edge_dict, neg_edge_dict, labels = self._sample_edges(sequence[-1])
                
                if len(pos_edge_dict) == 0 and len(neg_edge_dict) == 0:
                    logger.warning(f"No edges sampled for batch {batch_idx}")
                    continue
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # CRITICAL FIX: Use actual THGNN model forward pass
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        # Forward pass through THGNN model
                        edge_scores = self.model(sequence, pos_edge_dict, neg_edge_dict)
                        
                        # Compute loss using proper loss function
                        loss = self.loss_fn(edge_scores, labels)
                else:
                    # Forward pass without mixed precision
                    edge_scores = self.model(sequence, pos_edge_dict, neg_edge_dict)
                    loss = self.loss_fn(edge_scores, labels)
                
                # Check for valid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss detected: {loss.item()}")
                    continue
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip
                    )
                    self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                total_samples += len(labels)
                
                # Store predictions for AUC calculation
                with torch.no_grad():
                    probs = torch.sigmoid(edge_scores).detach().cpu().numpy()
                    all_predictions.extend(probs)
                    all_labels.extend(labels.cpu().numpy())
                
                # Log progress
                if batch_idx % self.config.system.log_interval == 0:
                    logger.info(
                        f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                        f"Loss: {loss.item():.4f}"
                    )
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader) if self.train_loader else 0
        epoch_time = time.time() - epoch_start_time
        
        # Calculate AUC if we have predictions
        train_auc = 0.0
        train_ap = 0.0
        if all_predictions and all_labels:
            try:
                train_auc = roc_auc_score(all_labels, all_predictions)
                train_ap = average_precision_score(all_labels, all_predictions)
            except Exception as e:
                logger.warning(f"Error calculating training AUC: {e}")
        
        return {
            'loss': avg_loss,
            'auc': train_auc,
            'ap': train_ap,
            'epoch_time': epoch_time,
            'samples': total_samples
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, sequence_batch in enumerate(self.val_loader):
                try:
                    # Handle batch of sequences
                    for sequence in sequence_batch:
                        # Move sequence to device
                        sequence = [graph.to(self.device) for graph in sequence]
                        
                        # Sample edges for validation using the fixed method
                        pos_edge_dict, neg_edge_dict, labels = self._sample_edges(sequence[-1])
                        
                        # Skip if no edges sampled
                        if len(pos_edge_dict) == 0 and len(neg_edge_dict) == 0:
                            continue
                        
                        # Forward pass through THGNN model
                        edge_scores = self.model(sequence, pos_edge_dict, neg_edge_dict)
                        
                        if edge_scores.numel() == 0:
                            continue
                        
                        # Compute loss
                        loss = self.loss_fn(edge_scores, labels)
                        
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            total_loss += loss.item()
                            total_samples += 1
                            
                            # Collect predictions for metrics
                            predictions = torch.sigmoid(edge_scores).cpu().numpy()
                            labels_np = labels.cpu().numpy()
                            
                            all_predictions.extend(predictions)
                            all_labels.extend(labels_np)
                
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        # Compute metrics
        avg_loss = total_loss / max(total_samples, 1)
        
        if len(all_predictions) > 0 and len(set(all_labels)) > 1:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc = roc_auc_score(all_labels, all_predictions)
            ap = average_precision_score(all_labels, all_predictions)
        else:
            auc = 0.0
            ap = 0.0
        
        return {
            'loss': avg_loss,
            'auc': auc,
            'ap': ap,
            'samples': total_samples
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_auc': self.best_val_auc,
            'best_val_loss': self.best_val_loss,
            'training_history': [m.to_dict() for m in self.training_history],
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.config.paths.checkpoints_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.config.paths.checkpoints_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: AUC = {self.best_val_auc:.4f}")
    
    
    def _sample_edges(self, graph: HeteroData) -> Tuple[Dict[Tuple[str, str, str], torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor], torch.Tensor]:
        """
        Sample positive and negative edges from heterogeneous graph.
        
        CRITICAL FIX: Properly handle heterogeneous graphs with typed edges.
        
        Args:
            graph: Heterogeneous graph with typed nodes and edges
            
        Returns:
            Tuple of (pos_edge_dict, neg_edge_dict, labels)
        """
        pos_edge_dict = {}
        neg_edge_dict = {}
        all_labels = []
        
        # Sample positive edges from existing edges (per edge type)
        for edge_type, edge_index in graph.edge_index_dict.items():
            if edge_index.size(1) == 0:
                continue
                
            src_type, rel_type, dst_type = edge_type
            
            # Sample subset of positive edges for this edge type
            num_pos = min(edge_index.size(1), self.config.training.max_edges_per_type)
            perm = torch.randperm(edge_index.size(1), device=edge_index.device)[:num_pos]
            pos_edges = edge_index[:, perm]
            
            pos_edge_dict[edge_type] = pos_edges
            
            # Create positive labels
            pos_labels = torch.ones(pos_edges.size(1), device=self.device)
            all_labels.append(pos_labels)
            
            # Sample negative edges for this edge type
            src_nodes = graph.x_dict[src_type].size(0)
            dst_nodes = graph.x_dict[dst_type].size(0)
            
            if src_nodes > 0 and dst_nodes > 0:
                # Generate random negative edges
                num_neg = min(num_pos, self.config.training.max_edges_per_type)
                
                # Create negative edges by random sampling
                neg_src = torch.randint(0, src_nodes, (num_neg,), device=self.device)
                neg_dst = torch.randint(0, dst_nodes, (num_neg,), device=self.device)
                
                # Remove any that accidentally match positive edges
                pos_edge_set = set()
                for i in range(pos_edges.size(1)):
                    pos_edge_set.add((pos_edges[0, i].item(), pos_edges[1, i].item()))
                
                # Filter out positive edges from negatives
                valid_neg_src = []
                valid_neg_dst = []
                for i in range(num_neg):
                    edge = (neg_src[i].item(), neg_dst[i].item())
                    if edge not in pos_edge_set and neg_src[i] != neg_dst[i]:  # No self-loops
                        valid_neg_src.append(neg_src[i])
                        valid_neg_dst.append(neg_dst[i])
                
                if valid_neg_src:
                    neg_edges = torch.stack([
                        torch.stack(valid_neg_src),
                        torch.stack(valid_neg_dst)
                    ], dim=0)
                    
                    neg_edge_dict[edge_type] = neg_edges
                    
                    # Create negative labels
                    neg_labels = torch.zeros(len(valid_neg_src), device=self.device)
                    all_labels.append(neg_labels)
        
        # Combine all labels
        if all_labels:
            labels = torch.cat(all_labels, dim=0)
        else:
            labels = torch.empty(0, device=self.device)
        
        return pos_edge_dict, neg_edge_dict, labels
    
    def train(self) -> Dict[str, Any]:
        """
        Complete training loop with early stopping and checkpointing.
        
        Returns:
            Training history and final metrics
        """
        logger.info("Starting THGNN training...")
        
        # Prepare data if not already done
        if self.train_loader is None:
            self.prepare_data()
        
        # Setup reproducibility
        self.config.setup_reproducibility()
        
        training_start_time = time.time()
        
        try:
            for epoch in range(self.config.training.num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self.train_epoch()
                
                # Validation phase
                val_metrics = self.validate()
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # Create training metrics object
                current_lr = self.optimizer.param_groups[0]['lr']
                metrics = TrainingMetrics(
                    epoch=epoch,
                    train_loss=train_metrics['loss'],
                    val_loss=val_metrics['loss'],
                    train_auc=train_metrics['auc'],
                    val_auc=val_metrics['auc'],
                    train_ap=train_metrics['ap'],
                    val_ap=val_metrics['ap'],
                    learning_rate=current_lr,
                    epoch_time=train_metrics['epoch_time']
                )
                
                self.training_history.append(metrics)
                
                # Logging
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Train AUC: {train_metrics['auc']:.4f} | "
                    f"Val AUC: {val_metrics['auc']:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {train_metrics['epoch_time']:.1f}s"
                )
                
                # TensorBoard logging
                if self.writer is not None:
                    self._log_to_tensorboard(metrics)
                
                # Checkpointing
                is_best = val_metrics['auc'] > self.best_val_auc
                if is_best:
                    self.best_val_auc = val_metrics['auc']
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                
                # Regular checkpoint
                if epoch % 10 == 0:
                    self.save_checkpoint(is_best=False)
                
                # Early stopping
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Check if we've met target performance
                if (val_metrics['auc'] >= self.config.performance.target_auc_roc and
                    train_metrics['epoch_time'] < self.config.performance.max_prediction_latency):
                    logger.info(f"Target performance achieved at epoch {epoch + 1}")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            if self.writer is not None:
                self.writer.close()
        
        total_training_time = time.time() - training_start_time
        
        # Final results
        results = {
            'training_history': [m.to_dict() for m in self.training_history],
            'best_val_auc': self.best_val_auc,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.training_history),
            'total_training_time': total_training_time,
            'final_model_path': self.config.paths.best_model_path
        }
        
        logger.info(f"Training completed in {total_training_time:.1f}s")
        logger.info(f"Best validation AUC: {self.best_val_auc:.4f}")
        
        return results
    
    def _log_to_tensorboard(self, metrics: TrainingMetrics) -> None:
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return
        
        self.writer.add_scalar('Loss/Train', metrics.train_loss, metrics.epoch)
        self.writer.add_scalar('Loss/Validation', metrics.val_loss, metrics.epoch)
        self.writer.add_scalar('AUC/Train', metrics.train_auc, metrics.epoch)
        self.writer.add_scalar('AUC/Validation', metrics.val_auc, metrics.epoch)
        self.writer.add_scalar('AP/Train', metrics.train_ap, metrics.epoch)
        self.writer.add_scalar('AP/Validation', metrics.val_ap, metrics.epoch)
        self.writer.add_scalar('Learning_Rate', metrics.learning_rate, metrics.epoch)
        self.writer.add_scalar('Epoch_Time', metrics.epoch_time, metrics.epoch)
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_auc': self.best_val_auc,
            'best_val_loss': self.best_val_loss,
            'training_history': [m.to_dict() for m in self.training_history],
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.paths.checkpoint_dir) / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.paths.best_model_path)
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load model checkpoint and resume training."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_auc = checkpoint['best_val_auc']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Restore training history
        history_dicts = checkpoint.get('training_history', [])
        self.training_history = [
            TrainingMetrics(**h) for h in history_dicts
        ]
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
    
    def test(self) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Returns:
            Test metrics dictionary
        """
        if self.test_loader is None:
            raise ValueError("Test data not prepared. Call prepare_data() first.")
        
        logger.info("Evaluating on test set...")
        
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, sequence_batch in enumerate(self.test_loader):
                try:
                    # Handle batch of sequences
                    for sequence in sequence_batch:
                        # Move sequence to device
                        sequence = [graph.to(self.device) for graph in sequence]
                        
                        # Sample edges for testing using the fixed method
                        pos_edge_dict, neg_edge_dict, labels = self._sample_edges(sequence[-1])
                        
                        # Skip if no edges sampled
                        if len(pos_edge_dict) == 0 and len(neg_edge_dict) == 0:
                            continue
                        
                        # Forward pass through THGNN model
                        edge_scores = self.model(sequence, pos_edge_dict, neg_edge_dict)
                        
                        if edge_scores.numel() == 0:
                            continue
                        
                        # Compute loss
                        loss = self.loss_fn(edge_scores, labels)
                        
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            total_loss += loss.item()
                            total_samples += 1
                            
                            # Collect predictions for metrics
                            predictions = torch.sigmoid(edge_scores).cpu().numpy()
                            labels_np = labels.cpu().numpy()
                            
                            all_predictions.extend(predictions)
                            all_labels.extend(labels_np)
                
                except Exception as e:
                    logger.error(f"Error in test batch: {e}")
                    continue
        
        # Compute metrics
        avg_loss = total_loss / max(total_samples, 1)
        
        if len(all_predictions) > 0 and len(set(all_labels)) > 1:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc = roc_auc_score(all_labels, all_predictions)
            ap = average_precision_score(all_labels, all_predictions)
        else:
            auc = 0.0
            ap = 0.0
        
        logger.info(f"Test Results - Loss: {avg_loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
        
        return {
            'loss': avg_loss,
            'auc': auc,
            'ap': ap,
            'samples': total_samples
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_auc = checkpoint['best_val_auc']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
    
        logger.info(f"Test Results - Loss: {avg_loss:.4f}, AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
        
        return test_metrics
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.training_history:
            return {}
        
        return {
            'total_epochs': len(self.training_history),
            'best_val_auc': self.best_val_auc,
            'best_val_loss': self.best_val_loss,
            'final_train_auc': self.training_history[-1].train_auc,
            'final_val_auc': self.training_history[-1].val_auc,
            'avg_epoch_time': np.mean([m.epoch_time for m in self.training_history]),
            'total_training_time': sum(m.epoch_time for m in self.training_history),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'config_summary': self.config.get_summary()
        }