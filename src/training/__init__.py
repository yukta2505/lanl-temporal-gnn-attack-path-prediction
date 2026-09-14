#!/usr/bin/env python3
"""
THGNN Training Infrastructure

Production-grade training pipeline for Temporal Heterogeneous Graph Neural Networks
with comprehensive monitoring, checkpointing, and evaluation capabilities.
"""

from .config import TrainingConfiguration

# Lazy imports to avoid circular dependencies
def get_trainer():
    from .trainer import THGNNTrainer
    return THGNNTrainer

def get_evaluator():
    from .evaluator import ModelEvaluator
    return ModelEvaluator

__all__ = [
    'TrainingConfiguration',
    'get_trainer',
    'get_evaluator'
]