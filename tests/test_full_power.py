#!/usr/bin/env python3
"""Test if model loads with 100% weights"""
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
from models.legacy_thgnn import UltimateSupremeTHGNN

print('Loading checkpoint...')
checkpoint_path = project_root / 'checkpoints' / 'REAL_TRAINED_MODEL_AUC0.8545.pth'
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

config = ckpt.get('config', {})
print(f'Config: {config.get("HIDDEN_DIM", 256)} hidden_dim, {config.get("NUM_CONV_LAYERS", 3)} layers')

print('Creating model...')
model = UltimateSupremeTHGNN(
    hidden_dim=config.get('HIDDEN_DIM', 256),
    num_conv_layers=config.get('NUM_CONV_LAYERS', 3),
    num_heads=config.get('NUM_HEADS', 8),
    dropout=config.get('DROPOUT', 0.2),
    feature_dim=config.get('FEATURE_DIM', 16),
    node_types=config.get('NODE_TYPES', ['user', 'host', 'process', 'file', 'registry']),
    edge_types=config.get('EDGE_TYPES', [])
)

print('Loading weights...')
missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)

print(f'\n=== RESULTS ===')
print(f'Missing keys: {len(missing)}')
print(f'Unexpected keys: {len(unexpected)}')
print(f'\nValidation AUC: {ckpt.get("best_val_auc", "N/A")}')
print(f'Epoch: {ckpt.get("epoch", "N/A")}')

if len(missing) == 0:
    print('\n🔥🔥🔥 SUCCESS! 100% WEIGHTS LOADED! FULL POWER! 🔥🔥🔥')
else:
    print(f'\n⚠️ Still missing {len(missing)} keys')
    if len(missing) < 10:
        print('Missing:', missing)
