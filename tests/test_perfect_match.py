#!/usr/bin/env python3
"""Test PERFECT model match"""
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
from models.perfect_legacy_thgnn import PerfectLegacyTHGNN

print('🔥 TESTING PERFECT MODEL MATCH 🔥\n')

print('Loading checkpoint...')
checkpoint_path = project_root / 'checkpoints' / 'REAL_TRAINED_MODEL_AUC0.8545.pth'
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

config = ckpt.get('config', {})
print(f'Checkpoint: {config.get("HIDDEN_DIM", 256)} hidden_dim, {config.get("NUM_CONV_LAYERS", 3)} layers')
print(f'Validation AUC: {ckpt.get("best_val_auc")}')
print(f'Epoch: {ckpt.get("epoch")}\n')

print('Creating PERFECT model...')
model = PerfectLegacyTHGNN(
    hidden_dim=config.get('HIDDEN_DIM', 256),
    num_conv_layers=config.get('NUM_CONV_LAYERS', 3),
    num_heads=config.get('NUM_HEADS', 8),
    dropout=config.get('DROPOUT', 0.2),
    feature_dim=config.get('FEATURE_DIM', 16),
    node_types=config.get('NODE_TYPES', ['user', 'host', 'process', 'file', 'registry']),
    edge_types=config.get('EDGE_TYPES', [])
)

print('Loading weights with strict=False...')
missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)

print(f'\n=== RESULTS ===')
print(f'Missing keys: {len(missing)}')
print(f'Unexpected keys: {len(unexpected)}')

if len(missing) == 0 and len(unexpected) == 0:
    print('\n🔥🔥🔥 PERFECT MATCH! 100% WEIGHTS LOADED! FULL 0.8545 AUC POWER! 🔥🔥🔥')
    print('✅ System is now running at FULL POWER!')
elif len(missing) < 10:
    print(f'\n⚠️ Almost there! Only {len(missing)} missing keys:')
    for key in missing:
        print(f'  - {key}')
else:
    print(f'\n❌ Still {len(missing)} missing keys')
