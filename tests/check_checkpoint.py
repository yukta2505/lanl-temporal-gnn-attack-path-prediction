import torch

ckpt = torch.load('checkpoints/REAL_TRAINED_MODEL_AUC0.8545.pth', map_location='cpu', weights_only=False)
print('=== CHECKPOINT INFO ===')
print(f'Epoch: {ckpt.get("epoch")}')
print(f'AUC: {ckpt.get("best_val_auc")}')
print(f'Total keys: {len(ckpt["model_state_dict"])}')

# Check what's actually in there
all_keys = list(ckpt['model_state_dict'].keys())
print(f'\n=== ALL KEYS ({len(all_keys)}) ===')
for key in all_keys:
    print(key)
