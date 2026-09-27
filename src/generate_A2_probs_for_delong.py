"""
Generates A2's raw test-set probabilities using get_dataloaders_uniform
(matching the exact test-set order already saved in outputs/delong/test_labels.npy),
so A2 can be directly DeLong-compared against the full model and ResNet-1D.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, '/data2/huma/picknet/src')
from ablation import PICKNet_NoPhysics
from train_final import CFG, get_dataloaders_uniform

OUT_DIR = '/data2/huma/picknet/outputs/delong'
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device(CFG['device'])

print('Loading A2 checkpoint...')
model = PICKNet_NoPhysics(lambda1=0.05, lambda2=0.05).to(device)
ckpt = torch.load('/data2/huma/picknet/outputs/ablations/A2_NoPhysicsLoss_checkpoint.pt', weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()

print('Loading data via get_dataloaders_uniform (matching existing saved test_labels.npy order)...')
_, _, test_loader, _ = get_dataloaders_uniform(CFG['batch_size'], CFG['num_workers'])

preds, targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out, _, _, _ = model(xb)
        preds.append(out.cpu().numpy())
        targets.append(yb.numpy())
preds = np.concatenate(preds)
targets = np.concatenate(targets)
probs = 1 / (1 + np.exp(-preds))

from sklearn.metrics import roc_auc_score
auc_check = roc_auc_score(targets, probs, average='macro')
print(f'A2 test AUC_macro via get_dataloaders_uniform: {auc_check:.4f} '
      f'(A2 original recipe test value was 0.8952 -- should be close, not identical, '
      f'since this uses a different dataloader/sampler for evaluation, though same underlying fold-10 records)')

np.save(os.path.join(OUT_DIR, 'a2_test_probs.npy'), probs)

existing_labels = np.load(os.path.join(OUT_DIR, 'test_labels.npy'))
if np.array_equal(targets, existing_labels):
    print('CONFIRMED: label order matches existing test_labels.npy exactly. Safe to DeLong-compare.')
else:
    print('WARNING: label order does NOT match existing test_labels.npy. '
          'Do not use this for DeLong comparison without investigating why.')

print(f'Saved to {OUT_DIR}/a2_test_probs.npy')
