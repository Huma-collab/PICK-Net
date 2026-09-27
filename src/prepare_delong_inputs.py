"""
Retrains ResNet-1D (saving weights this time) and generates raw
probability arrays for both PICK-Net and ResNet-1D on the identical
fold-10 test set, ready for delong_test.py.
"""
import os, sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, '/data2/huma/picknet/src')
from baselines import ResNet1D, SimpleLoss, BCFG, get_dataloaders_uniform, compute_metrics
from pick_net import PICKNet
from train_final import CFG

OUT_DIR = '/data2/huma/picknet/outputs/delong'
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device(BCFG['device'])

print("Loading data (same split used for original baselines)...")
train_loader, val_loader, test_loader, _ = get_dataloaders_uniform(
    BCFG['batch_size'], BCFG['num_workers']
)

print("\n" + "="*60)
print("Retraining ResNet-1D (saving weights this time)")
print("="*60)

model = ResNet1D().to(device)
criterion = SimpleLoss(gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=BCFG['lr'], weight_decay=BCFG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)

best_auc, best_state, patience_counter = 0.0, None, 0
for epoch in range(1, BCFG['epochs'] + 1):
    model.train()
    for xb, yb in tqdm(train_loader, desc=f'Ep{epoch:02d}', leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out, z1, z2, z3 = model(xb)
        loss, _ = criterion(out, yb, z1, z2, z3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), BCFG['grad_clip'])
        optimizer.step()
    scheduler.step()

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out, _, _, _ = model(xb)
            all_preds.append(out.cpu().numpy())
            all_targets.append(yb.numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    metrics = compute_metrics(preds, targets)
    auc = metrics.get('AUC_macro', 0)
    print(f'  Ep {epoch:02d} | AUC {auc:.4f} | F1 {metrics.get("F1_macro",0):.4f}')

    if auc > best_auc:
        best_auc = auc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= BCFG['patience']:
            print(f'  Early stop @ epoch {epoch}')
            break

model.load_state_dict(best_state)
torch.save({'model_state': best_state, 'best_val_auc': best_auc},
           os.path.join(OUT_DIR, 'resnet1d_retrained.pt'))
print(f"\nResNet-1D retrained. Best val AUC: {best_auc:.4f}. Weights saved.")

model.eval()
resnet_preds, resnet_targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out, _, _, _ = model(xb)
        resnet_preds.append(out.cpu().numpy())
        resnet_targets.append(yb.numpy())
resnet_preds = np.concatenate(resnet_preds)
resnet_targets = np.concatenate(resnet_targets)
resnet_probs = 1 / (1 + np.exp(-resnet_preds))

from sklearn.metrics import roc_auc_score
resnet_auc_check = roc_auc_score(resnet_targets, resnet_probs, average='macro')
print(f"ResNet-1D retrained test AUC_macro: {resnet_auc_check:.4f} "
      f"(original paper value was 0.9497 -- compare these)")

np.save(os.path.join(OUT_DIR, 'resnet1d_test_probs.npy'), resnet_probs)
np.save(os.path.join(OUT_DIR, 'test_labels.npy'), resnet_targets)

print("\n" + "="*60)
print("Generating PICK-Net test probabilities (no retraining needed)")
print("="*60)

pick_device = torch.device(CFG['device'])
pick_model = PICKNet(lambda1=CFG['lambda1'], lambda2=CFG['lambda2']).to(pick_device)
ckpt = torch.load('/data2/huma/picknet/outputs/final/best_model.pt', weights_only=False)
pick_model.load_state_dict(ckpt['model_state'])
pick_model.eval()

pick_preds, pick_targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(pick_device)
        out, _, _, _ = pick_model(xb)
        pick_preds.append(out.cpu().numpy())
        pick_targets.append(yb.numpy())
pick_preds = np.concatenate(pick_preds)
pick_targets = np.concatenate(pick_targets)
pick_probs = 1 / (1 + np.exp(-pick_preds))

pick_auc_check = roc_auc_score(pick_targets, pick_probs, average='macro')
print(f"PICK-Net test AUC_macro: {pick_auc_check:.4f} "
      f"(original paper value was 0.9209 -- should match closely)")

np.save(os.path.join(OUT_DIR, 'picknet_test_probs.npy'), pick_probs)

assert np.array_equal(pick_targets, resnet_targets), \
    "WARNING: test set labels differ between the two runs!"
print("\nConfirmed: both models evaluated on the identical test set.")
print(f"\nAll arrays saved to {OUT_DIR}/")
print("Now run delong_test.py, pointing it at these three .npy files.")
