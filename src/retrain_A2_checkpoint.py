"""
Retrains A2 (PICKNet_NoPhysics), exactly matching ablation.py's original
recipe, but saves the model weights this time.
"""
import os, sys
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, '/data2/huma/picknet/src')
from ablation import PICKNet_NoPhysics, run_ablation
from pick_net import PICKNetLoss
from dataset import get_dataloaders
from train_final import CFG

SAVE_DIR = '/data2/huma/picknet/outputs/ablations'
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device(CFG['device'])

print('Loading data (same as original ablation.py)...')
train_loader, val_loader, test_loader, _ = get_dataloaders(
    batch_size=CFG['batch_size'], num_workers=CFG['num_workers']
)

model = PICKNet_NoPhysics(lambda1=0.05, lambda2=0.05).to(device)
criterion = PICKNetLoss(lambda_int=CFG['lambda_int'])
optimizer = optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

def evaluate(model, loader, criterion, device):
    from ablation import compute_metrics
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out, z_mi, z_avb, z_int = model(xb)
            all_preds.append(out.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    import numpy as np
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return compute_metrics(preds, targets)

best_auc, best_state, patience_counter = 0.0, None, 0
print('\nRetraining A2 (No Physics Loss)...')
for epoch in range(1, CFG['epochs'] + 1):
    model.train()
    for xb, yb in tqdm(train_loader, desc=f'Ep{epoch:02d}', leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out, z_mi, z_avb, z_int = model(xb)
        L_phys = model.compute_physics_loss(xb)
        L_task, _ = criterion(out, yb, z_int, z_mi, z_avb)
        loss = L_task + L_phys
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
        optimizer.step()

    metrics = evaluate(model, val_loader, criterion, device)
    scheduler.step(metrics['AUC_macro'])
    print(f"  Ep {epoch:02d} | AUC {metrics['AUC_macro']:.4f} | F1 {metrics['F1_macro']:.4f}")

    if metrics['AUC_macro'] > best_auc:
        best_auc = metrics['AUC_macro']
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= CFG['patience']:
            print(f'  Early stop at epoch {epoch}')
            break

model.load_state_dict(best_state)
test_metrics = evaluate(model, test_loader, criterion, device)
print('\nA2 TEST RESULTS (retrained, for comparison to original Table 2 numbers):')
for k, v in test_metrics.items():
    print(f'  {k:15s}: {v:.4f}')

torch.save({'model_state': best_state, 'best_val_auc': best_auc, 'test_metrics': test_metrics},
           os.path.join(SAVE_DIR, 'A2_NoPhysicsLoss_checkpoint.pt'))
print(f"\nWeights saved to {SAVE_DIR}/A2_NoPhysicsLoss_checkpoint.pt")
print("Compare test_metrics above to the original Table 2 A2 row (AUC_macro=0.8791, AUC_MI+AVB=0.8935)")
