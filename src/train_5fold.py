import os, sys, json
import numpy as np
import torch
import torch.optim as optim
import wandb

sys.path.insert(0, '/data2/huma/picknet/src')
from pick_net import PICKNet
from train_final import PICKNetFinalLoss
from train_final    import CFG, get_dataloaders_uniform_cv, train_epoch, evaluate
from evaluate_final  import find_best_thresholds, compute_metrics_with_thresholds

SAVE_DIR = '/data2/huma/picknet/outputs/cv5'
os.makedirs(SAVE_DIR, exist_ok=True)

# Fold 10 always held out as test. Rotate validation across 5 of folds 1-9.
VAL_FOLD_SETS  = [[1], [3], [5], [7], [9]]
ALL_TRAIN_POOL = list(range(1, 10))
TEST_FOLD      = [10]

@torch.no_grad()
def get_raw_preds(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out, _, _, _ = model(xb)
        all_preds.append(out.cpu().numpy())
        all_targets.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)

results = []

for i, val_folds in enumerate(VAL_FOLD_SETS):
    print(f"\n{'='*70}\nCV Run {i+1}/5 — val_fold={val_folds}\n{'='*70}")
    train_folds = [f for f in ALL_TRAIN_POOL if f not in val_folds]

    device = torch.device(CFG['device'])
    wandb.init(project='pick-net-5fold', config=CFG, mode='offline',
               name=f'cv_run{i+1}_val{val_folds[0]}')

    train_loader, val_loader, test_loader, _ = get_dataloaders_uniform_cv(
        train_folds, val_folds, TEST_FOLD,
        batch_size=CFG['batch_size'], num_workers=CFG['num_workers']
    )

    model = PICKNet(lambda1=CFG['lambda1'], lambda2=CFG['lambda2']).to(device)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = CFG['dropout']

    criterion = PICKNetFinalLoss(
        lambda_int=CFG['lambda_int'],
        gamma=CFG['focal_gamma'],
        smoothing=CFG['label_smooth']
    )
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_auc, best_epoch, patience_counter = 0.0, 0, 0
    best_path = os.path.join(SAVE_DIR, f'best_model_run{i+1}.pt')

    for epoch in range(1, CFG['epochs'] + 1):
        train_loss, bd = train_epoch(model, train_loader, optimizer, criterion, device, CFG)
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        gap = train_loss / (val_loss + 1e-8)
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'gap': gap, **metrics})
        print(f"  Ep {epoch:03d} | Tr {train_loss:.4f} | Vl {val_loss:.4f} | "
              f"Gap {gap:.2f} | F1 {metrics.get('F1_macro',0):.4f} | AUC {metrics.get('AUC_macro',0):.4f}")

        if metrics.get('AUC_macro', 0) > best_auc:
            best_auc, best_epoch = metrics['AUC_macro'], epoch
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'metrics': metrics, 'cfg': CFG}, best_path)
            print(f'    ✓ Best (AUC {best_auc:.4f} @ ep {epoch})')
        else:
            patience_counter += 1
            if patience_counter >= CFG['patience']:
                print(f'  Early stop @ epoch {epoch}')
                break

    # Reload best checkpoint, optimize thresholds on val, evaluate on held-out fold 10
    ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])

    thresholds = find_best_thresholds(model, val_loader, device)
    test_preds, test_targets = get_raw_preds(model, test_loader, device)
    test_metrics = compute_metrics_with_thresholds(test_preds, test_targets, thresholds)
    test_metrics['val_fold']   = val_folds
    test_metrics['train_folds'] = train_folds
    test_metrics['thresholds']  = thresholds
    results.append(test_metrics)

    print(f"\n  Run {i+1} TEST (fold 10, optimal thresholds):")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"    {k:15s}: {v:.4f}")

    wandb.finish()

# ── Aggregate across 5 runs ──────────────────────────────
with open(f"{SAVE_DIR}/cv5_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*70}\nFINAL 5-FOLD CV SUMMARY (all evaluated on held-out fold 10)\n{'='*70}")
for metric in ['AUC_macro', 'F1_macro', 'AUC_MI', 'AUC_AVB', 'AUC_MI+AVB', 'F1_MI', 'F1_AVB', 'F1_MI+AVB']:
    vals = [r[metric] for r in results if metric in r and isinstance(r[metric], float)]
    if vals:
        print(f"  {metric:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  (range {min(vals):.4f}-{max(vals):.4f})")
