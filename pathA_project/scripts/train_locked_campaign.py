"""
Locked-seed campaign: trains all four conditions (Full, A2, Rotating-Subset,
Unbounded) under an IDENTICAL protocol (same recipe as the headline model:
PICKNetFinalLoss + get_dataloaders_uniform + AdamW + cosine annealing),
each with multiple independent seeds, for a fully controlled comparison.

IMPORTANT: this produces a NEW A2 number, different from both:
  - Table 2's original A2 (0.8791 AUC_macro) -- ablation.py's own recipe
  - The DeLong-comparison A2 (0.8952 AUC_macro) -- retrained on ablation.py's
    recipe but via get_dataloaders_uniform for label-order matching
This script's A2 uses the headline recipe throughout, matching the other
three conditions exactly, which is required for a fair "identical protocol"
comparison. All three A2 numbers should be reported with clear labels if
used together in the paper.

Usage:
    python train_locked_campaign.py --seeds 42 123
    (add more seeds for a more rigorous campaign; each seed = one full
    training run per condition, so 4 conditions x 2 seeds = 8 full runs)

Writes to: /data2/huma/picknet/locked_outputs/<condition>/seed_<seed>/
    checkpoint.pt, test_probs.npy, test_labels.npy, test_metrics.json
"""
import os, sys, json, argparse, random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, '/data2/huma/picknet/src')
sys.path.insert(0, '/data2/huma/picknet/pathA_project/models')

from train_final import PICKNetFinalLoss, get_dataloaders_uniform, CFG
from pick_net import PICKNet
from ablation import PICKNet_NoPhysics

OUT_ROOT = '/data2/huma/picknet/locked_outputs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(condition):
    if condition == 'full':
        return PICKNet(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'])
    elif condition == 'a2':
        return PICKNet_NoPhysics(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'])
    elif condition == 'rotating_subset':
        from all_gen_variant import PICKNet_AllGenerators
        return PICKNet_AllGenerators(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'],
                                      subsample_generators=8)
    elif condition == 'unbounded':
        from unbounded_variant import PICKNet_Unbounded
        return PICKNet_Unbounded(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'])
    else:
        raise ValueError(f"Unknown condition: {condition}")


@torch.no_grad()
def evaluate_raw(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out, _, _, _ = model(xb)
        all_preds.append(out.cpu().numpy())
        all_targets.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def compute_metrics(preds, targets, thresholds=None):
    from sklearn.metrics import f1_score, roc_auc_score
    probs = 1 / (1 + np.exp(-preds))
    if thresholds is None:
        binary = (probs > 0.5).astype(int)
    else:
        binary = np.stack([(probs[:, i] > thresholds[i]).astype(int) for i in range(3)], axis=1)
    metrics = {}
    for i, name in enumerate(['MI', 'AVB', 'MI+AVB']):
        if targets[:, i].sum() == 0:
            continue
        metrics[f'F1_{name}'] = f1_score(targets[:, i], binary[:, i], zero_division=0)
        metrics[f'AUC_{name}'] = roc_auc_score(targets[:, i], probs[:, i])
    metrics['F1_macro'] = f1_score(targets, binary, average='macro', zero_division=0)
    try:
        metrics['AUC_macro'] = roc_auc_score(targets, probs, average='macro')
    except Exception:
        metrics['AUC_macro'] = 0.0
    return metrics


def find_best_thresholds(preds, targets):
    """Per-class threshold optimization on validation set (fixes the F1=0 bug
    from the earlier Path A trainer -- see conversation history)."""
    from sklearn.metrics import f1_score
    probs = 1 / (1 + np.exp(-preds))
    thresholds = []
    for i in range(3):
        best_t, best_f1 = 0.5, -1
        for t in np.linspace(0.05, 0.95, 91):
            f1 = f1_score(targets[:, i], (probs[:, i] > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return thresholds


def train_one(condition, seed, epochs_override=None):
    set_seed(seed)
    device = torch.device(CFG['device'])
    out_dir = os.path.join(OUT_ROOT, condition, f'seed_{seed}')
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n{"="*70}\nCondition: {condition} | Seed: {seed}\n{"="*70}')
    train_loader, val_loader, test_loader, _ = get_dataloaders_uniform(
        CFG['batch_size'], CFG['num_workers']
    )

    model = build_model(condition).to(device)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = CFG['dropout']

    criterion = PICKNetFinalLoss(
        lambda_int=CFG['lambda_int'], gamma=CFG['focal_gamma'], smoothing=CFG['label_smooth']
    )
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    epochs = epochs_override if epochs_override is not None else CFG['epochs']
    best_auc, best_epoch, best_state, patience_counter = 0.0, 0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f'{condition}/seed{seed} Ep{epoch:03d}', leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out, z_mi, z_avb, z_int = model(xb)
            L_phys = model.compute_physics_loss(xb)
            L_task, _ = criterion(out, yb, z_int, z_mi, z_avb)
            loss = L_task + L_phys
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            optimizer.step()
        scheduler.step()

        val_preds, val_targets = evaluate_raw(model, val_loader, device)
        metrics = compute_metrics(val_preds, val_targets)
        auc = metrics.get('AUC_macro', 0)
        print(f"  Ep {epoch:03d} | Val AUC_macro {auc:.4f} | F1_macro(@0.5) {metrics.get('F1_macro',0):.4f}")

        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CFG['patience']:
                print(f'  Early stop @ epoch {epoch}')
                break

    print(f'\nBest val AUC_macro: {best_auc:.4f} @ epoch {best_epoch}')
    model.load_state_dict(best_state)
    model.to(device)

    val_preds, val_targets = evaluate_raw(model, val_loader, device)
    thresholds = find_best_thresholds(val_preds, val_targets)

    test_preds, test_targets = evaluate_raw(model, test_loader, device)
    test_metrics = compute_metrics(test_preds, test_targets, thresholds=thresholds)
    print('\nTEST RESULTS (optimized thresholds):')
    for k, v in test_metrics.items():
        print(f'  {k:15s}: {v:.4f}')

    test_probs = 1 / (1 + np.exp(-test_preds))
    torch.save({'model_state': best_state, 'best_val_auc': best_auc, 'seed': seed,
                'condition': condition, 'thresholds': thresholds},
               os.path.join(out_dir, 'checkpoint.pt'))
    np.save(os.path.join(out_dir, 'test_probs.npy'), test_probs)
    np.save(os.path.join(out_dir, 'test_labels.npy'), test_targets)
    with open(os.path.join(out_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conditions', nargs='+',
                         default=['full', 'a2', 'rotating_subset', 'unbounded'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123])
    parser.add_argument('--epochs', type=int, default=None,
                         help='Override epoch count for a faster test run')
    args = parser.parse_args()

    print(f'Locked campaign: conditions={args.conditions}, seeds={args.seeds}')
    print(f'Total runs: {len(args.conditions) * len(args.seeds)}')

    all_results = {}
    for condition in args.conditions:
        all_results[condition] = []
        for seed in args.seeds:
            metrics = train_one(condition, seed, epochs_override=args.epochs)
            all_results[condition].append({'seed': seed, **metrics})

    print(f'\n{"="*70}\nCAMPAIGN SUMMARY (mean +/- std across seeds)\n{"="*70}')
    summary = {}
    for condition, runs in all_results.items():
        aucs = [r['AUC_macro'] for r in runs]
        auc_mi_avb = [r.get('AUC_MI+AVB', 0) for r in runs]
        print(f'{condition:20s}: AUC_macro = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  '
              f'(n={len(aucs)} seeds: {[f"{a:.4f}" for a in aucs]})')
        print(f'{"":20s}  AUC_MI+AVB = {np.mean(auc_mi_avb):.4f} +/- {np.std(auc_mi_avb):.4f}')
        summary[condition] = {
            'auc_macro_mean': float(np.mean(aucs)), 'auc_macro_std': float(np.std(aucs)),
            'auc_mi_avb_mean': float(np.mean(auc_mi_avb)), 'auc_mi_avb_std': float(np.std(auc_mi_avb)),
            'seeds': args.seeds, 'per_seed_results': runs,
        }

    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(os.path.join(OUT_ROOT, 'campaign_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nFull summary saved to {OUT_ROOT}/campaign_summary.json')


if __name__ == '__main__':
    main()
