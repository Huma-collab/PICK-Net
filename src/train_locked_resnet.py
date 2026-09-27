"""
Retrains ResNet-1D under the SAME locked protocol used for the PICK-Net
variants (seeds 42 and 123, get_dataloaders_uniform -- identical label
order to every other locked checkpoint), so the baseline comparison can
finally be a direct, paired, same-split, same-seed comparison instead of
the old single-split reference number.

ResNet-1D keeps its own baseline training recipe (SimpleLoss, Adam,
CosineAnnealingWarmRestarts -- as originally defined in baselines.py),
since forcing it through PICK-Net's own loss formulation would not be a
fair "best case" baseline comparison. Only the seed and data split are
matched, which is what DeLong pairing and CI comparison actually require.

Usage:
    python train_locked_resnet.py --seeds 42 123

Writes to: /data2/huma/picknet/locked_outputs/resnet1d/seed_<seed>/
    checkpoint.pt, test_probs.npy, test_labels.npy, test_metrics.json
"""
import os, sys, json, argparse, random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, '/data2/huma/picknet/src')
from baselines import ResNet1D, SimpleLoss, BCFG
from train_final import get_dataloaders_uniform, CFG

OUT_ROOT = '/data2/huma/picknet/locked_outputs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def train_one(seed):
    set_seed(seed)
    device = torch.device(BCFG['device'])
    out_dir = os.path.join(OUT_ROOT, 'resnet1d', f'seed_{seed}')
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n{"="*70}\nResNet-1D | Seed: {seed} | get_dataloaders_uniform\n{"="*70}')
    train_loader, val_loader, test_loader, _ = get_dataloaders_uniform(
        CFG['batch_size'], CFG['num_workers']
    )

    model = ResNet1D().to(device)
    criterion = SimpleLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=BCFG['lr'], weight_decay=BCFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)

    best_auc, best_epoch, best_state, patience_counter = 0.0, 0, None, 0

    for epoch in range(1, BCFG['epochs'] + 1):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f'resnet1d/seed{seed} Ep{epoch:02d}', leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out, z1, z2, z3 = model(xb)
            loss, _ = criterion(out, yb, z1, z2, z3)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), BCFG['grad_clip'])
            optimizer.step()
        scheduler.step()

        val_preds, val_targets = evaluate_raw(model, val_loader, device)
        metrics = compute_metrics(val_preds, val_targets)
        auc = metrics.get('AUC_macro', 0)
        print(f"  Ep {epoch:02d} | Val AUC_macro {auc:.4f} | F1_macro(@0.5) {metrics.get('F1_macro',0):.4f}")

        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= BCFG['patience']:
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
                'condition': 'resnet1d', 'thresholds': thresholds},
               os.path.join(out_dir, 'checkpoint.pt'))
    np.save(os.path.join(out_dir, 'test_probs.npy'), test_probs)
    np.save(os.path.join(out_dir, 'test_labels.npy'), test_targets)
    with open(os.path.join(out_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    ref_labels_path = f'/data2/huma/picknet/locked_outputs/full/seed_{seed}/test_labels.npy'
    if os.path.exists(ref_labels_path):
        ref_labels = np.load(ref_labels_path)
        if np.array_equal(ref_labels, test_targets):
            print(f'CONFIRMED: label order matches locked full/seed_{seed}/test_labels.npy exactly.')
        else:
            print(f'WARNING: label order does NOT match locked full/seed_{seed} -- investigate before using for DeLong.')

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123])
    args = parser.parse_args()

    all_results = []
    for seed in args.seeds:
        metrics = train_one(seed)
        all_results.append({'seed': seed, **metrics})

    print(f'\n{"="*70}\nRESNET-1D LOCKED SUMMARY (mean +/- std across seeds)\n{"="*70}')
    aucs = [r['AUC_macro'] for r in all_results]
    auc_mi_avb = [r.get('AUC_MI+AVB', 0) for r in all_results]
    print(f'AUC_macro   = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  (seeds: {[f"{a:.4f}" for a in aucs]})')
    print(f'AUC_MI+AVB  = {np.mean(auc_mi_avb):.4f} +/- {np.std(auc_mi_avb):.4f}')

    summary = {'auc_macro_mean': float(np.mean(aucs)), 'auc_macro_std': float(np.std(aucs)),
               'auc_mi_avb_mean': float(np.mean(auc_mi_avb)), 'auc_mi_avb_std': float(np.std(auc_mi_avb)),
               'seeds': args.seeds, 'per_seed_results': all_results}
    with open(os.path.join(OUT_ROOT, 'resnet1d', 'campaign_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved to {OUT_ROOT}/resnet1d/campaign_summary.json')


if __name__ == '__main__':
    main()
