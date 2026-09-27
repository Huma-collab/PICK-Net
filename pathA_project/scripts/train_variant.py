"""
Generic, seeded trainer for Path A model variants. Uses the SAME training
recipe as your headline model (PICKNetFinalLoss = focal loss + uniform
sampler, from train_final.py) rather than the older ablation.py recipe
(plain PICKNetLoss + get_dataloaders), since Path A variants are meant to
be directly comparable to your reported headline PICK-Net numbers, not to
the A1-A4 ablation table specifically.

Usage:
    python train_variant.py --variant all_gen --subsample-generators 8
    python train_variant.py --variant unbounded   # only after unbounded_variant.py is finalized

Writes to: pathA_outputs/<variant_name>/
    checkpoint.pt          -- model_state, best_val_auc, seed
    test_probs.npy         -- (N, 3) sigmoid probabilities on fold-10 test set
    test_labels.npy        -- (N, 3) ground truth
    test_metrics.json      -- AUC/F1 per class + macro
"""
import os, sys, json, argparse, random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, '/data2/huma/picknet/src')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_final import PICKNetFinalLoss, get_dataloaders_uniform, CFG

SEED = 42
OUT_ROOT = '/data2/huma/picknet/pathA_outputs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(variant, subsample_generators):
    if variant == 'all_gen':
        from all_gen_variant import PICKNet_AllGenerators
        return PICKNet_AllGenerators(
            lambda1=CFG['lambda1'], lambda2=CFG['lambda2'],
            subsample_generators=subsample_generators
        )
    elif variant == 'unbounded':
        from unbounded_variant import PICKNet_Unbounded
        return PICKNet_Unbounded(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'])
    else:
        raise ValueError(f"Unknown variant: {variant}")


@torch.no_grad()
def evaluate_raw(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out, _, _, _ = model(xb)
        all_preds.append(out.cpu().numpy())
        all_targets.append(yb.numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return preds, targets


def compute_metrics(preds, targets):
    from sklearn.metrics import f1_score, roc_auc_score
    probs = 1 / (1 + np.exp(-preds))
    binary = (probs > 0.5).astype(int)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=['all_gen', 'unbounded'])
    parser.add_argument('--subsample-generators', type=int, default=None,
                         help='For all_gen variant: randomly sample this many of the 32 '
                              'generators per forward pass instead of all 32 (cost tradeoff). '
                              'Omit to use all 32 every step (most correct, most expensive).')
    parser.add_argument('--epochs', type=int, default=None,
                         help='Override CFG epochs if you want a shorter run for testing.')
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(CFG['device'])
    out_dir = os.path.join(OUT_ROOT, args.variant)
    os.makedirs(out_dir, exist_ok=True)

    print(f'Variant: {args.variant} | subsample_generators={args.subsample_generators} | seed={SEED}')
    print('Loading data (get_dataloaders_uniform, same as headline model)...')
    train_loader, val_loader, test_loader, _ = get_dataloaders_uniform(
        CFG['batch_size'], CFG['num_workers']
    )

    model = build_model(args.variant, args.subsample_generators).to(device)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = CFG['dropout']

    criterion = PICKNetFinalLoss(
        lambda_int=CFG['lambda_int'], gamma=CFG['focal_gamma'], smoothing=CFG['label_smooth']
    )
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    epochs = args.epochs if args.epochs is not None else CFG['epochs']
    best_auc, best_epoch, best_state, patience_counter = 0.0, 0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f'Ep{epoch:03d}', leave=False):
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
        print(f"  Ep {epoch:03d} | Val AUC_macro {auc:.4f} | F1_macro {metrics.get('F1_macro',0):.4f}")

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

    test_preds, test_targets = evaluate_raw(model, test_loader, device)
    test_metrics = compute_metrics(test_preds, test_targets)
    print('\nTEST RESULTS:')
    for k, v in test_metrics.items():
        print(f'  {k:15s}: {v:.4f}')

    torch.save({'model_state': best_state, 'best_val_auc': best_auc,
                'seed': SEED, 'variant': args.variant,
                'subsample_generators': args.subsample_generators},
               os.path.join(out_dir, 'checkpoint.pt'))
    test_probs = 1 / (1 + np.exp(-test_preds))
    np.save(os.path.join(out_dir, 'test_probs.npy'), test_probs)
    np.save(os.path.join(out_dir, 'test_labels.npy'), test_targets)
    with open(os.path.join(out_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f'\nAll outputs saved to {out_dir}/')


if __name__ == '__main__':
    main()
