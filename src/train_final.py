import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb

sys.path.insert(0, '/data2/huma/picknet/src')
from dataset  import get_dataloaders, PICKNetDataset, PTBXL_PATH, CACHE_PATH, CACHE_IDX
from pick_net import PICKNet

CFG = {
    'batch_size'  : 32,
    'lr'          : 5e-4,
    'epochs'      : 60,
    'lambda1'     : 0.05,
    'lambda2'     : 0.05,
    'lambda_int'  : 0.05,
    'grad_clip'   : 1.0,
    'dropout'     : 0.3,
    'weight_decay': 1e-4,
    'patience'    : 12,
    'focal_gamma' : 2.0,        # focal loss gamma
    'label_smooth': 0.05,
    'device'      : 'cuda:0',
    'save_dir'    : '/data2/huma/picknet/outputs/final',
    'num_workers' : 4,
}

# ── Focal Loss replaces weighted sampler ─────────────────────
class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.gamma     = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets_s = targets * (1-self.smoothing) + 0.5*self.smoothing
        bce  = nn.functional.binary_cross_entropy_with_logits(
            logits, targets_s, reduction='none'
        )
        prob = torch.sigmoid(logits)
        pt   = targets * prob + (1-targets) * (1-prob)
        focal_weight = (1-pt) ** self.gamma
        return (focal_weight * bce).mean()

class PICKNetFinalLoss(nn.Module):
    def __init__(self, lambda_int=0.05, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.lambda_int = lambda_int
        self.focal = FocalBCELoss(gamma, smoothing)

    def forward(self, predictions, targets, z_int, z_mi, z_avb):
        L_MI    = self.focal(predictions[:,0], targets[:,0])
        L_AVB   = self.focal(predictions[:,1], targets[:,1])
        L_MIAVB = self.focal(predictions[:,2], targets[:,2])
        y_int_gt = (targets[:,0] * targets[:,1]).unsqueeze(1)
        L_int    = ((z_int.norm(dim=1,keepdim=True) - y_int_gt)**2).mean()
        L_total  = L_MI + L_AVB + L_MIAVB + self.lambda_int * L_int
        return L_total, {
            'L_MI':L_MI.item(), 'L_AVB':L_AVB.item(),
            'L_MIAVB':L_MIAVB.item(), 'L_int':L_int.item()
        }

# ── DataLoaders WITHOUT weighted sampler ─────────────────────
def get_dataloaders_uniform(batch_size=32, num_workers=4):
    import pandas as pd, ast
    from torch.utils.data import DataLoader
    from dataset import build_picknet_labels

    df = pd.read_csv(os.path.join(PTBXL_PATH,'ptbxl_database.csv'),
                     index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    X         = np.load(CACHE_PATH, allow_pickle=True)
    valid_idx = np.load(CACHE_IDX,  allow_pickle=True)
    df        = df.loc[valid_idx]
    labels_df = build_picknet_labels(df)

    train_idx = df[df.strat_fold <= 8].index
    val_idx   = df[df.strat_fold == 9].index
    test_idx  = df[df.strat_fold == 10].index

    def make_ds(idx):
        pos = [list(df.index).index(i) for i in idx]
        return PICKNetDataset(X[pos], labels_df.loc[idx])

    train_ds = make_ds(train_idx)
    val_ds   = make_ds(val_idx)
    test_ds  = make_ds(test_idx)

    # NO weighted sampler — focal loss handles imbalance
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader, test_loader, labels_df

def compute_metrics(preds, targets):
    from sklearn.metrics import f1_score, roc_auc_score
    probs  = 1 / (1 + np.exp(-preds))
    binary = (probs > 0.5).astype(int)
    metrics = {}
    for i, name in enumerate(['MI','AVB','MI+AVB']):
        if targets[:,i].sum() == 0: continue
        metrics[f'F1_{name}']  = f1_score(targets[:,i],binary[:,i],zero_division=0)
        metrics[f'AUC_{name}'] = roc_auc_score(targets[:,i],probs[:,i])
    metrics['F1_macro']  = f1_score(targets,binary,average='macro',zero_division=0)
    try:    metrics['AUC_macro'] = roc_auc_score(targets,probs,average='macro')
    except: metrics['AUC_macro'] = 0.0
    return metrics

def train_epoch(model, loader, optimizer, criterion, device, cfg):
    model.train()
    total_loss = 0
    bd_sum = {'L_MI':0,'L_AVB':0,'L_MIAVB':0,'L_int':0}
    for xb, yb in tqdm(loader, desc='Train', leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out, z_mi, z_avb, z_int = model(xb)
        L_phys = model.compute_physics_loss(xb)
        L_task, bd = criterion(out, yb, z_int, z_mi, z_avb)
        loss = L_task + L_phys
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optimizer.step()
        total_loss += loss.item()
        for k in bd_sum: bd_sum[k] += bd[k]
    n = len(loader)
    return total_loss/n, {k:v/n for k,v in bd_sum.items()}

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    for xb, yb in tqdm(loader, desc='Eval ', leave=False):
        xb, yb = xb.to(device), yb.to(device)
        out, z_mi, z_avb, z_int = model(xb)
        L_phys = model.compute_physics_loss(xb)
        L_task, _ = criterion(out, yb, z_int, z_mi, z_avb)
        total_loss += (L_task+L_phys).item()
        all_preds.append(out.cpu().numpy())
        all_targets.append(yb.cpu().numpy())
    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return total_loss/len(loader), compute_metrics(all_preds, all_targets)

def train():
    os.makedirs(CFG['save_dir'], exist_ok=True)
    device = torch.device(CFG['device'])
    wandb.init(project='pick-net-final', config=CFG, mode='offline')

    print('Loading data (uniform sampler)...')
    train_loader, val_loader, test_loader, _ = \
        get_dataloaders_uniform(CFG['batch_size'], CFG['num_workers'])

    model = PICKNet(lambda1=CFG['lambda1'], lambda2=CFG['lambda2']).to(device)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = CFG['dropout']

    criterion = PICKNetFinalLoss(
        lambda_int=CFG['lambda_int'],
        gamma=CFG['focal_gamma'],
        smoothing=CFG['label_smooth']
    )
    optimizer = optim.AdamW(model.parameters(),
                            lr=CFG['lr'],
                            weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )

    best_auc, best_epoch = 0.0, 0
    best_path = os.path.join(CFG['save_dir'], 'best_model.pt')
    patience_counter = 0

    print(f'PICK-Net FINAL — focal loss, uniform sampler, cosine restarts')
    print(f'batch={CFG["batch_size"]} lr={CFG["lr"]} gamma={CFG["focal_gamma"]}')
    print('-'*70)

    for epoch in range(1, CFG['epochs']+1):
        train_loss, bd = train_epoch(
            model, train_loader, optimizer, criterion, device, CFG
        )
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        gap = train_loss/(val_loss+1e-8)
        wandb.log({'epoch':epoch,'train_loss':train_loss,
                   'val_loss':val_loss,'gap':gap,**metrics})

        print(f"Ep {epoch:03d} | Tr {train_loss:.4f} | Vl {val_loss:.4f} | "
              f"Gap {gap:.2f} | F1 {metrics.get('F1_macro',0):.4f} | "
              f"AUC {metrics.get('AUC_macro',0):.4f}")

        if metrics.get('AUC_macro',0) > best_auc:
            best_auc, best_epoch = metrics['AUC_macro'], epoch
            patience_counter = 0
            torch.save({'epoch':epoch,'model_state':model.state_dict(),
                        'metrics':metrics,'cfg':CFG}, best_path)
            print(f'  ✓ Best (AUC {best_auc:.4f} @ ep {epoch})')
        else:
            patience_counter += 1
            if patience_counter >= CFG['patience']:
                print(f'\nEarly stop @ epoch {epoch}')
                break

    print(f'\nBest val AUC: {best_auc:.4f} @ epoch {best_epoch}')
    print('\nTest evaluation...')
    ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    _, test_metrics = evaluate(model, test_loader, criterion, device)

    print('\n' + '='*40)
    print('FINAL TEST RESULTS')
    print('='*40)
    for k,v in test_metrics.items():
        print(f'  {k:15s}: {v:.4f}')

    wandb.finish()
    return test_metrics

if __name__ == '__main__':
    train()

def get_dataloaders_uniform_cv(train_folds, val_folds, test_folds, batch_size=32, num_workers=4):
    """CV variant of get_dataloaders_uniform — same loading logic, parameterized folds."""
    import pandas as pd, ast
    from torch.utils.data import DataLoader
    from dataset import build_picknet_labels

    df = pd.read_csv(os.path.join(PTBXL_PATH,'ptbxl_database.csv'), index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    X         = np.load(CACHE_PATH, allow_pickle=True)
    valid_idx = np.load(CACHE_IDX,  allow_pickle=True)
    df        = df.loc[valid_idx]
    labels_df = build_picknet_labels(df)

    train_idx = df[df.strat_fold.isin(train_folds)].index
    val_idx   = df[df.strat_fold.isin(val_folds)].index
    test_idx  = df[df.strat_fold.isin(test_folds)].index

    def make_ds(idx):
        pos = [list(df.index).index(i) for i in idx]
        return PICKNetDataset(X[pos], labels_df.loc[idx])

    train_ds = make_ds(train_idx)
    val_ds   = make_ds(val_idx)
    test_ds  = make_ds(test_idx)

    # copy whatever sampler/DataLoader construction the original function uses below this point —
    # paste lines 93+ of the original get_dataloaders_uniform here so behavior matches exactly
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, labels_df
