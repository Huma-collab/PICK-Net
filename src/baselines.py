import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/data2/huma/picknet/src')
from train_final import (get_dataloaders_uniform, CFG,
                          compute_metrics, evaluate)
from pick_net import CNNFeatureExtractor, TransformerEncoder, PACIM

SAVE_DIR = '/data2/huma/picknet/outputs/baselines'
os.makedirs(SAVE_DIR, exist_ok=True)

BCFG = {
    'batch_size'  : 32,
    'lr'          : 5e-4,
    'epochs'      : 40,
    'grad_clip'   : 1.0,
    'weight_decay': 1e-4,
    'patience'    : 10,
    'device'      : 'cuda:1',
    'num_workers' : 4,
}

# ── Baseline 1: ResNet-1D ─────────────────────────────────
class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)
        self.relu  = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)

class ResNet1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4)
        )
        self.layer1 = nn.Sequential(ResidualBlock1D(64),  nn.MaxPool1d(4))
        self.layer2 = nn.Sequential(ResidualBlock1D(64),  nn.MaxPool1d(4))
        self.layer3 = nn.Sequential(ResidualBlock1D(64),  nn.MaxPool1d(2))
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.drop   = nn.Dropout(0.3)
        self.fc     = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x), x, x, x

    def compute_physics_loss(self, x):
        return torch.tensor(0.0, device=x.device)

# ── Baseline 2: Inception-1D ──────────────────────────────
class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        branch_ch = out_ch // 4
        self.b1 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, 1),
            nn.Conv1d(branch_ch, branch_ch, 3, padding=1)
        )
        self.b2 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, 1),
            nn.Conv1d(branch_ch, branch_ch, 5, padding=2)
        )
        self.b3 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, branch_ch, 1)
        )
        self.b4 = nn.Conv1d(in_ch, branch_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return torch.relu(self.bn(
            torch.cat([self.b1(x), self.b2(x),
                       self.b3(x), self.b4(x)], dim=1)
        ))

class Inception1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=3):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv1d(in_channels, 32, 15, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4)
        )
        self.inc1   = InceptionBlock1D(32,  64)
        self.inc2   = InceptionBlock1D(64, 128)
        self.pool   = nn.Sequential(
            nn.MaxPool1d(4), nn.AdaptiveAvgPool1d(1)
        )
        self.drop   = nn.Dropout(0.3)
        self.fc     = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x), x, x, x

    def compute_physics_loss(self, x):
        return torch.tensor(0.0, device=x.device)

# ── Baseline 3: Transformer-only (no physics, no PACIM) ───
class TransformerOnly(nn.Module):
    def __init__(self, in_channels=12, num_classes=3):
        super().__init__()
        self.cnn  = CNNFeatureExtractor(in_channels, 64)
        self.transformer = TransformerEncoder(256, 4, 2)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(256, num_classes)

    def forward(self, x):
        f = self.cnn(x)
        z = self.transformer(f)
        z = self.drop(z)
        return self.fc(z), z, z, z

    def compute_physics_loss(self, x):
        return torch.tensor(0.0, device=x.device)

# ── Baseline 4: CNN-only (no Transformer, no PACIM) ───────
class CNNOnly(nn.Module):
    def __init__(self, in_channels=12, num_classes=3):
        super().__init__()
        self.cnn  = CNNFeatureExtractor(in_channels, 64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(256, num_classes)

    def forward(self, x):
        f = self.cnn(x)
        z = self.pool(f).squeeze(-1)
        z = self.drop(z)
        return self.fc(z), z, z, z

    def compute_physics_loss(self, x):
        return torch.tensor(0.0, device=x.device)

# ── Simple BCE loss for baselines ─────────────────────────
class SimpleLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, predictions, targets, z_int, z_mi, z_avb):
        probs = torch.sigmoid(predictions)
        pt    = targets * probs + (1-targets) * (1-probs)
        bce   = nn.functional.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        loss  = ((1-pt)**self.gamma * bce).mean()
        return loss, {'L_MI':0,'L_AVB':0,'L_MIAVB':0,'L_int':0}

# ── Training function ──────────────────────────────────────
def run_baseline(name, model, train_loader, val_loader,
                 test_loader, device):
    print(f'\n{"="*50}')
    print(f'Baseline: {name}')
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'{"="*50}')

    criterion = SimpleLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(),
                            lr=BCFG['lr'],
                            weight_decay=BCFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20
    )

    best_auc, best_state = 0.0, None
    patience_counter = 0

    for epoch in range(1, BCFG['epochs']+1):
        model.train()
        for xb, yb in tqdm(train_loader,
                            desc=f'Ep{epoch:02d}', leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out, z1, z2, z3 = model(xb)
            loss, _ = criterion(out, yb, z1, z2, z3)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), BCFG['grad_clip']
            )
            optimizer.step()
        scheduler.step()

        # evaluate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out, _, _, _ = model(xb)
                all_preds.append(out.cpu().numpy())
                all_targets.append(yb.numpy())
        preds   = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        metrics = compute_metrics(preds, targets)
        auc     = metrics.get('AUC_macro', 0)

        print(f'  Ep {epoch:02d} | '
              f'AUC {auc:.4f} | '
              f'F1 {metrics.get("F1_macro",0):.4f}')

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= BCFG['patience']:
                print(f'  Early stop @ epoch {epoch}')
                break

    # test
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out, _, _, _ = model(xb)
            all_preds.append(out.cpu().numpy())
            all_targets.append(yb.numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # optimal thresholds on val set
    from sklearn.metrics import f1_score, roc_auc_score
    val_preds, val_targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out, _, _, _ = model(xb)
            val_preds.append(out.cpu().numpy())
            val_targets.append(yb.numpy())
    vp = np.concatenate(val_preds)
    vt = np.concatenate(val_targets)
    thresholds = []
    for i in range(3):
        best_t, best_f1 = 0.5, 0.0
        probs = 1/(1+np.exp(-vp[:,i]))
        for t in np.linspace(0.05, 0.95, 91):
            f1 = f1_score(vt[:,i], (probs>t).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)

    probs  = 1/(1+np.exp(-preds))
    binary = np.stack([(probs[:,i]>thresholds[i]).astype(int)
                       for i in range(3)], axis=1)
    f1     = f1_score(targets, binary,
                      average='macro', zero_division=0)
    try:
        auc_macro = roc_auc_score(targets, probs, average='macro')
    except:
        auc_macro = 0.0

    per_class = {}
    names = ['MI','AVB','MI+AVB']
    for i, n in enumerate(names):
        if targets[:,i].sum() == 0: continue
        per_class[f'F1_{n}']  = f1_score(
            targets[:,i], binary[:,i], zero_division=0)
        per_class[f'AUC_{n}'] = roc_auc_score(
            targets[:,i], probs[:,i])

    result = {**per_class,
              'F1_macro': f1, 'AUC_macro': auc_macro}

    print(f'\n  {name} TEST RESULTS:')
    for k, v in result.items():
        print(f'    {k:15s}: {v:.4f}')

    torch.save(result,
               os.path.join(SAVE_DIR, f'{name}_results.pt'))
    return result

def run_all():
    device = torch.device(BCFG['device'])
    print('Loading data...')
    train_loader, val_loader, test_loader, _ = \
        get_dataloaders_uniform(BCFG['batch_size'],
                                BCFG['num_workers'])

    baselines = {
        'ResNet-1D'         : ResNet1D(),
        'Inception-1D'      : Inception1D(),
        'Transformer-only'  : TransformerOnly(),
        'CNN-only'          : CNNOnly(),
    }

    all_results = {}
    for name, model in baselines.items():
        model = model.to(device)
        result = run_baseline(name, model,
                              train_loader, val_loader,
                              test_loader, device)
        all_results[name] = result

    # final comparison table
    print('\n' + '='*75)
    print('BASELINE COMPARISON TABLE')
    print('='*75)
    header = f'{"Model":<22} {"AUC_MI":>8} {"AUC_AVB":>8} '
    header += f'{"AUC_MI+AVB":>11} {"AUC_macro":>10} {"F1_macro":>9}'
    print(header)
    print('-'*75)
    for name, r in all_results.items():
        print(f'{name:<22} '
              f'{r.get("AUC_MI",0):>8.4f} '
              f'{r.get("AUC_AVB",0):>8.4f} '
              f'{r.get("AUC_MI+AVB",0):>11.4f} '
              f'{r.get("AUC_macro",0):>10.4f} '
              f'{r.get("F1_macro",0):>9.4f}')
    print('-'*75)
    print(f'{"PICK-Net (ours)":<22} '
          f'{"0.8887":>8} '
          f'{"0.9308":>8} '
          f'{"0.9430":>11} '
          f'{"0.9209":>10} '
          f'{"0.4673":>9}')
    print('='*75)

if __name__ == '__main__':
    run_all()
