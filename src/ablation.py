import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/data2/huma/picknet/src')
from dataset  import get_dataloaders
from pick_net import (PICKNet, PICKNetLoss, CNNFeatureExtractor,
                      TransformerEncoder, PACIM, PhysicsConstraintLoss)

# ─────────────────────────────────────────────────────────────
# Ablation variants — each removes one component of PICK-Net
# ─────────────────────────────────────────────────────────────

# A1: Fixed kernels — replace dynamic physics conv with standard Conv1d
class PICKNet_FixedKernels(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=51, padding=25),
            nn.BatchNorm1d(32), nn.ReLU()
        )
        self.cnn         = CNNFeatureExtractor(32, 64)
        self.transformer = TransformerEncoder(256, 4, 2)
        self.pacim       = PACIM(256, 128)

    def forward(self, x):
        x = self.conv1(x)
        f = self.cnn(x)
        z = self.transformer(f)
        return self.pacim(z)

    def compute_physics_loss(self, x):
        return torch.tensor(0.0, device=x.device)


# A2: No physics loss — dynamic kernels but λ1=λ2=0
class PICKNet_NoPhysics(PICKNet):
    def compute_physics_loss(self, x):
        return torch.tensor(0.0, device=x.device)


# A3: No PACIM — independent heads, no interaction
class IndependentHeads(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.head_MI    = nn.Linear(d_model, 1)
        self.head_AVB   = nn.Linear(d_model, 1)
        self.head_MIAVB = nn.Linear(d_model, 1)
        self.dropout    = nn.Dropout(0.3)

    def forward(self, z):
        z = self.dropout(z)
        y_mi    = self.head_MI(z)
        y_avb   = self.head_AVB(z)
        y_miavb = self.head_MIAVB(z)
        out = torch.cat([y_mi, y_avb, y_miavb], dim=1)
        # return same signature as PACIM
        return out, z, z, z

class PICKNet_NoPACIM(nn.Module):
    def __init__(self):
        super().__init__()
        from pick_net import DynamicPhysicsConvLayer
        self.dyn_conv    = DynamicPhysicsConvLayer(12, 32, 51)
        self.physics_loss = PhysicsConstraintLoss(D=0.01)
        self.cnn         = CNNFeatureExtractor(32, 64)
        self.transformer = TransformerEncoder(256, 4, 2)
        self.heads       = IndependentHeads(256)
        from pick_net import PhysicsInformedKernel, KernelParamGenerator
        self.kernel_fn   = PhysicsInformedKernel(51)
        self.lambda1 = 0.05
        self.lambda2 = 0.05

    def forward(self, x):
        x = self.dyn_conv(x)
        f = self.cnn(x)
        z = self.transformer(f)
        return self.heads(z)

    def compute_physics_loss(self, x):
        gen     = self.dyn_conv.generators[0]
        params  = gen(x)
        kernels = self.kernel_fn(params)
        L_smooth, L_phys = self.physics_loss(kernels)
        return self.lambda1 * L_smooth + self.lambda2 * L_phys


# A4: Symmetric PACIM — remove directionality (both gates each other)
class PACIM_Symmetric(nn.Module):
    def __init__(self, d_model=256, d_sub=128):
        super().__init__()
        self.W_MI  = nn.Linear(d_model, d_sub, bias=False)
        self.W_AVB = nn.Linear(d_model, d_sub, bias=False)
        self.W_g1  = nn.Linear(d_sub, d_sub, bias=True)
        self.W_g2  = nn.Linear(d_sub, d_sub, bias=True)
        self.head_MI    = nn.Linear(d_sub, 1)
        self.head_AVB   = nn.Linear(d_sub, 1)
        self.head_MIAVB = nn.Linear(d_sub, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, z):
        z_MI  = torch.relu(self.W_MI(z))
        z_AVB = torch.relu(self.W_AVB(z))
        # symmetric: each gates the other equally
        gate1 = torch.sigmoid(self.W_g1(z_AVB))
        gate2 = torch.sigmoid(self.W_g2(z_MI))
        z_int = (z_MI * gate1 + z_AVB * gate2) / 2
        z_MI  = self.dropout(z_MI)
        z_AVB = self.dropout(z_AVB)
        z_int = self.dropout(z_int)
        out = torch.cat([
            self.head_MI(z_MI),
            self.head_AVB(z_AVB),
            self.head_MIAVB(z_int)
        ], dim=1)
        return out, z_MI, z_AVB, z_int

class PICKNet_SymmetricPACIM(PICKNet):
    def __init__(self):
        super().__init__(lambda1=0.05, lambda2=0.05)
        self.pacim = PACIM_Symmetric(256, 128)


# ─────────────────────────────────────────────────────────────
# Training function (reused for all ablations)
# ─────────────────────────────────────────────────────────────
CFG = {
    'batch_size'  : 32,
    'lr'          : 3e-4,
    'epochs'      : 30,
    'lambda_int'  : 0.1,
    'grad_clip'   : 1.0,
    'weight_decay': 1e-4,
    'patience'    : 8,
    'device'      : 'cuda:1',    # use GPU 1, leave GPU 0 for v2
    'num_workers' : 4,
    'save_dir'    : '/data2/huma/picknet/outputs/ablations',
}

def compute_metrics(preds, targets):
    from sklearn.metrics import f1_score, roc_auc_score
    probs  = 1 / (1 + np.exp(-preds))
    binary = (probs > 0.5).astype(int)
    metrics = {}
    for i, name in enumerate(['MI','AVB','MI+AVB']):
        if targets[:,i].sum() == 0:
            continue
        metrics[f'F1_{name}']  = f1_score(targets[:,i], binary[:,i], zero_division=0)
        metrics[f'AUC_{name}'] = roc_auc_score(targets[:,i], probs[:,i])
    metrics['F1_macro']  = f1_score(targets, binary, average='macro', zero_division=0)
    try:
        metrics['AUC_macro'] = roc_auc_score(targets, probs, average='macro')
    except Exception:
        metrics['AUC_macro'] = 0.0
    return metrics

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out, z_mi, z_avb, z_int = model(xb)
        all_preds.append(out.cpu().numpy())
        all_targets.append(yb.cpu().numpy())
    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return compute_metrics(all_preds, all_targets)

def run_ablation(name, model, train_loader, val_loader, test_loader, device):
    print(f'\n{"="*50}')
    print(f'Ablation: {name}')
    print(f'{"="*50}')

    criterion = PICKNetLoss(lambda_int=CFG['lambda_int'])
    optimizer = optim.Adam(model.parameters(),
                           lr=CFG['lr'],
                           weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4
    )

    best_auc  = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, CFG['epochs']+1):
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
        print(f"  Epoch {epoch:02d} | AUC {metrics['AUC_macro']:.4f} | F1 {metrics['F1_macro']:.4f}")

        if metrics['AUC_macro'] > best_auc:
            best_auc   = metrics['AUC_macro']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CFG['patience']:
                print(f'  Early stop at epoch {epoch}')
                break

    # test with best state
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f'\n  {name} TEST RESULTS:')
    for k, v in test_metrics.items():
        print(f'    {k:15s}: {v:.4f}')
    return test_metrics


def run_all_ablations():
    os.makedirs(CFG['save_dir'], exist_ok=True)
    device = torch.device(CFG['device'])

    print('Loading data...')
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        batch_size=CFG['batch_size'], num_workers=CFG['num_workers']
    )

    ablations = {
        'A1_FixedKernels'   : PICKNet_FixedKernels(),
        'A2_NoPhysicsLoss'  : PICKNet_NoPhysics(lambda1=0.05, lambda2=0.05),
        'A3_NoPACIM'        : PICKNet_NoPACIM(),
        'A4_SymmetricPACIM' : PICKNet_SymmetricPACIM(),
    }

    all_results = {}
    for name, model in ablations.items():
        model = model.to(device)
        results = run_ablation(
            name, model, train_loader, val_loader, test_loader, device
        )
        all_results[name] = results
        # save
        torch.save(results,
                   os.path.join(CFG['save_dir'], f'{name}_results.pt'))

    # summary table
    print('\n' + '='*70)
    print('ABLATION SUMMARY TABLE')
    print('='*70)
    print(f'{"Model":<25} {"AUC_macro":>10} {"F1_macro":>10} {"AUC_MI+AVB":>12}')
    print('-'*70)
    for name, r in all_results.items():
        print(f'{name:<25} {r["AUC_macro"]:>10.4f} {r["F1_macro"]:>10.4f} {r.get("AUC_MI+AVB",0):>12.4f}')
    print('='*70)
    print('\nNote: Compare these against PICK-Net v2 full model results')


if __name__ == '__main__':
    run_all_ablations()
