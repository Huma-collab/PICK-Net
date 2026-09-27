import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
from tqdm import tqdm

sys.path.insert(0, '/data2/huma/picknet/src')
from pick_net    import PICKNet, KernelParamGenerator, PhysicsInformedKernel
from train_final import get_dataloaders_uniform, CFG

SAVE_DIR = '/data2/huma/picknet/outputs/kernel_analysis'
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = ['Normal', 'MI', 'AVB', 'MI+AVB']
COLORS      = ['#5F5E5A', '#D85A30', '#185FA5', '#1D9E75']

def extract_kernel_params(model, loader, device):
    """Extract alpha, omega, sigma for every sample with its class label."""
    model.eval()
    gen       = model.dyn_conv.generators[0].to(device)
    kernel_fn = model.kernel_fn.to(device)

    all_alpha, all_omega, all_sigma = [], [], []
    all_labels = []

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc='Extracting kernel params'):
            xb = xb.to(device)
            params  = gen(xb)               # (B, 3)
            alpha   = params[:,0].cpu().numpy()
            omega   = params[:,1].cpu().numpy()
            sigma   = params[:,2].cpu().numpy()
            all_alpha.append(alpha)
            all_omega.append(omega)
            all_sigma.append(sigma)
            all_labels.append(yb.numpy())

    alpha  = np.concatenate(all_alpha)
    omega  = np.concatenate(all_omega)
    sigma  = np.concatenate(all_sigma)
    labels = np.concatenate(all_labels)   # (N, 3): mi, avb, mi_avb

    # assign class: mi_avb > avb > mi > normal
    class_id = np.zeros(len(labels), dtype=int)
    class_id[labels[:,0] == 1] = 1   # MI
    class_id[labels[:,1] == 1] = 2   # AVB
    class_id[labels[:,2] == 1] = 3   # MI+AVB

    return alpha, omega, sigma, class_id

def plot_violin(param_values, class_ids, param_name, unit, save_path):
    """Violin plot of one parameter across 4 classes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data_by_class = [param_values[class_ids == i] for i in range(4)]

    parts = ax.violinplot(data_by_class, positions=range(4),
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLORS[i])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('#2C2C2A')
    parts['cmedians'].set_linewidth(2)

    ax.set_xticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_ylabel(f'{param_name} ({unit})', fontsize=12)
    ax.set_title(f'Kernel parameter {param_name} distribution by class', fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')

def statistical_test(param_values, class_ids, param_name):
    """Kruskal-Wallis test across 4 classes."""
    groups = [param_values[class_ids == i] for i in range(4)]
    stat, p = kruskal(*groups)
    print(f'\n{param_name} — Kruskal-Wallis test:')
    print(f'  H-statistic : {stat:.4f}')
    print(f'  p-value     : {p:.6f}')
    print(f'  Significant : {"YES ***" if p < 0.001 else "YES **" if p < 0.01 else "YES *" if p < 0.05 else "NO"}')
    for i, name in enumerate(CLASS_NAMES):
        g = param_values[class_ids == i]
        print(f'  {name:8s}: mean={g.mean():.4f}  std={g.std():.4f}  n={len(g)}')
    return stat, p

def plot_kernel_shapes(model, loader, device):
    """Plot example kernel shapes for one sample from each class."""
    model.eval()
    gen       = model.dyn_conv.generators[0].to(device)
    kernel_fn = model.kernel_fn.to(device)

    samples = {}
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            labels = yb.numpy()
            class_id = np.zeros(len(labels), dtype=int)
            class_id[labels[:,0]==1] = 1
            class_id[labels[:,1]==1] = 2
            class_id[labels[:,2]==1] = 3

            params  = gen(xb)
            kernels = kernel_fn(params)

            for i in range(len(xb)):
                cid = class_id[i]
                if cid not in samples:
                    samples[cid] = kernels[i].cpu().numpy()
                if len(samples) == 4:
                    break
            if len(samples) == 4:
                break

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    t = np.linspace(-25, 25, 51) / 500 * 1000   # ms

    for i, (cid, name) in enumerate(zip(range(4), CLASS_NAMES)):
        if cid in samples:
            axes[i].plot(t, samples[cid], color=COLORS[i], linewidth=2)
            axes[i].fill_between(t, samples[cid], alpha=0.15, color=COLORS[i])
        axes[i].set_title(name, fontsize=12, color=COLORS[i])
        axes[i].set_xlabel('Time (ms)', fontsize=10)
        axes[i].axhline(0, color='gray', linewidth=0.5, linestyle='--')
        axes[i].grid(alpha=0.2)
        sns.despine(ax=axes[i])

    axes[0].set_ylabel('Kernel amplitude', fontsize=10)
    fig.suptitle('Physics-informed kernel shapes per class', fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'kernel_shapes.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')

def run():
    device    = torch.device(CFG['device'])
    best_path = '/data2/huma/picknet/outputs/final/best_model.pt'

    print('Loading model...')
    model = PICKNet(lambda1=CFG['lambda1'], lambda2=CFG['lambda2']).to(device)
    ckpt  = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])

    print('Loading data...')
    _, _, test_loader, _ = get_dataloaders_uniform(
        CFG['batch_size'], CFG['num_workers']
    )

    print('Extracting kernel parameters from test set...')
    alpha, omega, sigma, class_ids = extract_kernel_params(
        model, test_loader, device
    )

    print(f'\nTotal samples: {len(alpha)}')
    print(f'Class distribution: ' +
          ', '.join([f'{n}={np.sum(class_ids==i)}'
                     for i, n in enumerate(CLASS_NAMES)]))

    # statistical tests
    print('\n' + '='*50)
    print('STATISTICAL ANALYSIS')
    print('='*50)
    results = {}
    for param, name, unit in [
        (alpha, 'Alpha', 'amplitude'),
        (omega, 'Omega', 'Hz'),
        (sigma, 'Sigma', 'smoothness'),
    ]:
        stat, p = statistical_test(param, class_ids, name)
        results[name] = {'stat': stat, 'p': p}

    # violin plots
    print('\nGenerating violin plots...')
    plot_violin(alpha, class_ids, 'Alpha', 'amplitude',
                os.path.join(SAVE_DIR, 'violin_alpha.png'))
    plot_violin(omega, class_ids, 'Omega', 'Hz',
                os.path.join(SAVE_DIR, 'violin_omega.png'))
    plot_violin(sigma, class_ids, 'Sigma', 'smoothness',
                os.path.join(SAVE_DIR, 'violin_sigma.png'))

    # kernel shape examples
    print('Generating kernel shape plots...')
    plot_kernel_shapes(model, test_loader, device)

    # summary
    print('\n' + '='*50)
    print('INTERPRETABILITY SUMMARY')
    print('='*50)
    for name, r in results.items():
        sig = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else 'ns'
        print(f'  {name:6s}: H={r["stat"]:.2f}  p={r["p"]:.6f}  {sig}')
    print(f'\nPlots saved to: {SAVE_DIR}')
    print('\nClaim 3 (interpretability) experimental evidence: COMPLETE')

if __name__ == '__main__':
    run()
