"""
Extracts and plots the REAL kernel waveform (generator[0]) for the SAME
real test patient, under the locked Full model vs. the locked Unbounded
model (both seed 42), directly visualizing what the per-generator CV
finding describes abstractly: whether the kernel converges on a real,
physiologically-scaled oscillation, or collapses toward a near-flat,
near-zero-frequency shape.
"""
import os, sys, ast
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/data2/huma/picknet/src')
sys.path.insert(0, '/data2/huma/picknet/pathA_project/models')
from dataset import PTBXL_PATH, build_picknet_labels, preprocess_signal
from train_final import CFG

LOCKED_ROOT = '/data2/huma/picknet/locked_outputs'
SAVE_ROOT = '/data2/huma/picknet/outputs/per_gen_cv_locked'


def build_model(condition):
    if condition == 'full':
        from pick_net import PICKNet
        return PICKNet(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'])
    elif condition == 'unbounded':
        from unbounded_variant import PICKNet_Unbounded
        return PICKNet_Unbounded(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'])


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    device = torch.device(CFG['device'])

    print('Loading locked Full and Unbounded models (seed 42)...')
    models = {}
    for cond in ['full', 'unbounded']:
        ckpt_path = os.path.join(LOCKED_ROOT, cond, 'seed_42', 'checkpoint.pt')
        model = build_model(cond).to(device)
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        models[cond] = model

    print('Building test set (strat_fold==10) and picking one real patient...')
    X = np.load('/data2/huma/picknet/data/raw500.npy', allow_pickle=True)
    valid_idx = np.load('/data2/huma/picknet/data/raw500_idx.npy', allow_pickle=True)
    df = pd.read_csv(os.path.join(PTBXL_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
    df = df.loc[valid_idx]
    test_df = df[df.strat_fold == 10]
    test_pos = [list(df.index).index(i) for i in test_df.index]
    test_X = X[test_pos]

    patient_idx = 0
    signal = test_X[patient_idx]
    processed = preprocess_signal(signal)
    xb = torch.tensor(processed.transpose(1, 0)[None], dtype=torch.float32).to(device)

    kernel_size = 51
    t = np.linspace(-25, 25, kernel_size) / 500 * 1000

    results = {}
    with torch.no_grad():
        for cond, model in models.items():
            gen = model.dyn_conv.generators[0]
            kernel_fn = model.kernel_fn
            params = gen(xb)
            kernel = kernel_fn(params)[0].cpu().numpy()
            alpha, omega, sigma = params[0].cpu().numpy()
            results[cond] = {'kernel': kernel, 'alpha': float(alpha), 'omega': float(omega), 'sigma': float(sigma)}
            print(f'{cond}: alpha={alpha:.4f}  omega={omega:.4f}  sigma={sigma:.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    titles = {'full': 'Full (bounded)', 'unbounded': 'Unbounded'}
    colors = {'full': '#4C72B0', 'unbounded': '#DD8452'}

    for ax, cond in zip(axes, ['full', 'unbounded']):
        r = results[cond]
        ax.plot(t, r['kernel'], color=colors[cond], linewidth=2)
        ax.fill_between(t, r['kernel'], alpha=0.15, color=colors[cond])
        ax.set_title(f"{titles[cond]}\n" + r"$\omega$" + f"={r['omega']:.2f} Hz, " +
                     r"$\sigma$" + f"={r['sigma']:.3f}", fontsize=9)
        ax.set_xlabel('Time (ms)', fontsize=9)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.grid(alpha=0.2)

    axes[0].set_ylabel('Kernel amplitude', fontsize=9)
    fig.suptitle(f'Real kernel waveform, same patient, generator[0] (seed 42)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = os.path.join(SAVE_ROOT, 'kernel_comparison_real.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
