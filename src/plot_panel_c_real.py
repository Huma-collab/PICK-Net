"""
Plots REAL per-generator parameter values (alpha, omega, sigma) for the
locked Full model vs. locked Unbounded model, using the actual saved
per-generator means from per_generator_cv_locked.py -- no synthetic data.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/data2/huma/picknet/outputs/per_gen_cv_locked'

with open(f'{ROOT}/full_seed42_per_gen_cv.json') as f:
    full_data = json.load(f)
with open(f'{ROOT}/unbounded_seed42_per_gen_cv.json') as f:
    unb_data = json.load(f)

params = ['omega', 'sigma', 'alpha']
param_labels = [r'$\omega$', r'$\sigma$', r'$\alpha$']

fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=False)

for ax, param, label in zip(axes, params, param_labels):
    full_vals = np.array(full_data[param]['per_generator_means'])
    unb_vals = np.array(unb_data[param]['per_generator_means'])

    bp = ax.boxplot([full_vals, unb_vals], positions=[1, 2], widths=0.5,
                     showfliers=False, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#4C72B0', '#DD8452']):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    rng = np.random.RandomState(0)
    x_full = 1 + rng.uniform(-0.15, 0.15, size=len(full_vals))
    x_unb = 2 + rng.uniform(-0.15, 0.15, size=len(unb_vals))
    ax.scatter(x_full, full_vals, color='#4C72B0', s=14, alpha=0.7, zorder=3, label='Full' if param == 'omega' else None)
    ax.scatter(x_unb, unb_vals, color='#DD8452', s=14, alpha=0.7, zorder=3, label='Unbounded' if param == 'omega' else None)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Full', 'Unbounded'], fontsize=9)
    ax.set_title(f'{label}  (CV: {full_data[param]["cv"]:.3f} vs. {unb_data[param]["cv"]:.3f})', fontsize=9)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

fig.suptitle('Per-generator parameter means, real data (32 generators, seed 42)', fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_path = '/data2/huma/picknet/outputs/per_gen_cv_locked/panel_c_real.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'\nFull omega CV: {full_data["omega"]["cv"]:.4f}  |  Unbounded omega CV: {unb_data["omega"]["cv"]:.4f}')
print(f'Full sigma CV: {full_data["sigma"]["cv"]:.4f}  |  Unbounded sigma CV: {unb_data["sigma"]["cv"]:.4f}')
print(f'Full alpha CV: {full_data["alpha"]["cv"]:.4f}  |  Unbounded alpha CV: {unb_data["alpha"]["cv"]:.4f}')
