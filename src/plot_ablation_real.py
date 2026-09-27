"""
Fig. 3 -- Ablation performance, from REAL Table V values (single-split,
already published in the paper -- no new computation needed).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

variants = ['Full', 'A1', 'A2', 'A3', 'A4']
auc_macro = [0.9209, 0.9175, 0.8791, 0.8957, 0.8780]
auc_miavb = [0.9430, 0.9151, 0.8935, 0.9238, 0.9115]
colors = ['#4C72B0'] + ['#B0B0B0'] * 4

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

for ax, values, title in zip(axes, [auc_macro, auc_miavb],
                               [r'AUC$_{macro}$', r'AUC$_{MI+AVB}$']):
    bars = ax.bar(variants, values, color=colors, edgecolor='black', linewidth=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.004, f'{v:.4f}',
                ha='center', fontsize=8.5)
    ax.set_ylim(0.70, 1.00)
    ax.set_ylabel(title, fontsize=10)
    ax.set_xlabel('Model variant', fontsize=10)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

fig.suptitle('Ablation Performance (Table V, Single-Split)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = '/data2/huma/picknet/outputs/ablation_performance_real.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
