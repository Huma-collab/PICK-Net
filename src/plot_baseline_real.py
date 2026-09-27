"""
Real baseline comparison, from Table IV (single-split, the only protocol
baselines were evaluated under -- no cross-validation exists for baselines,
so no error bars are shown). No 'Normal' class column, since AUC_Normal
was never reported for any model.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = ['ResNet-1D', 'Transformer\n-only', 'CNN-only', 'Inception-1D', 'PICK-Net\n(ours)']
auc_mi =    [0.9293, 0.9179, 0.9231, 0.9183, 0.8887]
auc_avb =   [0.9623, 0.9622, 0.9194, 0.7889, 0.9308]
auc_miavb = [0.9575, 0.9666, 0.9382, 0.8541, 0.9430]

x = np.arange(len(models))
width = 0.26

fig, ax = plt.subplots(figsize=(8, 4))
b1 = ax.bar(x - width, auc_mi, width, label='MI', color='#DD8452')
b2 = ax.bar(x, auc_avb, width, label='AVB', color='#55A868')
b3 = ax.bar(x + width, auc_miavb, width, label='MI+AVB', color='#C44E52')

for bars in [b1, b2, b3]:
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(1.4)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('AUC', fontsize=10)
ax.set_ylim(0.70, 1.00)
ax.legend(fontsize=9, loc='lower left')
ax.grid(axis='y', alpha=0.25, linestyle='--')
ax.set_title('Baseline Comparison (Table IV, Single-Split; No CV Exists for Baselines)', fontsize=10)

fig.tight_layout()
out_path = '/data2/huma/picknet/outputs/baseline_comparison_real.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
