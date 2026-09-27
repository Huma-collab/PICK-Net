"""
Real cross-dataset generalization -- ONLY the AVB class, since that is the
only class ever evaluated on CPSC-2018.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

labels = ['PTB-XL AVB\n(5-fold CV mean, in-distribution)', 'CPSC-2018 AVB\n(first-degree, zero-shot)']
values = [0.9367, 0.8847]
errors = [0.0146, None]

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(labels, values, color=['#4C72B0', '#DD8452'], edgecolor='black', linewidth=0.8, width=0.5)
ax.errorbar([0], [values[0]], yerr=[errors[0]], fmt='none', ecolor='black', capsize=5, linewidth=1.2)

for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f'{v:.4f}', ha='center', fontsize=10)

ax.set_ylim(0.5, 1.0)
ax.set_ylabel('AUC', fontsize=10)
ax.set_title('AVB Cross-Dataset Generalization\n(the only class tested on CPSC-2018)', fontsize=10)
ax.grid(axis='y', alpha=0.25, linestyle='--')
ax.text(0.5, 0.55, 'MI, MI+AVB, and higher-degree AVB\nwere not evaluated on CPSC-2018',
        ha='center', fontsize=8, style='italic', color='gray', transform=ax.transAxes)

fig.tight_layout()
out_path = '/data2/huma/picknet/outputs/cross_dataset_real.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
