"""
Fig. 5 -- Kernel parameter vs. clinical measurement correlation heatmap,
from REAL clinical_correlation.py output for the full model.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_PATH = '/data2/huma/picknet/outputs/clinical_correlation/correlation_results.csv'
df = pd.read_csv(CSV_PATH)

params = ['Omega (\u03c9)', 'Sigma (\u03c3)', 'Alpha (\u03b1)']
features = ['heart_rate', 'rr_interval_ms', 'qrs_duration_ms', 'st_level_mv', 'signal_power', 'dominant_freq_hz']
feature_labels = ['Heart rate', 'RR interval', 'QRS duration', 'ST level', 'Signal power', 'Dominant\nfrequency']

matrix = np.zeros((3, 6))
sig_matrix = np.zeros((3, 6), dtype=bool)
for i, p in enumerate(params):
    for j, f in enumerate(features):
        row = df[(df.kernel_param == p) & (df.ecg_feature == f)]
        matrix[i, j] = row.spearman_r.values[0]
        sig_matrix[i, j] = row.significant.values[0]

fig, ax = plt.subplots(figsize=(7.5, 3.2))
im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

ax.set_xticks(range(6))
ax.set_xticklabels(feature_labels, fontsize=9)
ax.set_yticks(range(3))
ax.set_yticklabels([r'$\omega$', r'$\sigma$', r'$\alpha$'], fontsize=12)

for i in range(3):
    for j in range(6):
        star = '*' if sig_matrix[i, j] else ''
        ax.text(j, i, f'{matrix[i,j]:.3f}{star}', ha='center', va='center', fontsize=9)

cbar = fig.colorbar(im, ax=ax, shrink=0.9)
cbar.set_label('Spearman correlation $r$', fontsize=9)

ax.set_title('Kernel Parameter -- Clinical Measurement Correlations (Full Model)', fontsize=10)
fig.tight_layout()

out_path = '/data2/huma/picknet/outputs/clinical_correlation/heatmap_real.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
