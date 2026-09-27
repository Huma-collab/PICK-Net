"""
Pairwise DeLong significance tests across all Path A variants plus your
existing full model / A2 / ResNet-1D checkpoints from the main project.
Reuses the same fastDeLong implementation already verified working this
session (src/delong_test.py), generalized to N-way comparison.

Usage:
    python delong_compare.py
(edit the MODELS dict below to point at whichever .npy prediction arrays
you want compared -- it will run every pairwise combination automatically)
"""
import os, itertools
import numpy as np
from scipy import stats

# ---- Add/remove entries here as your variants finish training ----
MODELS = {
    'Full model (headline)': (
        '/data2/huma/picknet/outputs/delong/picknet_test_probs.npy',
        '/data2/huma/picknet/outputs/delong/test_labels.npy',
    ),
    'ResNet-1D': (
        '/data2/huma/picknet/outputs/delong/resnet1d_test_probs.npy',
        '/data2/huma/picknet/outputs/delong/test_labels.npy',
    ),
    'All-Generator Physics Loss': (
        '/data2/huma/picknet/pathA_outputs/all_gen/test_probs.npy',
        '/data2/huma/picknet/pathA_outputs/all_gen/test_labels.npy',
    ),
    'A2 (No Physics Loss)': (
        '/data2/huma/picknet/outputs/delong/a2_test_probs.npy',
        '/data2/huma/picknet/outputs/delong/test_labels.npy',
    ),
    'Unbounded Gabor': (
        '/data2/huma/picknet/pathA_outputs/unbounded/test_probs.npy',
        '/data2/huma/picknet/pathA_outputs/unbounded/test_labels.npy',
    ),
}
# ---------------------------------------------------------------


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def fastDeLong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive = preds_sorted_transposed[:, :m]
    negative = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = compute_midrank(positive[r, :])
        ty[r, :] = compute_midrank(negative[r, :])
        tz[r, :] = compute_midrank(preds_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, probs_a, probs_b):
    order = (-y_true).argsort()
    y_true_sorted = y_true[order]
    m = int(y_true_sorted.sum())
    preds = np.vstack([probs_a, probs_b])[:, order]
    aucs, delongcov = fastDeLong(preds, m)
    diff = aucs[0] - aucs[1]
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], diff, z, p


def main():
    # only include models whose files actually exist yet
    available = {}
    for name, (probs_path, labels_path) in MODELS.items():
        if os.path.exists(probs_path) and os.path.exists(labels_path):
            available[name] = (np.load(probs_path), np.load(labels_path))
        else:
            print(f'SKIPPING "{name}" -- files not found yet ({probs_path})')

    if len(available) < 2:
        print('\nNeed at least 2 available models to compare. Train more variants first.')
        return

    class_names = ['MI', 'AVB', 'MI+AVB']
    pairs = list(itertools.combinations(available.keys(), 2))
    print(f'\nRunning {len(pairs)} pairwise comparisons across {len(available)} available models...\n')

    for name_a, name_b in pairs:
        probs_a, labels_a = available[name_a]
        probs_b, labels_b = available[name_b]
        if not np.array_equal(labels_a, labels_b):
            print(f'WARNING: {name_a} and {name_b} have different test labels -- '
                  f'skipping (not a valid comparison, likely different test set construction).')
            continue

        print(f'{"="*70}\n{name_a}  vs.  {name_b}\n{"="*70}')
        print(f"{'Class':<10}{'AUC_A':>10}{'AUC_B':>10}{'Diff':>10}{'z':>8}{'p-value':>10}")
        for i, cname in enumerate(class_names):
            auc_a, auc_b, diff, z, p = delong_roc_test(labels_a[:, i], probs_a[:, i], probs_b[:, i])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"{cname:<10}{auc_a:>10.4f}{auc_b:>10.4f}{diff:>10.4f}{z:>8.3f}{p:>10.4f}  {sig}")
        print()


if __name__ == '__main__':
    main()
