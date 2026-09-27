"""
Extends the locked-campaign DeLong comparison to report 95% confidence
intervals on the AUC DIFFERENCE itself, not just the p-value. This uses
the DeLong covariance matrix already computed by the test statistic --
no new training or bootstrap needed, just reporting what was already
calculated.
"""
import os, itertools
import numpy as np
from scipy import stats

ROOT = '/data2/huma/picknet/locked_outputs'
CONDITIONS = ['full', 'a2', 'rotating_subset', 'unbounded', 'resnet1d']
SEEDS = [42, 123]


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


def delong_roc_test_with_ci(y_true, probs_a, probs_b, alpha=0.05):
    order = (-y_true).argsort()
    y_true_sorted = y_true[order]
    m = int(y_true_sorted.sum())
    preds = np.vstack([probs_a, probs_b])[:, order]
    aucs, delongcov = fastDeLong(preds, m)
    diff = aucs[0] - aucs[1]
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    se = np.sqrt(var) if var > 0 else 0.0
    z = diff / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lo = diff - z_crit * se
    ci_hi = diff + z_crit * se
    return aucs[0], aucs[1], diff, se, ci_lo, ci_hi, z, p


def load(condition, seed):
    d = os.path.join(ROOT, condition, f'seed_{seed}')
    probs = np.load(os.path.join(d, 'test_probs.npy'))
    labels = np.load(os.path.join(d, 'test_labels.npy'))
    return probs, labels


def main():
    class_names = ['MI', 'AVB', 'MI+AVB']
    pairs = list(itertools.combinations(CONDITIONS, 2))

    for seed in SEEDS:
        print(f'\n{"#"*90}\nSEED {seed}  (AUC difference with 95% CI)\n{"#"*90}')
        data = {}
        for c in CONDITIONS:
            try:
                data[c] = load(c, seed)
            except FileNotFoundError:
                continue

        for cond_a, cond_b in pairs:
            if cond_a not in data or cond_b not in data:
                continue
            probs_a, labels_a = data[cond_a]
            probs_b, labels_b = data[cond_b]
            if not np.array_equal(labels_a, labels_b):
                print(f'WARNING: {cond_a} vs {cond_b} (seed {seed}) label mismatch -- skipping')
                continue

            print(f'\n{cond_a}  vs.  {cond_b}  (seed {seed})')
            print(f"{'Class':<10}{'AUC_A':>9}{'AUC_B':>9}{'Diff':>9}{'95% CI':>20}{'p-value':>10}")
            for i, cname in enumerate(class_names):
                auc_a, auc_b, diff, se, ci_lo, ci_hi, z, p = delong_roc_test_with_ci(
                    labels_a[:, i], probs_a[:, i], probs_b[:, i]
                )
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ci_str = f'[{ci_lo:+.4f}, {ci_hi:+.4f}]'
                print(f"{cname:<10}{auc_a:>9.4f}{auc_b:>9.4f}{diff:>9.4f}{ci_str:>20}{p:>10.4f}  {sig}")


if __name__ == '__main__':
    main()
