"""
Full interpretability battery for Path A variants: Kruskal-Wallis with
effect sizes, Spearman correlations with bootstrap 95% CIs, and a
per-generator consistency check across all 32 kernel generators.

Directly addresses the reviewer's specific asks:
  - "Effect sizes, not only p-values"           -> epsilon-squared reported
  - "Confidence intervals for the correlations"  -> bootstrap 95% CI reported
  - "Whether all 32 generators behave similarly" -> per-generator distributions compared

Test set construction matches clinical_correlation.py exactly (direct
strat_fold==10 filtering), same pattern verified working earlier this session.

Usage:
    python interpretability_battery.py --checkpoint /path/to/checkpoint.pt \
        --variant all_gen --n-generators 32 --label "All-Generator Physics Loss"

    python interpretability_battery.py --checkpoint /path/to/checkpoint.pt \
        --variant unbounded --n-generators 32 --label "Unbounded Gabor"
"""
import os, sys, ast, argparse, json
import numpy as np
import pandas as pd
import torch
from scipy.stats import kruskal, spearmanr
from tqdm import tqdm
import neurokit2 as nk

sys.path.insert(0, '/data2/huma/picknet/src')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from dataset import PTBXL_PATH, build_picknet_labels, preprocess_signal
from train_final import CFG

N_BOOTSTRAP = 2000


def extract_ecg_features(signal, fs=500):
    """Identical to clinical_correlation.py's extract_ecg_features."""
    features = {}
    try:
        lead = signal[:, 1]
        ecg_cleaned = nk.ecg_clean(lead, sampling_rate=fs)
        signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=fs)
        if 'ECG_Rate' in signals:
            features['heart_rate'] = signals['ECG_Rate'].mean()
        rpeaks = info['ECG_R_Peaks']
        if len(rpeaks) > 1:
            rr = np.diff(rpeaks) / fs * 1000
            features['rr_interval_ms'] = rr.mean()
        if 'ECG_Q_Peaks' in info and 'ECG_S_Peaks' in info:
            q, s = info.get('ECG_Q_Peaks', []), info.get('ECG_S_Peaks', [])
            if len(q) > 0 and len(s) > 0:
                qrs = (np.array(s[:len(q)]) - np.array(q[:len(s)])) / fs * 1000
                features['qrs_duration_ms'] = np.abs(qrs).mean()
        if 'ECG_T_Peaks' in info:
            r_peaks = info['ECG_R_Peaks']
            st_values = []
            for r in r_peaks:
                st_idx = r + int(0.08 * fs)
                if st_idx < len(lead):
                    st_values.append(lead[st_idx])
            if st_values:
                features['st_level_mv'] = np.mean(st_values)
        features['signal_power'] = np.mean(lead ** 2)
        fft_vals = np.abs(np.fft.rfft(lead))
        freqs = np.fft.rfftfreq(len(lead), 1 / fs)
        mask = (freqs >= 0.5) & (freqs <= 40)
        if mask.sum() > 0:
            dom_freq_idx = np.argmax(fft_vals[mask])
            features['dominant_freq_hz'] = freqs[mask][dom_freq_idx]
    except Exception:
        pass
    return features


def epsilon_squared(H, n, k):
    """Rank-based effect size for Kruskal-Wallis. 0.01=small, 0.06=medium, 0.14=large (Cohen-style bands)."""
    return (H - k + 1) / (n - k)


def bootstrap_spearman_ci(x, y, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(x[idx], y[idx])
        boot_rs.append(r)
    lo, hi = np.percentile(boot_rs, [2.5, 97.5])
    return lo, hi


def load_model(variant, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    if variant == 'all_gen':
        from all_gen_variant import PICKNet_AllGenerators
        model = PICKNet_AllGenerators(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'])
    elif variant == 'unbounded':
        from unbounded_variant import PICKNet_Unbounded
        model = PICKNet_Unbounded(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'])
    else:
        raise ValueError(f"Unknown variant: {variant}")
    model.load_state_dict(ckpt['model_state'])
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--variant', required=True, choices=['all_gen', 'unbounded'])
    parser.add_argument('--label', required=True, help='Human-readable name for this variant in the report')
    parser.add_argument('--n-generators', type=int, default=32,
                         help='How many of the 32 generators to include in the per-generator '
                              'consistency check (default: all 32). Lower for a faster run.')
    args = parser.parse_args()

    device = torch.device(CFG['device'])
    out_dir = os.path.join('/data2/huma/picknet/pathA_outputs', args.variant, 'interpretability')
    os.makedirs(out_dir, exist_ok=True)

    print(f'Loading {args.label} ({args.variant}) from {args.checkpoint}...')
    model = load_model(args.variant, args.checkpoint, device)

    print('Building test set (strat_fold==10, identical to clinical_correlation.py)...')
    X = np.load('/data2/huma/picknet/data/raw500.npy', allow_pickle=True)
    valid_idx = np.load('/data2/huma/picknet/data/raw500_idx.npy', allow_pickle=True)
    df = pd.read_csv(os.path.join(PTBXL_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
    df = df.loc[valid_idx]
    test_df = df[df.strat_fold == 10]
    test_pos = [list(df.index).index(i) for i in test_df.index]
    test_X = X[test_pos]

    df_copy = df.copy()
    df_copy['scp_codes'] = df_copy['scp_codes'].apply(ast.literal_eval)
    labels_df = build_picknet_labels(df_copy)
    test_labels = labels_df.loc[test_df.index]

    # ---- Extract params from generator[0] (for correlation analysis, comparable to existing results) ----
    print(f'Extracting kernel params (generator[0]) and ECG features for {len(test_X)} samples...')
    gen0 = model.dyn_conv.generators[0].to(device)
    all_omega, all_sigma, all_alpha, all_ecg_features, all_class_ids = [], [], [], [], []
    batch_size = 32
    for start in tqdm(range(0, len(test_X), batch_size)):
        batch_signals = test_X[start:start + batch_size]
        batch_labels = test_labels.iloc[start:start + batch_size]
        processed = np.stack([preprocess_signal(s) for s in batch_signals])
        xb = torch.tensor(processed.transpose(0, 2, 1), dtype=torch.float32).to(device)
        with torch.no_grad():
            params = gen0(xb)
        all_omega.append(params[:, 1].cpu().numpy())
        all_sigma.append(params[:, 2].cpu().numpy())
        all_alpha.append(params[:, 0].cpu().numpy())
        for sig in batch_signals:
            all_ecg_features.append(extract_ecg_features(sig))
        lv = batch_labels[['mi', 'avb', 'mi_avb']].values
        cids = np.zeros(len(lv), dtype=int)
        cids[lv[:, 0] == 1] = 1
        cids[lv[:, 1] == 1] = 2
        cids[lv[:, 2] == 1] = 3
        all_class_ids.append(cids)

    omega = np.concatenate(all_omega)
    sigma = np.concatenate(all_sigma)
    alpha = np.concatenate(all_alpha)
    class_ids = np.concatenate(all_class_ids)
    ecg_df = pd.DataFrame(all_ecg_features)

    # ---- Kruskal-Wallis with effect size ----
    print(f'\n{"="*70}\n{args.label} -- KRUSKAL-WALLIS WITH EFFECT SIZE\n{"="*70}')
    kw_results = {}
    for param, name in [(omega, 'Omega'), (sigma, 'Sigma'), (alpha, 'Alpha')]:
        groups = [param[class_ids == i] for i in range(4)]
        stat, p = kruskal(*groups)
        eps2 = epsilon_squared(stat, len(param), 4)
        size_band = 'large' if eps2 >= 0.14 else 'medium' if eps2 >= 0.06 else 'small' if eps2 >= 0.01 else 'negligible'
        print(f'{name}: H={stat:.4f}  p={p:.6f}  epsilon^2={eps2:.4f} ({size_band})')
        kw_results[name] = {'H': stat, 'p': p, 'epsilon_squared': eps2, 'effect_size_band': size_band}

    # ---- Correlations with bootstrap CI ----
    print(f'\n{"="*70}\n{args.label} -- CORRELATIONS WITH 95% BOOTSTRAP CI ({N_BOOTSTRAP} resamples)\n{"="*70}')
    corr_results = []
    for kvals, kname in [(omega, 'Omega'), (sigma, 'Sigma'), (alpha, 'Alpha')]:
        for feat in ecg_df.columns:
            feat_vals = ecg_df[feat].values
            mask = ~np.isnan(feat_vals) & ~np.isnan(kvals)
            if mask.sum() < 50:
                continue
            x, y = kvals[mask], feat_vals[mask]
            r, p = spearmanr(x, y)
            ci_lo, ci_hi = bootstrap_spearman_ci(x, y)
            ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)
            corr_results.append({
                'kernel_param': kname, 'ecg_feature': feat, 'spearman_r': r, 'p_value': p,
                'ci_95_lo': ci_lo, 'ci_95_hi': ci_hi, 'ci_excludes_zero': ci_excludes_zero,
                'n': mask.sum(), 'significant_p': p < 0.05
            })
            print(f'  {kname:6s} <-> {feat:20s}: r={r:+.4f}  95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  p={p:.4f}')

    corr_df = pd.DataFrame(corr_results)
    corr_df.to_csv(os.path.join(out_dir, 'correlations_with_ci.csv'), index=False)

    # ---- Per-generator consistency check ----
    print(f'\n{"="*70}\n{args.label} -- PER-GENERATOR CONSISTENCY ({args.n_generators} of 32 generators)\n{"="*70}')
    n_gen_check = min(args.n_generators, len(model.dyn_conv.generators))
    gen_means = {'omega': [], 'sigma': [], 'alpha': []}
    # use a fixed subsample of the test set for speed (this check doesn't need the full set)
    sample_idx = np.random.RandomState(42).choice(len(test_X), size=min(200, len(test_X)), replace=False)
    sample_signals = test_X[sample_idx]
    processed_sample = np.stack([preprocess_signal(s) for s in sample_signals])
    xb_sample = torch.tensor(processed_sample.transpose(0, 2, 1), dtype=torch.float32).to(device)

    with torch.no_grad():
        for gi in tqdm(range(n_gen_check), desc='Checking generators'):
            gen_i = model.dyn_conv.generators[gi]
            params_i = gen_i(xb_sample)
            gen_means['alpha'].append(params_i[:, 0].mean().item())
            gen_means['omega'].append(params_i[:, 1].mean().item())
            gen_means['sigma'].append(params_i[:, 2].mean().item())

    consistency_summary = {}
    for pname, vals in gen_means.items():
        vals = np.array(vals)
        cv = vals.std() / (abs(vals.mean()) + 1e-8)  # coefficient of variation across generators
        print(f'  {pname}: mean-across-generators={vals.mean():.4f}  std-across-generators={vals.std():.4f}  CV={cv:.4f}')
        consistency_summary[pname] = {'mean': float(vals.mean()), 'std': float(vals.std()),
                                       'cv': float(cv), 'per_generator_means': vals.tolist()}

    # ---- Save full summary ----
    summary = {
        'variant': args.variant, 'label': args.label, 'checkpoint': args.checkpoint,
        'n_test_samples': len(test_X), 'kruskal_wallis': kw_results,
        'n_significant_correlations': int(corr_df['significant_p'].sum()),
        'n_correlations_tested': len(corr_df),
        'n_ci_excludes_zero': int(corr_df['ci_excludes_zero'].sum()),
        'per_generator_consistency': consistency_summary,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n{"="*70}\nSUMMARY: {args.label}\n{"="*70}')
    print(f"Significant correlations (p<0.05): {summary['n_significant_correlations']}/{summary['n_correlations_tested']}")
    print(f"Correlations with 95% CI excluding zero: {summary['n_ci_excludes_zero']}/{summary['n_correlations_tested']}")
    print(f'Saved to: {out_dir}/')


if __name__ == '__main__':
    main()
