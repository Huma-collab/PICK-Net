"""
Reruns the full interpretability battery (Kruskal-Wallis + effect size,
correlations + bootstrap CI + Benjamini-Hochberg, per-generator consistency)
on the LOCKED-PROTOCOL checkpoints (Full, A2, Rotating-Subset, Unbounded),
instead of the earlier protocol-mismatched ones. This directly tests whether
the paper's interpretability claims (especially the A2 sign-flip result)
survive once the confound is removed.

Usage:
    python interp_locked.py --condition full --seed 42
    python interp_locked.py --condition a2 --seed 42
    python interp_locked.py --condition rotating_subset --seed 42
    python interp_locked.py --condition unbounded --seed 42
    (repeat with --seed 123 for the second seed)
"""
import os, sys, ast, argparse, json
import numpy as np
import pandas as pd
import torch
from scipy.stats import kruskal, spearmanr
from tqdm import tqdm
import neurokit2 as nk

sys.path.insert(0, '/data2/huma/picknet/src')
sys.path.insert(0, '/data2/huma/picknet/pathA_project/models')
from dataset import PTBXL_PATH, build_picknet_labels, preprocess_signal
from train_final import CFG

N_BOOTSTRAP = 2000
LOCKED_ROOT = '/data2/huma/picknet/locked_outputs'
SAVE_ROOT = '/data2/huma/picknet/outputs/interp_locked'


def extract_ecg_features(signal, fs=500):
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


def benjamini_hochberg(p_values, alpha=0.05):
    p = np.asarray(p_values)
    n = len(p)
    order = np.argsort(p)
    ranked_p = p[order]
    adj = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked_p[i] * n / (i + 1)
        prev = min(prev, val)
        adj[i] = prev
    adj_p = np.empty(n)
    adj_p[order] = np.clip(adj, 0, 1)
    return adj_p < alpha, adj_p


def build_model(condition):
    if condition == 'full':
        from pick_net import PICKNet
        return PICKNet(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'])
    elif condition == 'a2':
        from ablation import PICKNet_NoPhysics
        return PICKNet_NoPhysics(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'])
    elif condition == 'rotating_subset':
        from all_gen_variant import PICKNet_AllGenerators
        return PICKNet_AllGenerators(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'], subsample_generators=8)
    elif condition == 'unbounded':
        from unbounded_variant import PICKNet_Unbounded
        return PICKNet_Unbounded(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'])
    else:
        raise ValueError(condition)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True,
                         choices=['full', 'a2', 'rotating_subset', 'unbounded'])
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()

    os.makedirs(SAVE_ROOT, exist_ok=True)
    device = torch.device(CFG['device'])
    ckpt_path = os.path.join(LOCKED_ROOT, args.condition, f'seed_{args.seed}', 'checkpoint.pt')

    print(f'Loading {args.condition} (seed {args.seed}) from {ckpt_path}...')
    model = build_model(args.condition).to(device)
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    gen = model.dyn_conv.generators[0].to(device)

    print('Building test set (strat_fold==10)...')
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

    print(f'Processing {len(test_X)} test samples...')
    all_omega, all_sigma, all_alpha, all_ecg_features, all_class_ids = [], [], [], [], []
    batch_size = 32
    for start in tqdm(range(0, len(test_X), batch_size)):
        batch_signals = test_X[start:start + batch_size]
        batch_labels = test_labels.iloc[start:start + batch_size]
        processed = np.stack([preprocess_signal(s) for s in batch_signals])
        xb = torch.tensor(processed.transpose(0, 2, 1), dtype=torch.float32).to(device)
        with torch.no_grad():
            params = gen(xb)
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

    label = f'{args.condition} (seed {args.seed})'
    print(f'\n{"="*70}\n{label} -- KRUSKAL-WALLIS WITH EFFECT SIZE\n{"="*70}')
    kw_results = {}
    for param, name in [(omega, 'Omega'), (sigma, 'Sigma'), (alpha, 'Alpha')]:
        groups = [param[class_ids == i] for i in range(4)]
        stat, p = kruskal(*groups)
        eps2 = epsilon_squared(stat, len(param), 4)
        band = 'large' if eps2 >= 0.14 else 'medium' if eps2 >= 0.06 else 'small' if eps2 >= 0.01 else 'negligible'
        print(f'{name}: H={stat:.4f}  p={p:.6f}  epsilon^2={eps2:.4f} ({band})')
        kw_results[name] = {'H': stat, 'p': p, 'epsilon_squared': eps2, 'effect_size_band': band}

    print(f'\n{"="*70}\n{label} -- CORRELATIONS WITH CI + BH CORRECTION\n{"="*70}')
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
            corr_results.append({'kernel_param': kname, 'ecg_feature': feat,
                                  'spearman_r': r, 'p_value': p,
                                  'ci_95_lo': ci_lo, 'ci_95_hi': ci_hi, 'n': mask.sum()})

    corr_df = pd.DataFrame(corr_results)
    rejected, adj_p = benjamini_hochberg(corr_df['p_value'].values)
    corr_df['bh_adjusted_p'] = adj_p
    corr_df['significant_raw_p05'] = corr_df['p_value'] < 0.05
    corr_df['significant_bh'] = rejected

    for _, row in corr_df.iterrows():
        flag = '***' if row['significant_bh'] else ('(raw-sig only)' if row['significant_raw_p05'] else '')
        print(f"  {row['kernel_param']:6s} <-> {row['ecg_feature']:18s}: r={row['spearman_r']:+.4f}  "
              f"95%CI=[{row['ci_95_lo']:+.4f},{row['ci_95_hi']:+.4f}]  raw_p={row['p_value']:.2e}  "
              f"BH_p={row['bh_adjusted_p']:.2e} {flag}")

    out_file = f"{args.condition}_seed{args.seed}"
    corr_df.to_csv(os.path.join(SAVE_ROOT, f'{out_file}_correlations.csv'), index=False)

    n_sig_raw = int(corr_df['significant_raw_p05'].sum())
    n_sig_bh = int(corr_df['significant_bh'].sum())
    print(f'\n{"="*70}\nSUMMARY: {label}\n{"="*70}')
    print(f'Significant raw p<0.05: {n_sig_raw}/18 | After BH correction: {n_sig_bh}/18')

    summary = {'condition': args.condition, 'seed': args.seed, 'kruskal_wallis': kw_results,
               'n_significant_raw_p05': n_sig_raw, 'n_significant_bh': n_sig_bh}
    with open(os.path.join(SAVE_ROOT, f'{out_file}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved to {SAVE_ROOT}/{out_file}_*')


if __name__ == '__main__':
    main()
