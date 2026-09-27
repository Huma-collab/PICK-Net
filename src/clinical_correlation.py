import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import neurokit2 as nk

sys.path.insert(0, '/data2/huma/picknet/src')
from pick_net    import PICKNet, KernelParamGenerator
from train_final import get_dataloaders_uniform, CFG
from dataset     import PTBXL_PATH

SAVE_DIR = '/data2/huma/picknet/outputs/clinical_correlation'
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_ecg_features(signal, fs=500):
    """
    Extract clinical ECG features from raw signal.
    signal: (5000, 12) numpy array
    Returns dict of clinical measurements.
    """
    features = {}
    try:
        # use lead II (index 1) — standard clinical lead
        lead = signal[:, 1]

        # process with neurokit2
        ecg_cleaned = nk.ecg_clean(lead, sampling_rate=fs)
        signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=fs)

        # heart rate
        if 'ECG_Rate' in signals:
            features['heart_rate'] = signals['ECG_Rate'].mean()

        # RR interval (ms)
        rpeaks = info['ECG_R_Peaks']
        if len(rpeaks) > 1:
            rr = np.diff(rpeaks) / fs * 1000
            features['rr_interval_ms'] = rr.mean()

        # QRS duration (ms)
        if 'ECG_Q_Peaks' in info and 'ECG_S_Peaks' in info:
            q = info.get('ECG_Q_Peaks', [])
            s = info.get('ECG_S_Peaks', [])
            if len(q) > 0 and len(s) > 0:
                qrs = (np.array(s[:len(q)]) -
                       np.array(q[:len(s)])) / fs * 1000
                features['qrs_duration_ms'] = np.abs(qrs).mean()

        # ST deviation (mV) — mean signal between S and T peaks
        if 'ECG_T_Peaks' in info:
            t_peaks = info['ECG_T_Peaks']
            r_peaks = info['ECG_R_Peaks']
            if len(t_peaks) > 0 and len(r_peaks) > 0:
                # ST segment: 80ms after R peak
                st_values = []
                for r in r_peaks:
                    st_idx = r + int(0.08 * fs)
                    if st_idx < len(lead):
                        st_values.append(lead[st_idx])
                if st_values:
                    features['st_level_mv'] = np.mean(st_values)

        # signal power (related to amplitude)
        features['signal_power'] = np.mean(lead**2)

        # dominant frequency via FFT
        fft_vals = np.abs(np.fft.rfft(lead))
        freqs    = np.fft.rfftfreq(len(lead), 1/fs)
        # focus on 0.5-40 Hz range
        mask = (freqs >= 0.5) & (freqs <= 40)
        if mask.sum() > 0:
            dom_freq_idx = np.argmax(fft_vals[mask])
            features['dominant_freq_hz'] = freqs[mask][dom_freq_idx]

    except Exception:
        pass

    return features

def run():
    device    = torch.device(CFG['device'])
    best_path = '/data2/huma/picknet/outputs/final/best_model.pt'

    print('Loading model...')
    model = PICKNet(lambda1=CFG['lambda1'],
                    lambda2=CFG['lambda2']).to(device)
    ckpt  = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    gen = model.dyn_conv.generators[0].to(device)

    print('Loading data...')
    # load raw signals for feature extraction
    X         = np.load('/data2/huma/picknet/data/raw500.npy',
                        allow_pickle=True)
    valid_idx = np.load('/data2/huma/picknet/data/raw500_idx.npy',
                        allow_pickle=True)

    df = pd.read_csv(
        os.path.join(PTBXL_PATH, 'ptbxl_database.csv'),
        index_col='ecg_id'
    )
    df = df.loc[valid_idx]

    # use test fold only
    test_df  = df[df.strat_fold == 10]
    test_pos = [list(df.index).index(i) for i in test_df.index]
    test_X   = X[test_pos]

    print(f'Processing {len(test_X)} test samples...')
    print('Extracting kernel params and ECG features...')

    all_omega, all_sigma, all_alpha = [], [], []
    all_ecg_features = []
    all_class_ids    = []

    from dataset import build_picknet_labels
    import ast
    df_copy = df.copy()
    df_copy['scp_codes'] = df_copy['scp_codes'].apply(ast.literal_eval)
    labels_df = build_picknet_labels(df_copy)
    test_labels = labels_df.loc[test_df.index]

    batch_size = 32
    for start in tqdm(range(0, len(test_X), batch_size)):
        batch_signals = test_X[start:start+batch_size]
        batch_labels  = test_labels.iloc[start:start+batch_size]

        # preprocess for model
        from dataset import preprocess_signal
        processed = np.stack([
            preprocess_signal(s) for s in batch_signals
        ])
        xb = torch.tensor(processed.transpose(0,2,1),
                          dtype=torch.float32).to(device)

        with torch.no_grad():
            params = gen(xb)

        all_omega.append(params[:,1].cpu().numpy())
        all_sigma.append(params[:,2].cpu().numpy())
        all_alpha.append(params[:,0].cpu().numpy())

        # extract clinical features from raw signal
        for i, sig in enumerate(batch_signals):
            feats = extract_ecg_features(sig)
            all_ecg_features.append(feats)

        # class ids
        lv = batch_labels[['mi','avb','mi_avb']].values
        cids = np.zeros(len(lv), dtype=int)
        cids[lv[:,0]==1] = 1
        cids[lv[:,1]==1] = 2
        cids[lv[:,2]==1] = 3
        all_class_ids.append(cids)

    omega     = np.concatenate(all_omega)
    sigma     = np.concatenate(all_sigma)
    alpha     = np.concatenate(all_alpha)
    class_ids = np.concatenate(all_class_ids)
    ecg_df    = pd.DataFrame(all_ecg_features)

    print(f'\nExtracted features for {len(ecg_df)} samples')
    print(f'Available ECG features: {list(ecg_df.columns)}')

    # ── CORRELATION ANALYSIS ─────────────────────────────────
    print('\n' + '='*60)
    print('CORRELATION: Kernel Parameters vs Clinical ECG Features')
    print('='*60)

    results = []
    for kparam, kvals, kname in [
        (omega, omega, 'Omega (ω)'),
        (sigma, sigma, 'Sigma (σ)'),
        (alpha, alpha, 'Alpha (α)'),
    ]:
        for feat in ecg_df.columns:
            feat_vals = ecg_df[feat].values
            # remove NaN pairs
            mask = ~np.isnan(feat_vals) & ~np.isnan(kvals)
            if mask.sum() < 50:
                continue
            r, p = spearmanr(kvals[mask], feat_vals[mask])
            results.append({
                'kernel_param': kname,
                'ecg_feature' : feat,
                'spearman_r'  : r,
                'p_value'     : p,
                'n'           : mask.sum(),
                'significant' : p < 0.05
            })
            if abs(r) > 0.1:
                sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
                print(f'  {kname:15s} ↔ {feat:25s}: '
                      f'r={r:+.4f}  p={p:.4f} {sig}')

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(SAVE_DIR, 'correlation_results.csv'),
        index=False
    )

    # ── KEY PLOT: Omega vs Dominant Frequency ─────────────────
    if 'dominant_freq_hz' in ecg_df.columns:
        mask = ~np.isnan(ecg_df['dominant_freq_hz'].values)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#5F5E5A','#D85A30','#185FA5','#1D9E75']
        names  = ['Normal','MI','AVB','MI+AVB']
        for i, (name, color) in enumerate(zip(names, colors)):
            m = mask & (class_ids == i)
            if m.sum() == 0: continue
            ax.scatter(
                ecg_df['dominant_freq_hz'].values[m],
                omega[m],
                c=color, alpha=0.5, s=20,
                label=f'{name} (n={m.sum()})'
            )
        r, p = spearmanr(
            ecg_df['dominant_freq_hz'].values[mask],
            omega[mask]
        )
        ax.set_xlabel('ECG dominant frequency (Hz)', fontsize=12)
        ax.set_ylabel('Kernel ω parameter (Hz)', fontsize=12)
        ax.set_title(
            f'Kernel ω vs ECG dominant frequency\n'
            f'Spearman r={r:.4f}  p={p:.4f}',
            fontsize=12
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_DIR, 'omega_vs_dominant_freq.png'),
            dpi=150
        )
        plt.close()
        print(f'\nKey result: ω ↔ ECG dominant frequency: '
              f'r={r:.4f}  p={p:.4f}')

    # ── SUMMARY ───────────────────────────────────────────────
    print('\n' + '='*60)
    print('INTERPRETABILITY PROOF SUMMARY')
    print('='*60)
    sig_results = results_df[results_df['significant']]
    print(f'Total significant correlations: '
          f'{len(sig_results)} / {len(results_df)}')
    print('\nTop correlations (|r| > 0.15):')
    top = results_df[results_df['spearman_r'].abs() > 0.15].sort_values(
        'spearman_r', key=abs, ascending=False
    )
    print(top[['kernel_param','ecg_feature',
               'spearman_r','p_value']].to_string(index=False))
    print(f'\nPlots saved to: {SAVE_DIR}')

if __name__ == '__main__':
    run()
