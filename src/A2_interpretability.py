"""
Combined Kruskal-Wallis + clinical correlation analysis on A2 (No Physics Loss),
built the same way as clinical_correlation.py -- directly from strat_fold==10,
sidestepping any get_dataloaders vs get_dataloaders_uniform mismatch.
Directly comparable to the full model's existing Section V-F numbers.
"""
import os, sys, ast
import numpy as np
import pandas as pd
import torch
from scipy.stats import kruskal, spearmanr
from tqdm import tqdm
import neurokit2 as nk

sys.path.insert(0, '/data2/huma/picknet/src')
from ablation import PICKNet_NoPhysics
from dataset  import PTBXL_PATH, build_picknet_labels, preprocess_signal
from train_final import CFG

SAVE_DIR = '/data2/huma/picknet/outputs/A2_interpretability'
os.makedirs(SAVE_DIR, exist_ok=True)
CLASS_NAMES = ['Normal', 'MI', 'AVB', 'MI+AVB']

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
        features['signal_power'] = np.mean(lead**2)
        fft_vals = np.abs(np.fft.rfft(lead))
        freqs = np.fft.rfftfreq(len(lead), 1/fs)
        mask = (freqs >= 0.5) & (freqs <= 40)
        if mask.sum() > 0:
            dom_freq_idx = np.argmax(fft_vals[mask])
            features['dominant_freq_hz'] = freqs[mask][dom_freq_idx]
    except Exception:
        pass
    return features

device = torch.device(CFG['device'])
ckpt_path = '/data2/huma/picknet/outputs/ablations/A2_NoPhysicsLoss_checkpoint.pt'

print('Loading A2 model...')
model = PICKNet_NoPhysics(lambda1=0.05, lambda2=0.05).to(device)
ckpt = torch.load(ckpt_path, weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()
gen = model.dyn_conv.generators[0].to(device)
kernel_fn = model.kernel_fn.to(device)

print('Loading data (identical strat_fold==10 test set construction as clinical_correlation.py)...')
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
all_omega, all_sigma, all_alpha = [], [], []
all_ecg_features, all_class_ids = [], []

batch_size = 32
for start in tqdm(range(0, len(test_X), batch_size)):
    batch_signals = test_X[start:start+batch_size]
    batch_labels = test_labels.iloc[start:start+batch_size]
    processed = np.stack([preprocess_signal(s) for s in batch_signals])
    xb = torch.tensor(processed.transpose(0,2,1), dtype=torch.float32).to(device)
    with torch.no_grad():
        params = gen(xb)
    all_omega.append(params[:,1].cpu().numpy())
    all_sigma.append(params[:,2].cpu().numpy())
    all_alpha.append(params[:,0].cpu().numpy())
    for sig in batch_signals:
        all_ecg_features.append(extract_ecg_features(sig))
    lv = batch_labels[['mi','avb','mi_avb']].values
    cids = np.zeros(len(lv), dtype=int)
    cids[lv[:,0]==1] = 1
    cids[lv[:,1]==1] = 2
    cids[lv[:,2]==1] = 3
    all_class_ids.append(cids)

omega = np.concatenate(all_omega)
sigma = np.concatenate(all_sigma)
alpha = np.concatenate(all_alpha)
class_ids = np.concatenate(all_class_ids)
ecg_df = pd.DataFrame(all_ecg_features)

print(f'\n{"="*60}\nA2 (NO PHYSICS LOSS) -- KRUSKAL-WALLIS ACROSS 4 CLASSES\n{"="*60}')
for param, name in [(omega, 'Omega'), (sigma, 'Sigma')]:
    groups = [param[class_ids == i] for i in range(4)]
    stat, p = kruskal(*groups)
    sig = 'YES ***' if p<0.001 else 'YES **' if p<0.01 else 'YES *' if p<0.05 else 'NO'
    print(f'{name}: H={stat:.4f}  p={p:.6f}  Significant={sig}')
    print(f'  (Full model comparison: Omega H=200.95 p<0.001 | Sigma H=57.76 p<0.001)')

print(f'\n{"="*60}\nA2 -- CORRELATION: Kernel Params vs Clinical ECG Features\n{"="*60}')
results = []
for kvals, kname in [(omega, 'Omega (w)'), (sigma, 'Sigma'), (alpha, 'Alpha')]:
    for feat in ecg_df.columns:
        feat_vals = ecg_df[feat].values
        mask = ~np.isnan(feat_vals) & ~np.isnan(kvals)
        if mask.sum() < 50:
            continue
        r, p = spearmanr(kvals[mask], feat_vals[mask])
        results.append({'kernel_param': kname, 'ecg_feature': feat,
                         'spearman_r': r, 'p_value': p, 'n': mask.sum(), 'significant': p < 0.05})

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(SAVE_DIR, 'A2_correlation_results.csv'), index=False)

sig_results = results_df[results_df['significant']]
print(f'\nA2 total significant correlations: {len(sig_results)} / {len(results_df)}')
print('(Full model comparison: 17/18 significant correlations)')
print(f'\nFull A2 correlation table:')
print(results_df[['kernel_param','ecg_feature','spearman_r','p_value','significant']].to_string(index=False))
print(f'\nSaved to {SAVE_DIR}/A2_correlation_results.csv')
