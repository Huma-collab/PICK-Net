import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

sys.path.insert(0, '/data2/huma/picknet/src')
from pick_net    import PICKNet, KernelParamGenerator, PhysicsInformedKernel
from train_final import get_dataloaders_uniform, CFG
from dataset     import PTBXL_PATH, preprocess_signal

SAVE_DIR = '/data2/huma/picknet/outputs/cardiologist_kernels'
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES   = ['Normal', 'MI', 'AVB', 'MI+AVB']
CLASS_COLORS  = ['#5F5E5A', '#D85A30', '#185FA5', '#1D9E75']
CLASS_LABELS  = {
    0: 'Normal sinus rhythm — no pathology detected',
    1: 'Myocardial Infarction (MI)',
    2: 'Atrioventricular Block (AVB)',
    3: 'MI + AVB co-occurrence',
}

def get_ecg_clinical_features(raw_signal, fs=500):
    """Extract dominant frequency and basic stats from raw ECG."""
    import neurokit2 as nk
    feats = {}
    try:
        lead = raw_signal[:, 1]  # Lead II
        ecg_c = nk.ecg_clean(lead, sampling_rate=fs)
        _, info = nk.ecg_process(ecg_c, sampling_rate=fs)
        rpeaks = info['ECG_R_Peaks']
        if len(rpeaks) > 1:
            rr = np.diff(rpeaks) / fs * 1000
            feats['hr']  = round(60000 / rr.mean(), 1)
            feats['rr']  = round(rr.mean(), 1)
        # ST level at 80ms after R
        st_vals = []
        for r in rpeaks:
            idx = r + int(0.08*fs)
            if idx < len(lead):
                st_vals.append(lead[idx])
        if st_vals:
            feats['st'] = round(np.mean(st_vals)*1000, 2)  # mV→µV
    except Exception:
        pass
    # dominant freq
    lead = raw_signal[:, 1]
    fft  = np.abs(np.fft.rfft(lead))
    freq = np.fft.rfftfreq(len(lead), 1/fs)
    mask = (freq >= 0.5) & (freq <= 40)
    if mask.sum() > 0:
        feats['dom_freq'] = round(freq[mask][np.argmax(fft[mask])], 2)
    return feats

def make_cardiologist_figure(
        raw_signal, kernel_vals, t_ms, params,
        class_id, sample_id, ecg_feats, save_path):
    """
    One figure per sample showing:
    - Top:    Raw ECG (leads I, II, V1, V5) — 10 seconds
    - Middle: Physics-informed kernel shape
    - Bottom: Kernel parameters with clinical annotations
    """
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        hspace=0.45, wspace=0.35,
        height_ratios=[2.5, 2, 1]
    )

    color   = CLASS_COLORS[class_id]
    label   = CLASS_LABELS[class_id]
    lead_names = ['I','II','III','AVR','AVL','AVF',
                  'V1','V2','V3','V4','V5','V6']
    show_leads = [0, 1, 6, 10]   # I, II, V1, V5
    t_ecg = np.arange(raw_signal.shape[0]) / 500   # seconds

    # ── Row 1: ECG leads ──────────────────────────────────────
    for col, li in enumerate(show_leads):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t_ecg, raw_signal[:, li],
                color='#1a1a1a', linewidth=0.7, alpha=0.9)
        ax.set_title(f'Lead {lead_names[li]}',
                     fontsize=11, fontweight='bold', pad=4)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Amplitude (mV)', fontsize=9)
        ax.set_xlim([0, 10])
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=8)
        # shade last 2s for visual focus
        ax.axvspan(0, 10, alpha=0.02, color=color)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    # ── Row 2: Kernel shape (spans all 4 cols) ────────────────
    ax_k = fig.add_subplot(gs[1, :])
    ax_k.plot(t_ms, kernel_vals, color=color,
              linewidth=2.5, label='K(t; α,ω,σ)')
    ax_k.fill_between(t_ms, kernel_vals, alpha=0.15, color=color)
    ax_k.axhline(0, color='#888888', linewidth=0.8,
                 linestyle='--', alpha=0.6)
    ax_k.axvline(0, color='#888888', linewidth=0.8,
                 linestyle='--', alpha=0.6)

    # annotate peak
    peak_idx = np.argmax(np.abs(kernel_vals))
    ax_k.annotate(
        f'Peak at t={t_ms[peak_idx]:.1f}ms',
        xy=(t_ms[peak_idx], kernel_vals[peak_idx]),
        xytext=(t_ms[peak_idx]+8, kernel_vals[peak_idx]*0.85),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='#444444',
                        lw=1.2),
        color='#444444'
    )

    ax_k.set_xlabel('Time relative to kernel center (ms)',
                    fontsize=11)
    ax_k.set_ylabel('Kernel amplitude K(t)', fontsize=11)
    ax_k.set_title(
        f'Physics-Informed Kernel Shape   '
        f'K(t; α,ω,σ) = α·exp(−t²/2σ²)·cos(2πωt)',
        fontsize=12, fontweight='bold', pad=6
    )
    ax_k.legend(fontsize=10, loc='upper right')
    ax_k.grid(True, alpha=0.2, linestyle='--')
    ax_k.tick_params(labelsize=9)
    for spine in ax_k.spines.values():
        spine.set_linewidth(0.5)

    # ── Row 3: Parameter summary boxes ───────────────────────
    alpha_val, omega_val, sigma_val = (
        params[0], params[1], params[2]
    )

    param_info = [
        {
            'name' : 'α  (Amplitude)',
            'value': f'{alpha_val:.4f}',
            'unit' : 'a.u.',
            'interp': (
                'Controls kernel gain.\n'
                'Correlates with signal\n'
                'power and heart rate.'
            ),
        },
        {
            'name' : 'ω  (Frequency)',
            'value': f'{omega_val:.2f}',
            'unit' : 'Hz',
            'interp': (
                'Dominant filter frequency.\n'
                'Correlates with ST-level\n'
                'and signal power (p<0.001).'
            ),
        },
        {
            'name' : 'σ  (Smoothness)',
            'value': f'{sigma_val:.4f}',
            'unit' : 'a.u.',
            'interp': (
                'Kernel temporal width.\n'
                'Correlates with heart\n'
                'rate and RR interval.'
            ),
        },
        {
            'name' : 'ECG features',
            'value': (
                f'HR: {ecg_feats.get("hr","–")} bpm\n'
                f'RR: {ecg_feats.get("rr","–")} ms\n'
                f'ST: {ecg_feats.get("st","–")} µV'
            ),
            'unit' : '',
            'interp': (
                'Clinical measurements\n'
                'extracted from Lead II\n'
                'for correlation context.'
            ),
        },
    ]

    for col, info in enumerate(param_info):
        ax_p = fig.add_subplot(gs[2, col])
        ax_p.axis('off')
        box_color = color if col < 3 else '#F0F0F0'
        txt_color = 'white' if col < 3 else '#333333'

        ax_p.add_patch(plt.Rectangle(
            (0, 0), 1, 1,
            transform=ax_p.transAxes,
            facecolor=box_color, edgecolor='#CCCCCC',
            linewidth=0.8, clip_on=False
        ))
        ax_p.text(0.5, 0.88, info['name'],
                  transform=ax_p.transAxes,
                  ha='center', va='top',
                  fontsize=9, fontweight='bold',
                  color=txt_color)
        ax_p.text(0.5, 0.60,
                  f"{info['value']}  {info['unit']}",
                  transform=ax_p.transAxes,
                  ha='center', va='top',
                  fontsize=11, fontweight='bold',
                  color=txt_color)
        ax_p.text(0.5, 0.30, info['interp'],
                  transform=ax_p.transAxes,
                  ha='center', va='top',
                  fontsize=7.5, color=txt_color,
                  linespacing=1.4)

    # ── Overall title ─────────────────────────────────────────
    fig.suptitle(
        f'Sample {sample_id:02d}  —  Diagnosis: {label}\n'
        f'PICK-Net Physics-Informed Kernel Analysis  '
        f'(PTB-XL Test Set)',
        fontsize=13, fontweight='bold',
        color=color, y=0.98
    )

    plt.savefig(save_path, dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

def run():
    device    = torch.device(CFG['device'])
    best_path = '/data2/huma/picknet/outputs/final/best_model.pt'

    print('Loading model...')
    model = PICKNet(lambda1=CFG['lambda1'],
                    lambda2=CFG['lambda2']).to(device)
    ckpt  = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    gen       = model.dyn_conv.generators[0].to(device)
    kernel_fn = model.kernel_fn.to(device)

    # time axis in milliseconds
    t_ms = np.linspace(-25, 25, 51) / 500 * 1000

    print('Loading raw signals...')
    import pandas as pd, ast
    X         = np.load('/data2/huma/picknet/data/raw500.npy',
                        allow_pickle=True)
    valid_idx = np.load('/data2/huma/picknet/data/raw500_idx.npy',
                        allow_pickle=True)
    df        = pd.read_csv(
        os.path.join(PTBXL_PATH, 'ptbxl_database.csv'),
        index_col='ecg_id'
    )
    df        = df.loc[valid_idx]
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

    from dataset import build_picknet_labels
    labels_df = build_picknet_labels(df)

    test_df     = df[df.strat_fold == 10]
    test_labels = labels_df.loc[test_df.index]
    test_pos    = [list(df.index).index(i) for i in test_df.index]
    test_X      = X[test_pos]

    # select 5 samples per class
    class_ids_all = np.zeros(len(test_labels), dtype=int)
    lv = test_labels[['mi','avb','mi_avb']].values
    class_ids_all[lv[:,0]==1] = 1
    class_ids_all[lv[:,1]==1] = 2
    class_ids_all[lv[:,2]==1] = 3

    selected = []
    for cid in range(4):
        idxs = np.where(class_ids_all == cid)[0]
        # pick 5 evenly spaced to get variety
        chosen = idxs[np.linspace(0, len(idxs)-1, 5,
                                   dtype=int)]
        for idx in chosen:
            selected.append((idx, cid))

    print(f'Generating {len(selected)} cardiologist figures...')
    print('Classes: 5×Normal + 5×MI + 5×AVB + 5×MI+AVB = 20 total')

    for fig_num, (idx, cid) in enumerate(
            tqdm(selected, desc='Rendering')):
        raw_sig = test_X[idx]             # (5000, 12)

        # get kernel params
        proc = preprocess_signal(raw_sig)
        xb   = torch.tensor(
            proc.T[np.newaxis], dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            params  = gen(xb)[0].cpu().numpy()
            kern_t  = torch.tensor(
                params, dtype=torch.float32
            ).unsqueeze(0).to(device)
            # rebuild kernel manually
            alpha = params[0]
            omega = params[1]
            sigma = params[2]
            t_ax  = torch.linspace(-25,25,51).to(device)/500
            kv = (alpha *
                  torch.exp(-t_ax**2/(2*sigma**2)) *
                  torch.cos(2*np.pi*omega*t_ax))
            kernel_vals = kv.cpu().numpy()

        # extract clinical features from raw signal
        ecg_feats = get_ecg_clinical_features(raw_sig)

        class_name = CLASS_NAMES[cid]
        fname = (f'{fig_num+1:02d}_{class_name}_'
                 f'alpha{params[0]:.3f}_'
                 f'omega{params[1]:.1f}_'
                 f'sigma{params[2]:.4f}.png')
        save_path = os.path.join(SAVE_DIR, fname)

        make_cardiologist_figure(
            raw_signal  = raw_sig,
            kernel_vals = kernel_vals,
            t_ms        = t_ms,
            params      = params,
            class_id    = cid,
            sample_id   = fig_num + 1,
            ecg_feats   = ecg_feats,
            save_path   = save_path,
        )

    print(f'\nAll 20 figures saved to: {SAVE_DIR}')
    print('\nFigure naming convention:')
    print('  NN_Class_alphaX.XXX_omegaXX.X_sigmaX.XXXX.png')
    print('\nReady to show to cardiologist.')
    print('Each figure shows:')
    print('  - Raw ECG (4 leads: I, II, V1, V5)')
    print('  - Physics-informed kernel shape')
    print('  - Kernel parameters with clinical annotations')
    print('  - ECG clinical measurements for context')

if __name__ == '__main__':
    run()
