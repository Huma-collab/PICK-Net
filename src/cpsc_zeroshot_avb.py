import os, glob, sys
import numpy as np
import scipy.io
import torch

sys.path.insert(0, '/data2/huma/picknet/src')
from pick_net import PICKNet
from train_final import CFG
from dataset import preprocess_signal

CPSC_DIR = '/data2/huma/cpsc_2018'
CKPT     = '/data2/huma/picknet/outputs/final/best_model.pt'
OUT_DIR  = '/data2/huma/picknet/outputs/cpsc_zeroshot'
os.makedirs(OUT_DIR, exist_ok=True)

AVB_CODE      = '270492004'   # 1st-degree AV Block, confirmed n=722
THRESHOLD_AVB = 0.36           # reuse PTB-XL-optimal AVB threshold

def load_cpsc_ecg(hea_path):
    mat_path = hea_path.replace('.hea', '.mat')
    data = scipy.io.loadmat(mat_path)
    signal = data['val'].astype(np.float32)      # (12, N) raw ADC
    signal = signal / 1000.0                      # gain correction -> mV
    signal = signal[:, :5000]                     # first 10s @ 500Hz
    if signal.shape[1] < 5000:
        signal = np.pad(signal, ((0,0),(0, 5000 - signal.shape[1])))
    signal = signal.T                              # -> (5000, 12) to match preprocess_signal's expected axis
    signal = preprocess_signal(signal)             # SAME filtering/normalization as training data
    return signal.T                                # -> (12, 5000) for model input

def run():
    device = torch.device(CFG['device'])
    model = PICKNet(lambda1=CFG['lambda1'], lambda2=CFG['lambda2']).to(device)
    ckpt = torch.load(CKPT, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    hea_files = glob.glob(f'{CPSC_DIR}/**/*.hea', recursive=True)
    print(f"Found {len(hea_files)} CPSC-2018 records")

    avb_probs, avb_labels = [], []
    skipped = 0

    with torch.no_grad():
        for hea in hea_files:
            with open(hea) as f:
                content = f.read()
            dx_line = [l for l in content.splitlines() if l.startswith('# Dx:')]
            if not dx_line:
                skipped += 1
                continue
            codes = [c.strip() for c in dx_line[0].replace('# Dx:', '').split(',')]
            label = 1 if AVB_CODE in codes else 0

            try:
                sig = load_cpsc_ecg(hea)
            except Exception:
                skipped += 1
                continue

            x = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).to(device)
            out, _, _, _ = model(x)
            probs = torch.sigmoid(out).cpu().numpy()[0]

            avb_probs.append(probs[1])
            avb_labels.append(label)

    print(f"Processed {len(avb_probs)} records, skipped {skipped}")

    avb_probs  = np.array(avb_probs)
    avb_labels = np.array(avb_labels)

    from sklearn.metrics import roc_auc_score, f1_score
    auc = roc_auc_score(avb_labels, avb_probs)
    binary = (avb_probs > THRESHOLD_AVB).astype(int)
    f1 = f1_score(avb_labels, binary)

    print(f"\n{'='*50}")
    print("CPSC-2018 Zero-Shot AVB Results")
    print(f"{'='*50}")
    print(f"  n positive (AVB): {avb_labels.sum()}")
    print(f"  n negative:       {(avb_labels==0).sum()}")
    print(f"  AUC_AVB (CPSC):   {auc:.4f}")
    print(f"  F1_AVB (CPSC, threshold={THRESHOLD_AVB}): {f1:.4f}")
    print(f"\n  Compare to PTB-XL AUC_AVB: 0.9308 (single-split) / 0.9367±0.0071 (5-fold CV)")

    np.savez(f'{OUT_DIR}/avb_zeroshot_results.npz',
             probs=avb_probs, labels=avb_labels, auc=auc, f1=f1)

if __name__ == '__main__':
    run()
