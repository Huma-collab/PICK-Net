"""
Lightweight extraction of real per-sample omega values for Full and
Unbounded locked checkpoints (seed 42), for a real frequency-distribution
histogram. Skips neurokit2 entirely (not needed -- we only want omega
itself, not clinical correlations), so this runs in well under a minute.
"""
import os, sys, ast, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '/data2/huma/picknet/src')
sys.path.insert(0, '/data2/huma/picknet/pathA_project/models')
from dataset import PTBXL_PATH, build_picknet_labels, preprocess_signal
from train_final import CFG

LOCKED_ROOT = '/data2/huma/picknet/locked_outputs'
SAVE_ROOT = '/data2/huma/picknet/outputs/per_gen_cv_locked'


def build_model(condition):
    if condition == 'full':
        from pick_net import PICKNet
        return PICKNet(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'])
    elif condition == 'unbounded':
        from unbounded_variant import PICKNet_Unbounded
        return PICKNet_Unbounded(lambda1=CFG['lambda1'], lambda2=CFG['lambda2'])
    else:
        raise ValueError(condition)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True, choices=['full', 'unbounded'])
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()

    os.makedirs(SAVE_ROOT, exist_ok=True)
    device = torch.device(CFG['device'])
    ckpt_path = os.path.join(LOCKED_ROOT, args.condition, f'seed_{args.seed}', 'checkpoint.pt')

    print(f'Loading {args.condition} (seed {args.seed})...')
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

    print(f'Extracting omega for {len(test_X)} real test samples (no neurokit2 needed)...')
    all_omega = []
    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(test_X), batch_size):
            batch_signals = test_X[start:start + batch_size]
            processed = np.stack([preprocess_signal(s) for s in batch_signals])
            xb = torch.tensor(processed.transpose(0, 2, 1), dtype=torch.float32).to(device)
            params = gen(xb)
            all_omega.append(params[:, 1].cpu().numpy())

    omega = np.concatenate(all_omega)
    out_path = os.path.join(SAVE_ROOT, f'{args.condition}_seed{args.seed}_omega_raw.npy')
    np.save(out_path, omega)
    print(f'Saved {len(omega)} real omega values to {out_path}')
    print(f'omega range: [{omega.min():.3f}, {omega.max():.3f}]  mean={omega.mean():.3f}  std={omega.std():.3f}')


if __name__ == '__main__':
    main()
