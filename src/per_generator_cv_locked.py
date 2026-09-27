"""
Recomputes the cross-generator coefficient of variation (CV) for
alpha/omega/sigma -- the 77x finding -- under the LOCKED protocol,
for any two locked-protocol conditions (typically full vs. unbounded).

This was previously only measured on the protocol-mismatched Path A
checkpoints; this script reruns it on locked_outputs/ checkpoints instead.

Usage:
    python per_generator_cv_locked.py --condition full --seed 42
    python per_generator_cv_locked.py --condition unbounded --seed 42
    (repeat for seed 123 if desired)
"""
import os, sys, argparse, json
import numpy as np
import torch

sys.path.insert(0, '/data2/huma/picknet/src')
sys.path.insert(0, '/data2/huma/picknet/pathA_project/models')
from dataset import preprocess_signal, PTBXL_PATH
from train_final import CFG

LOCKED_ROOT = '/data2/huma/picknet/locked_outputs'
SAVE_ROOT = '/data2/huma/picknet/outputs/per_gen_cv_locked'


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
    parser.add_argument('--n-samples', type=int, default=200,
                         help='Test samples used to estimate each generator average (fixed subsample for speed)')
    args = parser.parse_args()

    os.makedirs(SAVE_ROOT, exist_ok=True)
    device = torch.device(CFG['device'])
    ckpt_path = os.path.join(LOCKED_ROOT, args.condition, f'seed_{args.seed}', 'checkpoint.pt')

    print(f'Loading {args.condition} (seed {args.seed}) from {ckpt_path}...')
    model = build_model(args.condition).to(device)
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print('Loading a fixed test-set subsample (strat_fold==10, deterministic seed=42 selection)...')
    X = np.load('/data2/huma/picknet/data/raw500.npy', allow_pickle=True)
    valid_idx = np.load('/data2/huma/picknet/data/raw500_idx.npy', allow_pickle=True)
    import pandas as pd
    df = pd.read_csv(os.path.join(PTBXL_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
    df = df.loc[valid_idx]
    test_df = df[df.strat_fold == 10]
    test_pos = [list(df.index).index(i) for i in test_df.index]
    test_X = X[test_pos]

    sample_idx = np.random.RandomState(42).choice(len(test_X), size=min(args.n_samples, len(test_X)), replace=False)
    sample_signals = test_X[sample_idx]
    processed_sample = np.stack([preprocess_signal(s) for s in sample_signals])
    xb_sample = torch.tensor(processed_sample.transpose(0, 2, 1), dtype=torch.float32).to(device)

    n_gen = len(model.dyn_conv.generators)
    gen_means = {'alpha': [], 'omega': [], 'sigma': []}

    print(f'Extracting per-generator parameter means across all {n_gen} generators...')
    with torch.no_grad():
        for gi in range(n_gen):
            gen_i = model.dyn_conv.generators[gi]
            params_i = gen_i(xb_sample)
            gen_means['alpha'].append(params_i[:, 0].mean().item())
            gen_means['omega'].append(params_i[:, 1].mean().item())
            gen_means['sigma'].append(params_i[:, 2].mean().item())

    print(f'\n{"="*60}\n{args.condition} (seed {args.seed}) -- CROSS-GENERATOR CV\n{"="*60}')
    results = {}
    for pname, vals in gen_means.items():
        vals = np.array(vals)
        cv = vals.std() / (abs(vals.mean()) + 1e-8)
        print(f'  {pname}: mean={vals.mean():.4f}  std={vals.std():.4f}  CV={cv:.4f}')
        results[pname] = {'mean': float(vals.mean()), 'std': float(vals.std()), 'cv': float(cv),
                           'per_generator_means': vals.tolist()}

    out_file = f'{args.condition}_seed{args.seed}_per_gen_cv.json'
    with open(os.path.join(SAVE_ROOT, out_file), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {SAVE_ROOT}/{out_file}')


if __name__ == '__main__':
    main()
