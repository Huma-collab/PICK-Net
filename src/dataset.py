import os
import sys
import ast
import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# benchmarking utils for evaluation only
BENCH_PATH = '/data2/huma/picknet/ecg_ptbxl_benchmarking/code/utils'
sys.path.insert(0, BENCH_PATH)
from utils import evaluate_experiment, apply_thresholds

# paths
PTBXL_PATH    = '/data2/huma/picknet/data/ptbxl/'
CACHE_PATH    = '/data2/huma/picknet/data/raw500.npy'
CACHE_IDX     = '/data2/huma/picknet/data/raw500_idx.npy'
SAMPLING_RATE = 500

# SCP codes
MI_CODES = {
    'AMI','IMI','ASMI','ILMI','ALMI','INJAS','LMI',
    'INJAL','IPLMI','IPMI','INJIN','INJLA','PMI','INJIL'
}
AVB_CODES = {
    '1AVB','2AVB','3AVB'
}

def load_raw_signals(df, ptbxl_path, cache_path, cache_idx_path):
    if os.path.exists(cache_path) and os.path.exists(cache_idx_path):
        print('Loading from cache...')
        data      = np.load(cache_path, allow_pickle=True)
        valid_idx = np.load(cache_idx_path, allow_pickle=True)
        return data, valid_idx

    print('Reading signals from disk (first time only)...')
    data      = []
    valid_idx = []

    for ecg_id, row in tqdm(df.iterrows(), total=len(df)):
        path = os.path.join(ptbxl_path, row['filename_hr'])
        hea  = path + '.hea'
        if not os.path.exists(hea):
            continue
        try:
            signal, _ = wfdb.rdsamp(path)
            if signal.shape == (5000, 12):
                data.append(signal.astype(np.float32))
                valid_idx.append(ecg_id)
        except Exception:
            continue

    data      = np.array(data, dtype=np.float32)
    valid_idx = np.array(valid_idx)

    print(f'Loaded {len(data)} records, skipped {len(df)-len(data)} corrupt/missing')
    np.save(cache_path,     data)
    np.save(cache_idx_path, valid_idx)
    return data, valid_idx

def preprocess_signal(signal, fs=500):
    low  = 0.5  / (fs / 2)
    high = 40.0 / (fs / 2)
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, signal, axis=0).astype(np.float32)
    mu  = filtered.mean(axis=0, keepdims=True)
    std = filtered.std(axis=0,  keepdims=True) + 1e-8
    return ((filtered - mu) / std).astype(np.float32)

def build_picknet_labels(Y):
    scp_df     = pd.read_csv(os.path.join(PTBXL_PATH, 'scp_statements.csv'), index_col=0)
    diag_codes = set(scp_df[scp_df['diagnostic'] == 1].index)
    records    = []
    for _, row in Y.iterrows():
        codes   = {k for k in row['scp_codes'] if k in diag_codes}
        has_mi  = bool(codes & MI_CODES)
        has_avb = bool(codes & AVB_CODES)
        records.append({
            'mi'    : int(has_mi),
            'avb'   : int(has_avb),
            'mi_avb': int(has_mi and has_avb),
            'normal': int(not has_mi and not has_avb),
        })
    return pd.DataFrame(records, index=Y.index)

class PICKNetDataset(Dataset):
    def __init__(self, X, labels_df, preprocess=True):
        self.X          = X
        self.labels     = labels_df[['mi','avb','mi_avb']].values.astype(np.float32)
        self.do_preproc = preprocess

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        signal = self.X[idx]
        if self.do_preproc:
            signal = preprocess_signal(signal)
        signal = signal.T
        return torch.tensor(signal, dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.float32)

def make_weighted_sampler(labels_df):
    labels   = labels_df[['mi','avb','mi_avb']].values
    class_id = np.zeros(len(labels), dtype=int)
    class_id[labels[:,0] == 1] = 1
    class_id[labels[:,1] == 1] = 2
    class_id[labels[:,2] == 1] = 3
    counts  = np.bincount(class_id)
    weights = 1.0 / counts[class_id]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(weights),
        replacement=True
    )

def get_dataloaders(batch_size=32, num_workers=4):
    df = pd.read_csv(os.path.join(PTBXL_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    X, valid_idx = load_raw_signals(df, PTBXL_PATH, CACHE_PATH, CACHE_IDX)

    df        = df.loc[valid_idx]
    labels_df = build_picknet_labels(df)

    train_idx = df[df.strat_fold <= 8].index
    val_idx   = df[df.strat_fold == 9].index
    test_idx  = df[df.strat_fold == 10].index

    def make_ds(idx):
        pos = [list(df.index).index(i) for i in idx]
        return PICKNetDataset(X[pos], labels_df.loc[idx])

    train_ds     = make_ds(train_idx)
    val_ds       = make_ds(val_idx)
    test_ds      = make_ds(test_idx)
    sampler      = make_weighted_sampler(labels_df.loc[train_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, labels_df

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(PTBXL_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    X, valid_idx = load_raw_signals(df, PTBXL_PATH, CACHE_PATH, CACHE_IDX)
    df           = df.loc[valid_idx]
    labels_df    = build_picknet_labels(df)

    mi_only  = ((labels_df['mi']==1) & (labels_df['avb']==0)).sum()
    avb_only = ((labels_df['avb']==1) & (labels_df['mi']==0)).sum()

    print('\n=== Label Distribution ===')
    print('Normal  :', labels_df['normal'].sum())
    print('MI only :', mi_only)
    print('AVB only:', avb_only)
    print('MI+AVB  :', labels_df['mi_avb'].sum())
    print('Total   :', len(labels_df))

    train_loader, val_loader, test_loader, _ = get_dataloaders(batch_size=32)
    xb, yb = next(iter(train_loader))
    print('\nBatch signal :', xb.shape)
    print('Batch labels :', yb.shape)
    print('Label sample :', yb[:4])
    print('\nPipeline OK')
