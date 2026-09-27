import os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/data2/huma/picknet/src')
from pick_net    import PICKNet
from baselines   import ResNet1D, Inception1D, TransformerOnly, CNNOnly
from train_final import CFG

OUT_PATH = '/data2/huma/picknet/outputs/computational_overhead.txt'
device = torch.device(CFG['device'])

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def measure_inference(model, batch_size=32, n_warmup=10, n_trials=50):
    model.eval()
    x = torch.randn(batch_size, 12, 5000, device=device)

    # warmup
    for _ in range(n_warmup):
        _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(n_trials):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms for the whole batch

    times = np.array(times)
    per_sample_ms = times / batch_size
    return {
        'batch_ms_mean': times.mean(),
        'batch_ms_std':  times.std(),
        'per_sample_ms': per_sample_ms.mean(),
    }

def measure_train_step(model, criterion_fn, batch_size=32, n_warmup=5, n_trials=20):
    model.train()
    x = torch.randn(batch_size, 12, 5000, device=device, requires_grad=False)
    y = torch.randint(0, 2, (batch_size, 3), device=device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step():
        optimizer.zero_grad()
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = criterion_fn(logits, y)
        loss.backward()
        optimizer.step()

    for _ in range(n_warmup):
        step()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(n_trials):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    return {'step_ms_mean': times.mean(), 'step_ms_std': times.std()}

bce = torch.nn.BCEWithLogitsLoss()

def picknet_criterion(logits, y):
    return bce(logits, y)

MODELS = {
    'PICK-Net (ours)': lambda: PICKNet(lambda1=CFG['lambda1'], lambda2=CFG['lambda2']),
    'ResNet-1D':        ResNet1D,
    'Inception-1D':     Inception1D,
    'Transformer-only': TransformerOnly,
    'CNN-only':         CNNOnly,
}

results = {}
print(f"Device: {device}\n")

for name, ctor in MODELS.items():
    print(f"Measuring {name}...")
    model = ctor().to(device)
    n_params = count_params(model)
    inf = measure_inference(model)
    train = measure_train_step(model, picknet_criterion)

    results[name] = {
        'params': n_params,
        'inf_per_sample_ms': inf['per_sample_ms'],
        'inf_batch_ms': inf['batch_ms_mean'],
        'inf_batch_std': inf['batch_ms_std'],
        'train_step_ms': train['step_ms_mean'],
        'train_step_std': train['step_ms_std'],
    }
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# Estimate per-epoch training time using real dataset size
N_TRAIN_SAMPLES = 17418  # confirmed: PTB-XL folds 1-8
BATCH_SIZE = CFG['batch_size']
steps_per_epoch = N_TRAIN_SAMPLES // BATCH_SIZE

print(f"\n{'='*90}")
print(f"{'Model':<20}{'Params':>12}{'Infer/sample(ms)':>18}{'Train step(ms)':>17}{'Est. epoch(s)':>15}")
print('='*90)

lines = [f"{'='*90}",
         f"{'Model':<20}{'Params':>12}{'Infer/sample(ms)':>18}{'Train step(ms)':>17}{'Est. epoch(s)':>15}",
         '='*90]

for name, r in results.items():
    epoch_sec = (r['train_step_ms'] * steps_per_epoch) / 1000
    line = f"{name:<20}{r['params']:>12,}{r['inf_per_sample_ms']:>18.4f}{r['train_step_ms']:>17.2f}{epoch_sec:>15.1f}"
    print(line)
    lines.append(line)

lines.append('='*90)
lines.append(f"\nNote: steps_per_epoch estimated from N_TRAIN_SAMPLES={N_TRAIN_SAMPLES}, batch_size={BATCH_SIZE}")
lines.append("Inference measured on batch_size=32, averaged over 50 trials after 10 warmup iterations.")
lines.append("Training step measured on one forward+backward+optimizer.step(), averaged over 20 trials after 5 warmup iterations.")

with open(OUT_PATH, 'w') as f:
    f.write('\n'.join(lines))

print(f"\nSaved to {OUT_PATH}")
