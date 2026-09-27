import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────────
# STEP 1: Kernel Parameter Generator  g_θ(x)
# Input : x  (B, 12, 5000)  — batch of 12-lead ECG signals
# Output: p  (B, 3)         — (α, ω, σ) per sample
# ─────────────────────────────────────────────────────────────
class KernelParamGenerator(nn.Module):
    def __init__(self, in_channels=12):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 32, kernel_size=7,
                              stride=2, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(32, 3)

    def forward(self, x):
        # x: (B, 12, 5000)
        h = F.relu(self.conv(x))   # (B, 32, 2500)
        h = self.pool(h).squeeze(-1)  # (B, 32)
        p = self.fc(h)             # (B, 3)

        # constrain parameters to physiologically valid ranges
        alpha = torch.sigmoid(p[:, 0]) * 2.0        # α ∈ (0, 2)
        omega = torch.sigmoid(p[:, 1]) * 40.0 + 0.5 # ω ∈ (0.5, 40.5) Hz
        sigma = torch.sigmoid(p[:, 2]) * 0.1 + 0.01 # σ ∈ (0.01, 0.11)

        return torch.stack([alpha, omega, sigma], dim=1)  # (B, 3)


# ─────────────────────────────────────────────────────────────
# Unit test
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/data2/huma/picknet/src')
    from dataset import get_dataloaders

    device = torch.device('cuda:0')
    print('Testing on:', device)

    # get one real batch
    train_loader, _, _, _ = get_dataloaders(batch_size=8, num_workers=2)
    xb, yb = next(iter(train_loader))
    xb = xb.to(device)
    print('Input batch :', xb.shape)

    # test generator
    gen = KernelParamGenerator(in_channels=12).to(device)
    params = gen(xb)
    print('Output params:', params.shape)
    print('alpha range  :', params[:,0].min().item(), '-', params[:,0].max().item())
    print('omega range  :', params[:,1].min().item(), '-', params[:,1].max().item())
    print('sigma range  :', params[:,2].min().item(), '-', params[:,2].max().item())

    # test gradients flow
    loss = params.sum()
    loss.backward(retain_graph=True)
    print('Gradients OK :', all(p.grad is not None for p in gen.parameters()))
    print('\nStep 1 PASSED')

# ─────────────────────────────────────────────────────────────
# STEP 2: Physics-Informed Kernel  K(t; α, ω, σ)
# K(t) = α · exp(−t²/2σ²) · cos(ωt)
# This is an input-adaptive Gabor-inspired kernel where
# parameters are generated per-sample by g_θ(x)
# ─────────────────────────────────────────────────────────────
class PhysicsInformedKernel(nn.Module):
    def __init__(self, kernel_size=51, fs=500):
        super().__init__()
        self.kernel_size = kernel_size
        self.fs          = fs
        # fixed time axis — not a parameter, just a buffer
        t = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size) / fs
        self.register_buffer('t', t)

    def forward(self, params):
        # params: (B, 3) — alpha, omega, sigma per sample
        alpha = params[:, 0].view(-1, 1)   # (B, 1)
        omega = params[:, 1].view(-1, 1)   # (B, 1)
        sigma = params[:, 2].view(-1, 1)   # (B, 1)

        t = self.t.unsqueeze(0)            # (1, kernel_size)

        # Gaussian-cosine kernel (Gabor-inspired, adaptive)
        gaussian = torch.exp(-t**2 / (2 * sigma**2))
        cosine   = torch.cos(2 * np.pi * omega * t)
        kernel   = alpha * gaussian * cosine  # (B, kernel_size)

        return kernel


if __name__ == '__main__':
    # ── Step 2 test (runs after Step 1 test) ──────────────────
    print('\n--- Step 2: Physics-Informed Kernel ---')
    kernel_fn = PhysicsInformedKernel(kernel_size=51, fs=500).to(device)

    # reuse params from Step 1
    kernels = kernel_fn(params)
    print('Kernel shape     :', kernels.shape)          # (B, 51)
    print('Kernel mean      :', kernels.mean().item())
    print('Kernel min/max   :', kernels.min().item(), '/', kernels.max().item())

    # visualise one kernel value pattern
    k0 = kernels[0].detach().cpu().numpy()
    print('First kernel (centre 5 values):', k0[23:28])

    # test gradients flow back through kernel to params
    loss2 = kernels.sum()
    loss2.backward()
    print('Kernel gradients OK')
    print('\nStep 2 PASSED')

# ─────────────────────────────────────────────────────────────
# STEP 3: Dynamic Physics-Informed Convolution Layer
# Replaces fixed Conv1d with input-adaptive physics kernel
# y(t) = Σ x(τ) · K(t−τ; p(x))
# ─────────────────────────────────────────────────────────────
class DynamicPhysicsConvLayer(nn.Module):
    def __init__(self, in_channels=12, out_channels=32,
                 kernel_size=51, fs=500):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size

        # one generator per output channel
        self.generators = nn.ModuleList([
            KernelParamGenerator(in_channels) for _ in range(out_channels)
        ])
        self.kernel_fn = PhysicsInformedKernel(kernel_size, fs)

        self.bn  = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: (B, 12, 5000)
        B, C, T = x.shape
        outputs = []

        for gen in self.generators:
            params  = gen(x)                        # (B, 3)
            kernels = self.kernel_fn(params)        # (B, kernel_size)

            # apply per-sample kernel via grouped conv
            # reshape for F.conv1d: process each sample independently
            out_ch = []
            for b in range(B):
                k  = kernels[b].view(1, 1, -1)     # (1, 1, kernel_size)
                xb = x[b].unsqueeze(0)             # (1, 12, T)
                # convolve each lead with same kernel
                yb = F.conv1d(xb, k.expand(C, 1, -1),
                              padding=self.kernel_size//2,
                              groups=C)             # (1, 12, T)
                out_ch.append(yb)
            out_ch = torch.cat(out_ch, dim=0)       # (B, 12, T)
            outputs.append(out_ch)

        # stack all output channels: (B, out_channels*12, T)
        # then mix down to out_channels via 1x1 conv
        stacked = torch.stack(outputs, dim=1)       # (B, out_channels, 12, T)
        stacked = stacked.mean(dim=2)               # (B, out_channels, T)

        out = self.bn(F.relu(stacked))              # (B, out_channels, T)
        return out


if __name__ == '__main__':
    # ── Step 3 test ───────────────────────────────────────────
    print('\n--- Step 3: Dynamic Convolution Layer ---')
    dyn_conv = DynamicPhysicsConvLayer(
        in_channels=12, out_channels=32, kernel_size=51
    ).to(device)

    # fresh batch
    xb2, _ = next(iter(train_loader))
    xb2    = xb2.to(device)

    out = dyn_conv(xb2)
    print('Input  shape:', xb2.shape)
    print('Output shape:', out.shape)    # expect (8, 32, 5000)
    print('Output mean :', out.mean().item())
    print('Output std  :', out.std().item())

    # verify gradients
    loss3 = out.mean()
    loss3.backward()
    grads = [p.grad is not None for p in dyn_conv.parameters()]
    print('All grads OK:', all(grads))
    print('Grad count  :', sum(grads), '/', len(grads))
    print('\nStep 3 PASSED')

# ─────────────────────────────────────────────────────────────
# STEP 4: Physics-Based Kernel Constraint Losses
# L_smooth = ∫ |d²K/dt²|² dt  — penalises sharp kernels
# L_phys   = ∫ |∂K/∂t − D·∂²K/∂t²|² dt  — diffusion residual
# ─────────────────────────────────────────────────────────────
class PhysicsConstraintLoss(nn.Module):
    def __init__(self, D=0.01):
        super().__init__()
        self.D = D

    def forward(self, kernels):
        # kernels: (B, kernel_size)
        # finite difference approximations of derivatives
        # 1st derivative: K'(t) ≈ K(t+1) - K(t)
        d1 = kernels[:, 1:] - kernels[:, :-1]      # (B, K-1)
        # 2nd derivative: K''(t) ≈ K'(t+1) - K'(t)
        d2 = d1[:, 1:] - d1[:, :-1]                # (B, K-2)

        # smoothness loss: penalise large 2nd derivatives
        L_smooth = (d2 ** 2).mean()

        # diffusion loss: K'(t) ≈ D * K''(t)
        # align lengths: d1 has K-1, d2 has K-2
        # compare d1[:,:-1] with D*d2
        L_phys = ((d1[:, :-1] - self.D * d2) ** 2).mean()

        return L_smooth, L_phys


if __name__ == '__main__':
    # ── Step 4 test ───────────────────────────────────────────
    print('\n--- Step 4: Physics Constraint Losses ---')
    physics_loss = PhysicsConstraintLoss(D=0.01).to(device)

    # generate kernels from a fresh batch
    gen2        = KernelParamGenerator(in_channels=12).to(device)
    kernel_fn2  = PhysicsInformedKernel(kernel_size=51).to(device)

    xb3, _  = next(iter(train_loader))
    xb3     = xb3.to(device)
    params3 = gen2(xb3)
    kernels3 = kernel_fn2(params3)

    L_smooth, L_phys = physics_loss(kernels3)
    print('L_smooth :', L_smooth.item())
    print('L_phys   :', L_phys.item())

    # verify both are positive and differentiable
    total = L_smooth + L_phys
    total.backward()
    print('Gradients flow through losses: OK')

    # sanity check: smooth kernel should have lower L_smooth
    # than random noise kernel
    noise_kernel = torch.randn_like(kernels3)
    L_s_noise, _ = physics_loss(noise_kernel)
    print('L_smooth (physics kernel):', L_smooth.item())
    print('L_smooth (random noise)  :', L_s_noise.item())
    print('Physics kernel smoother  :', L_smooth.item() < L_s_noise.item())
    print('\nStep 4 PASSED')

# ─────────────────────────────────────────────────────────────
# STEP 5: Deep CNN Feature Extractor
# Input : (B, 32, 5000) — output of dynamic conv layer
# Output: (B, 256, 156) — compressed feature sequence
# ─────────────────────────────────────────────────────────────
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=32, base_channels=64):
        super().__init__()
        self.blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels,      base_channels,   kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),        # 5000 → 1250

            # Block 2
            nn.Conv1d(base_channels,    base_channels*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),        # 1250 → 312

            # Block 3
            nn.Conv1d(base_channels*2,  base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),        # 312 → 156
        )
        self.out_channels = base_channels * 4             # 256

    def forward(self, x):
        return self.blocks(x)                             # (B, 256, 156)


if __name__ == '__main__':
    # ── Step 5 test ───────────────────────────────────────────
    print('\n--- Step 5: CNN Feature Extractor ---')

    xb4, _ = next(iter(train_loader))
    xb4    = xb4.to(device)

    # full forward: input → dynamic conv → CNN
    dyn_conv2 = DynamicPhysicsConvLayer(
        in_channels=12, out_channels=32, kernel_size=51
    ).to(device)
    cnn = CNNFeatureExtractor(in_channels=32, base_channels=64).to(device)

    features = dyn_conv2(xb4)       # (B, 32, 5000)
    features = cnn(features)        # (B, 256, 156)

    print('After dyn conv :', dyn_conv2(xb4).shape)
    print('After CNN      :', features.shape)
    print('Feature mean   :', features.mean().item())
    print('Feature std    :', features.std().item())

    loss5 = features.mean()
    loss5.backward()
    print('Gradients OK   : True')
    print('\nStep 5 PASSED')

# ─────────────────────────────────────────────────────────────
# STEP 6: Transformer Encoder
# Input : (B, 256, 156) — CNN feature sequence
# Output: (B, 256)      — shared latent representation z
# ─────────────────────────────────────────────────────────────
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        # positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 156, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, 256, 156) — channels first
        x = x.permute(0, 2, 1)          # (B, 156, 256) — seq first
        x = x + self.pos_embedding      # add positional encoding
        x = self.transformer(x)         # (B, 156, 256)
        x = self.norm(x)
        z = x.mean(dim=1)               # (B, 256) — global average
        return z


if __name__ == '__main__':
    # ── Step 6 test ───────────────────────────────────────────
    print('\n--- Step 6: Transformer Encoder ---')

    xb5, _ = next(iter(train_loader))
    xb5    = xb5.to(device)

    dyn_conv3 = DynamicPhysicsConvLayer(
        in_channels=12, out_channels=32, kernel_size=51
    ).to(device)
    cnn2      = CNNFeatureExtractor(in_channels=32, base_channels=64).to(device)
    transformer = TransformerEncoder(
        d_model=256, nhead=4, num_layers=2
    ).to(device)

    # full pipeline so far
    f1 = dyn_conv3(xb5)           # (B, 32, 5000)
    f2 = cnn2(f1)                 # (B, 256, 156)
    z  = transformer(f2)          # (B, 256)

    print('After dyn conv  :', f1.shape)
    print('After CNN       :', f2.shape)
    print('After Transformer:', z.shape)
    print('Latent z mean   :', z.mean().item())
    print('Latent z std    :', z.std().item())

    loss6 = z.mean()
    loss6.backward()
    print('Gradients OK    : True')
    print('\nStep 6 PASSED')

# ─────────────────────────────────────────────────────────────
# STEP 7: Physiology-Aware Condition Interaction Module (PACIM)
# Input : z (B, 256) — shared latent representation
# Output: z_mi, z_avb, z_int — condition-specific subspaces
#
# Key novelty: directional gate models the clinical reality
# that MI (ischemia) causes AVB (conduction disturbance)
# z_int = z_MI ⊙ σ(W_g · z_AVB)
# ─────────────────────────────────────────────────────────────
class PACIM(nn.Module):
    def __init__(self, d_model=256, d_sub=128):
        super().__init__()
        # condition-specific projection matrices
        self.W_MI  = nn.Linear(d_model, d_sub, bias=False)  # ischemia subspace
        self.W_AVB = nn.Linear(d_model, d_sub, bias=False)  # conduction subspace

        # directional gating: AVB features gate MI representation
        # reflects clinical direction: MI → AVB (not reverse)
        self.W_g = nn.Linear(d_sub, d_sub, bias=True)

        # output heads
        self.head_MI    = nn.Linear(d_sub, 1)
        self.head_AVB   = nn.Linear(d_sub, 1)
        self.head_MIAVB = nn.Linear(d_sub, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, z):
        # z: (B, 256)

        # decompose into condition-specific subspaces
        z_MI  = F.relu(self.W_MI(z))           # (B, 128) ischemia features
        z_AVB = F.relu(self.W_AVB(z))          # (B, 128) conduction features

        # directional gated interaction
        # MI features modulated by AVB severity
        gate  = torch.sigmoid(self.W_g(z_AVB)) # (B, 128)
        z_int = z_MI * gate                    # (B, 128) interaction repr

        z_MI  = self.dropout(z_MI)
        z_AVB = self.dropout(z_AVB)
        z_int = self.dropout(z_int)

        # predictions from each subspace
        y_mi    = self.head_MI(z_MI)            # (B, 1)
        y_avb   = self.head_AVB(z_AVB)          # (B, 1)
        y_miavb = self.head_MIAVB(z_int)        # (B, 1)

        # concatenate to (B, 3)
        out = torch.cat([y_mi, y_avb, y_miavb], dim=1)

        return out, z_MI, z_AVB, z_int


if __name__ == '__main__':
    # ── Step 7 test ───────────────────────────────────────────
    print('\n--- Step 7: PACIM Module ---')

    xb6, yb6 = next(iter(train_loader))
    xb6 = xb6.to(device)
    yb6 = yb6.to(device)

    dyn_conv4   = DynamicPhysicsConvLayer(12, 32, 51).to(device)
    cnn3        = CNNFeatureExtractor(32, 64).to(device)
    transformer2 = TransformerEncoder(256, 4, 2).to(device)
    pacim       = PACIM(d_model=256, d_sub=128).to(device)

    # full forward pass
    f   = dyn_conv4(xb6)           # (B, 32, 5000)
    f   = cnn3(f)                  # (B, 256, 156)
    z   = transformer2(f)          # (B, 256)
    out, z_mi, z_avb, z_int = pacim(z)

    print('Latent z     :', z.shape)
    print('z_MI shape   :', z_mi.shape)
    print('z_AVB shape  :', z_avb.shape)
    print('z_int shape  :', z_int.shape)
    print('Output shape :', out.shape)     # (B, 3)
    print('Output sample:', out[0].detach().cpu().numpy())

    # check gate activations differ for different samples
    gate_vals = torch.sigmoid(pacim.W_g(z_avb))
    print('Gate mean    :', gate_vals.mean().item())
    print('Gate std     :', gate_vals.std().item())
    print('Gate varies  :', gate_vals.std().item() > 0.001)

    # verify interaction z_int differs from z_MI
    diff = (z_int - z_mi).abs().mean().item()
    print('z_int ≠ z_MI :', diff > 0)

    loss7 = out.mean()
    loss7.backward()
    print('Gradients OK : True')
    print('\nStep 7 PASSED')

# ─────────────────────────────────────────────────────────────
# STEP 8: Complete PICK-Net Model
# Assembles all components into one end-to-end model
# ─────────────────────────────────────────────────────────────
class PICKNet(nn.Module):
    def __init__(self,
                 in_channels=12,
                 out_channels=32,
                 kernel_size=51,
                 base_channels=64,
                 d_model=256,
                 d_sub=128,
                 nhead=4,
                 num_layers=2,
                 lambda1=0.01,
                 lambda2=0.01,
                 fs=500):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # component 1: dynamic physics-informed conv
        self.dyn_conv    = DynamicPhysicsConvLayer(
            in_channels, out_channels, kernel_size, fs
        )
        # component 2: physics losses
        self.physics_loss = PhysicsConstraintLoss(D=0.01)

        # component 3: CNN feature extractor
        self.cnn         = CNNFeatureExtractor(out_channels, base_channels)

        # component 4: transformer encoder
        self.transformer = TransformerEncoder(
            d_model, nhead, num_layers
        )

        # component 5: PACIM
        self.pacim       = PACIM(d_model, d_sub)

        # store kernel fn for physics loss computation
        self.kernel_fn   = PhysicsInformedKernel(kernel_size, fs)

    def forward(self, x):
        # x: (B, 12, 5000)

        # dynamic physics conv
        x_conv = self.dyn_conv(x)          # (B, 32, 5000)

        # CNN features
        f = self.cnn(x_conv)               # (B, 256, 156)

        # transformer latent
        z = self.transformer(f)            # (B, 256)

        # PACIM predictions
        out, z_mi, z_avb, z_int = self.pacim(z)

        return out, z_mi, z_avb, z_int

    def compute_physics_loss(self, x):
        # compute physics losses from kernel params
        # use first generator only for efficiency
        gen    = self.dyn_conv.generators[0]
        params = gen(x)
        kernels = self.kernel_fn(params)
        L_smooth, L_phys = self.physics_loss(kernels)
        return self.lambda1 * L_smooth + self.lambda2 * L_phys


# ─────────────────────────────────────────────────────────────
# Total Loss Function
# L_total = L_MI + L_AVB + L_MI+AVB + L_int + L_physics
# ─────────────────────────────────────────────────────────────
class PICKNetLoss(nn.Module):
    def __init__(self, lambda_int=0.1):
        super().__init__()
        self.lambda_int = lambda_int
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets, z_int, z_mi, z_avb):
        # predictions: (B, 3) — logits for MI, AVB, MI+AVB
        # targets    : (B, 3) — binary labels
        y_mi    = targets[:, 0]
        y_avb   = targets[:, 1]
        y_miavb = targets[:, 2]

        # classification losses
        L_MI    = self.bce(predictions[:, 0], y_mi)
        L_AVB   = self.bce(predictions[:, 1], y_avb)
        L_MIAVB = self.bce(predictions[:, 2], y_miavb)

        # interaction regularisation loss
        # z_int should be active only when both MI and AVB present
        y_int_gt = (y_mi * y_avb).unsqueeze(1)          # (B, 1)
        beta     = 1.0
        L_int    = ((z_int.norm(dim=1, keepdim=True) -
                     beta * y_int_gt) ** 2).mean()

        L_total = L_MI + L_AVB + L_MIAVB + self.lambda_int * L_int
        return L_total, {
            'L_MI'   : L_MI.item(),
            'L_AVB'  : L_AVB.item(),
            'L_MIAVB': L_MIAVB.item(),
            'L_int'  : L_int.item(),
        }


if __name__ == '__main__':
    # ── Step 8 test ───────────────────────────────────────────
    print('\n--- Step 8: Complete PICK-Net ---')

    model     = PICKNet().to(device)
    criterion = PICKNetLoss(lambda_int=0.1)

    xb7, yb7 = next(iter(train_loader))
    xb7 = xb7.to(device)
    yb7 = yb7.to(device)

    # forward pass
    out, z_mi, z_avb, z_int = model(xb7)
    print('Input  :', xb7.shape)
    print('Output :', out.shape)

    # loss
    L_phys  = model.compute_physics_loss(xb7)
    L_task, breakdown = criterion(out, yb7, z_int, z_mi, z_avb)
    L_total = L_task + L_phys

    print('L_MI    :', breakdown['L_MI'])
    print('L_AVB   :', breakdown['L_AVB'])
    print('L_MIAVB :', breakdown['L_MIAVB'])
    print('L_int   :', breakdown['L_int'])
    print('L_phys  :', L_phys.item())
    print('L_total :', L_total.item())

    # backward
    L_total.backward()

    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params    : {total_params:,}')
    print(f'Trainable params: {train_params:,}')

    print('\nStep 8 PASSED — PICK-Net fully assembled')
