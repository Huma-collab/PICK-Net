"""
PICK-Net variant with the 0.5-40 Hz frequency bound on omega removed,
isolating whether that specific physiological constraint matters (as
opposed to the physics losses, or the Gabor parametric form generally).

Verified against the real KernelParamGenerator.forward() (confirmed this
session):

    alpha = torch.sigmoid(p[:, 0]) * 2.0        # alpha in (0, 2)
    omega = torch.sigmoid(p[:, 1]) * 40.0 + 0.5 # omega in (0.5, 40.5) Hz  <- THIS is the bound removed here
    sigma = torch.sigmoid(p[:, 2]) * 0.1 + 0.01 # sigma in (0.01, 0.11)

Design choice: alpha and sigma keep their original bounds. Only omega's
bound is removed, since that is the specific "physiological frequency
range" claim under test (Contribution 1 of the paper); alpha and sigma's
bounds are about Gabor kernel numerical stability, not the physiological
claim being isolated here.

Omega's bound is replaced with the RAW linear output of the same fc layer
(no sigmoid, no rescale) -- i.e., the network can produce any real-valued
"frequency", unconstrained in either direction. This is the strictest,
most literal interpretation of "unbounded". Two things worth watching once
this trains:

  1. cos(2*pi*omega*t) in PhysicsInformedKernel is mathematically well-defined
     for any real omega (including negative or very large values), so this
     should not crash -- but the network is free to learn a large, physically
     meaningless frequency value if that's what minimizes the loss, which is
     exactly the behavior this experiment is designed to detect.
  2. If training is numerically unstable (loss becomes NaN, or omega explodes
     to very large magnitudes early in training), that instability is itself
     a meaningful finding about the value of the bound, not just an
     implementation bug -- note it rather than "fixing" it by re-adding a
     bound, which would defeat the point of the experiment.
"""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/data2/huma/picknet/src')
from pick_net import PICKNet, KernelParamGenerator


class KernelParamGenerator_Unbounded(KernelParamGenerator):
    """
    Identical to the original KernelParamGenerator, except omega's
    sigmoid-rescale-to-[0.5,40.5]Hz is replaced with the raw linear
    output (no bound). Alpha and sigma are unchanged.
    """
    def forward(self, x):
        h = F.relu(self.conv(x))
        h = self.pool(h).squeeze(-1)
        p = self.fc(h)

        alpha = torch.sigmoid(p[:, 0]) * 2.0          # unchanged
        omega = p[:, 1]                                # UNBOUNDED -- raw linear output, no frequency constraint
        sigma = torch.sigmoid(p[:, 2]) * 0.1 + 0.01    # unchanged

        return torch.stack([alpha, omega, sigma], dim=1)


class PICKNet_Unbounded(PICKNet):
    """
    PICK-Net using the unbounded generator in every one of its 32 kernel
    generators (each independently instantiated, same as the original
    architecture's per-channel generator design).
    """
    def __init__(self, lambda1=0.05, lambda2=0.05):
        super().__init__(lambda1=lambda1, lambda2=lambda2)
        in_channels = self.dyn_conv.generators[0].conv.in_channels
        n_generators = len(self.dyn_conv.generators)
        self.dyn_conv.generators = torch.nn.ModuleList([
            KernelParamGenerator_Unbounded(in_channels) for _ in range(n_generators)
        ])
