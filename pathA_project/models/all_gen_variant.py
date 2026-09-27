"""
PICK-Net variant with physics constraint losses extended to ALL kernel
generators, not just generators[0].

Confirmed against your real pick_net.py (verified this session):
  - PICKNet.compute_physics_loss() originally uses self.dyn_conv.generators[0] only
  - self.dyn_conv.generators is an nn.ModuleList of 32 independent KernelParamGenerator
    instances (one per output channel)
  - self.kernel_fn (on PICKNet itself, not dyn_conv) converts (alpha,omega,sigma) -> kernel
  - self.physics_loss is a PhysicsConstraintLoss(D=0.01) instance returning (L_smooth, L_phys)

This subclass only reuses confirmed attributes/methods -- no unverified
assumptions about internals you haven't shown me.

IMPORTANT COST WARNING: looping over all 32 generators makes the physics
loss ~32x more expensive per forward pass than the original (1-generator)
version. Given PICK-Net already trains 16-27x slower than your baselines
(Table VII in the paper), a naive all-32 loop could push training time from
~108s/epoch to potentially 1000s+/epoch. Use --subsample-generators (see
train_variant.py) to compute the loss on a random subset of generators per
step instead of all 32 -- this still extends regularization far beyond the
current 1-of-32, at a fraction of the cost, and the subset changes every
step so all 32 generators still get regularized over the course of training.
"""
import torch
import sys
sys.path.insert(0, '/data2/huma/picknet/src')
from pick_net import PICKNet


class PICKNet_AllGenerators(PICKNet):
    """
    Physics constraint losses computed over a (possibly subsampled) set of
    kernel generators, instead of only generators[0].
    """
    def __init__(self, lambda1=0.05, lambda2=0.05, subsample_generators=None):
        """
        subsample_generators: None -> use all 32 generators every step (expensive, most correct).
                               int  -> randomly sample this many generators per forward call
                                       (cheaper; different subset each call, so all 32 generators
                                       are regularized over the course of many training steps).
        """
        super().__init__(lambda1=lambda1, lambda2=lambda2)
        self.subsample_generators = subsample_generators

    def compute_physics_loss(self, x):
        all_generators = self.dyn_conv.generators
        n_total = len(all_generators)

        if self.subsample_generators is not None and self.subsample_generators < n_total:
            idx = torch.randperm(n_total)[:self.subsample_generators]
            generators_to_use = [all_generators[i] for i in idx]
        else:
            generators_to_use = list(all_generators)

        total_L_smooth = 0.0
        total_L_phys = 0.0
        for gen in generators_to_use:
            params = gen(x)
            kernels = self.kernel_fn(params)
            L_smooth, L_phys = self.physics_loss(kernels)
            total_L_smooth = total_L_smooth + L_smooth
            total_L_phys = total_L_phys + L_phys

        n_used = len(generators_to_use)
        total_L_smooth = total_L_smooth / n_used
        total_L_phys = total_L_phys / n_used

        return self.lambda1 * total_L_smooth + self.lambda2 * total_L_phys

    def compute_physics_loss_per_generator(self, x):
        """
        Diagnostic method (not used during training): returns per-generator
        (L_smooth, L_phys) as lists, so you can check whether all 32
        generators are similarly well-behaved -- addresses the reviewer's
        'whether all 32 generators behave similarly' ask directly.
        Not used in the training loop (too expensive to call every step);
        intended for the interpretability battery only.
        """
        smooth_per_gen, phys_per_gen = [], []
        with torch.no_grad():
            for gen in self.dyn_conv.generators:
                params = gen(x)
                kernels = self.kernel_fn(params)
                L_smooth, L_phys = self.physics_loss(kernels)
                smooth_per_gen.append(L_smooth.item())
                phys_per_gen.append(L_phys.item())
        return smooth_per_gen, phys_per_gen
