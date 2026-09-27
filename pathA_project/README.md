# PICK-Net Path A -- Frequency-Bound & Full-Generator Ablation Campaign

This is a **separate, self-contained project**. It does not modify any file in
`/data2/huma/picknet/` -- it only *imports* from your existing `src/` modules
(read-only) and writes all its own outputs to a new directory tree. You can
run this alongside your existing reduced-Path-B work with zero risk of the
two interfering with each other.

## What this answers

Per the reviewer's critique: your current physics losses are computed using
only 1 of 32 independent kernel generators, and there is no ablation that
isolates the 0.5-40 Hz frequency bound specifically (as opposed to the whole
physics-loss mechanism, which A1-A4 already test). This project trains the
two missing model variants and runs the full interpretability battery on all
of them, so you have evidence -- not just a disclosed limitation -- for
whether the frequency bound and the all-generator physics loss matter.

## Directory layout

```
pathA_project/
  README.md                      <- this file
  models/
    all_gen_variant.py            <- PICKNet with physics loss over all 32 generators (READY)
    unbounded_variant.py          <- PICKNet with no frequency bound (NEEDS ONE VERIFICATION STEP -- see below)
  scripts/
    train_variant.py              <- generic trainer, seeded, saves checkpoint + predictions
    interpretability_battery.py   <- full analysis: correlations w/ effect sizes + bootstrap CIs,
                                      per-generator consistency, class-separation
    delong_compare.py             <- pairwise DeLong tests across all trained variants
    run_all.sh                    <- orchestrates the full campaign in order
  outputs/                        <- everything this project writes goes here, nothing touches
                                     your existing outputs/ directories
```

## Before running anything: one real gap to close

I have not seen the source of `KernelParamGenerator` or `PhysicsInformedKernel`
in `pick_net.py` -- specifically, I don't know the exact mechanism that
constrains omega to 0.5-40 Hz (a `torch.clamp`? a `sigmoid` rescaled into that
range? a `tanh`?). `models/unbounded_variant.py` is written as a clearly-marked
scaffold with the bound-removal logic left as a `# TODO: VERIFY` block --
**do not run it until this is confirmed**, or you risk silently training a
model that isn't actually unbounded, wasting the GPU time.

To close this gap, run on your server:
```bash
grep -n "class KernelParamGenerator" -A 30 /data2/huma/picknet/src/pick_net.py
```
and send me the output. I'll then finalize `unbounded_variant.py` to match
your real code exactly, the same way every other script this session was
verified before running.

**Everything else in this project (`all_gen_variant.py`, the trainer, the
interpretability battery, DeLong comparisons) is built from code you've
already shown me and is ready to run now**, without waiting on that one item.

## Recommended order

1. **Now, in parallel with Path B**: run `train_variant.py --variant all_gen`
   (extends the physics loss to all 32 generators -- this is ready).
2. **Send me the `KernelParamGenerator` grep output** whenever convenient.
3. Once I confirm `unbounded_variant.py`: run
   `train_variant.py --variant unbounded`.
4. Run `interpretability_battery.py` on all three variants (all-gen, unbounded,
   and your existing A2 checkpoint from the reduced-Path-B work) plus the
   original full model, for a complete 4-way comparison.
5. Run `delong_compare.py` for pairwise significance across all variants and
   your existing ResNet-1D checkpoint.

## Design choices worth knowing

- **Fixed seed**: all training in this project uses `torch.manual_seed(42)`
  and `np.random.seed(42)` for reproducibility, unlike the original codebase.
  This means results here are *not* expected to numerically match your
  original single-seed runs -- that's expected and correct; the point of
  Path A is a clean, reproducible, from-scratch campaign.
- **Effect sizes**: the interpretability battery reports epsilon-squared
  (rank-based effect size for Kruskal-Wallis) alongside the H-statistic and
  p-value, and bootstrap 95% CIs (2000 resamples) for every Spearman
  correlation, addressing the reviewer's specific ask for effect sizes and
  CIs, not p-values alone.
- **Per-generator consistency**: unlike every prior interpretability script
  in this project (which used only `generators[0]`), `interpretability_battery.py`
  extracts kernel parameters from a configurable sample of generators (default:
  all 32, or a subset via `--n-generators` for speed) and reports whether their
  distributions are consistent with each other -- directly answering "whether
  all 32 generators behave similarly."
