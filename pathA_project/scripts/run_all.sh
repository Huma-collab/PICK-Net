#!/bin/bash
# Orchestrates the full Path A campaign, in the recommended order.
# Run from inside pathA_project/scripts/
# Stops on first error (set -e) so you don't waste GPU time on a broken step.
set -e

echo "=============================================="
echo "Path A campaign -- step 1: All-Generator Physics Loss"
echo "(ready to run now)"
echo "=============================================="
python train_variant.py --variant all_gen --subsample-generators 8

echo ""
echo "=============================================="
echo "Path A campaign -- step 2: interpretability on All-Generator variant"
echo "=============================================="
python interpretability_battery.py \
    --checkpoint /data2/huma/picknet/pathA_outputs/all_gen/checkpoint.pt \
    --variant all_gen --label "All-Generator Physics Loss" --n-generators 32

echo ""
echo "=============================================="
echo "STOP HERE if unbounded_variant.py is not yet finalized."
echo "Send Claude the KernelParamGenerator source first (see README.md)."
echo "Once confirmed, uncomment the two blocks below and re-run this script."
echo "=============================================="

# echo "Path A campaign -- step 3: Unbounded Gabor variant"
# python train_variant.py --variant unbounded
#
# echo "Path A campaign -- step 4: interpretability on Unbounded variant"
# python interpretability_battery.py \
#     --checkpoint /data2/huma/picknet/pathA_outputs/unbounded/checkpoint.pt \
#     --variant unbounded --label "Unbounded Gabor" --n-generators 32

echo ""
echo "=============================================="
echo "Path A campaign -- step 5: pairwise DeLong comparisons"
echo "(runs on whichever variants have finished so far)"
echo "=============================================="
python delong_compare.py

echo ""
echo "Campaign step(s) complete. Check pathA_outputs/ for all results."
