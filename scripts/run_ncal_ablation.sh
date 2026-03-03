#!/bin/bash
# Run the n_cal ablation experiment on the server.
#
# Step 1 (GPU): Cache model outputs on unused training samples
# Step 2 (CPU): Run ablation across n_cal values
#
# Usage:
#   bash scripts/run_ncal_ablation.sh

set -euo pipefail

echo "============================================"
echo "Phase 6: n_cal Ablation Experiment"
echo "============================================"

# Step 1: Cache unused samples (requires GPU for model inference)
echo ""
echo "Step 1: Caching unused training samples (GPU)..."
echo "--------------------------------------------"
python scripts/cache_unused_samples.py \
    --datasets sst2 agnews \
    --cache_dir cache/paper \
    --seeds 42 123 456

# Step 2: Run n_cal ablation (CPU only)
echo ""
echo "Step 2: Running n_cal ablation (CPU)..."
echo "--------------------------------------------"
python scripts/run_ncal_ablation.py \
    --cache_dir cache/paper \
    --output_dir results/ncal_ablation \
    --n_draws 20

echo ""
echo "============================================"
echo "Done! Results saved to results/ncal_ablation/"
echo ""
echo "Next: download results and run plotting locally:"
echo "  rsync -avz upnquick:~/error_detection/uncertainty_calibration_llm/results/ncal_ablation/ results/ncal_ablation/"
echo "  python scripts/plot_ncal_ablation.py"
echo "============================================"
