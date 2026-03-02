#!/bin/bash
# Phase 4: Compute paper metrics (ROCAUC, ECE, MCE) for all 36 experiments.
#
# 3 models × 4 datasets × 3 seeds = 36 cache prefixes.
# Scores: SP, PE, Doctor, Energy, MD (RDE not implemented, will be skipped).
# For each score: metrics before and after Uniform Mass calibration.
#
# CPU-only — no GPU needed.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODELS="electra bert deberta"
DATASETS="mrpc sst2 cola agnews"
SEEDS="42 123 456"
CACHE_DIR="cache/paper"
OUTPUT_DIR="results/paper_metrics"

# Build list of all 36 cache prefixes
PREFIXES=()
for model in $MODELS; do
    for dataset in $DATASETS; do
        for seed in $SEEDS; do
            PREFIXES+=("${CACHE_DIR}/${model}_${dataset}_seed${seed}")
        done
    done
done

echo "========================================================================"
echo "  Phase 4: Compute Paper Metrics"
echo "  ${#PREFIXES[@]} experiments, 5 scores each"
echo "  Output: ${OUTPUT_DIR}/"
echo "  $(date)"
echo "========================================================================"

python scripts/compute_paper_metrics.py \
    --cache_prefixes "${PREFIXES[@]}" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "  Phase 4 DONE. $(date)"
echo "  Results: ${OUTPUT_DIR}/paper_metrics.json"
echo "========================================================================"
