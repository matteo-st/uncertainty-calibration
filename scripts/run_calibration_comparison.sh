#!/bin/bash
# Run calibration method comparison on all 36 cached experiments.
# CPU-only — no GPU needed. Should take < 1 minute.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODELS="electra bert deberta"
DATASETS="mrpc sst2 cola agnews"
SEEDS="42 123 456"
CACHE_DIR="cache/paper"
OUTPUT_DIR="results/calibration_comparison"

PREFIXES=()
for model in $MODELS; do
    for dataset in $DATASETS; do
        for seed in $SEEDS; do
            PREFIXES+=("${CACHE_DIR}/${model}_${dataset}_seed${seed}")
        done
    done
done

echo "========================================================================"
echo "  Calibration Method Comparison"
echo "  ${#PREFIXES[@]} experiments, 5 scores, 4 methods each"
echo "  Output: ${OUTPUT_DIR}/"
echo "  $(date)"
echo "========================================================================"

python scripts/compute_calibration_comparison.py \
    --cache_prefixes "${PREFIXES[@]}" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "  Comparison DONE. $(date)"
echo "========================================================================"
