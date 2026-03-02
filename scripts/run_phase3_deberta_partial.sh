#!/bin/bash
# Phase 3: Fine-tune DeBERTa on MRPC, SST-2, CoLA with best HPs across 3 seeds.
# Runs on GPU 1 while AG News HP search finishes on GPU 0.
#
# 3 datasets × 3 seeds = 9 jobs.
# AG News DeBERTa will be run separately once its HP search completes.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=1

SEEDS="42 123 456"
CONFIGS=(
    configs/paper/mrpc_deberta.yaml
    configs/paper/sst2_deberta.yaml
    configs/paper/cola_deberta.yaml
)

TOTAL=$((${#CONFIGS[@]} * 3))
COUNT=0

for config in "${CONFIGS[@]}"; do
    for seed in $SEEDS; do
        COUNT=$((COUNT + 1))
        echo ""
        echo "========================================================================"
        echo "  Job $COUNT/$TOTAL: $(basename $config .yaml) seed=$seed"
        echo "  $(date)"
        echo "========================================================================"
        python scripts/finetune_encoder.py --config "$config" --seed "$seed"
    done
done

echo ""
echo "========================================================================"
echo "  Phase 3 DeBERTa (partial) complete: $TOTAL jobs finished."
echo "  $(date)"
echo "========================================================================"
