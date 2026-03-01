#!/bin/bash
# Phase 3: Fine-tune ELECTRA + BERT with best HPs across 3 seeds.
# Runs on GPU 1 while DeBERTa HP search finishes on GPU 0.
#
# 8 (model, dataset) pairs × 3 seeds = 24 jobs.
# Each job: fine-tune, predict, extract features, fit MD scorer, cache everything.
# Estimated total: ~3-4 hours on A100.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=1

SEEDS="42 123 456"
CONFIGS=(
    configs/paper/mrpc_electra.yaml
    configs/paper/mrpc_bert.yaml
    configs/paper/sst2_electra.yaml
    configs/paper/sst2_bert.yaml
    configs/paper/cola_electra.yaml
    configs/paper/cola_bert.yaml
    configs/paper/agnews_electra.yaml
    configs/paper/agnews_bert.yaml
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
echo "  Phase 3 complete (ELECTRA + BERT): $TOTAL jobs finished."
echo "  $(date)"
echo "========================================================================"
