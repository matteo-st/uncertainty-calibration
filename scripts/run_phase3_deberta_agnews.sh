#!/bin/bash
# Phase 3: Fine-tune DeBERTa on AG News with best HPs across 3 seeds.
# Config already has correct HPs from HP search (val_acc=0.934).

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=1
SEEDS="42 123 456"
COUNT=0

for seed in $SEEDS; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "========================================================================"
    echo "  Job $COUNT/3: agnews_deberta seed=$seed"
    echo "  $(date)"
    echo "========================================================================"
    python scripts/finetune_encoder.py --config configs/paper/agnews_deberta.yaml --seed "$seed"
done

echo ""
echo "========================================================================"
echo "  AG News DeBERTa Phase 3 DONE. $(date)"
echo "  ALL 36/36 Phase 3 jobs now complete."
echo "========================================================================"
