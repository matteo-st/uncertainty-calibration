#!/bin/bash
# Auto-launch: waits for AG News DeBERTa HP search + partial Phase 3 to finish,
# then runs the last 3 fine-tuning jobs (agnews_deberta × 3 seeds).
# Reads best HPs directly from the HP search output file.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

RESULT_FILE="results/hp_search_v2/agnews_deberta_best_hps.yaml"
PARTIAL_LOG="logs/phase3_deberta_partial.log"

echo "=== Waiting for AG News DeBERTa HP search to complete... ==="
while [ ! -f "$RESULT_FILE" ]; do
    sleep 60
done
echo "HP search complete at $(date)"
cat "$RESULT_FILE"

echo ""
echo "=== Waiting for partial DeBERTa Phase 3 to finish (GPU 1 busy)... ==="
while ! grep -q "Phase 3 DeBERTa (partial) complete" "$PARTIAL_LOG" 2>/dev/null; do
    sleep 30
done
echo "Partial Phase 3 done at $(date)"

# Read best HPs from the search results
LR=$(grep learning_rate "$RESULT_FILE" | awk '{print $2}')
EPOCHS=$(grep num_train_epochs "$RESULT_FILE" | awk '{print $2}')
BS=$(grep per_device_train_batch_size "$RESULT_FILE" | awk '{print $2}')
WD=$(grep weight_decay "$RESULT_FILE" | awk '{print $2}')

echo ""
echo "Best HPs: lr=$LR, epochs=$EPOCHS, bs=$BS, wd=$WD"

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
    python scripts/finetune_encoder.py \
        --config configs/paper/agnews_deberta.yaml \
        --seed "$seed" \
        --learning_rate "$LR" \
        --num_train_epochs "$EPOCHS" \
        --per_device_train_batch_size "$BS" \
        --weight_decay "$WD"
done

echo ""
echo "========================================================================"
echo "  ALL Phase 3 COMPLETE (36/36 jobs). $(date)"
echo "========================================================================"
