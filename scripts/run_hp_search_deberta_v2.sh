#!/bin/bash
# Re-run DeBERTa HP searches with tighter, model-specific search space
# (He et al. ICLR 2023, Table 11). Results saved to results/hp_search_v2/.
# Runs on GPU 0 while GPU 1 finishes the original recovery batch.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

NTRIALS=20
OUTDIR=results/hp_search_v2
mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES=0 python scripts/hp_search.py --config configs/paper/mrpc_deberta.yaml    --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=0 python scripts/hp_search.py --config configs/paper/sst2_deberta.yaml    --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=0 python scripts/hp_search.py --config configs/paper/cola_deberta.yaml    --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=0 python scripts/hp_search.py --config configs/paper/agnews_deberta.yaml  --n_trials $NTRIALS --output_dir $OUTDIR

echo "DeBERTa v2 HP search complete (all 4 datasets)."
