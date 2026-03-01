#!/bin/bash
# Recovery script: re-run the 6 HP searches that failed on GPU 1
# due to DeBERTa-v3 tokenizer crash (fixed in 7c039ae).
# GPU 0 is still running mrpc_bert + cola_bert — do NOT touch it.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

NTRIALS=20
OUTDIR=results/hp_search

# DeBERTa (4 configs) — these crashed due to tiktoken/SentencePiece issue
CUDA_VISIBLE_DEVICES=1 python scripts/hp_search.py --config configs/paper/mrpc_deberta.yaml    --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=1 python scripts/hp_search.py --config configs/paper/sst2_deberta.yaml    --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=1 python scripts/hp_search.py --config configs/paper/cola_deberta.yaml    --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=1 python scripts/hp_search.py --config configs/paper/agnews_deberta.yaml  --n_trials $NTRIALS --output_dir $OUTDIR && \

# BERT configs that were queued behind DeBERTa on GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/hp_search.py --config configs/paper/sst2_bert.yaml       --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=1 python scripts/hp_search.py --config configs/paper/agnews_bert.yaml     --n_trials $NTRIALS --output_dir $OUTDIR

echo "GPU 1 recovery: all 6 HP searches complete."
