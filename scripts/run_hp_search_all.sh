#!/bin/bash
# Run HP search for all 12 (model, dataset) pairs on 2 GPUs.
# GPU 0: 6 configs, GPU 1: 6 configs — running sequentially within each GPU.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

NTRIALS=20
OUTDIR=results/hp_search
mkdir -p "$OUTDIR"

# GPU 0 stream: ELECTRA (4) + BERT-mrpc + BERT-cola
CUDA_VISIBLE_DEVICES=0 python scripts/hp_search.py --config configs/paper/mrpc_electra.yaml    --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=0 python scripts/hp_search.py --config configs/paper/sst2_electra.yaml    --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=0 python scripts/hp_search.py --config configs/paper/cola_electra.yaml    --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=0 python scripts/hp_search.py --config configs/paper/agnews_electra.yaml  --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=0 python scripts/hp_search.py --config configs/paper/mrpc_bert.yaml       --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=0 python scripts/hp_search.py --config configs/paper/cola_bert.yaml       --n_trials $NTRIALS --output_dir $OUTDIR &

# GPU 1 stream: DeBERTa (4) + BERT-sst2 + BERT-agnews
CUDA_VISIBLE_DEVICES=1 python scripts/hp_search.py --config configs/paper/mrpc_deberta.yaml    --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=1 python scripts/hp_search.py --config configs/paper/sst2_deberta.yaml    --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=1 python scripts/hp_search.py --config configs/paper/cola_deberta.yaml    --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=1 python scripts/hp_search.py --config configs/paper/agnews_deberta.yaml  --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=1 python scripts/hp_search.py --config configs/paper/sst2_bert.yaml       --n_trials $NTRIALS --output_dir $OUTDIR && \
CUDA_VISIBLE_DEVICES=1 python scripts/hp_search.py --config configs/paper/agnews_bert.yaml     --n_trials $NTRIALS --output_dir $OUTDIR &

wait
echo "All 12 HP searches complete."
