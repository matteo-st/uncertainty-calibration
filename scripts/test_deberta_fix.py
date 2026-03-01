#!/usr/bin/env python3
"""
Quick diagnostic: fine-tune DeBERTa-v3-base on SST-2 with paper HPs
to verify that adam_beta2=0.98 fix works.

Expected: ~93-94% val accuracy (vs ~85% without the fix).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.encoder_data import load_encoder_dataset
from src.encoder_models import EncoderClassifier
from src.utils import set_seed, setup_logging

logger = setup_logging()
set_seed(42)

# Load SST-2 with DeBERTa tokenizer
logger.info("Loading SST-2 dataset...")
data = load_encoder_dataset("sst2", "microsoft/deberta-v3-base", n_cal=1000, seed=42)

# HP split (same as hp_search.py)
n = len(data["train_dataset"])
rng = np.random.RandomState(42)
indices = rng.permutation(n).tolist()
n_val = int(n * 0.2)
hp_train = data["train_dataset"].select(indices[: n - n_val])
hp_val = data["train_dataset"].select(indices[n - n_val :])
logger.info(f"hp_train={len(hp_train)}, hp_val={len(hp_val)}")

# Fine-tune with DeBERTa paper HPs
classifier = EncoderClassifier(
    model_name="microsoft/deberta-v3-base",
    num_labels=data["num_labels"],
)

classifier.finetune(
    train_dataset=hp_train,
    val_dataset=hp_val,
    output_dir="/tmp/deberta_diag",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    seed=42,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Evaluate
probs, preds, labels, _ = classifier.predict(hp_val)
val_accuracy = float((preds == labels).mean())
logger.info(f"\n{'='*60}")
logger.info(f"DIAGNOSTIC RESULT: val_accuracy={val_accuracy:.4f}")
logger.info(f"Expected: ~0.93-0.94 (with fix)")
logger.info(f"Before fix: ~0.85")
logger.info(f"{'='*60}")
