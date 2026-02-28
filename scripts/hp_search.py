#!/usr/bin/env python3
"""
Bayesian hyperparameter search for encoder fine-tuning using Optuna TPE.

Tunes (learning_rate, num_train_epochs, batch_size, weight_decay) per
(model, dataset) pair.  The calibration set is already carved out by
load_encoder_dataset(); this script further splits the remaining training
data into 80 % hp_train / 20 % hp_val for the search objective.

Objective: maximise validation accuracy (following Vazhentsev 2022).

Usage:
    # Single pair, 20 trials
    python scripts/hp_search.py --config configs/paper/mrpc_electra.yaml --n_trials 20

    # Quick smoke-test (1 trial)
    python scripts/hp_search.py --config configs/paper/mrpc_electra.yaml --n_trials 1

    # Override output directory
    python scripts/hp_search.py --config configs/paper/mrpc_electra.yaml \
        --n_trials 20 --output_dir results/hp_search
"""

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import optuna
import torch
import yaml
from optuna.samplers import TPESampler
from transformers import EarlyStoppingCallback

from src.encoder_data import load_encoder_dataset
from src.encoder_models import EncoderClassifier
from src.utils import load_config, set_seed, setup_logging

MODEL_SHORT_NAMES = {
    "google/electra-base-discriminator": "electra",
    "microsoft/deberta-v3-base": "deberta",
    "google-bert/bert-base-uncased": "bert",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bayesian HP search for encoder fine-tuning (Optuna TPE)"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fraction of training data held out for HP validation")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early-stopping patience (epochs)")
    parser.add_argument("--output_dir", type=str, default="results/hp_search",
                        help="Directory for saving best HPs and study results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data splitting and Optuna sampler")
    return parser.parse_args()


def load_config_and_data(config_path: str, seed: int):
    """Load YAML config and dataset (with calibration set carved out)."""
    config = load_config(config_path)

    ds_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})

    dataset_name = ds_cfg["name"]
    model_name = model_cfg["name"]
    n_cal = ds_cfg.get("n_cal", 600)
    n_train = ds_cfg.get("n_train", None)
    max_length = ds_cfg.get("max_length", 128)

    data = load_encoder_dataset(
        dataset_name=dataset_name,
        model_name=model_name,
        n_train=n_train,
        n_cal=n_cal,
        max_length=max_length,
        seed=seed,
    )
    return config, data


def split_hp_train_val(train_dataset, val_ratio: float, seed: int):
    """Split train_dataset into hp_train / hp_val using seeded shuffle."""
    n = len(train_dataset)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n).tolist()

    n_val = int(n * val_ratio)
    n_train = n - n_val

    hp_train = train_dataset.select(indices[:n_train])
    hp_val = train_dataset.select(indices[n_train:])
    return hp_train, hp_val


def make_objective(config, hp_train, hp_val, patience, seed, output_dir, logger):
    """Return the Optuna objective function (closure over data)."""
    model_cfg = config.get("model", {})
    model_name = model_cfg["name"]
    num_labels = config.get("_num_labels", 2)
    use_spectral_norm = model_cfg.get("use_spectral_norm", False)

    def objective(trial: optuna.Trial) -> float:
        # ---- Suggest hyperparameters ----
        lr = trial.suggest_float("learning_rate", 5e-6, 1e-4, log=True)
        epochs = trial.suggest_int("num_train_epochs", 3, 15)
        batch_size = trial.suggest_categorical(
            "per_device_train_batch_size", [4, 16, 32, 64]
        )
        wd = trial.suggest_float("weight_decay", 0.0, 0.1)

        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {trial.number}: lr={lr:.2e}, epochs={epochs}, "
                     f"bs={batch_size}, wd={wd:.4f}")
        logger.info(f"{'='*60}")

        # ---- Create fresh model ----
        classifier = EncoderClassifier(
            model_name=model_name,
            num_labels=num_labels,
            use_spectral_norm=use_spectral_norm,
        )

        trial_dir = str(Path(output_dir) / "trials" / f"trial_{trial.number}")

        # ---- Fine-tune with early stopping ----
        classifier.finetune(
            train_dataset=hp_train,
            val_dataset=hp_val,
            output_dir=trial_dir,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=wd,
            warmup_ratio=0.1,
            seed=seed,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )

        # ---- Evaluate on hp_val ----
        probs, preds, labels, _ = classifier.predict(hp_val)
        val_accuracy = float((preds == labels).mean())
        logger.info(f"Trial {trial.number} val_accuracy={val_accuracy:.4f}")

        # ---- GPU cleanup ----
        del classifier
        gc.collect()
        torch.cuda.empty_cache()

        return val_accuracy

    return objective


def main():
    args = parse_args()
    logger = setup_logging()
    set_seed(args.seed)

    # ---- Load data ----
    logger.info("Loading config and dataset...")
    config, data = load_config_and_data(args.config, args.seed)

    ds_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    model_short = MODEL_SHORT_NAMES.get(model_name, model_name.split("/")[-1])
    config["_num_labels"] = data["num_labels"]

    logger.info(f"Dataset: {ds_name}  Model: {model_name}")
    logger.info(f"Train samples (after cal carve-out): {data['n_train']}")

    # ---- Split train -> hp_train / hp_val ----
    hp_train, hp_val = split_hp_train_val(
        data["train_dataset"], args.val_ratio, args.seed
    )
    logger.info(f"HP split: hp_train={len(hp_train)}, hp_val={len(hp_val)}")

    # ---- Create Optuna study ----
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        study_name=f"{ds_name}_{model_short}",
    )

    objective = make_objective(
        config, hp_train, hp_val, args.patience, args.seed, args.output_dir, logger
    )

    logger.info(f"\nStarting Optuna search: {args.n_trials} trials, "
                f"patience={args.patience}")
    study.optimize(objective, n_trials=args.n_trials)

    # ---- Results ----
    best = study.best_trial
    logger.info(f"\n{'='*60}")
    logger.info("BEST TRIAL")
    logger.info(f"{'='*60}")
    logger.info(f"  Trial:          {best.number}")
    logger.info(f"  Val accuracy:   {best.value:.4f}")
    for k, v in best.params.items():
        logger.info(f"  {k}: {v}")

    # ---- Save best HPs as YAML ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_hps = {
        "best_trial": best.number,
        "best_val_accuracy": round(best.value, 4),
        **{k: v for k, v in best.params.items()},
    }
    hp_path = output_dir / f"{ds_name}_{model_short}_best_hps.yaml"
    with open(hp_path, "w") as f:
        yaml.dump(best_hps, f, default_flow_style=False, sort_keys=False)
    logger.info(f"\nBest HPs saved to {hp_path}")

    # ---- Save full study as JSON ----
    trials_data = []
    for t in study.trials:
        trials_data.append({
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        })
    study_path = output_dir / f"{ds_name}_{model_short}_study.json"
    with open(study_path, "w") as f:
        json.dump(trials_data, f, indent=2)
    logger.info(f"Full study saved to {study_path}")


if __name__ == "__main__":
    main()
