#!/usr/bin/env python3
"""
Validate Phase 3 results: check all 36 cache files exist with correct HPs
and reasonable test accuracies.

Compares cached metadata against HP search results to catch any mismatch
(e.g., wrong epochs from CLI override bug).

Usage:
    python scripts/validate_phase3.py
"""

import json
import sys
from pathlib import Path

import yaml

CACHE_DIR = Path("cache/paper")
SEEDS = [42, 123, 456]

# Model-dataset pairs and their HP search result files
PAIRS = {
    ("electra", "mrpc"): "results/hp_search/mrpc_electra_best_hps.yaml",
    ("electra", "sst2"): "results/hp_search/sst2_electra_best_hps.yaml",
    ("electra", "cola"): "results/hp_search/cola_electra_best_hps.yaml",
    ("electra", "agnews"): "results/hp_search/agnews_electra_best_hps.yaml",
    ("bert", "mrpc"): "results/hp_search/mrpc_bert_best_hps.yaml",
    ("bert", "sst2"): "results/hp_search/sst2_bert_best_hps.yaml",
    ("bert", "cola"): "results/hp_search/cola_bert_best_hps.yaml",
    ("bert", "agnews"): "results/hp_search/agnews_bert_best_hps.yaml",
    ("deberta", "mrpc"): "results/hp_search_v2/mrpc_deberta_best_hps.yaml",
    ("deberta", "sst2"): "results/hp_search_v2/sst2_deberta_best_hps.yaml",
    ("deberta", "cola"): "results/hp_search_v2/cola_deberta_best_hps.yaml",
    ("deberta", "agnews"): "results/hp_search_v2/agnews_deberta_best_hps.yaml",
}

# Minimum expected test accuracy per (model, dataset) — sanity floor
MIN_ACC = {
    ("electra", "mrpc"): 0.82,
    ("electra", "sst2"): 0.90,
    ("electra", "cola"): 0.82,
    ("electra", "agnews"): 0.90,
    ("bert", "mrpc"): 0.80,
    ("bert", "sst2"): 0.87,
    ("bert", "cola"): 0.80,
    ("bert", "agnews"): 0.90,
    ("deberta", "mrpc"): 0.85,
    ("deberta", "sst2"): 0.92,
    ("deberta", "cola"): 0.82,
    ("deberta", "agnews"): 0.90,
}


def load_hp_results(path):
    """Load best HPs from HP search YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def check_hp_match(metadata, hp_results, pair_name):
    """Check that metadata HPs match HP search results. Returns list of issues."""
    issues = []
    checks = [
        ("num_train_epochs", "num_train_epochs"),
        ("per_device_train_batch_size", "per_device_train_batch_size"),
    ]
    for meta_key, hp_key in checks:
        meta_val = metadata.get(meta_key)
        hp_val = hp_results.get(hp_key)
        if meta_val != hp_val:
            issues.append(
                f"  MISMATCH {meta_key}: cache={meta_val} vs hp_search={hp_val}"
            )

    # Learning rate: allow small rounding differences (config rounds to 4-5 digits)
    meta_lr = metadata.get("learning_rate", 0)
    hp_lr = hp_results.get("learning_rate", 0)
    if hp_lr > 0 and abs(meta_lr - hp_lr) / hp_lr > 0.01:
        issues.append(
            f"  MISMATCH learning_rate: cache={meta_lr:.6e} vs hp_search={hp_lr:.6e}"
        )

    return issues


def main():
    errors = []
    warnings = []
    results = {}  # (model, dataset) -> [acc1, acc2, acc3]

    print("=" * 70)
    print("Phase 3 Validation")
    print("=" * 70)

    # Check all 36 cache files exist and have correct HPs
    for (model, dataset), hp_path in sorted(PAIRS.items()):
        pair_name = f"{model}_{dataset}"
        results[(model, dataset)] = []

        # Load HP search results
        if not Path(hp_path).exists():
            errors.append(f"MISSING HP file: {hp_path}")
            continue
        hp_results = load_hp_results(hp_path)

        for seed in SEEDS:
            prefix = f"{model}_{dataset}_seed{seed}"

            # Check metadata exists
            meta_path = CACHE_DIR / f"{prefix}_metadata.json"
            if not meta_path.exists():
                errors.append(f"MISSING: {meta_path}")
                continue

            with open(meta_path) as f:
                metadata = json.load(f)

            # Check HPs match
            hp_issues = check_hp_match(metadata, hp_results, pair_name)
            if hp_issues:
                errors.append(f"HP MISMATCH in {prefix}:")
                errors.extend(hp_issues)

            # Check test accuracy
            acc = metadata.get("test_accuracy", 0)
            results[(model, dataset)].append(acc)
            min_acc = MIN_ACC.get((model, dataset), 0.75)
            if acc < min_acc:
                warnings.append(
                    f"LOW ACC: {prefix} test_acc={acc:.4f} (min={min_acc})"
                )

            # Check all cache files exist
            for split in ["train", "cal", "test"]:
                npz_path = CACHE_DIR / f"{prefix}_{split}.npz"
                if not npz_path.exists():
                    errors.append(f"MISSING: {npz_path}")

            scorer_path = CACHE_DIR / f"{prefix}_md_scorer.pkl"
            if not scorer_path.exists():
                errors.append(f"MISSING: {scorer_path}")

    # Print results table
    print("\nTest Accuracy (3 seeds):")
    print(f"{'':>15s}  {'ELECTRA':>22s}  {'BERT':>22s}  {'DeBERTa':>22s}")
    print("-" * 70)
    for dataset in ["mrpc", "sst2", "cola", "agnews"]:
        row = f"{dataset:>15s}"
        for model in ["electra", "bert", "deberta"]:
            accs = results.get((model, dataset), [])
            if len(accs) == 3:
                mean = sum(accs) / 3
                std = (sum((a - mean) ** 2 for a in accs) / 2) ** 0.5
                row += f"  {mean:.3f} ± {std:.3f}"
            elif accs:
                row += f"  {len(accs)}/3 done"
            else:
                row += f"  {'MISSING':>22s}"
        print(row)

    # Print issues
    if warnings:
        print(f"\n{'WARNINGS':}")
        for w in warnings:
            print(f"  ⚠ {w}")

    if errors:
        print(f"\n{'ERRORS':}")
        for e in errors:
            print(f"  ✗ {e}")
        print(f"\nVALIDATION FAILED: {len(errors)} error(s)")
        return 1
    else:
        print(f"\nVALIDATION PASSED: 36/36 jobs OK, all HPs match HP search results.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
