# NLP Experiment Plan — ICML Paper

## Core Claim

Post-hoc uniform-mass (UM) calibration preserves discrimination (ROCAUC) while providing calibrated error probabilities (ECE) with finite-sample guarantees.

---

## Status Summary

| Phase | Status | Notes |
|-------|--------|-------|
| **Phase 1**: Code changes | **DONE** | All scores, datasets, configs, HP search script |
| **Phase 2**: HP search | **DONE** | All 12 pairs complete (Optuna TPE) |
| **Phase 3**: Fine-tuning | **DONE** | All 36 jobs validated |
| **Phase 4**: Evaluation | **DONE** | `paper_metrics.json` + `calibration_comparison.json` |
| **Phase 5**: Paper writing | **IN PROGRESS** | Experiment section written, LaTeX compilation issue pending |
| **Phase 6**: n\_cal ablation | **PENDING** | Scripts ready, needs server run |

---

## Datasets (4)

| Dataset | K | Full train | Subsample | Available | n\_cal | n\_train | n\_test | Cal % | Source |
|---------|---|-----------|-----------|-----------|--------|---------|--------|-------|--------|
| **SST-2** | 2 | 67,349 | 10% | 6,735 | 1,000 | 5,735 | 872 | 15% | GLUE val |
| **MRPC** | 2 | 3,668 | No | 3,668 | 1,000 | 2,668 | 408 | 27% | GLUE val |
| **CoLA** | 2 | 8,551 | No | 8,551 | 1,000 | 7,551 | 1,043 | 12% | GLUE val |
| **AG News** | 4 | 120,000 | 10% | 12,000 | 1,000 | 11,000 | 7,600 | 8% | Own test |

**Calibration guarantee** (n\_cal=1000, B=19 via Scott's rule, alpha=0.05):
- B = int(2 × 1000^(1/3)) = **19** (floating-point: 1000^(1/3) ≈ 9.9999...)
- n\_per\_bin = floor(1001/19) − 1 = **51**
- epsilon = sqrt(log(2×19/0.05) / (2×51)) ≈ **0.255**

---

## Models (3)

| Model | Params | Pretraining | HuggingFace ID |
|-------|--------|------------|----------------|
| **ELECTRA-base** | 110M | Replaced Token Detection | `google/electra-base-discriminator` |
| **BERT-base** | 110M | Masked LM | `google-bert/bert-base-uncased` |
| **DeBERTa-v3-base** | 138M | Disentangled Attention | `microsoft/deberta-v3-base` |

3 models × 4 datasets × 3 seeds = **36 fine-tuning runs**.

---

## Uncertainty Scores (5)

| # | Score | Type | Formula | Bounded | Status |
|---|-------|------|---------|---------|--------|
| 1 | **SP** (MaxProb) | Softmax / aleatoric | `1 − max_k p_k` | [0, 1−1/K] | Done |
| 2 | **PE** (Predictive Entropy) | Softmax / aleatoric | `−Σ p_k log(p_k)` | [0, log K] | Done |
| 3 | **Doctor** | Softmax / aleatoric | `1 − Σ p_k²` | [0, 1−1/K] | Done |
| 4 | **Energy** | Logit-based | `−log Σ exp(l_k)` | (−∞, 0) | Done |
| 5 | **MD** | Density / epistemic | `min_c (h−μ_c)ᵀ Σ⁻¹ (h−μ_c)` | [0, +∞) | Done |

**Note**: RDE was planned but never implemented — skipped. SP/PE/Doctor are equivalent (identical ROCAUC) for K=2 tasks.

---

## Key Results (Phase 4)

### ROCAUC preservation after UM calibration
| Score | Avg ROCAUC drop | Range |
|-------|----------------|-------|
| SP    | −0.017 | −0.034 to +0.005 |
| Energy | −0.007 | −0.020 to +0.012 |
| MD    | −0.021 | −0.041 to −0.010 |

### ECE after UM calibration
| Score | Avg ECE(UM) | Range |
|-------|-------------|-------|
| SP    | 0.038 | 0.020 – 0.061 |
| Energy | 0.036 | 0.020 – 0.066 |
| MD    | 0.037 | 0.017 – 0.068 |

### MCE ≤ ε guarantee (ε = 0.255)
| Score | UM | Platt | Isotonic |
|-------|----|-------|----------|
| SP    | **94%** | 50% | 92% |
| Energy | **100%** | 86% | 86% |
| MD    | **86%** | 61% | 94% |

### Calibration method comparison (aggregated over 36 experiments)
| Method | ROCAUC | ECE | MCE | MCE≤ε |
|--------|--------|-----|-----|-------|
| Raw SP | .792 | .097 | .347 | 25% |
| UM | .775 | .038 | .140 | **94%** |
| Platt | .792 | .073 | .249 | 50% |
| Isotonic | .788 | .028 | .126 | 92% |

### MD dominance
- Best raw ROCAUC in **9/12** (dataset, model) pairs

---

## Paper Structure (Experiment Section)

### Main paper (Section 4)
- **Setup**: 4 datasets, 3 models, 5 scores, 3 seeds, n_cal=1000
- **Metrics**: ROCAUC, ECE, MCE (formal definitions)
- **Results**: 4 findings paragraphs:
  1. UM preserves discrimination (ROCAUC drop < 0.02)
  2. MD is strongest discriminator (9/12 wins)
  3. Low ECE after calibration (0.02–0.07)
  4. MCE≤ε guarantee holds (86–100%), vs Platt (50–86%)
- **Table 2** (`nlp_main_results.tex`): SP/Energy/MD for all 12 pairs
- **Table 3** (`calibration_comparison.tex`): UM vs Platt vs Isotonic aggregated

### Appendix
- **Dataset stats table** (`dataset_stats.tex`)
- **Uncertainty score descriptions**: SP, PE, Doctor, Energy, MD
- **SP/PE/Doctor equivalence** for K=2 explained
- **Full results table** (`full_results.tex`): all 5 scores × 12 pairs

### Files
- `docs/paper/main.tex` — main paper (experiment section at ~line 538)
- `docs/paper/tables/` — all table .tex files
- `docs/paper/main.bib` — bibliography (includes BERT, CoLA, AG News entries)

---

## Potential Additional Experiments

1. **n\_cal ablation** — Calibrate with n\_cal ∈ {100, 250, 500, 1000}. Shows how guarantee degrades with smaller calibration sets. CPU-only, fast.
2. **B ablation** — Vary number of bins beyond Scott's rule. CPU-only, fast.
3. **Reliability diagrams** — Visualization of calibration for appendix. CPU-only.
4. **AURC metric** — Area Under Risk-Coverage curve. Common in selective prediction literature. Code addition only.

---

## Seeds & Reporting

- Seeds: **42, 123, 456**
- All results reported as **mean ± std** across 3 seeds
- Each seed controls: data shuffling, train/cal split, model initialization, training

---

## Result Files (downloaded locally)

- `results/paper_metrics/paper_metrics.json` — Phase 4 output (36 experiments × 5 scores)
- `results/calibration_comparison/calibration_comparison.json` — Comparison output (36 experiments × 5 scores × 5 methods)
- Scripts: `scripts/run_phase4.sh`, `scripts/compute_paper_metrics.py`, `scripts/compute_calibration_comparison.py`, `scripts/run_calibration_comparison.sh`

---

## Phase 6: Calibration Sample Size Ablation (n\_cal)

**Status: PENDING**

### Goal

Show how calibration performance (MCE, ROCAUC) scales with the size of the calibration set. Demonstrates that UM's theoretical MCE guarantee decreases with more calibration data and that UM provides reliable calibration even with small n\_cal.

### Design

- **Datasets**: SST-2 and AG News only (these were subsampled to 10%, leaving ~60K and ~108K unused training samples as a pool for n\_cal > 1000)
- **Score**: MD only (strongest discriminator)
- **Methods**: Uniform Mass, Platt Scaling, Isotonic Regression
- **n\_cal values**: 50, 100, 200, 500, 1000, 2000, 5000, 10000
- **Random draws**: 20 per n\_cal value (sample from pool without replacement)
- **Aggregation**: Mean ± std over (2 datasets × 3 models × 3 seeds × 20 draws) = 360 points per n\_cal

### Scripts

1. `scripts/cache_unused_samples.py` — **GPU, server**: Run inference on unused training data (18 jobs: 2 datasets × 3 models × 3 seeds)
2. `scripts/run_ncal_ablation.py` — **CPU**: Fit calibration methods at each n\_cal, evaluate on fixed test set
3. `scripts/plot_ncal_ablation.py` — **CPU, local**: Generate 2-panel figure (MCE + ROCAUC vs n\_cal)
4. `scripts/run_ncal_ablation.sh` — Server orchestration (runs steps 1+2)

### Output

- `cache/paper/{variant}_seed{seed}_unused.npz` — Cached model outputs on unused samples
- `results/ncal_ablation/ncal_ablation.json` — Full ablation results
- `docs/paper/figures/ncal_ablation.pdf` — Figure for paper

### Expected runtime

- Step 1 (cache unused): ~30–60 min on A100 (18 jobs, each runs inference + feature extraction)
- Step 2 (ablation): ~5–10 min on CPU (pure numpy, no GPU)
- Step 3 (plot): seconds

### Paper integration

The figure replaces Table 3 (`calibration_comparison.tex`) in Section 5.3.
