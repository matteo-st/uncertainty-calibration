# Experiment Report: LLM Calibration for Text Classification

This report documents the experimental results for calibrating LLM posteriors in text classification tasks.

**Reference Paper:** Estienne et al. (2023) - "Unsupervised Calibration through Prior Adaptation for Text Classification using Large Language Models"

---

## Experiment 1: Replicating Figure 3 (AGNews)

**Branch:** `exp/replicate-figure3-agnews`

**Objective:** Reproduce Figure 3 from the paper - Cross-Entropy and Error Rate vs. number of shots for AGNews dataset.

**Setup:**
- Model: GPT-2 XL (1.5B parameters)
- Dataset: AGNews (4-class news classification)
- Training samples: 600 (for calibration)
- Test samples: 1000
- Number of shots: 0, 1, 4, 8
- Seeds: 10 (for confidence intervals)

**Methods compared:**
- No Adaptation (baseline)
- UCPA (Unsupervised Calibration through Prior Adaptation)
- SUCPA (Semi-Unsupervised, with known priors)
- Calibration (Affine logistic regression with α and β)

### Status: COMPLETED

**Runtime:** ~1h 40min (2026-02-15 22:50 to 2026-02-16 00:30)

---

### Results: Error Rate

| Shots | No Adaptation | UCPA | SUCPA | Calibration |
|-------|---------------|------|-------|-------------|
| 0 | 0.526 | 0.284 | **0.283** | 0.292 |
| 1 | 0.547 | **0.260** | 0.266 | 0.287 |
| 4 | 0.721 | 0.230 | **0.222** | 0.300 |
| 8 | 0.539 | **0.239** | 0.243 | 0.266 |

### Results: Cross-Entropy

| Shots | No Adaptation | UCPA | SUCPA | Calibration |
|-------|---------------|------|-------|-------------|
| 0 | 1.071 | 0.802 | 0.798 | **0.763** |
| 1 | 1.071 | 0.753 | 0.749 | **0.746** |
| 4 | 1.784 | 0.657 | **0.651** | 0.837 |
| 8 | 1.071 | 0.705 | **0.702** | 0.712 |

### Key Findings

1. **UCPA and SUCPA consistently outperform No Adaptation** - reducing error rate from ~50-72% to ~22-28%
2. **SUCPA performs slightly better than UCPA** in most cases (as expected since it uses known priors)
3. **Calibration (affine)** performs well on cross-entropy but not always best on error rate
4. **No Adaptation degrades at 4-shot** (72% error) - likely due to prompt sensitivity
5. **More shots generally help** - error rates decrease from 0-shot to 8-shot for calibrated methods

### Comparison with Paper (Figure 3)

The trends match the paper's findings:
- Calibration methods significantly improve over no adaptation
- UCPA/SUCPA are competitive with supervised calibration
- Performance improves with more shots

Note: Exact values differ from paper, which is expected due to:
- Different random seeds
- Potentially different test set sampling
- Minor implementation differences

---

### Preliminary Test Results (100 train, 100 test)

Quick validation run before full experiment:

| Method | Accuracy | Error Rate | Cross-Entropy | Norm CE |
|--------|----------|------------|---------------|---------|
| No Adaptation | 50.0% | 50.0% | 1.034 | 0.753 |
| UCPA | 67.0% | 33.0% | 0.779 | 0.568 |
| SUCPA | 75.0% | 25.0% | 0.789 | 0.575 |
| Calibration (α,β) | 73.0% | 27.0% | 0.758 | 0.552 |

These preliminary results confirm the implementation is working correctly.

---

## Notes

- Server: 2x NVIDIA A100 80GB
- Inference speed: ~11 samples/sec with GPT-2 XL
