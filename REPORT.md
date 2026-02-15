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

### Status: IN PROGRESS

Results will be added below as experiments complete.

---

### Results Table

| Shots | Method | Error Rate | Cross-Entropy | Normalized CE |
|-------|--------|------------|---------------|---------------|
| | | | | |

*Table will be populated as experiments run.*

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
