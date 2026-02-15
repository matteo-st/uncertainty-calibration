# Uncertainty Calibration for LLM Text Classification

Implementation of calibration techniques for Large Language Models in text classification tasks, based on the paper:

> **Unsupervised Calibration through Prior Adaptation for Text Classification using Large Language Models**
> Estienne et al., 2023 ([arXiv:2307.06713](https://arxiv.org/abs/2307.06713))

## Overview

This project implements several calibration methods to improve the quality of posterior probabilities from LLMs when used for text classification:

- **No Calibration**: Baseline using raw LLM posteriors
- **Content-Free Calibration** (Zhao et al., 2021): Uses neutral inputs to estimate model bias
- **UCPA-Naive**: Unsupervised prior adaptation using averaged posteriors
- **UCPA**: Iterative unsupervised calibration (equivalent to logistic regression with α=1)
- **SUCPA**: Semi-unsupervised variant using known class priors
- **Affine Calibration**: Supervised method learning α and β parameters

## Project Structure

```
uncertainty-calibration/
├── src/
│   ├── models.py          # LLM loading and inference
│   ├── data.py            # Dataset loading (AGNews, SST-2, TREC, DBPedia)
│   ├── calibration.py     # Calibration methods
│   ├── evaluation.py      # Metrics (accuracy, cross-entropy)
│   └── utils.py           # Utilities
├── scripts/
│   ├── run_inference.py   # Extract probabilities from LLM
│   ├── run_calibration.py # Apply calibration and evaluate
│   └── run_experiment.py  # Full experiment pipeline
├── configs/               # Experiment configurations
├── data/                  # Datasets (gitignored)
├── results/               # Outputs (gitignored)
└── checkpoints/           # Model checkpoints (gitignored)
```

## Setup

### Local Development
```bash
# Clone the repository
git clone https://github.com/matteo-st/uncertainty-calibration.git
cd uncertainty-calibration

# Install dependencies (for local testing/development)
pip install -r requirements.txt
```

### Server Setup
```bash
# Connect to server
ssh lamsade
ssh upnquick

# Navigate to project
cd ~/error_detection/uncertainty_calibration_llm

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Extract Probabilities
```bash
python scripts/run_inference.py \
    --dataset agnews \
    --model gpt2-xl \
    --n_shots 0 \
    --n_train 600 \
    --n_test 1000
```

### 2. Run Calibration
```bash
python scripts/run_calibration.py \
    --probs_dir results/probabilities \
    --output_dir results/calibration
```

### 3. Full Experiment
```bash
python scripts/run_experiment.py \
    --dataset agnews \
    --model gpt2-xl \
    --n_shots_list 0 1 4 8 \
    --n_train_list 10 20 40 80 200 400 600
```

## Datasets

| Dataset | Classes | Task |
|---------|---------|------|
| AGNews | 4 | News classification (World, Sports, Business, Technology) |
| SST-2 | 2 | Sentiment analysis (Positive, Negative) |
| TREC | 6 | Question classification |
| DBPedia | 14 | Ontology classification |

## Reference

```bibtex
@article{estienne2023unsupervised,
  title={Unsupervised Calibration through Prior Adaptation for Text Classification using Large Language Models},
  author={Estienne, Lautaro and Ferrer, Luciana and Vera, Mat{\'\i}as and Piantanida, Pablo},
  journal={arXiv preprint arXiv:2307.06713},
  year={2023}
}
```

## Development Workflow

1. Edit code locally
2. Commit and push to GitHub
3. SSH to server and `git pull`
4. Run experiments on server
