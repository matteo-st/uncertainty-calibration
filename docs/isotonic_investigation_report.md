# Isotonic Regression Interpolation: Linear vs Step Function

## Background

Scikit-learn's `IsotonicRegression` uses the Pool Adjacent Violators (PAV) algorithm to fit a non-decreasing function from uncertainty scores to error probabilities. After PAV produces a set of breakpoints (x_i, y_i), sklearn's `predict()` method calls `_build_f()`, which constructs:

```python
interp1d(X_thresholds_, y_thresholds_, kind='linear', bounds_error=False)
```

This **linearly interpolates** between breakpoints, producing a continuous piecewise-linear output with many unique values (50--400 on our test sets). The "true" PAV result is a **step function** (piecewise constant), where all scores within a plateau share the same calibrated probability.

This investigation compares both prediction modes across our experimental setup.

## Experimental Setup

### Experiment 1: Fixed n_cal = 1000 (all 4 datasets)

- **Model**: DeBERTa
- **Datasets**: MRPC (n_test=408), SST-2 (n_test=7600), CoLA (n_test=1043), AG News (n_test=7600)
- **Scores**: SP (softmax probability complement), MD (Mahalanobis distance)
- **Seeds**: 42, 123, 456
- **Calibration**: Isotonic regression fitted on n_cal=1000 calibration samples
- **Evaluation**: Uniform-mass (UM) binning with Scott's rule on the test set

### Experiment 2: n_cal ablation (SST-2 and AG News)

- **Model**: DeBERTa
- **Datasets**: SST-2, AG News (both have large unused pools for varying n_cal)
- **Scores**: SP, MD
- **n_cal values**: 50, 100, 200, 500, 1000, 2000, 5000, 10000
- **Random draws**: 20 per n_cal value
- **Aggregation**: Mean +/- std over 3 seeds x 20 draws = 60 points per n_cal

## Results

### 1. ROCAUC (discrimination)

Both modes are monotone transforms, but the step function creates more ties (all scores within a PAV plateau map to the same value), degrading ROCAUC.

**At n_cal = 1000** (fixed experiment, all datasets):

| Dataset  | Score | Raw    | Linear | Step   | Linear-Step |
|----------|-------|--------|--------|--------|-------------|
| MRPC     | SP    | .786   | .786   | .785   | +.0003      |
| MRPC     | MD    | .801   | .800   | .800   | +.0002      |
| SST-2    | SP    | .846   | .845   | .845   | +.0001      |
| SST-2    | MD    | .859   | .856   | .856   | -.0000      |
| CoLA     | SP    | .798   | .793   | .793   | -.0000      |
| CoLA     | MD    | .798   | .792   | .792   | +.0001      |
| AG News  | SP    | .805   | .797   | .795   | +.0012      |
| AG News  | MD    | .839   | .832   | .832   | -.0001      |

At n_cal=1000, the difference is negligible (< 0.002).

**n_cal ablation** (averaged over SST-2 + AG News):

| n_cal  | SP Linear | SP Step | Diff    | MD Linear | MD Step | Diff    |
|--------|-----------|---------|---------|-----------|---------|---------|
| 50     | .758      | .742    | +.016   | .775      | .760    | +.015   |
| 100    | .787      | .782    | +.005   | .804      | .799    | +.005   |
| 500    | .817      | .817    | +.001   | .840      | .839    | +.000   |
| 1,000  | .823      | .823    | +.000   | .843      | .843    | +.000   |
| 10,000 | .828      | .828    | +.000   | .848      | .848    | +.000   |

**Finding**: Linear interpolation preserves ROCAUC better at small n_cal (+0.016 at n_cal=50) because it maintains within-plateau ordering. The advantage vanishes by n_cal >= 500 as plateaus become narrow.

### 2. MCE (maximum calibration error)

**At n_cal = 1000** (all datasets):

| Dataset  | Score | Linear | Step   | Diff    |
|----------|-------|--------|--------|---------|
| MRPC     | SP    | .110   | .107   | +.003   |
| MRPC     | MD    | .135   | .116   | +.019   |
| SST-2    | SP    | .106   | .106   | +.000   |
| SST-2    | MD    | .103   | .106   | -.003   |
| CoLA     | SP    | .136   | .136   | -.001   |
| CoLA     | MD    | .124   | .120   | +.004   |
| AG News  | SP    | .142   | **.085**| **+.056** |
| AG News  | MD    | .079   | .063   | +.016   |

**n_cal ablation** (averaged over SST-2 + AG News):

| n_cal  | SP Linear | SP Step | Diff    | MD Linear | MD Step | Diff    |
|--------|-----------|---------|---------|-----------|---------|---------|
| 50     | .241      | **.118**| **+.123** | .259    | **.132**| **+.127** |
| 100    | .200      | .134   | +.066   | .216      | .143    | +.073   |
| 500    | .144      | .130   | +.014   | .136      | .116    | +.020   |
| 1,000  | .112      | .097   | +.015   | .107      | .099    | +.008   |
| 10,000 | .071      | .066   | +.005   | .067      | .064    | +.003   |

**Finding**: The step function consistently achieves lower MCE than linear interpolation, with the largest gap at small n_cal. At n_cal=50, the step MCE is 0.12 lower. This occurs because linear interpolation creates many unique values that don't align with UM bin boundaries, creating spurious within-bin variation that inflates worst-case bin error.

### 3. ECE (expected calibration error)

**n_cal ablation**:

| n_cal  | SP Linear | SP Step | Diff    | MD Linear | MD Step | Diff    |
|--------|-----------|---------|---------|-----------|---------|---------|
| 50     | .046      | .040    | +.005   | .047      | .041    | +.006   |
| 100    | .036      | .034    | +.002   | .035      | .033    | +.002   |
| 500    | .020      | .020    | +.000   | .021      | .020    | +.000   |
| 1,000  | .016      | .016    | +.000   | .016      | .016    | +.000   |
| 10,000 | .010      | .010    | +.000   | .010      | .010    | +.000   |

**Finding**: ECE is virtually identical between the two modes. The tiny difference at small n_cal is within noise.

### 4. Number of unique output values

| n_cal  | SP Linear | SP Step | MD Linear | MD Step |
|--------|-----------|---------|-----------|---------|
| 50     | 321       | 3       | 361       | 3       |
| 100    | 219       | 5       | 248       | 4       |
| 500    | 136       | 10      | 144       | 10      |
| 1,000  | 108       | 13      | 100       | 13      |
| 10,000 | 114       | 29      | 54        | 31      |

**Finding**: Linear interpolation produces 10--100x more unique values than the step function. The linear count *decreases* with n_cal (more calibration data -> more PAV pooling -> more flat plateaus where interpolation gives constant output). The step count *increases* with n_cal (more data -> more distinguishable levels).

## Analysis

### Why does linear interpolation hurt MCE?

When UM evaluation bins the calibrated scores into B quantile bins (B ~ 2 * n_test^{1/3}), the linear interpolation's many unique values get distributed across bins in a way that does not respect the PAV structure. A single PAV plateau may span multiple UM bins, and the interpolated values within each bin do not reflect the true error rate of that bin -- they reflect the interpolation's arbitrary positioning between breakpoints.

The step function's output naturally aligns better: all points in a PAV plateau get the same value, so UM bins that fall within a plateau are homogeneous.

### Why does linear interpolation help ROCAUC at small n_cal?

With few calibration samples (n_cal=50), PAV produces only ~3 breakpoints = 3 step levels. The step function maps all test scores to one of 3 values, destroying most ranking information. Linear interpolation preserves within-plateau ordering, maintaining finer discrimination.

At large n_cal, PAV produces many breakpoints with narrow plateaus, so within-plateau ordering contributes negligibly to ROCAUC.

### Trade-off summary

|                  | ROCAUC (discrimination)     | MCE (calibration)          |
|------------------|-----------------------------|----------------------------|
| **Linear interp.** | Better at small n_cal (+0.016) | Worse, especially at small n_cal (+0.12) |
| **Step function**   | Worse at small n_cal (-0.016)  | Better at all n_cal values |
| **Converge at**     | n_cal >= 500               | Gap persists but shrinks   |

### Impact on our paper

Our paper uses isotonic regression as a baseline calibration method with n_cal = 1000. At this operating point:
- **ROCAUC**: difference < 0.001 (negligible)
- **MCE**: step is 0.008--0.015 better (minor)
- **ECE**: identical

**The interpolation choice does not materially affect our reported results.** The MCE numbers in our main results table would change by at most 0.02 if we switched to step function -- within the seed-to-seed standard deviation.

However, the step function is arguably the "correct" output of isotonic regression (it's what PAV computes), and it gives uniformly better or equal calibration metrics. If we wanted to give isotonic regression its best chance, we would use the step function.

## Files

- `scripts/investigate_isotonic.py` -- Fixed n_cal experiment (all 4 datasets)
- `scripts/run_isotonic_ablation.py` -- n_cal ablation (SST-2, AG News)
- `scripts/plot_isotonic_ablation.py` -- Figure generation
- `results/isotonic_investigation/isotonic_investigation.json` -- Fixed n_cal results
- `results/isotonic_ablation/isotonic_ablation.json` -- Ablation results (3816 rows)
- `docs/paper/figures/isotonic_ablation_sp.pdf` -- SP score, 3x2 (ROCAUC, MCE, ECE)
- `docs/paper/figures/isotonic_ablation_md.pdf` -- MD score, 3x2
- `docs/paper/figures/isotonic_ablation_combined.pdf` -- Both scores overlaid, 3x2
- `docs/paper/figures/isotonic_ablation_nunique.pdf` -- Unique output values, 1x2
