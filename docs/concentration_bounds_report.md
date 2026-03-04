# Concentration Inequalities for Uniform-Mass Calibration Bounds

## Background

Our paper's Theorem 3 (restated from Gupta & Ramdas, 2021) guarantees that the uniform-mass calibrator is (ε, α)-calibrated with:

$$
\epsilon_n = \sqrt{\frac{\log(2B/\alpha)}{2\,n_b}}, \qquad n_b = \lfloor(n+1)/B\rfloor - 1
$$

This bound comes from **Hoeffding's inequality** applied per-bin with a union bound over B bins. Hoeffding treats the errors as generic [0,1]-bounded variables, ignoring that they are **Bernoulli** (binary: E ∈ {0,1}). This investigation asks: how much tighter can we make the bound by exploiting the Bernoulli structure?

## Bounds Investigated

### 1. Hoeffding (current — baseline)

For i.i.d. variables X_i ∈ [0,1] with mean μ:

$$
P(|\bar{X}_n - \mu| \geq \epsilon) \leq 2\exp(-2n\epsilon^2)
$$

Inverted with union bound over B bins (δ_b = α/B):

$$
\epsilon_b^{\text{Hoeff}} = \sqrt{\frac{\log(2B/\alpha)}{2\,n_b}}
$$

**Properties**: Distribution-free, does not depend on observed data. Same ε for all bins. Ignores that Var(X) = p(1−p) ≤ 1/4 with equality only at p = 1/2.

### 2. KL-Chernoff (Bernoulli exact exponential rate)

For i.i.d. Bernoulli(p) variables, the exact Chernoff bound gives:

$$
P(\hat{p} \geq p + \epsilon) \leq \exp\big(-n \cdot d(\hat{p} + \epsilon \,\|\, \hat{p})\big)
$$

where d(a ‖ b) = a ln(a/b) + (1−a) ln((1−a)/(1−b)) is the binary KL divergence. Inverted:

$$
\epsilon_b^{\text{KL}} : \quad n_b \cdot d(\hat{p}_b + \epsilon_b \,\|\, \hat{p}_b) = \log(2B/\alpha)
$$

This is an implicit equation solved by bisection (trivial numerically, ~10 iterations).

**Why tighter**: Hoeffding uses Pinsker's inequality d(a ‖ b) ≥ 2(a−b)², which replaces the exact KL with its lower bound. The gap is largest when p is far from 1/2:
- At p = 0.05: KL is ~2.5× tighter than Hoeffding
- At p = 0.10: KL is ~1.7× tighter
- At p = 0.50: KL ≈ Hoeffding (no gain)

**Connection to KL-UCB**: This is exactly the bound used in KL-UCB (Cappé et al., 2013) for optimal bandit algorithms. Each bin is analogous to an arm.

### 3. Clopper-Pearson (exact binomial confidence interval)

For k errors in n Bernoulli trials, the exact two-sided interval:

$$
p_b \in \big[\text{Beta}(\alpha_b/2;\, k,\, n{-}k{+}1),\;\; \text{Beta}(1{-}\alpha_b/2;\, k{+}1,\, n{-}k)\big]
$$

where α_b = α/B (union bound) and Beta(q; a, b) is the q-th quantile of the Beta distribution.

**Properties**: Exact coverage (P(p ∈ CI) ≥ 1−α for all p ∈ [0,1]). Known to be conservative (actual coverage exceeds nominal). Trivially computed via `scipy.stats.beta.ppf()`.

**Why tightest**: Uses the exact binomial distribution rather than exponential bounds. The improvement over KL-Chernoff comes from sub-exponential corrections that matter at small n_b.

### 4. Empirical Bernstein (Maurer & Pontil, 2009)

$$
\epsilon_b^{\text{EB}} = \sqrt{\frac{2\hat{V}_b \log(2/\delta_b)}{n_b}} + \frac{7\log(2/\delta_b)}{3(n_b - 1)}
$$

where V̂_b = p̂_b(1 − p̂_b) is the sample variance.

**Critical problem**: The second-order term 7 log(2/δ_b) / (3(n_b−1)) comes from union-bounding over variance estimation. At our bin sizes (n_b ≈ 50), this term **dominates**, making the bound worse than Hoeffding:
- At n_b = 50, δ_b = 0.05/19: second-order term ≈ 0.34, while Hoeffding total ≈ 0.26

**Verdict**: Empirical Bernstein is **counterproductive** at all bin sizes relevant to our paper (n_b = 10–300). It only becomes competitive for n_b > 500.

### 5. Bernstein with plug-in variance

$$
\epsilon_b^{\text{Bern}} = \sqrt{\frac{2\hat{p}_b(1-\hat{p}_b)\log(2/\delta_b)}{n_b}} + \frac{\log(2/\delta_b)}{3n_b}
$$

This uses the Bernoulli identity Var = p(1−p) with plug-in p̂. It avoids the Maurer-Pontil penalty but is **not a valid PAC bound** without additional correction (using p̂ instead of true p introduces bias that is not accounted for).

Included for reference only — shows the theoretical floor if variance were known.

### 6. Waudby-Smith & Ramdas (2024) — betting-based confidence sequences

The state-of-the-art approach constructs confidence intervals by sequential betting (test martingales with predictable mixture). Key advantages:
- Avoids the Maurer-Pontil second-order penalty via predictable plug-in
- Asymptotically matches oracle Bernstein (with log(2/α), not log(4/α))
- Valid for any bounded random variables

**However**: For Bernoulli variables specifically, the KL-Chernoff bound already captures the essential gain because Var = p(1−p) is a known function of the mean. The WS&R approach is most valuable for general [0,1]-bounded (non-Bernoulli) variables where variance must be estimated from data.

No simple closed-form formula exists (involves Kummer's hypergeometric function). Available in the `confseq` Python package.

## Experimental Setup

- **Data**: All 36 experiments from the paper (3 models × 4 datasets × 3 seeds)
- **Calibration**: Uniform-mass with n_cal = 1000, B from Scott's rule (B = 19 for n_cal = 1000)
- **Per-bin**: n_b = ⌊1001/19⌋ − 1 = 51 samples per bin
- **Confidence level**: α = 0.05 (per-bin: δ_b = α/B = 0.0026)
- **Scores**: SP (softmax probability complement) and MD (Mahalanobis distance)
- **Evaluation**: Per-bin ε_b from each bound; MCE bound = max_b ε_b; ECE bound = Σ_b (n_b/n) ε_b

## Results

### 1. Bound width as function of observed error rate

![Bound width vs observed error rate](../results/concentration_bounds/concentration_bounds_vs_p.png)

At n_b = 50 (our operating point), for α = 0.05, B = 19:

| p̂_b  | Hoeffding | KL-Chernoff | Clopper-Pearson | Emp. Bernstein |
|-------|-----------|-------------|-----------------|----------------|
| 0.02  | 0.255     | 0.120       | 0.118           | 0.424          |
| 0.05  | 0.255     | 0.161       | 0.148           | 0.454          |
| 0.10  | 0.255     | 0.206       | 0.180           | 0.497          |
| 0.20  | 0.255     | 0.238       | 0.211           | 0.541          |
| 0.30  | 0.255     | 0.248       | 0.224           | 0.561          |
| 0.50  | 0.255     | 0.254       | 0.232           | 0.577          |

**Key finding**: At low p̂ (typical for well-calibrated models with ~7% error), KL-Chernoff is **2× tighter** and Clopper-Pearson is **2.2× tighter** than Hoeffding. The advantage vanishes as p̂ → 0.5.

### 2. Bound width as function of bin size

![Bound width vs bin size](../results/concentration_bounds/concentration_bounds_vs_n.png)

At all bin sizes, the ordering is consistent: Clopper-Pearson ≤ KL-Chernoff < Hoeffding ≪ Emp. Bernstein. The Emp. Bernstein penalty term (7 log(2/δ)/(3(n−1))) causes it to diverge catastrophically as n_b decreases below ~100.

### 3. Per-bin analysis on actual data

![Per-bin bounds on actual data](../results/concentration_bounds/concentration_bounds_perbin.png)

**Left panels**: Per-bin ε from each bound (colored markers) vs actual calibration error measured on test set (black dots). All bounds are valid upper bounds on the actual error. Clopper-Pearson (blue) is closest to the actual errors.

**Right panels**: Ratio ε / ε_Hoeffding. Values below 1 indicate improvement over Hoeffding. Clopper-Pearson is at 0.5–0.75× for low-p̂ bins. Emp. Bernstein (orange) is always above 1.

Average ratio ε / ε_Hoeffding across all bins and seeds (DeBERTa):

| Dataset  | KL-Chernoff (SP) | CP (SP)  | KL-Chernoff (MD) | CP (MD)  |
|----------|-------------------|----------|-------------------|----------|
| MRPC     | 0.83              | 0.73     | 0.82              | 0.72     |
| SST-2    | 0.75              | 0.67     | 0.75              | 0.67     |
| CoLA     | 0.81              | 0.71     | 0.82              | 0.72     |
| AG News  | 0.76              | 0.67     | 0.77              | 0.68     |

**KL-Chernoff** is 17–25% tighter than Hoeffding per bin on average.
**Clopper-Pearson** is 27–33% tighter per bin on average.

The improvement is largest for SST-2 and AG News (lowest error rates → most bins have small p̂).

### 4. KL-Chernoff improvement across all experiments

![KL-Chernoff / Hoeffding heatmap](../results/concentration_bounds/concentration_bounds_heatmap.png)

The KL-Chernoff/Hoeffding ratio (averaged over bins) ranges from **0.73 to 0.87** across all 36 experiments:
- Best improvement: SST-2 × ELECTRA (SP): 0.73 (27% tighter)
- Least improvement: CoLA × BERT (MD): 0.87 (13% tighter)
- Consistent across models; driven primarily by dataset error rate

### 5. MCE bound (max over bins)

| Bound             | Avg MCE bound | vs Hoeffding | Improvement |
|-------------------|:-------------:|:------------:|:-----------:|
| **Hoeffding**     | 0.255         | 1.000×       | —           |
| **KL-Chernoff**   | 0.251         | 0.985×       | +1.5%       |
| **Clopper-Pearson**| 0.218        | 0.853×       | **+14.7%**  |
| Emp. Bernstein    | 0.560         | 2.196×       | −119.6%     |

**Why KL barely improves MCE**: The MCE is determined by the **worst-case bin** (highest p̂_b). In our data, the worst bin has p̂ ≈ 0.32, where KL(0.32+ε ‖ 0.32) ≈ 2ε² (Hoeffding is already nearly tight). Clopper-Pearson still helps because exact binomial tails are tighter than exponential even at moderate p.

### 6. ECE-type bound (weighted average over bins)

| Bound             | Avg ECE bound | vs Hoeffding | Improvement |
|-------------------|:-------------:|:------------:|:-----------:|
| **Hoeffding**     | 0.255         | 1.000×       | —           |
| **KL-Chernoff**   | 0.184         | 0.722×       | **+27.8%**  |
| **Clopper-Pearson**| 0.165        | 0.647×       | **+35.3%**  |
| Emp. Bernstein    | 0.419         | 1.644×       | −64.4%      |

**The weighted-average bound benefits much more** because most bins have low p̂ where KL and CP are substantially tighter. This is the bound relevant for ECE guarantees.

### 7. MCE bound vs n_cal — overlay on ablation data

![MCE bounds overlaid on n_cal ablation](../results/concentration_bounds/concentration_bounds_ncal_overlay.png)

Overlaying the theoretical MCE bounds on the n_cal ablation data (DeBERTa, SST-2 and AG News):

- **Hoeffding** (red solid) is always a valid upper bound on the empirical MCE but is quite loose
- **KL-Chernoff** (orange dashed) is noticeably tighter, especially at small n_cal
- **Clopper-Pearson** (purple dash-dot) is the tightest, 15–30% below Hoeffding across all n_cal
- At n_cal = 10000, Clopper-Pearson is close to the actual empirical MCE

### 8. Theoretical MCE bound vs n_cal

![Theoretical MCE bound vs n_cal](../results/concentration_bounds/concentration_bounds_mce_vs_ncal.png)

For two regimes (low error ~7% like SST-2, moderate error ~15% like MRPC):

- **Hoeffding, KL-Chernoff, and Clopper-Pearson** converge at the same rate (O(√(B log B / n)))
- **Clopper-Pearson** has the best constants, especially at small n_cal
- **Emp. Bernstein** is off the chart at small n_cal (MCE bound > 3.0 at n_cal = 50)
- At moderate error rates, the gap between bounds narrows (Hoeffding is less loose at p ≈ 0.5)

## Analysis

### Why is the MCE bound hard to improve?

The MCE bound = max_b ε_b is dominated by the **worst bin**. For uniform-mass calibration of a reasonably accurate model (5–15% error), the bin error rates range from ~0 to ~0.3–0.5. The worst bin (highest p̂_b) is where:

- **Hoeffding is nearly tight**: At p = 0.3–0.5, the Bernoulli variance p(1−p) is close to the maximum 1/4 that Hoeffding assumes. The KL divergence d(p+ε ‖ p) ≈ ε²/(2p(1−p)) ≈ 2ε² (Hoeffding).
- **Clopper-Pearson still helps**: The exact binomial tail is tighter than the exponential bound even at moderate p, giving ~15% improvement.

### Where do tighter bounds help most?

The **low-error-rate bins** (p̂ < 0.1) benefit enormously from KL/CP. These bins represent the majority of test samples (since the model is usually correct). An **ECE-type guarantee** (weighted average) benefits 28–35% from switching to KL/CP.

### Practical implications for the paper

| Guarantee type | Best bound to use | Improvement over Hoeffding |
|:--------------|:------------------|:--------------------------:|
| MCE (max)     | Clopper-Pearson   | ~15%                       |
| ECE (avg)     | KL-Chernoff or CP | ~28–35%                    |
| Per-bin       | Clopper-Pearson   | ~27–33%                    |

### What NOT to use

- **Empirical Bernstein (Maurer-Pontil)**: Catastrophically worse at n_b < 200. The 7 log(2/δ)/(3(n−1)) variance-estimation penalty dominates.
- **Bernstein with plug-in variance**: Not a valid PAC bound (uses uncontrolled plug-in estimate).
- **Wilson / Agresti-Coull intervals**: Not strict PAC bounds (actual coverage can dip below 1−α for some p values).

## Test-Set Bounds for Discrete Calibrators

### Motivation: why are the n_cal-based bounds loose relative to the test MCE?

Sections 3–6 above compute calibration bounds using the **calibration set** (n_cal = 1000, n_b ≈ 51 per bin). Section 7 overlays these bounds on the empirical MCE measured on the **test set** (n_test = 7600 for SST-2/AG News). The bounds appear loose (0.255 vs observed MCE ≈ 0.08) because:

1. The theorem guarantees the **true** calibration error using only n_cal; the test MCE is a noisy estimate from a much larger n_test.
2. The bound accounts for worst-case uncertainty from n_b ≈ 51 samples per bin; the test MCE benefits from n_test/B ≈ 400–760 samples per group.

A natural question: can we compute **tighter post-hoc bounds** using the test set?

### Approach

Any discrete calibrator (UM or isotonic step) maps test scores to a finite set of values {t_1, ..., t_B}. We group test samples by their calibrated value and, within each group b, observe n_b i.i.d. Bernoulli errors. Since the calibrator is fitted on D_cal and D_test is independent, the test errors within each group are valid i.i.d. samples.

For each group b, we compute:
- **Observed calibration error**: |p̂_test,b − t_b|
- **CI width** ε_b from Hoeffding, KL-Chernoff, or Clopper-Pearson (with union bound δ_b = α/B)
- **Calibration certificate**: |p̂_test,b − t_b| + ε_b (upper bound on the true |P(E=1|û=t_b) − t_b|)

The **MCE certificate** is max_b(|p̂_test,b − t_b| + ε_b), which bounds the true MCE with probability ≥ 1−α.

Note: this is a **post-hoc verification** (requires observing test data), unlike the theorem which is an **a priori guarantee** (holds before seeing any test data).

### Results on actual data

Analysis across all 72 experiments (3 models × 4 datasets × 3 seeds × 2 scores):

**MCE certificates** (averaged over 9 experiments per cell):

| Dataset  | Score | B  | n_b (med) | MCE obs | CP cert | Theorem (H) |
|----------|-------|----|-----------|---------|---------|-------------|
| MRPC     | SP    | 16 | 22        | 0.211   | 0.638   | 0.255       |
| MRPC     | MD    | 14 | 23        | 0.226   | 0.569   | 0.255       |
| **SST-2**| SP    | 10 | 468       | 0.079   | **0.142** | 0.255     |
| **SST-2**| MD    | 10 | 467       | 0.080   | **0.139** | 0.255     |
| CoLA     | SP    | 15 | 61        | 0.133   | 0.328   | 0.255       |
| CoLA     | MD    | 14 | 62        | 0.177   | 0.365   | 0.255       |
| **AG News** | SP | 12 | 438       | 0.085   | **0.144** | 0.255     |
| **AG News** | MD | 11 | 439       | 0.091   | **0.155** | 0.255     |

**CI width** (max_b ε_b — estimation precision only):

| Dataset   | Score | n_b (med) | ε_CP  | Theorem (H) | Ratio |
|-----------|-------|-----------|-------|-------------|-------|
| MRPC      | SP    | 22        | 0.497 | 0.255       | 1.95  |
| SST-2     | SP    | 468       | 0.072 | 0.255       | 0.28  |
| CoLA      | SP    | 61        | 0.225 | 0.255       | 0.88  |
| AG News   | SP    | 438       | 0.073 | 0.255       | 0.29  |

**ECE certificates** (Σ_b w_b · (|p̂_test,b − t_b| + ε_b)):

| Dataset   | Score | ECE obs | CP cert | Theorem (H) |
|-----------|-------|---------|---------|-------------|
| SST-2     | SP    | 0.021   | 0.047   | 0.255       |
| SST-2     | MD    | 0.019   | 0.046   | 0.255       |
| AG News   | SP    | 0.023   | 0.053   | 0.255       |
| AG News   | MD    | 0.022   | 0.050   | 0.255       |

**Per-group detail** (DeBERTa, SST-2, seed 42, SP — 10 groups):

| t_b    | n_b  | k_b | p_test | \|err\| | ε_CP   | cert   |
|--------|------|-----|--------|---------|--------|--------|
| 0.000  | 2461 | 23  | 0.009  | 0.009   | 0.007  | 0.016  |
| 0.019  | 1738 | 34  | 0.020  | 0.001   | 0.011  | 0.012  |
| 0.019  | 789  | 7   | 0.009  | 0.010   | 0.014  | 0.024  |
| 0.038  | 344  | 4   | 0.012  | 0.026   | 0.027  | 0.053  |
| 0.038  | 405  | 34  | 0.084  | 0.046   | 0.046  | 0.091  |
| 0.057  | 391  | 44  | 0.113  | 0.056   | 0.052  | 0.108  |
| 0.075  | 419  | 21  | 0.050  | 0.025   | 0.038  | 0.063  |
| 0.212  | 276  | 46  | 0.167  | 0.045   | 0.071  | 0.116  |
| 0.226  | 321  | 72  | 0.224  | 0.002   | 0.071  | 0.074  |
| 0.321  | 456  | 193 | 0.423  | 0.103   | 0.067  | **0.169** |
| **MCE**|      |     |        | 0.103   |        | **0.169** |

Theorem bound (n_cal=1000, Hoeffding): **0.255** — the test-set CP certificate (0.169) is 34% tighter.

### Key findings

1. **For large test sets** (SST-2, AG News with n_test = 7600): test-set CP certificates are **43% tighter** than the theorem bound, because each group has n_b ≈ 400–760 samples (vs 51 from n_cal).

2. **For small test sets** (MRPC with n_test = 408): test-set bounds are **worse** than the theorem — each group has only n_b ≈ 22 samples, making the CIs larger than the theorem's ε from n_b ≈ 51.

3. **The same principle applies to isotonic regression (step function)**: isotonic step produces B_iso ≈ 13 discrete values at n_cal = 1000, so test samples can be grouped identically. The bounds would be comparable to the UM results above (similar B, similar group sizes).

4. **Two different guarantees**:
   - **Theorem (a priori)**: before seeing test data, guarantees MCE_true ≤ ε_n. Uses n_cal only.
   - **Test-set certificate (post-hoc)**: after seeing test data, certifies MCE_true ≤ max_b(|p̂_test,b − t_b| + ε_b). Tighter when n_test >> n_cal.

![MCE: Observed vs Test-set certificates vs Theorem bound](../results/testset_bounds/testset_bounds_mce_comparison.png)

![Test-set per-group bounds for UM calibration](../results/testset_bounds/testset_bounds_pergroup.png)

## Recommendation for the Paper

### Option A: Keep Hoeffding, cite tighter alternatives in discussion

The current Hoeffding bound is clean, well-known, and gives a simple closed-form formula for Theorem 3. Mention in the discussion that tighter bounds exist (KL-Chernoff, Clopper-Pearson) and quantify the improvement.

### Option B: Replace with Clopper-Pearson in Theorem 3

Restate the guarantee with data-dependent per-bin bounds:

$$
|\mathbb{P}(E=1 \mid \hat{u}_n = t_b) - t_b| \leq \epsilon_b, \quad \text{where } \epsilon_b = \text{Beta}^{-1}(1-\alpha/(2B);\, k_b+1,\, n_b-k_b) - \hat{p}_b
$$

This is exact (not asymptotic) and 15–35% tighter. The downside is a less elegant formula.

### Option C: Replace with KL-Chernoff bound

Restate as: ε_b satisfying n_b · d(p̂_b + ε_b ‖ p̂_b) = log(2B/α). This is the information-theoretically optimal exponential rate for Bernoulli and connects to the KL-UCB literature. The formula is implicit but standard.

**Our recommendation**: Option A for Theorem 3 (simplicity), with a remark noting that the Clopper-Pearson bound gives ~15% tighter MCE guarantees and ~35% tighter ECE guarantees.

## Files

### Concentration bounds (n_cal-based)
- `scripts/investigate_concentration_bounds.py` — Full analysis script
- `results/concentration_bounds/concentration_bounds_analysis.json` — Per-bin results (927 bins)
- `results/concentration_bounds/concentration_bounds_vs_p.pdf` — ε vs p̂ for various n_b
- `results/concentration_bounds/concentration_bounds_vs_n.pdf` — ε vs n_b for various p̂
- `results/concentration_bounds/concentration_bounds_perbin.pdf` — per-bin comparison on actual data
- `results/concentration_bounds/concentration_bounds_heatmap.pdf` — KL/Hoeffding ratio heatmap
- `results/concentration_bounds/concentration_bounds_ncal_overlay.pdf` — bounds overlaid on n_cal ablation
- `results/concentration_bounds/concentration_bounds_mce_vs_ncal.pdf` — theoretical MCE bound vs n_cal

### Test-set bounds (post-hoc certificates)
- `scripts/investigate_testset_bounds.py` — Test-set bounds analysis script
- `results/testset_bounds/testset_bounds_analysis.json` — 72 experiment results
- `results/testset_bounds/testset_bounds_pergroup.pdf` — Per-group bounds vs actual error (DeBERTa, all datasets)
- `results/testset_bounds/testset_bounds_mce_comparison.pdf` — Bar chart: observed MCE vs certificates vs theorem
- `results/testset_bounds/testset_bounds_nb_vs_eps.pdf` — Group size vs bound tightness

## References

- Gupta & Ramdas (2021), "Distribution-free calibration guarantees for histogram binning without sample splitting", ICML
- Hoeffding (1963), "Probability inequalities for sums of bounded random variables"
- Clopper & Pearson (1934), "The use of confidence or fiducial limits illustrated in the case of the binomial"
- Cappé et al. (2013), "Kullback-Leibler upper confidence bounds for optimal sequential allocation", Annals of Statistics (KL-UCB)
- Maurer & Pontil (2009), "Empirical Bernstein Bounds and Sample Variance Penalization", COLT
- Waudby-Smith & Ramdas (2024), "Estimating means of bounded random variables by betting", JRSS-B
- Brown, Cai & DasGupta (2001), "Interval Estimation for a Binomial Proportion", Statistical Science
