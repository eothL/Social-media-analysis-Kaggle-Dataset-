# Academic Impact Analysis — Technical Walkthrough

This document explains, step-by-step, how the analysis in `scripts/academic_impact_analysis.py` answers the question described in `reports/academic_impact_question_plan.md`, and how to interpret each output.

The **results summary** is in `reports/academic_impact_results_report.md`.

---

## Research question
**Which factors best explain whether a student reports that social media affects their academic performance (`Affects_Academic_Performance`: Yes/No), and how large are those effects?**

---

## Data and preprocessing

**Input dataset**
- `Data/media_addiction.csv`

**Outcome encoding**
- `Affects_Academic_Performance` is converted to a binary target:
  - Yes → `y = 1`
  - No → `y = 0`

**Numeric variables used**
- `Avg_Daily_Usage_Hours`, `Sleep_Hours_Per_Night`, `Mental_Health_Score`, `Addicted_Score`, `Age`, `Conflicts_Over_Social_Media`

**Categorical variables used**
- `Gender`, `Academic_Level`, `Relationship_Status`, `Most_Used_Platform`

**Platform regrouping**
- Platforms are grouped into “top K” (default `K=6`) by frequency plus `Other`.
- This reduces sparse contingency tables and makes the chi-square test more reliable.

**Standardization (for logistic regression only)**
- Each numeric predictor is standardized to z-scores:
  - `z = (x − mean(x)) / sd(x)`
- This makes odds ratios comparable across numeric predictors and improves numeric stability.

**One-hot encoding**
- Categorical predictors are converted to dummy variables.
- One level is dropped per categorical variable (a “reference” level) to avoid perfect multicollinearity.
- Interpretation: coefficients are *relative to the reference level*.

---

## Step 1 — Descriptives + confidence interval for the “Yes” rate

**What is estimated**
- The proportion of students who report academic impact:
  - `p̂ = (# Yes) / n`

**Why a confidence interval**
- A CI quantifies uncertainty about the population proportion.

**Method used**
- Wilson 95% CI (more stable than the naive normal CI when proportions aren’t near 0.5 or when n is moderate).

**Where it appears**
- In `results/academic_impact*/academic_impact_report.md` and `reports/academic_impact_results_report.md`.

---

## Step 2 — Numeric comparisons (Yes vs No)

**Goal**
- Quantify how different the Yes and No groups are on each numeric variable.

**What is reported**
- Mean(Yes), mean(No)
- Difference in means: `Δ = mean(Yes) − mean(No)`
- 95% CI for `Δ` using a normal approximation:
  - `Δ ± 1.96 × SE(Δ)`
- **Effect size (Cohen’s d)**:
  - `d = (mean(Yes) − mean(No)) / pooled_SD`
  - Rough scale: 0.2 small, 0.5 medium, 0.8 large

**Why effect size matters**
- It measures “how big” the difference is in standard-deviation units, not just whether it is statistically different.
- This is especially useful when sample sizes are uneven.

**Outputs**
- `results/academic_impact/numeric_group_comparisons.csv`
- Also summarized in `reports/academic_impact_results_report.md`

---

## Step 3 — Categorical associations (chi-square + effect size)

**Goal**
- Test whether “academic impact Yes/No” is associated with categorical variables.

**Method**
- Chi-square test of independence on a 2-column contingency table (Yes vs No).
- Test statistic:
  - `χ² = Σ (O − E)² / E`
  - Degrees of freedom: `(r − 1)(c − 1)`

**Why Cramér’s V**
- Chi-square p-values can be tiny with large samples even for small practical differences.
- **Cramér’s V** is a standardized effect size for contingency tables:
  - `V = sqrt( χ² / (n × min(r−1, c−1)) )`
  - Ranges from 0 (no association) to 1 (strong association).

**Where Cramér’s V comes from**
- The notions summary explicitly covers **chi-square tests** (contingency tables).
- The file also recommends reporting **effect sizes** rather than only “significant / not significant”.
- Cramér’s V is a standard effect-size companion to the chi-square test (widely used in applied statistics); it wasn’t copied as a formula from the summary file, but it is directly aligned with the summary’s “chi-square + effect size” workflow.

**Expected-count warning**
- Chi-square’s approximation is less reliable when expected counts are very small (common rule of thumb: < 5).
- The analysis outputs `min_expected` to flag this.

**Outputs**
- `results/academic_impact/categorical_chi_square_summary.csv`

---

## Step 4 — Logistic regression (multivariable model)

**Goal**
- Model the probability of reporting academic impact while controlling for multiple predictors:
  - `P(Impact = Yes | predictors)`

**Model**
- Logistic regression:
  - `logit(p) = log(p/(1-p)) = β0 + β1 x1 + ... + βk xk`
  - Predicted probability: `p = 1 / (1 + exp(−η))`

**Estimation**
- The script fits coefficients via IRLS (iteratively reweighted least squares), a standard algorithm for logistic regression.

**Odds ratios**
- For a coefficient `β`, the odds ratio is `OR = exp(β)`.
- For standardized numeric predictors, OR is per **+1 SD** change.

**Collinearity note**
- Usage and addiction tend to be highly correlated; the script fits three variants:
  - `full`: usage + addiction + other controls
  - `usage_only`: usage + controls
  - `addiction_only`: addiction + controls

**Perfect separation warning**
- In this dataset, the logistic models achieve perfect classification (AUC≈1, FP=0, FN=0).
- When separation occurs, unregularized logistic regression can have coefficients that diverge toward ±∞; ORs and CIs become unstable.

**Regularization (“ridge”)**
- Running with `--l2 1.0` applies an L2 penalty to stabilize coefficients:
  - It shrinks extreme coefficients and keeps ORs finite.

**Outputs**
- Coefficients:
  - `results/academic_impact_ridge/logistic_full_odds_ratios.csv` (recommended to interpret)
- Diagnostics:
  - `results/academic_impact_ridge/logistic_full_predictions.csv` (row-level `y_true` and `p_hat`)
  - `results/academic_impact_ridge/fig_logistic_roc.svg`
  - `results/academic_impact_ridge/fig_logistic_calibration.svg`
  - `results/academic_impact_ridge/fig_logistic_or_forest.svg`

---

## Step 5 — Robustness / sensitivity to small High School group

**Motivation**
- High School is a small subgroup (n=27), so it can distort comparisons involving `Academic_Level`.

**Check performed**
- Rerun the analysis excluding High School:
  - `python3 scripts/academic_impact_analysis.py --exclude-high-school --out-dir results/academic_impact_no_hs`
  - and similarly for the ridge run.

**What to look for**
- If numeric effect sizes and mean differences remain similar, the main conclusions are robust.
- If `Academic_Level` association disappears, it suggests that signal is driven mostly by the High School group.

**Outputs**
- `results/academic_impact_no_hs/*`
- `results/academic_impact_ridge_no_hs/*`

---

## How to reproduce

```bash
python3 scripts/academic_impact_analysis.py
python3 scripts/academic_impact_analysis.py --exclude-high-school --out-dir results/academic_impact_no_hs
python3 scripts/academic_impact_analysis.py --l2 1.0 --out-dir results/academic_impact_ridge
python3 scripts/academic_impact_analysis.py --l2 1.0 --exclude-high-school --out-dir results/academic_impact_ridge_no_hs
python3 scripts/generate_academic_impact_results_report.py
```
