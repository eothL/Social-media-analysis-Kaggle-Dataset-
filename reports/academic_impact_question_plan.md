# Research Question + Methods Plan (Course-Technique Aligned)

## Research question (data-driven)
**Which factors best explain whether a student reports that social media affects their academic performance (`Affects_Academic_Performance`: Yes/No), and how large are those effects after controlling for other variables?**

This question fits the dataset well because it has:
- A clear binary outcome (`Yes/No`) suitable for contingency methods and logistic regression.
- Multiple plausible predictors: usage, addiction, sleep, mental health, age, relationship conflicts, plus categorical demographics/platform.

---

## Variables

**Outcome (binary)**
- `Affects_Academic_Performance` (Yes/No)

**Numeric predictors**
- `Avg_Daily_Usage_Hours`
- `Addicted_Score`
- `Sleep_Hours_Per_Night`
- `Mental_Health_Score`
- `Age`
- `Conflicts_Over_Social_Media`

**Categorical predictors (controls / subgroup tests)**
- `Gender`
- `Academic_Level` (note: High School group is small)
- `Most_Used_Platform` (many levels; may need regrouping)
- `Relationship_Status`
- `Country` (too many levels for simple inference; use cautiously)

---

## Hypotheses (examples you can report)

1) **Usage/addiction link to academic impact**
- H0: Mean(`Avg_Daily_Usage_Hours`) is equal for Yes vs No.
- H0: Mean(`Addicted_Score`) is equal for Yes vs No.

2) **Sleep/mental health differences**
- H0: Mean(`Sleep_Hours_Per_Night`) is equal for Yes vs No.
- H0: Mean(`Mental_Health_Score`) is equal for Yes vs No.

3) **Categorical associations**
- H0: `Affects_Academic_Performance` is independent of `Gender` / `Academic_Level` / `Relationship_Status` / `Most_Used_Platform`.

4) **Multivariable explanation**
- H0 (logistic regression): each coefficient = 0 (no contribution) after controlling for others.

---

## Analysis workflow (mapped to course notions)

### Step 1 — Descriptives + uncertainty (CLT / CI)
**Goal:** quantify the baseline prevalence of “academic impact”.
- Compute the sample proportion: \n
  \t`p_hat = (# Yes) / n`\n
- 95% CI for the proportion (use a standard approximation taught in class, or a Wilson CI if you want better small-sample behavior).
- For each numeric predictor: mean/SD overall, and by Yes/No.

Deliverables:
- Table: overall summaries and Yes/No summaries.
- A short paragraph interpreting the CI (what we can say about the population proportion).

### Step 2 — Two-group comparisons (t-test logic + effect size)
**Goal:** compare numeric predictors between Yes vs No.
- For each numeric predictor:
  - Difference in means (Yes − No)
  - 95% CI for the mean difference (two-sample t logic)
  - **Effect size** (Cohen’s d) to describe magnitude (recommended, especially when sample sizes differ)

Deliverables:
- Table: mean(Yes), mean(No), mean diff, CI, Cohen’s d.
- Plot: side-by-side boxplots/violin plots for the most important predictors (usage, addiction, sleep, mental health).

### Step 3 — Categorical vs categorical (contingency tables, chi-square / Fisher)
**Goal:** test whether academic impact depends on categorical attributes.
- Build contingency tables for:
  - `Affects_Academic_Performance` × `Gender`
  - `Affects_Academic_Performance` × `Academic_Level`
  - `Affects_Academic_Performance` × `Relationship_Status`
  - `Affects_Academic_Performance` × `Most_Used_Platform` (consider regrouping small platforms)
- Apply **chi-square test of independence** where expected counts are adequate.
- If expected counts are small (common with many platform levels), use **Fisher’s exact** or merge rare categories.

Deliverables:
- Table: observed counts + row/column percentages.
- Report: chi-square statistic, df, p-value (and a short interpretation).

### Step 4 — Logistic regression (binary outcome modeling)
**Goal:** model the probability of academic impact while controlling for multiple predictors.

Model form:
- `logit(P(Impact=Yes)) = β0 + β1*Usage + β2*Addiction + β3*Sleep + β4*MentalHealth + β5*Age + β6*Conflicts + categorical controls`

Report:
- Coefficients as **odds ratios** (`exp(β)`) with 95% CIs
- Interpretation in plain language (e.g., “+1 point addiction score multiplies odds by X”)

Important modeling note (multicollinearity):
- `Avg_Daily_Usage_Hours` and `Addicted_Score` are usually highly correlated; check VIF or correlation.
- If collinearity is strong, consider:
  - A model with only one of them (usage-only vs addiction-only)
  - A model with both but interpret carefully (shared variance)
  - Standardize numeric predictors for comparability

Deliverables:
- Table: ORs + CIs + p-values.
- Plot: coefficient/OR “forest plot” for readability.

### Step 5 — Diagnostics + robustness (model-building)
**Goal:** ensure conclusions aren’t driven by small subgroups or unstable modeling choices.
- Sensitivity: rerun main conclusions excluding the small `High School` group (or merge HS with UG if justified).
- Platform levels: regroup rare platforms and check whether conclusions change.
- Basic fit checks for logistic regression:
  - Confusion matrix at a chosen threshold
  - AUC/ROC (prediction quality)
  - Calibration check (predicted vs observed)

Deliverables:
- Short “robustness” section: what changes, what stays stable.

---

## Suggested “story” for the report
1) How common is reported academic impact (with CI)?
2) Which numeric factors differ most between Yes and No (effect sizes)?
3) Which categorical factors are associated (chi-square / Fisher)?
4) When considered together, which predictors remain important (logistic regression)?
5) Are results robust to small groups / category choices?

---

## Optional extensions (if you have time)
- Interaction: does the usage→impact relationship differ by `Academic_Level` or `Gender`?
- Mediation-style framing (course-dependent): usage → addiction → impact (but be clear this is observational, not causal).

