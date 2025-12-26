# Quantitative Analysis — Notions Summary (auto-compiled, software-agnostic)

This file summarizes the main statistical/quantitative notions that appear across the files in this folder (lectures, homeworks, solutions, datasets, and the group-project archive), explains how they connect, and ends with an end-to-end workflow for analyzing data.

## Scope

- Included: all files in the current folder.
- Excluded folders (not scanned): `QA Project 2025Fall/`, `R_Recitation1/`, `R_Recitation2/`.
- Notes on extraction:
  - PDFs were text-extracted (some slide elements can be missing if they are embedded as images).
  - Office files (`.xlsx/.docx/.pptx`) were text-extracted from their internal XML (formatting, formulas, and charts may not be fully captured).
  - `.RData` is a binary workspace file and is not reliably readable without the originating software.

## Data in this folder (examples you can analyze)

- `class.csv`: student attributes (e.g., `sex`, `age`, `height`, `weight`)
- `scores.csv`: names + `scores` (can be joined to `class.csv` by `name`)
- `pups.csv`: measurements (`weight`, `length`, `age`, `clutch`)

These datasets are useful to illustrate: descriptive statistics, inference (means/variances), contingency reasoning, regression, and ANOVA-like comparisons.

---

## Big picture: how the topics connect

You can view the course as one connected pipeline:

1. **Probability** provides a language for uncertainty (events, conditional probability, distributions).
2. **Random variables + distributions** let you model noisy measurements.
3. **Sampling distributions + CLT** explain why statistics (like $\bar X$) fluctuate from sample to sample.
4. **Estimation + confidence intervals** quantify uncertainty about unknown parameters.
5. **Hypothesis tests** turn uncertainty into decisions under controlled error rates.
6. **Independence/contingency methods** specialize testing/estimation for categorical data.
7. **Regression + ANOVA** model relationships, compare groups, and generalize inference to multi-variable settings.
8. **Model building + diagnostics** ensure the model fits data and assumptions are plausible.
9. **Time series** adapts regression/inference ideas when observations are ordered and correlated over time.
10. **CART / ML (e.g., LSTM)** emphasize prediction, nonlinearity, and algorithmic modeling (often with different validation practices).

---

## 1) Probability theory (events and conditioning)

**Notions**

- **Sample space** $S$ is the set of all possible outcomes; an **event** is a subset $A\subseteq S$.
- **Set operations** describe compound events:
  - Complement: $A^c$ (“not $A$”)
  - Union: $A\cup B$ (“$A$ or $B$”)
  - Intersection: $A\cap B$ (“$A$ and $B$”)
- Probability rules (axioms) guarantee coherent “degrees of belief”:
  - $P(A)\ge 0$
  - $P(S)=1$
  - If $A\cap B=\emptyset$, then $P(A\cup B)=P(A)+P(B)$
- **Counting** (permutations/combinations) lets you compute probabilities in finite spaces by “favorable outcomes / total outcomes.”
- **Conditional probability** updates probabilities when you know $B$ occurred:

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)} \quad (P(B)>0)
$$

- **Independence** means learning $B$ doesn’t change $A$:

$$
A\perp B \iff P(A\mid B)=P(A) \iff P(A\cap B)=P(A)P(B)
$$

- **Law of total probability** (for a partition $\{B_i\}$ of $S$):

$$
P(A)=\sum_i P(A\mid B_i)P(B_i)
$$

- **Bayes’ theorem** (a re-expression of conditional probability):

$$
P(B\mid A)=\frac{P(A\mid B)P(B)}{P(A)}
$$

**Numerical example (Bayes, fully worked)**

Suppose disease prevalence is $P(D)=0.01$, sensitivity $P(+\mid D)=0.95$, and false positive $P(+\mid D^c)=0.02$.

1) Total probability of testing positive:

$$
P(+)=P(+\mid D)P(D)+P(+\mid D^c)P(D^c)=0.95(0.01)+0.02(0.99)=0.0293
$$

2) Posterior probability of disease after a positive test:

$$
P(D\mid +)=\frac{P(+\mid D)P(D)}{P(+)}=\frac{0.95\cdot 0.01}{0.0293}\approx 0.324
$$

**How it connects**

- Conditional probability and Bayes show up again in inference (likelihood/posterior thinking), classification, and decision-making under uncertainty.
- Counting and event algebra show up whenever you build probability models for discrete outcomes (binomial/hypergeometric) and for “exact” tests.

**Small counting example**

If you draw 2 items from 5 without order, the number of possible pairs is:

$$
\binom{5}{2}=\frac{5!}{2!\,3!}=10
$$

---

## 2) Random variables and distributions

**Notions**

- A **random variable** $X$ assigns a number to each outcome in $S$.
- Discrete vs continuous:
  - Discrete: $P(X=x)$ is meaningful (PMF).
  - Continuous: probabilities come from areas under a density (PDF).
- **CDF** $F(x)=P(X\le x)$ unifies both cases.
- **Expectation** and **variance** summarize location and spread:

$$
E[X]=\sum_x x\,p(x)\ \text{(discrete)} \qquad E[X]=\int x f(x)\,dx\ \text{(continuous)}
$$

$$
\mathrm{Var}(X)=E[(X-E[X])^2]=E[X^2]-(E[X])^2
$$

**Core distributions (and why they matter)**

- **Bernoulli/Binomial**: counts of successes in $n$ independent trials (quality control, click-through, defect rates).
- **Poisson**: counts of rare events per interval (calls per minute, arrivals).
- **Normal**: arises from aggregation (measurement error, CLT).
- **Exponential**: waiting times for Poisson events.
- **$t$, $\chi^2$, $F$**: appear as sampling distributions of statistics, powering CIs and tests (means, variances, regression, ANOVA).

**Numerical examples**

- Binomial mean and variance for $X\sim \mathrm{Bin}(n,p)$:
  - $E[X]=np$, $\mathrm{Var}(X)=np(1-p)$
  - Example: $n=20$, $p=0.3$ gives $E[X]=6$, $\mathrm{Var}(X)=4.2$.
- Exponential tail probability for $T\sim \mathrm{Exp}(\lambda)$:

$$
P(T>t)=e^{-\lambda t}
$$

  - Example: $\lambda=0.5$, $t=3$ gives $P(T>3)=e^{-1.5}\approx 0.223$.

**How it connects**

- Distributions are the “assumption layer” behind inference (CI/tests) and modeling (regression errors, ANOVA errors, time series noise).

---

## 3) Joint behavior: covariance and correlation

**Notions**

- **Covariance** measures joint linear co-movement:

$$
\mathrm{Cov}(X,Y)=E[(X-E[X])(Y-E[Y])]
$$

- **Correlation** standardizes covariance to $[-1,1]$:

$$
\rho=\mathrm{Corr}(X,Y)=\frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}
$$

- Independence $\Rightarrow$ zero covariance, but zero covariance $\not\Rightarrow$ independence (nonlinear relationships can exist).

**Numerical example**

If you have paired observations $(x_i,y_i)$, sample correlation is:

$$
r=\frac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_i(x_i-\bar x)^2}\sqrt{\sum_i(y_i-\bar y)^2}}
$$

This directly connects to regression: in simple linear regression, the estimated slope is proportional to this numerator.

**How it connects**

- Correlation is a quick association measure; regression formalizes association into a predictive/explanatory model with uncertainty quantification.

---

## 4) Sampling, estimators, and the Central Limit Theorem (CLT)

**Notions**

- A **statistic** (like $\bar X$) is a function of the sample; it is random because the sample is random.
- The **sampling distribution** is the distribution of that statistic across repeated sampling.
- **Standard error (SE)** is the standard deviation of a statistic’s sampling distribution; for the mean:

$$
SE(\bar X)=\frac{\sigma}{\sqrt{n}}
$$

- **CLT** (core approximation idea): for large $n$, regardless of the original distribution (under mild conditions),

$$
\frac{\bar X-\mu}{\sigma/\sqrt{n}} \approx N(0,1)
$$

- **Point estimators** aim to estimate unknown parameters (like $\mu$, $\sigma^2$). Key properties:
  - **Bias**: $E[\hat\theta]-\theta$
  - **Variance**: variability of $\hat\theta$ across samples
  - (Often) **MSE**: $\mathrm{MSE}(\hat\theta)=\mathrm{Var}(\hat\theta)+\mathrm{Bias}(\hat\theta)^2$

**Numerical example (why $1/\sqrt{n}$ matters)**

If $\sigma=10$:

- For $n=25$, $SE(\bar X)=10/5=2$
- For $n=100$, $SE(\bar X)=10/10=1$

So quadrupling sample size halves typical error bars.

**How it connects**

- Sampling distributions justify confidence intervals and p-values; “model error bars” come from SEs.

---

## 5) Confidence intervals (CI)

**Notions**

- A **confidence interval** is a random interval with long-run coverage (e.g., 95%).
- Typical patterns:
  - mean (unknown variance): use $t$
  - variance: use $\chi^2$
  - ratio of variances: use $F$

**Mean CI (unknown $\sigma$)**

If $X_1,\dots,X_n$ are i.i.d. normal (or $n$ is reasonably large), a common $95\%$ CI for $\mu$ is:

$$
\bar X \pm t_{0.975,\ n-1}\cdot \frac{S}{\sqrt{n}}
$$

Interpretation: the interval is random; the parameter $\mu$ is fixed.

**Variance CI (normality assumption)**

Under normality,

$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}
$$

so a $(1-\alpha)$ CI for $\sigma^2$ is:

$$
\left[\frac{(n-1)S^2}{\chi^2_{1-\alpha/2,\ n-1}},\ \frac{(n-1)S^2}{\chi^2_{\alpha/2,\ n-1}}\right]
$$

**Numerical example (mean CI by hand)**

Suppose $n=10$, $\bar X=82$, $S=6$. With $t_{0.975,9}\approx 2.262$:

$$
82 \pm 2.262\cdot \frac{6}{\sqrt{10}} = 82 \pm 4.29 \Rightarrow [77.71,\ 86.29]
$$

**How it connects**

- CIs are “estimation-mode” inference; hypothesis tests are the “decision-mode” counterpart (often dual to each other).

---

## 6) Hypothesis testing (p-values, errors, power)

**Notions**

- **Null** $H_0$ vs **alternative** $H_1$.
- **Test statistic**, **critical region**, **p-value**.
- **Type I error** $\alpha$: reject $H_0$ when it’s true.
- **Type II error** $\beta$: fail to reject $H_0$ when it’s false.
- **Power** $1-\beta$.
- **One-sided** vs **two-sided** tests.

**p-value (meaning)**

The p-value is:

> the probability, assuming $H_0$ is true, of observing a test statistic at least as extreme as what you observed.

It is *not* the probability that $H_0$ is true.

**Numerical example (z-test style intuition)**

Suppose your standardized statistic is $z=2.0$ for a two-sided test. Then:

$$
p = 2\cdot P(Z\ge 2.0)\approx 2\cdot 0.0228 = 0.0456
$$

At $\alpha=0.05$, you reject $H_0$.

**Power (why sample size matters)**

If effects are small, power can be low at fixed $n$. Increasing $n$ decreases SE, making real effects easier to detect.

**How it connects**

- Tests can be derived from CIs (e.g., $H_0:\mu=\mu_0$ is rejected at level $\alpha$ iff $\mu_0$ is outside the $(1-\alpha)$ CI).

---

## 7) Independence and contingency tables (categorical data)

**Notions**

- **Chi-square test of independence** in a contingency table.
- **Expected counts** under independence.
- **Chi-square goodness-of-fit** compares observed counts to expected probabilities.
- When expected counts are small (esp. 2×2), consider **Fisher’s exact test**.

**Independence test (table form)**

Given counts $O_{ij}$ in a table (rows = categories of $A$, columns = categories of $B$), under independence the expected counts are:

$$
E_{ij}=\frac{(\text{row }i\text{ total})(\text{col }j\text{ total})}{n}
$$

and the statistic is:

$$
\chi^2=\sum_{i,j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}}
$$

Under $H_0$ (independence), $\chi^2$ is approximately chi-square distributed with degrees of freedom:

$$
df=(r-1)(c-1)
$$

**Goodness-of-fit**

You compare observed counts $(O_1,\dots,O_k)$ to expected probabilities $(p_1,\dots,p_k)$ via $E_i=np_i$ and the same $\chi^2$ form.

**How it connects**

- This is “hypothesis testing for categorical variables,” parallel to $t$-tests for means.
- Fisher’s exact test is an “exact probability model” approach (hypergeometric), closely tied to the same counting ideas used in probability.

---

## 8) Experimental design + ANOVA (comparing groups)

**Notions**

- **Factor**, **levels**, **treatments**, **replication**, **randomization**.
- **Completely randomized design** vs **randomized block design**.
- **ANOVA** decomposes variability (sum of squares) and uses an **F-test**.
- (Often) post-hoc comparisons after rejecting the overall null.

**One-way ANOVA model**

If you have $g$ groups, a common model is:

$$
Y_{ij}=\mu+\tau_i+\varepsilon_{ij}
$$

where $\tau_i$ is the effect of group $i$ and $\varepsilon_{ij}$ are random errors.

The ANOVA **F-test** compares “between-group variation” to “within-group variation.” If between-group variation is large relative to within-group, group means likely differ.

In one-way ANOVA, a common test statistic is:

$$
F=\frac{\mathrm{MS}_{\text{between}}}{\mathrm{MS}_{\text{within}}}
$$

**Randomization and blocking (why they matter)**

- **Randomization** helps make groups comparable and supports causal interpretation.
- **Blocking** reduces noise by comparing treatments within similar blocks (e.g., batches, days), improving power.

**How it connects**

- ANOVA is a special case of regression (with categorical predictors encoded via indicators).

---

## 9) Simple linear regression (SLR)

**Notions**

- Model: $Y = \beta_0 + \beta_1 X + \varepsilon$.
- **Least squares** estimates minimize sum of squared residuals.
- Inference for slope/intercept (t-tests), overall model (F-test).
- **R-squared** as a variance-explained measure.
- **Prediction interval** vs **confidence interval**.
- **Residual diagnostics** (nonlinearity, non-constant variance, outliers).

**Least squares (closed form)**

Given data $(x_i,y_i)$, the fitted line $\hat y=\hat\beta_0+\hat\beta_1 x$ has:

$$
\hat\beta_1=\frac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sum_i (x_i-\bar x)^2},\qquad
\hat\beta_0=\bar y-\hat\beta_1\bar x
$$

**Residuals**

Residuals are $\hat\varepsilon_i=y_i-\hat y_i$. Patterns in residuals indicate model problems (missing nonlinearity, changing variance, outliers).

**Confidence vs prediction**

- CI answers: “Where is the *mean response* at $x$ likely to be?”
- PI answers: “Where is a *new observation* at $x$ likely to fall?” (wider than CI).

**How it connects**

- Correlation links to slope (the numerator is shared).
- Hypothesis tests/CI apply to $\beta_1$ to decide whether $X$ is informative.
- $R^2$ connects to variance decomposition: it measures how much of the variability in $Y$ is explained by the fitted line (in-sample).

---

## 10) Multiple regression (MLR)

**Notions**

- Model: $Y = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p + \varepsilon$.
- **Partial effects** (“holding other variables constant”).
- **Dummy (indicator) variables** for categorical predictors (e.g., sex).
- **Multicollinearity**: correlated predictors inflate uncertainty.

**Interpretation**

$\beta_j$ is the expected change in $Y$ for a one-unit increase in $X_j$ *holding other predictors fixed* (a ceteris paribus statement).

**Dummy variables**

A categorical variable with levels (e.g., sex = F/M) is represented by indicator(s), turning group differences into regression coefficients (link to ANOVA).

**Multicollinearity**

If predictors are highly correlated, you can still predict well, but individual coefficients become unstable (wide SEs; sensitive to small data changes).

**How it connects**

- Regression is the “unifying engine” behind ANOVA and many inference procedures (via general linear models).

---

## 11) Logistic regression (binary outcomes)

**Why it appears**

Some slides mention “logistic” in the context of extending regression to classification problems.

**Notions**

- Used when $Y\in\{0,1\}$ (e.g., pass/fail, default/no default).
- Models the probability of $Y=1$ given predictors $X$ using the logistic function:

$$
P(Y=1\mid X)=\frac{1}{1+e^{-(\beta_0+\beta^\top X)}}
$$

- Equivalent “log-odds” form:

$$
\log\left(\frac{P(Y=1\mid X)}{1-P(Y=1\mid X)}\right)=\beta_0+\beta^\top X
$$

**How it connects**

- It’s a close cousin of linear regression (same idea of linear predictor + parameters), but with a probability output and different error model.
- In practice, it sits between classical inference and ML: you still interpret coefficients and test significance, but you also validate predictive performance.

---

## 12) Regression model building + diagnostics

**Notions**

- Variable selection ideas: **forward selection**, **backward elimination**, **stepwise**.
- Criteria: adjusted $R^2$, AIC/BIC (often mentioned), validation mindset.
- **Multicollinearity** checks (e.g., VIF).
- **Autocorrelation** of residuals (esp. time-ordered data).
- **Durbin–Watson test** for first-order autocorrelation.
- Transformations (when needed) to improve model adequacy.

**VIF (variance inflation factor)**

For predictor $X_j$:

$$
\mathrm{VIF}_j=\frac{1}{1-R_j^2}
$$

where $R_j^2$ is from regressing $X_j$ on the other predictors. Large VIF indicates collinearity.

**Autocorrelation and Durbin–Watson (idea)**

If residuals are correlated over time (common in operational/financial series), standard regression SEs can be wrong. The Durbin–Watson statistic tests for first-order autocorrelation.

**How it connects**

- Diagnostics protect inference: if assumptions fail, CIs/tests can mislead. In time series, autocorrelation is the rule, not the exception.
- Variable selection without validation can overfit: it may look great on the same data used to choose the model but generalize poorly.

---

## 13) Time series analysis + forecasting

**Notions**

- Time series components: **level**, **trend**, **seasonality**, noise.
- Forecasting families: qualitative vs quantitative.
- **Moving average** smoothing.
- **Exponential smoothing** (and **Holt / Holt–Winters** extensions).
- **Stationarity** (important for many time series methods).
- Autocorrelation concepts and diagnostic tests (linked to regression too).

**Moving average (definition)**

For a window size $m$, a simple moving average is:

$$
\hat y_t=\frac{1}{m}\sum_{k=0}^{m-1} y_{t-k}
$$

This reduces noise but can lag behind trend changes.

**Exponential smoothing (core idea)**

One common form:

$$
\hat y_{t+1}=\alpha y_t+(1-\alpha)\hat y_t,\quad 0<\alpha<1
$$

Large $\alpha$ reacts quickly; small $\alpha$ smooths more.

**Trend/seasonality extensions**

Holt and Holt–Winters add components for trend and seasonal patterns.

**Stationarity (why it matters)**

Many time-series models assume stable mean/variance and stable dependence structure over time. Non-stationarity often requires detrending, differencing, or decomposition.

**How it connects**

- Time series connects back to regression via residual autocorrelation: “independent errors” becomes a *modeling choice*, not a default.

---

## 14) CART (decision trees)

**Notion (appears in the syllabus)**

- **CART** (Classification and Regression Trees): recursively split the predictor space to predict a class label or a numeric response.
- Key ideas: impurity reduction (classification) / squared-error reduction (regression), pruning/validation.

**Example (conceptual)**

- Regression tree splits `height` and `age` to predict `weight`, choosing splits that reduce within-node variance.

---

## 15) Group project notions (stock forecasting, ML)

The archive `Group Project..zip` contains a prior-year project about **predicting stock market trends**, including ML terms such as **LSTM**.

**Notions**

- Returns (simple or log returns), train/test split, feature engineering.
- Forecast evaluation: error metrics (MAE/RMSE), backtesting mindset.
- LSTM (recurrent neural network) for sequence modeling (when you have enough data).

**Numerical example (log return)**

If price goes from $P_t$ to $P_{t+1}$, log return is:

$$
r_{t+1}=\log\left(\frac{P_{t+1}}{P_t}\right)
$$

Example: $P_t=100$, $P_{t+1}=101.5$ gives $r\approx \log(1.015)\approx 0.0149$ (about $1.49\%$).

---

## 16) Putting it all together: how to analyze real data with these tools

This is a practical “end-to-end” workflow that shows how the notions interlock in a typical analysis.

### Step 0 — Define the question and variables

- Decide the goal: **estimation** (what is the mean?) vs **testing** (is there a difference?) vs **prediction** (forecast future values?).
- Identify response $Y$ and predictors/features $X$.
- Identify data types: numeric vs categorical; independent sample vs paired; time-ordered vs i.i.d.

### Step 1 — Data checks + descriptive statistics

- Summaries: means, medians, variances, quantiles; group summaries.
- Visuals: histograms, boxplots, scatterplots, time plots.
- Spot missing values, outliers, impossible values.

Connected notions: random variables, expectation/variance, correlation.

### Step 2 — Choose a probability/statistical model (assumptions)

- For numeric outcomes: normal-error models are common starting points.
- For counts: Poisson-like models; for waiting times: exponential-like models.
- For categorical outcomes: contingency-table logic, classification models.

Connected notions: distributions, independence, conditioning.

### Step 3 — Estimate parameters and quantify uncertainty

- Point estimates ($\bar X$, regression coefficients).
- Standard errors (sampling distribution mindset).
- Confidence intervals to communicate uncertainty in parameters.

Connected notions: sampling distributions, CLT, $t/\chi^2/F$.

### Step 4 — Test hypotheses when decisions are needed

- Define $H_0$/$H_1$ and an acceptable Type I error rate $\alpha$.
- Use p-values as evidence *under $H_0$*.
- Consider power: if you can’t detect the effect you care about, redesign or collect more data.

Connected notions: Type I/II errors, power, CI/test duality.

### Step 5 — Model relationships (regression/ANOVA) and interpret

- Use regression to model $Y$ from multiple predictors.
- Use ANOVA to compare group means (which is regression with categorical predictors).
- Interpret partial effects cautiously (causal vs associational depends on design).

Connected notions: covariance/correlation, regression, ANOVA, dummy variables.

### Step 6 — Diagnose and iterate

- Check residual patterns (nonlinearity, heteroscedasticity, outliers).
- Check multicollinearity (VIF-like reasoning).
- If time-ordered, check autocorrelation; if present, use time-series-aware methods.

Connected notions: residuals, Durbin–Watson, stationarity.

### Step 7 — Forecasting (if the target is future values)

- Decompose series: level + trend + seasonality.
- Start simple (moving average/exponential smoothing), then increase complexity.
- Evaluate forecasts with held-out periods (time-respecting validation).

Connected notions: smoothing, seasonality, time dependence.

### Step 8 — Communicate results

- Report effect sizes + uncertainty (CI) rather than only “significant/not significant.”
- Separate **estimation**, **testing**, and **prediction** claims.
- Make assumptions explicit (normality, independence, stationarity, etc.).

---

## 17) “Which tool should I use?” (quick mapping)

- One numeric variable: summarize (mean/median/variance) + CI for $\mu$ if needed.
- Compare two numeric groups: CI/test for mean difference (two-sample $t$ logic), check variance assumptions if relevant.
- Compare many numeric groups: ANOVA (then post-hoc comparisons if overall difference is detected).
- Relationship between numeric variables: correlation (quick) → regression (model + uncertainty).
- Many predictors: multiple regression; check multicollinearity (VIF), do validated model selection if selecting variables.
- Categorical vs categorical: chi-square independence test; Fisher’s exact for small counts.
- Numeric over time: time plots; test/diagnose autocorrelation; forecasting via smoothing or more structured time-series models.
- Nonlinear patterns / interactions / mixed data types: CART as an interpretable baseline; ML (e.g., LSTM) if you have enough data and a prediction-first goal.

## Files scanned (in this folder)

- Scripts / data: `.Rhistory`, `QA_6.R`, `R script.R`, `class.csv`, `scores.csv`, `pups.csv`, `.RData` (binary)
- Lectures: `lecture_0_Introduction to the course.pdf`, `lecture_1_Probability_Theory_Introduction_Final.pdf`, …, `lecture_11_Time_Series_Analysis.pdf`
- Homeworks + solutions: `Homework_*.pdf`, `homework *.pdf`, `Sol_Quantitative Analysis2025_HW*.pdf`
- Reference tables: `chi-square-table.pdf`, `chisqtab.pdf`
- Spreadsheets: `lecture_5_Example.xlsx`, `Lecture 7_in class exercise.xlsx`, `Lecture 7_solution.xlsx`, `Lecture 9 example before class.xlsx`, `homework_QA_*.xlsx`, `Group Information.xlsx`, `Classeur1.xlsx`
- Archives: `Group Project..zip`, `Homework_*.zip`
