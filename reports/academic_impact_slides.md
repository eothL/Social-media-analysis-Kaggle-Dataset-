# Academic Impact (Yes/No): Hypothesis Testing Setup (H1–H4)

- Outcome: `Affects_Academic_Performance` (Yes = reports academic impact).
- Sample: `n=705` (Yes=453, No=252) → Yes rate `64.3%` (95% Wilson CI `[60.6%, 67.7%]`).
- Significance level: `α = 0.05`.
- Sources: `results/academic_impact/numeric_group_comparisons.csv`, `results/academic_impact/categorical_chi_square_summary.csv`.

**Numeric hypotheses (H1–H2): Welch two-sample t-tests**
- Example (for a numeric X):
  - `H0: μ_yes = μ_no`
  - `H1 (directional): μ_yes > μ_no` (H1 variables) or `μ_yes < μ_no` (H2 variables)
- Test statistic: `t = (x̄_yes − x̄_no) / sqrt(s_yes²/n_yes + s_no²/n_no)` (Welch df)
- Decision: reject `H0` if `p < α` (directional one-sided p-values)

**Categorical hypotheses (H3–H4): Chi-square independence tests**
- `H0: variable ⟂ academic impact` vs `H1: association exists`
- Report: `χ²(df)`, `p`, and effect size `Cramér’s V`

---

# H1: Higher usage, addiction, conflicts → higher P(Impact = Yes)

For each variable X ∈ {usage, addiction, conflicts}:
- `H0: μ_yes(X) = μ_no(X)`
- `H1: μ_yes(X) > μ_no(X)` (directional)
- Test: Welch two-sample t-test (one-sided)

| Variable | Test result | p-value | Effect size | Decision |
|---|---:|---:|---:|---|
| `Avg_Daily_Usage_Hours` | t(646.7)=25.39 | <1e-16 | d=1.84 | Reject `H0` |
| `Addicted_Score` | t(582.3)=47.81 | <1e-16 | d=3.61 | Reject `H0` |
| `Conflicts_Over_Social_Media` | t(623.9)=42.51 | <1e-16 | d=3.13 | Reject `H0` |

- Conclusion: evidence supports H1 (all three means are significantly higher in the Yes group).

---

# H2: Lower sleep hours and mental health → higher P(Impact = Yes)

For each variable X ∈ {sleep, mental health}:
- `H0: μ_yes(X) = μ_no(X)`
- `H1: μ_yes(X) < μ_no(X)` (directional)
- Test: Welch two-sample t-test (one-sided)

| Variable | Test result | p-value | Effect size | Decision |
|---|---:|---:|---:|---|
| `Sleep_Hours_Per_Night` | t(661.7)=−23.34 | <1e-16 | d=−1.67 | Reject `H0` |
| `Mental_Health_Score` | t(667.0)=−40.24 | <1e-16 | d=−2.87 | Reject `H0` |

- Conclusion: evidence supports H2 (sleep and mental health means are significantly lower in the Yes group).

---

# H3–H4: Platform/Relationship associated; Gender weak/no association

For each categorical variable C:
- `H0: C ⟂ academic impact` (independent)
- `H1: association exists`
- Test: χ² independence (α=0.05)

| Variable | Test result | p-value | Effect size | Decision |
|---|---:|---:|---:|---|
| `Most_Used_Platform` | χ²(6)=209.43 | 1.86e-42 | V=0.545 | Reject `H0` |
| `Relationship_Status` | χ²(2)=22.71 | 1.17e-05 | V=0.179 | Reject `H0` |
| `Gender` | χ²(1)=0.43 | 0.511 | V=0.025 | Fail to reject `H0` |

- Conclusion: H3 supported (platform + relationship status associated); H4 supported (gender weak/no association).
