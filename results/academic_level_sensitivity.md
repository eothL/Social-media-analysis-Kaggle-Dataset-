# Academic Level Sample Size Check

## Group sizes
- High School: 27
- Undergraduate: 353
- Graduate: 325

## 1) Confidence intervals for group means (95%)
Normal-approximate 95% CIs for each outcome by academic level:

| Outcome | High School (mean, CI) | Undergraduate (mean, CI) | Graduate (mean, CI) |
|---|---|---|---|
| Avg_Daily_Usage_Hours | 5.54 [5.28, 5.80] | 5.00 [4.87, 5.14] | 4.78 [4.64, 4.91] |
| Addicted_Score | 8.04 [7.61, 8.46] | 6.49 [6.32, 6.66] | 6.24 [6.08, 6.40] |
| Mental_Health_Score | 5.11 [4.89, 5.33] | 6.18 [6.06, 6.30] | 6.37 [6.26, 6.49] |
| Sleep_Hours_Per_Night | 5.46 [5.30, 5.61] | 6.83 [6.70, 6.95] | 7.03 [6.92, 7.14] |
| Conflicts_Over_Social_Media | 3.74 [3.43, 4.05] | 2.92 [2.82, 3.01] | 2.70 [2.60, 2.81] |

Interpretation: High School intervals are wider due to n=27; treat HS comparisons as exploratory.

## 2) Sensitivity check: correlations with and without High School
If correlations are stable after excluding High School, results are less sensitive to the small group.

| Pair | All students (r) | Excluding High School (r) | Δ |
|---|---|---|---|
| Avg_Daily_Usage_Hours vs Mental_Health_Score | -0.801 | -0.801 | +0.000 |
| Avg_Daily_Usage_Hours vs Sleep_Hours_Per_Night | -0.791 | -0.794 | -0.004 |
| Avg_Daily_Usage_Hours vs Conflicts_Over_Social_Media | 0.805 | 0.804 | -0.000 |
| Addicted_Score vs Mental_Health_Score | -0.945 | -0.944 | +0.001 |
| Addicted_Score vs Sleep_Hours_Per_Night | -0.765 | -0.753 | +0.012 |
| Addicted_Score vs Conflicts_Over_Social_Media | 0.934 | 0.932 | -0.001 |

Interpretation: Small deltas indicate that excluding HS does not materially change the main relationships.

## 3) Effect sizes (Cohen’s d) for High School vs other levels
Magnitude guide: ~0.2 small, ~0.5 medium, ~0.8 large (direction shows HS relative to comparison).

| Outcome | HS vs Undergraduate (d) | HS vs Graduate (d) |
|---|---|---|
| Avg_Daily_Usage_Hours | 0.43 | 0.64 |
| Addicted_Score | 0.96 | 1.22 |
| Mental_Health_Score | -0.98 | -1.22 |
| Sleep_Hours_Per_Night | -1.19 | -1.63 |
| Conflicts_Over_Social_Media | 0.90 | 1.09 |

Interpretation: Use effect sizes rather than p-values given the small HS sample.