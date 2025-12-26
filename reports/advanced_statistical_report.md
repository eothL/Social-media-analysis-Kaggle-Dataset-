# Advanced Statistical Analysis Report
Methodology: Standard Library Implementation (ANOVA, OLS, VIF)

## 1. One-Way ANOVA: Usage vs Academic Level
- **F-statistic**: 6.2655
- **p-value**: 2.0088e-03
- **Degrees of Freedom**: (2, 702)

**Group Means:**
- Undergraduate: 5.00 hours
- Graduate: 4.78 hours
- High School: 5.54 hours

**Conclusion**: The difference in usage across academic levels is **Significant**.

## 2. Linear Regression: Predicting Mental Health
- **R-squared**: 0.8954
- **Adj R-squared**: 0.8945

| Feature | Coef | SE | t | p-value |
|---|---|---|---|---|
| Intercept | 11.0673 | 0.3226 | 34.3029 | 5.6983e-152 |
| Usage_Hours | -0.0733 | 0.0216 | -3.3931 | 7.3004e-04 |
| Sleep_Hours | -0.0666 | 0.0207 | -3.2244 | 1.3213e-03 |
| Age | 0.0066 | 0.0113 | 0.5867 | 5.5756e-01 |
| Gender_Male | -0.0146 | 0.0313 | -0.4674 | 6.4035e-01 |
| Addiction_Score | -0.6451 | 0.0163 | -39.4564 | 5.3145e-180 |

> **Note**: This model includes 'Addiction_Score' to test its effect alongside Usage.

## 3. Multicollinearity Check (VIF)
| Feature | VIF |
|---|---|
| Usage_Hours | 4.04 |
| Sleep_Hours | 2.96 |
| Age | 1.37 |
| Gender_Male | 1.34 |
| Addiction_Score | 3.69 |

**Interpretation**:
- No severe multicollinearity detected (VIF < 5).