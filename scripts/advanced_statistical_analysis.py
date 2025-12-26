#!/usr/bin/env python3
"""
Advanced Statistical Analysis (ANOVA, OLS, VIF) - Stdlib Only.

Implements:
1. One-way ANOVA for Usage vs Academic Level.
2. Multiple Linear Regression (OLS) for Mental Health.
3. Variance Inflation Factor (VIF) analysis.
"""

import math
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Sequence, Optional

# --- Math & Stat Helpers ---

def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2: return 0.0
    m = mean(values)
    return sum((x - m)**2 for x in values) / (len(values) - 1)

def sample_std(values: Sequence[float]) -> float:
    return math.sqrt(sample_variance(values))

def dot(v1: List[float], v2: List[float]) -> float:
    return sum(x * y for x, y in zip(v1, v2))

def transpose(m: List[List[float]]) -> List[List[float]]:
    return list(map(list, zip(*m)))

def mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    Bt = transpose(B)
    return [[dot(row, col) for col in Bt] for row in A]

def _solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(a)
    # Copy A and b to avoid modification
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            # Singular matrix, return zeros or handle gracefully
            return [0.0] * n
        aug[col], aug[pivot] = aug[pivot], aug[col]

        inv_pivot = 1.0 / aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] *= inv_pivot

        for r in range(n):
            if r != col:
                factor = aug[r][col]
                for j in range(col, n + 1):
                    aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]

def _invert_matrix(a: List[List[float]]) -> List[List[float]]:
    n = len(a)
    cols = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        cols.append(_solve_linear_system(a, e))
    return transpose(cols)

# --- Probability Functions (Beta, F, T) ---

def betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta function."""
    MAXIT = 100
    EPS = 3.0e-7
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        del_val = d * c
        h *= del_val
        if abs(del_val - 1.0) < EPS: break
    return h

def betainc(x: float, a: float, b: float) -> float:
    """Regularized Incomplete Beta Function Ix(a,b)."""
    if x < 0.0 or x > 1.0: return 0.0
    if x == 0.0 or x == 1.0: return x
    
    # Logs of Gamma function
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    factor = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)
    
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * betacf(a, b, x) / a
    else:
        return 1.0 - factor * betacf(b, a, 1.0 - x) / b

def f_test_p_value(f_stat: float, df1: int, df2: int) -> float:
    """P-value for F-statistic (right-tailed)."""
    if f_stat <= 0: return 1.0
    x = df2 / (df2 + df1 * f_stat)
    return betainc(x, df2 / 2.0, df1 / 2.0)

def t_test_p_value(t_stat: float, df: int) -> float:
    """Two-tailed P-value for t-statistic."""
    if df <= 0: return 1.0
    t_abs = abs(t_stat)
    x = df / (df + t_abs**2)
    return betainc(x, df / 2.0, 0.5)

# --- Analysis Functions ---

def run_anova(data: List[Dict], group_col: str, value_col: str) -> Dict:
    groups = {}
    for row in data:
        g = row[group_col]
        try:
            val = float(row[value_col])
            groups.setdefault(g, []).append(val)
        except ValueError:
            continue
            
    all_values = [v for g_vals in groups.values() for v in g_vals]
    grand_mean = mean(all_values)
    
    ss_between = sum(len(vals) * (mean(vals) - grand_mean)**2 for vals in groups.values())
    ss_within = sum(sum((v - mean(vals))**2 for v in vals) for vals in groups.values())
    
    df_between = len(groups) - 1
    df_within = len(all_values) - len(groups)
    
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0
    
    f_stat = ms_between / ms_within if ms_within > 0 else 0.0
    p_val = f_test_p_value(f_stat, df_between, df_within)
    
    return {
        "groups": list(groups.keys()),
        "means": {g: mean(vals) for g, vals in groups.items()},
        "F": f_stat,
        "p_value": p_val,
        "df_between": df_between,
        "df_within": df_within
    }

def fit_ols(X: List[List[float]], y: List[float], feature_names: List[str]) -> Dict:
    n = len(y)
    p = len(X[0])
    
    # B = (Xt X)^-1 Xt y
    Xt = transpose(X)
    XtX = mat_mul(Xt, X)
    try:
        XtX_inv = _invert_matrix(XtX)
    except Exception:
        return {"error": "Singular Matrix"}
        
    Xty = [dot(row, y) for row in Xt]
    beta = [dot(row, Xty) for row in XtX_inv]
    
    # Predictions & Residuals
    y_pred = [dot(row, beta) for row in X]
    residuals = [yi - ypi for yi, ypi in zip(y, y_pred)]
    ssr = sum(r**2 for r in residuals)
    
    # R-squared
    y_mean = mean(y)
    sst = sum((yi - y_mean)**2 for yi in y)
    r_squared = 1.0 - (ssr / sst) if sst > 0 else 0.0
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - p - 1)
    
    # Standard Errors
    sigma2 = ssr / (n - p)
    se_beta = [math.sqrt(max(0, sigma2 * XtX_inv[j][j])) for j in range(p)]
    
    # t-stats & p-values
    results = []
    for j in range(p):
        se = se_beta[j]
        t = beta[j] / se if se > 0 else 0.0
        p_val = t_test_p_value(t, n - p)
        results.append({
            "feature": feature_names[j],
            "coef": beta[j],
            "se": se,
            "t": t,
            "p": p_val
        })
        
    return {
        "coefficients": results,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "sigma2": sigma2
    }

def calculate_vif(X: List[List[float]], feature_names: List[str]) -> List[Dict]:
    # X includes intercept at index 0
    # For each feature j (skip intercept), regress Xj on other X's
    vifs = []
    p = len(X[0])
    
    for j in range(1, p): # Skip intercept
        target_col = [row[j] for row in X]
        # Predictor cols: Intercept + all other features
        predictor_cols = [row[:j] + row[j+1:] for row in X]
        predictor_names = feature_names[:j] + feature_names[j+1:]
        
        # We only need R-squared
        res = fit_ols(predictor_cols, target_col, predictor_names)
        if "error" in res:
            vif = float('inf')
        else:
            r2 = res["r_squared"]
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
            
        vifs.append({"feature": feature_names[j], "VIF": vif})
        
    return vifs

def main():
    data_path = Path("Data/media_addiction.csv")
    out_path = Path("reports/advanced_statistical_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with data_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    report = []
    report.append("# Advanced Statistical Analysis Report")
    report.append("Methodology: Standard Library Implementation (ANOVA, OLS, VIF)")
    report.append("")
    
    # 1. ANOVA: Usage vs Academic Level
    report.append("## 1. One-Way ANOVA: Usage vs Academic Level")
    anova_res = run_anova(rows, "Academic_Level", "Avg_Daily_Usage_Hours")
    report.append(f"- **F-statistic**: {anova_res['F']:.4f}")
    report.append(f"- **p-value**: {anova_res['p_value']:.4e}")
    report.append(f"- **Degrees of Freedom**: ({anova_res['df_between']}, {anova_res['df_within']})")
    report.append("\n**Group Means:**")
    for g, m in anova_res["means"].items():
        report.append(f"- {g}: {m:.2f} hours")
    
    sig = "Significant" if anova_res['p_value'] < 0.05 else "Not Significant"
    report.append(f"\n**Conclusion**: The difference in usage across academic levels is **{sig}**.")
    report.append("")
    
    # 2. Linear Regression: Mental Health
    report.append("## 2. Linear Regression: Predicting Mental Health")
    # Features: Usage, Sleep, Age, Gender(Male=1)
    features = []
    target = []
    feature_names = ["Intercept", "Usage_Hours", "Sleep_Hours", "Age", "Gender_Male", "Addiction_Score"]
    
    for r in rows:
        try:
            y = float(r["Mental_Health_Score"])
            usage = float(r["Avg_Daily_Usage_Hours"])
            sleep = float(r["Sleep_Hours_Per_Night"])
            age = float(r["Age"])
            gender = 1.0 if r["Gender"] == "Male" else 0.0
            addiction = float(r["Addicted_Score"])
            
            target.append(y)
            features.append([1.0, usage, sleep, age, gender, addiction])
        except ValueError:
            continue
            
    ols_res = fit_ols(features, target, feature_names)
    
    if "error" in ols_res:
        report.append(f"Error fitting model: {ols_res['error']}")
    else:
        report.append(f"- **R-squared**: {ols_res['r_squared']:.4f}")
        report.append(f"- **Adj R-squared**: {ols_res['adj_r_squared']:.4f}")
        report.append("\n| Feature | Coef | SE | t | p-value |")
        report.append("|---|---|---|---|---|")
        for row in ols_res["coefficients"]:
            report.append(f"| {row['feature']} | {row['coef']:.4f} | {row['se']:.4f} | {row['t']:.4f} | {row['p']:.4e} |")
            
    report.append("")
    report.append("> **Note**: This model includes 'Addiction_Score' to test its effect alongside Usage.")
    report.append("")

    # 3. VIF Analysis
    report.append("## 3. Multicollinearity Check (VIF)")
    vifs = calculate_vif(features, feature_names)
    report.append("| Feature | VIF |")
    report.append("|---|---|")
    for row in vifs:
        report.append(f"| {row['feature']} | {row['VIF']:.2f} |")
        
    report.append("\n**Interpretation**:")
    high_vif = [r for r in vifs if r['VIF'] > 5.0]
    if high_vif:
        report.append(f"- High multicollinearity detected in: {', '.join(r['feature'] for r in high_vif)}.")
        report.append("- This confirms that Usage and Addiction are strongly entangled, making their individual regression coefficients unstable.")
    else:
        report.append("- No severe multicollinearity detected (VIF < 5).")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print(f"Analysis complete. Report saved to {out_path}")

if __name__ == "__main__":
    main()
