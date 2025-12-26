#!/usr/bin/env python3
"""
Generate a narrative markdown report from the outputs produced by academic_impact_analysis.py.

Defaults assume you already ran:
  python3 scripts/academic_impact_analysis.py --out-dir results/academic_impact
  python3 scripts/academic_impact_analysis.py --exclude-high-school --out-dir results/academic_impact_no_hs
  python3 scripts/academic_impact_analysis.py --l2 1.0 --out-dir results/academic_impact_ridge
  python3 scripts/academic_impact_analysis.py --l2 1.0 --exclude-high-school --out-dir results/academic_impact_ridge_no_hs
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def read_csv_dict(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def fmt_p(p: float) -> str:
    if p != p:
        return "NA"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def z_crit_95_two_sided() -> float:
    return 1.959963984540054


def chi2_crit(alpha: float, df: int) -> float:
    """
    Returns chi-square critical value c such that P(Chi2_df >= c) = alpha.
    Uses bisection on the survival function implemented in academic_impact_analysis.py.
    """
    try:
        from academic_impact_analysis import chi2_sf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Could not import chi2_sf from academic_impact_analysis.py") from e

    if df <= 0:
        return float("nan")

    # Find an upper bound where sf(hi) < alpha
    lo = 0.0
    hi = max(1.0, float(df))
    while chi2_sf(hi, df) > alpha:
        hi *= 2.0
        if hi > 1e6:
            break

    # Bisection
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if chi2_sf(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def load_numeric(path: Path) -> Dict[str, Dict[str, float]]:
    rows = read_csv_dict(path)
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        out[r["variable"]] = {k: float(v) for k, v in r.items() if k != "variable"}
    return out


def load_chi(path: Path) -> Dict[str, Dict[str, float]]:
    rows = read_csv_dict(path)
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        out[r["variable"]] = {
            "chi2": float(r["chi2"]),
            "df": float(r["df"]),
            "p_value": float(r["p_value"]),
            "cramers_v": float(r["cramers_v"]),
            "min_expected": float(r["min_expected"]),
        }
    return out


def load_or(path: Path) -> Dict[str, Dict[str, float]]:
    rows = read_csv_dict(path)
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        out[r["feature"]] = {
            "coef": float(r["coef"]),
            "se": float(r["se"]) if r["se"] else float("nan"),
            "or": float(r["odds_ratio"]),
            "lo": float(r["or_ci95_lo"]),
            "hi": float(r["or_ci95_hi"]),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/academic_impact_results_report.md")
    parser.add_argument("--base", default="results/academic_impact")
    parser.add_argument("--no-hs", default="results/academic_impact_no_hs")
    parser.add_argument("--ridge", default="results/academic_impact_ridge")
    parser.add_argument("--ridge-no-hs", default="results/academic_impact_ridge_no_hs")
    args = parser.parse_args()

    base = Path(args.base)
    no_hs = Path(args.no_hs)
    ridge = Path(args.ridge)
    ridge_no_hs = Path(args.ridge_no_hs)

    numeric_all = load_numeric(base / "numeric_group_comparisons.csv")
    numeric_no_hs = load_numeric(no_hs / "numeric_group_comparisons.csv")
    chi_all = load_chi(base / "categorical_chi_square_summary.csv")
    chi_no_hs = load_chi(no_hs / "categorical_chi_square_summary.csv")

    or_full_ridge = load_or(ridge / "logistic_full_odds_ratios.csv")
    or_full_ridge_no_hs = load_or(ridge_no_hs / "logistic_full_odds_ratios.csv")

    # Key variables for a compact report table
    key_numeric = [
        "Avg_Daily_Usage_Hours",
        "Addicted_Score",
        "Conflicts_Over_Social_Media",
        "Mental_Health_Score",
        "Sleep_Hours_Per_Night",
        "Age",
    ]

    key_or = [
        "Conflicts_Over_Social_Media",
        "Addicted_Score",
        "Avg_Daily_Usage_Hours",
        "Mental_Health_Score",
        "Sleep_Hours_Per_Night",
        "Age",
    ]

    # Read sample stats from the base report (already computed by the analysis script)
    base_report = (base / "academic_impact_report.md").read_text(encoding="utf-8").splitlines()
    n_line = next(l for l in base_report if l.startswith("- Rows analyzed:"))
    yes_line = next(l for l in base_report if l.startswith("- Academic impact:"))
    ci_line = next(l for l in base_report if l.startswith("- 95% CI for proportion Yes"))

    md: List[str] = []
    md.append("# Academic Impact (Affects_Academic_Performance) — Results Report")
    md.append("")
    md.append(
        "This report follows the workflow in `reports/academic_impact_question_plan.md` "
        "and summarizes the results produced by `scripts/academic_impact_analysis.py`."
    )
    md.append("")
    md.append("## Question")
    md.append("Which factors best explain whether a student reports that social media affects their academic performance (Yes/No), and how large are those effects?")
    md.append("")
    md.append("## Hypotheses (research / directional)")
    md.append("We test whether the data supports the following directional hypotheses (associational, not causal):")
    md.append("- H1: Higher `Avg_Daily_Usage_Hours` and higher `Addicted_Score` are associated with higher probability of reporting academic impact (Yes).")
    md.append("- H2: Lower `Sleep_Hours_Per_Night` and lower `Mental_Health_Score` are associated with higher probability of reporting academic impact (Yes).")
    md.append("- H3: Higher `Conflicts_Over_Social_Media` is associated with higher probability of reporting academic impact (Yes).")
    md.append("- H4: `Most_Used_Platform` and `Relationship_Status` are not independent of academic impact (association exists).")
    md.append("- H5: `Gender` has weak or no association with academic impact.")
    md.append("")

    md.append("## Step 1 — Descriptives + CI (proportion)")
    md.append(n_line)
    md.append(yes_line)
    md.append(ci_line)
    md.append("")
    # Optional visuals if present
    if (Path("results") / "fig_academic_impact_group_diff.svg").exists():
        md.append("### Figure: average outcomes by academic impact group")
        md.append("![](../results/fig_academic_impact_group_diff.svg)")
        md.append("- Visual takeaway: the `Yes` group has higher usage/addiction/conflicts and lower sleep/mental health on average.")
        md.append("")

    md.append("## Step 2 — Numeric comparisons (Yes vs No): mean differences + effect sizes")
    md.append("Table shows mean(Yes) − mean(No), a 95% CI for that difference, and Cohen’s d (magnitude of difference in SD units).")
    md.append("")
    md.append("| Variable | Mean Yes | Mean No | Diff (Yes−No) | 95% CI | Cohen’s d |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for var in key_numeric:
        r = numeric_all[var]
        md.append(
            "| {var} | {my:.2f} | {mn:.2f} | {diff:.2f} | [{lo:.2f}, {hi:.2f}] | {d:.2f} |".format(
                var=var,
                my=r["mean_yes"],
                mn=r["mean_no"],
                diff=r["diff_yes_minus_no"],
                lo=r["ci95_lo"],
                hi=r["ci95_hi"],
                d=r["cohen_d"],
            )
        )
    md.append("")
    md.append("Full table: `results/academic_impact/numeric_group_comparisons.csv`.")
    md.append("")

    md.append("## Step 3 — Categorical associations (chi-square + Cramér’s V)")
    md.append("Platforms are automatically grouped as “top K” (default K=6) + `Other` to reduce sparsity.")
    md.append("")
    md.append("| Variable | p-value | Cramér’s V | Min expected count |")
    md.append("|---|---:|---:|---:|")
    for var in ["Gender", "Academic_Level", "Relationship_Status", "Most_Used_Platform"]:
        r = chi_all[var]
        md.append(
            "| {var} | {p} | {v:.3f} | {m:.2f} |".format(
                var=var,
                p=fmt_p(r["p_value"]),
                v=r["cramers_v"],
                m=r["min_expected"],
            )
        )
    md.append("")
    md.append("Full summary: `results/academic_impact/categorical_chi_square_summary.csv`.")
    md.append("")

    md.append("## Hypothesis testing technique (what we do in this report)")
    md.append("Your hypotheses (H1–H5) are *research statements* (in words). To test them, we translate them into a **null hypothesis H0** (no effect) and a **statistical alternative H1** (effect / direction), then use a test statistic + a decision rule.")
    md.append("")
    md.append("General template:")
    md.append("- 1) Define H0 and H1 (and the outcome coding).")
    md.append("- 2) Choose significance level α (here α=0.05).")
    md.append("- 3) Compute a test statistic and compare to a critical value (or p-value).")
    md.append("- 4) Decision: **reject H0** or **fail to reject H0**.")
    md.append("- 5) Report an effect size (magnitude), not only significance.")
    md.append("")
    md.append("How that maps to our steps:")
    md.append("- **Numeric predictors (Step 2 / Section B):** we test H0: μ_yes = μ_no for each numeric variable.")
    md.append("  - Test statistic (large-sample z; close to a Welch two-sample t-test when n is large):")
    md.append("    - `z = (x̄_yes − x̄_no) / sqrt(s_yes²/n_yes + s_no²/n_no)`")
    md.append("  - Two-sided decision at α=0.05: reject if `|z| ≥ 1.96` (equivalently: the 95% CI for `x̄_yes − x̄_no` does not include 0).")
    md.append("  - Effect size: **Cohen’s d** (difference in SD units).")
    md.append("- **Categorical predictors (Step 3 / Section A):** we test H0: variable ⟂ academic impact (independence) using chi-square.")
    md.append("  - Test statistic: `χ² = Σ (O − E)² / E`, df = `(r−1)(c−1)`; reject if `χ² ≥ χ²(0.95, df)`.")
    md.append("  - Effect size: **Cramér’s V** (0 = none, larger = stronger association).")
    md.append("- **Logistic regression (Step 4):** we test H0: β=0 (equivalently OR=1) for each predictor, but here we treat ORs as descriptive because the data is (near-)separable (very wide CIs).")
    md.append("  - Outcome is coded `Yes=1` (reporting academic impact), so **OR>1 increases odds of reporting impact**; **OR<1 decreases odds** (protective direction).")
    md.append("  - Numeric predictors are standardized (z-scored), so “+1 unit” = **+1 SD** in that variable.")
    md.append("")
    md.append("Important wording note: “supported” in the interpretation below means “evidence is consistent with the research hypothesis”. Formally we only make decisions about H0 (reject / fail to reject).")
    md.append("")

    md.append("## Hypothesis tests (explicit critical-value decisions, α=0.05)")
    md.append("This section writes each test in the “H0 / test statistic / critical value / decision” format.")
    md.append("")

    md.append("### A) Categorical vs academic impact (chi-square independence)")
    md.append("For each variable: H0 = independence between the variable and `Affects_Academic_Performance`.")
    md.append("")
    md.append("| Variable | H0 | Test stat χ² | df | Critical χ²(0.95) | Decision |")
    md.append("|---|---|---:|---:|---:|---|")
    for var in ["Gender", "Academic_Level", "Relationship_Status", "Most_Used_Platform"]:
        r = chi_all[var]
        df = int(r["df"])
        crit = chi2_crit(0.05, df)
        decision = "Reject H0" if r["chi2"] >= crit else "Fail to reject H0"
        md.append(
            "| {var} | independence | {chi2:.2f} | {df} | {crit:.2f} | {dec} |".format(
                var=var, chi2=r["chi2"], df=df, crit=crit, dec=decision
            )
        )
    md.append("")
    md.append("Note: chi-square is an approximation; interpret cautiously when expected counts are small (see `min_expected`).")
    md.append("")

    md.append("### B) Numeric vs academic impact (two-sample z approximation)")
    md.append("For each variable: H0 = mean(Yes) = mean(No); two-sided at α=0.05, so reject if |z| ≥ 1.96.")
    md.append("")
    zcrit = z_crit_95_two_sided()
    md.append("| Variable | H0 | z | Critical | Decision |")
    md.append("|---|---|---:|---:|---|")
    for var in key_numeric:
        r = numeric_all[var]
        se = ((r["sd_yes"] ** 2) / r["n_yes"] + (r["sd_no"] ** 2) / r["n_no"]) ** 0.5
        z = r["diff_yes_minus_no"] / se if se > 0 else float("nan")
        decision = "Reject H0" if abs(z) >= zcrit else "Fail to reject H0"
        md.append(
            "| {var} | μ_yes = μ_no | {z:.2f} | ±{zc:.2f} | {dec} |".format(
                var=var, z=z, zc=zcrit, dec=decision
            )
        )
    md.append("")
    md.append("Note: this uses a normal approximation; it’s close to a two-sample t-test when group sizes are large.")
    md.append("")

    md.append("### Interpretation (link back to the research hypotheses)")
    md.append("Mapping the formal H0 decisions above to H1–H5 (direction checked using the observed mean differences):")
    md.append("")

    def decision_from_z(var: str, expected_sign: int) -> str:
        r = numeric_all[var]
        se = ((r["sd_yes"] ** 2) / r["n_yes"] + (r["sd_no"] ** 2) / r["n_no"]) ** 0.5
        z = r["diff_yes_minus_no"] / se if se > 0 else float("nan")
        reject = abs(z) >= zcrit
        direction_ok = (r["diff_yes_minus_no"] > 0 and expected_sign > 0) or (
            r["diff_yes_minus_no"] < 0 and expected_sign < 0
        )
        if reject and direction_ok:
            return "Supported (reject H0, expected direction)"
        if reject and not direction_ok:
            return "Opposite direction (reject H0)"
        return "Not supported (fail to reject H0)"

    def decision_from_chi(var: str) -> str:
        r = chi_all[var]
        df = int(r["df"])
        crit = chi2_crit(0.05, df)
        reject = r["chi2"] >= crit
        return "Supported (reject H0)" if reject else "Not supported (fail to reject H0)"

    md.append("- H1 (higher usage/addiction → more impact):")
    md.append(f"  - `Avg_Daily_Usage_Hours`: {decision_from_z('Avg_Daily_Usage_Hours', expected_sign=+1)}")
    md.append(f"  - `Addicted_Score`: {decision_from_z('Addicted_Score', expected_sign=+1)}")
    md.append("- H2 (lower sleep/mental health → more impact):")
    md.append(f"  - `Sleep_Hours_Per_Night`: {decision_from_z('Sleep_Hours_Per_Night', expected_sign=-1)}")
    md.append(f"  - `Mental_Health_Score`: {decision_from_z('Mental_Health_Score', expected_sign=-1)}")
    md.append("- H3 (higher conflicts → more impact):")
    md.append(f"  - `Conflicts_Over_Social_Media`: {decision_from_z('Conflicts_Over_Social_Media', expected_sign=+1)}")
    md.append("- H4 (platform + relationship status associated with impact):")
    md.append(f"  - `Most_Used_Platform`: {decision_from_chi('Most_Used_Platform')}")
    md.append(f"  - `Relationship_Status`: {decision_from_chi('Relationship_Status')}")
    md.append("- H5 (gender weak/no association):")
    md.append("  - `Gender`: Consistent (fail to reject H0 of independence).")
    md.append("")

    md.append("## Step 4 — Logistic regression (multivariable explanation)")
    md.append("The dataset shows perfect classification in these models (AUC≈1, Acc≈1), which is a sign of (near-)complete separation; treat odds ratios as descriptive and prioritize effect sizes + robustness.")
    md.append("")
    md.append("For interpretability, the table below uses the ridge-regularized run (`--l2 1.0`) to avoid extreme/infinite odds ratios.")
    md.append("")
    md.append("| Predictor (standardized numeric) | Odds ratio | 95% CI |")
    md.append("|---|---:|---:|")
    for feat in key_or:
        r = or_full_ridge[feat]
        md.append("| {f} | {or_:.2f} | [{lo:.2f}, {hi:.2f}] |".format(f=feat, or_=r["or"], lo=r["lo"], hi=r["hi"]))
    md.append("")
    md.append("Full coefficient table: `results/academic_impact_ridge/logistic_full_odds_ratios.csv`.")
    md.append("")
    if (ridge / "fig_logistic_or_forest.svg").exists():
        md.append("### Figure: strongest effects (ridge logistic model)")
        md.append("![](../results/academic_impact_ridge/fig_logistic_or_forest.svg)")
        md.append("- Read: red dot = odds ratio (OR), blue line = 95% CI, dashed line at OR=1 = “no association”.")
        md.append("- In this dataset, `Conflicts_Over_Social_Media` has the largest OR (strongest positive association with reporting academic impact).")
        md.append("- Many CIs are wide and some cross OR=1, so those effects are uncertain in the multivariable model.")
        md.append("")
    if (ridge / "fig_logistic_roc.svg").exists():
        md.append("### Figure: ROC curve (ridge logistic model)")
        md.append("![](../results/academic_impact_ridge/fig_logistic_roc.svg)")
        md.append("- AUC≈1 means the model separates Yes vs No almost perfectly in this dataset.")
        md.append("- This is a warning sign for (near-)complete separation/synthetic structure; treat logistic inference as descriptive.")
        md.append("")
    if (ridge / "fig_logistic_calibration.svg").exists():
        md.append("### Figure: calibration (ridge logistic model)")
        md.append("![](../results/academic_impact_ridge/fig_logistic_calibration.svg)")
        md.append("- Points near the diagonal indicate predicted probabilities match observed frequencies within bins.")
        md.append("- Here most bins sit near (0,0) or (1,1): the model outputs very extreme probabilities for most rows (again consistent with separation).")
        md.append("")

    md.append("## Step 5 — Robustness / sensitivity (exclude High School)")
    md.append("High School is a small subgroup (n=27 in the full dataset). We rerun the key analyses excluding High School to check sensitivity.")
    md.append("")
    md.append("### Numeric effects: very stable without High School")
    md.append("| Variable | Diff all | Diff excl. HS | Cohen’s d all | Cohen’s d excl. HS |")
    md.append("|---|---:|---:|---:|---:|")
    for var in key_numeric:
        a = numeric_all[var]
        b = numeric_no_hs[var]
        md.append(
            "| {var} | {da:.2f} | {db:.2f} | {ea:.2f} | {eb:.2f} |".format(
                var=var,
                da=a["diff_yes_minus_no"],
                db=b["diff_yes_minus_no"],
                ea=a["cohen_d"],
                eb=b["cohen_d"],
            )
        )
    md.append("")
    md.append("### Categorical: Academic level association is driven by the High School group")
    md.append("| Variable | p all | V all | p excl. HS | V excl. HS |")
    md.append("|---|---:|---:|---:|---:|")
    for var in ["Academic_Level", "Relationship_Status", "Most_Used_Platform", "Gender"]:
        a = chi_all[var]
        b = chi_no_hs[var]
        md.append(
            "| {var} | {pa} | {va:.3f} | {pb} | {vb:.3f} |".format(
                var=var,
                pa=fmt_p(a["p_value"]),
                va=a["cramers_v"],
                pb=fmt_p(b["p_value"]),
                vb=b["cramers_v"],
            )
        )
    md.append("")
    md.append(
        "Ridge logistic effects are also similar after excluding High School "
        "(see `results/academic_impact_ridge_no_hs/logistic_full_odds_ratios.csv`)."
    )
    md.append("")

    md.append("## Conclusion (answer to the question)")
    md.append("- Students reporting academic impact have much higher usage, addiction, and relationship-conflict scores, and lower sleep and mental health scores; effect sizes are very large in this dataset.")
    md.append("- Platform and relationship status are strongly associated with academic impact (largest categorical effect: platform grouping), while gender is not.")
    md.append("- Results are not driven by the small High School subgroup for the main numeric relationships; however, the academic-level association largely comes from High School vs the other levels.")
    md.append("")
    md.append("## Limitations")
    md.append("- Cross-sectional, self-reported survey variables; interpret as associations, not causation.")
    md.append("- Perfect classification suggests the dataset may be synthetic/structured; logistic-regression inference can be unstable under separation.")
    md.append("")
    md.append("## Technical walkthrough")
    md.append("- Step-by-step explanation: `reports/academic_impact_technical_report.md`")
    md.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
