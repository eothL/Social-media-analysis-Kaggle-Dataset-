#!/usr/bin/env python3
"""
Academic impact analysis (stdlib-only).

Creates:
- `results/academic_impact/academic_impact_report.md`
- CSV tables for group comparisons, chi-square summaries, and logistic ORs
- An SVG forest plot for the main logistic model

Run:
  python3 scripts/academic_impact_analysis.py
  python3 scripts/academic_impact_analysis.py --exclude-high-school
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


Z_95 = 1.959963984540054  # two-sided 95% normal critical value


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def safe_log(value: float, eps: float = 1e-15) -> float:
    return math.log(max(value, eps))


def exp_or_inf(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf") if value > 0 else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return sum((v - avg) ** 2 for v in values) / (len(values) - 1)


def sample_std(values: Sequence[float]) -> float:
    return math.sqrt(sample_variance(values))


def wilson_ci_95(successes: int, total: int) -> Tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denom = 1.0 + (Z_95**2) / total
    center = (phat + (Z_95**2) / (2.0 * total)) / denom
    half = (
        Z_95
        * math.sqrt((phat * (1.0 - phat) / total) + (Z_95**2) / (4.0 * total**2))
        / denom
    )
    return (max(0.0, center - half), min(1.0, center + half))


def diff_in_means_ci_95(
    group_yes: Sequence[float], group_no: Sequence[float]
) -> Tuple[float, float, float]:
    """
    Returns (diff, lo, hi) for diff = mean(Yes) - mean(No),
    using a normal approximation CI.
    """
    diff = mean(group_yes) - mean(group_no)
    se = math.sqrt(
        sample_variance(group_yes) / max(1, len(group_yes))
        + sample_variance(group_no) / max(1, len(group_no))
    )
    return (diff, diff - Z_95 * se, diff + Z_95 * se)


def cohen_d(group_yes: Sequence[float], group_no: Sequence[float]) -> float:
    n_yes = len(group_yes)
    n_no = len(group_no)
    if n_yes < 2 or n_no < 2:
        return float("nan")
    var_yes = sample_variance(group_yes)
    var_no = sample_variance(group_no)
    pooled = math.sqrt(((n_yes - 1) * var_yes + (n_no - 1) * var_no) / (n_yes + n_no - 2))
    if pooled == 0:
        return 0.0
    return (mean(group_yes) - mean(group_no)) / pooled


# --- Special functions for chi-square p-values (no scipy) ---


def _regularized_gamma_q(a: float, x: float) -> float:
    """
    Regularized upper incomplete gamma Q(a, x) = Γ(a, x) / Γ(a).
    Adapted from Numerical Recipes (series/continued fraction switch).
    """
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0:
        return 1.0

    if x < a + 1.0:
        # Series for P(a, x), then Q = 1 - P.
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(200):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        log_prefactor = -x + a * math.log(x) - math.lgamma(a)
        p = total * math.exp(log_prefactor)
        return max(0.0, min(1.0, 1.0 - p))

    # Continued fraction for Q(a, x)
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    prefactor = math.exp(log_prefactor)
    b0 = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b0
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b0 += 2.0
        d = an * d + b0
        if abs(d) < 1e-30:
            d = 1e-30
        c = b0 + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    q = prefactor * h
    return max(0.0, min(1.0, q))


def chi2_sf(chi2: float, df: int) -> float:
    if df <= 0:
        return float("nan")
    return _regularized_gamma_q(df / 2.0, chi2 / 2.0)


def cramer_v(chi2: float, n: int, r: int, c: int) -> float:
    if n <= 0:
        return float("nan")
    denom = n * max(1, min(r - 1, c - 1))
    return math.sqrt(max(0.0, chi2 / denom))


def chi_square_independence(counts: List[List[int]]) -> Tuple[float, int, float, float]:
    """
    Returns (chi2, df, p_value, min_expected).
    """
    r = len(counts)
    c = len(counts[0]) if counts else 0
    n = sum(sum(row) for row in counts)
    row_totals = [sum(row) for row in counts]
    col_totals = [sum(counts[i][j] for i in range(r)) for j in range(c)]

    chi2 = 0.0
    min_expected = float("inf")
    for i in range(r):
        for j in range(c):
            expected = (row_totals[i] * col_totals[j]) / n if n else 0.0
            min_expected = min(min_expected, expected)
            if expected > 0:
                chi2 += (counts[i][j] - expected) ** 2 / expected
    df = (r - 1) * (c - 1)
    return chi2, df, chi2_sf(chi2, df), min_expected


# --- Logistic regression (IRLS) ---


@dataclass
class LogisticFit:
    feature_names: List[str]
    coefficients: List[float]
    standard_errors: Optional[List[float]]
    log_likelihood: float
    converged: bool
    iterations: int

    def predict_proba(self, x: Sequence[float]) -> float:
        eta = sum(b * xj for b, xj in zip(self.coefficients, x))
        return sigmoid(eta)


def _solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting (copies inputs)."""
    n = len(a)
    aug = [row[:] + [b_i] for row, b_i in zip(a, b)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]

        inv_pivot = 1.0 / aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] *= inv_pivot

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def _invert_matrix(a: List[List[float]]) -> List[List[float]]:
    n = len(a)
    cols: List[List[float]] = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        cols.append(_solve_linear_system(a, e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def fit_logistic_regression_irls(
    x_rows: List[List[float]],
    y: List[int],
    feature_names: List[str],
    l2_penalty: float = 0.0,
    max_iter: int = 60,
    tol: float = 1e-8,
) -> LogisticFit:
    n = len(x_rows)
    p = len(x_rows[0]) if x_rows else 0
    beta = [0.0] * p

    def penalty(j: int) -> float:
        return 0.0 if feature_names[j] == "Intercept" else l2_penalty

    last_ll = -float("inf")
    converged = False
    iterations = 0
    for it in range(1, max_iter + 1):
        iterations = it
        probs: List[float] = []
        weights: List[float] = []
        z: List[float] = []
        ll = 0.0
        for xi, yi in zip(x_rows, y):
            eta = sum(b * xj for b, xj in zip(beta, xi))
            pi = sigmoid(eta)
            probs.append(pi)
            wi = max(pi * (1.0 - pi), 1e-9)
            weights.append(wi)
            z.append(eta + (yi - pi) / wi)
            ll += yi * safe_log(pi) + (1 - yi) * safe_log(1.0 - pi)
        ll -= 0.5 * sum(penalty(j) * beta[j] ** 2 for j in range(p))

        if abs(ll - last_ll) < tol:
            converged = True
            break
        last_ll = ll

        h = [[0.0 for _ in range(p)] for _ in range(p)]
        rhs = [0.0 for _ in range(p)]
        for wi, zi, xi in zip(weights, z, x_rows):
            for j in range(p):
                rhs[j] += xi[j] * wi * zi
                for k in range(p):
                    h[j][k] += xi[j] * wi * xi[k]
        for j in range(p):
            h[j][j] += penalty(j)

        beta_new = _solve_linear_system(h, rhs)
        max_step = max(abs(bn - b) for bn, b in zip(beta_new, beta)) if p else 0.0
        beta = beta_new
        if max_step < tol:
            converged = True
            break

    # Standard errors from inverse Fisher information (normal approximation).
    standard_errors: Optional[List[float]] = None
    try:
        probs = [sigmoid(sum(b * xj for b, xj in zip(beta, xi))) for xi in x_rows]
        weights = [max(pi * (1.0 - pi), 1e-9) for pi in probs]
        info = [[0.0 for _ in range(p)] for _ in range(p)]
        for wi, xi in zip(weights, x_rows):
            for j in range(p):
                for k in range(p):
                    info[j][k] += xi[j] * wi * xi[k]
        for j in range(p):
            info[j][j] += 1e-9
        inv = _invert_matrix(info)
        standard_errors = [math.sqrt(max(0.0, inv[j][j])) for j in range(p)]
    except Exception:
        standard_errors = None

    final_ll = 0.0
    for xi, yi in zip(x_rows, y):
        pi = sigmoid(sum(b * xj for b, xj in zip(beta, xi)))
        final_ll += yi * safe_log(pi) + (1 - yi) * safe_log(1.0 - pi)

    return LogisticFit(
        feature_names=feature_names,
        coefficients=beta,
        standard_errors=standard_errors,
        log_likelihood=final_ll,
        converged=converged,
        iterations=iterations,
    )


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    paired = sorted(zip(y_score, y_true), key=lambda t: t[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    rank = 1
    sum_ranks_pos = 0.0
    i = 0
    while i < len(paired):
        j = i
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            if paired[k][1] == 1:
                sum_ranks_pos += avg_rank
        rank += (j - i)
        i = j

    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


# --- Minimal SVG charting (no external deps) ---


def _svg_header(width: int, height: int) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>"
    )


def _svg_footer() -> str:
    return "</svg>\n"


def save_roc_curve_svg(
    y_true: Sequence[int],
    y_score: Sequence[float],
    path: Path,
    title: str,
) -> None:
    width, height = 640, 520
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    pairs = list(zip(y_score, y_true))
    pairs.sort(reverse=True)
    p = sum(y_true)
    n = len(y_true) - p
    if p == 0 or n == 0:
        return

    tpr_points: List[Tuple[float, float]] = []
    tp = fp = 0
    last_score: Optional[float] = None
    for score, label in pairs:
        if last_score is None or score != last_score:
            tpr = tp / p
            fpr = fp / n
            tpr_points.append((fpr, tpr))
            last_score = score
        if label == 1:
            tp += 1
        else:
            fp += 1
    tpr_points.append((fp / n, tp / p))

    def sx(x: float) -> float:
        return margin + x * plot_w

    def sy(y: float) -> float:
        return margin + (1.0 - y) * plot_h

    auc = roc_auc(y_true, y_score)

    svg: List[str] = []
    svg.append(_svg_header(width, height))
    svg.append("<rect width='100%' height='100%' fill='#ffffff' />")
    svg.append(
        f"<text x='{width/2:.1f}' y='28' text-anchor='middle' font-family='Georgia, serif' "
        f"font-size='16' fill='#222'>{title}</text>"
    )
    svg.append(
        f"<text x='{width/2:.1f}' y='48' text-anchor='middle' font-family='Georgia, serif' "
        f"font-size='12' fill='#444'>AUC = {auc:.3f}</text>"
    )

    # Axes
    svg.append(f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#222' />")
    svg.append(f"<line x1='{margin}' y1='{height - margin}' x2='{margin}' y2='{margin}' stroke='#222' />")
    svg.append(
        f"<text x='{width/2:.1f}' y='{height - 20}' text-anchor='middle' font-family='Georgia, serif' font-size='12' fill='#222'>False Positive Rate</text>"
    )
    svg.append(
        f"<text x='18' y='{height/2:.1f}' text-anchor='middle' font-family='Georgia, serif' font-size='12' fill='#222' transform='rotate(-90 18 {height/2:.1f})'>True Positive Rate</text>"
    )

    # Diagonal baseline
    svg.append(f"<line x1='{sx(0):.2f}' y1='{sy(0):.2f}' x2='{sx(1):.2f}' y2='{sy(1):.2f}' stroke='#bbb' stroke-dasharray='4 4' />")

    # ROC polyline
    points_attr = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in tpr_points)
    svg.append(f"<polyline points='{points_attr}' fill='none' stroke='#1f4e5f' stroke-width='2' />")

    svg.append(_svg_footer())
    path.write_text("\n".join(svg), encoding="utf-8")


def save_calibration_plot_svg(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    path: Path,
    title: str,
    bins: int = 10,
) -> None:
    width, height = 640, 520
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    if not y_true:
        return

    # Bin by predicted probability
    idx = sorted(range(len(y_prob)), key=lambda i: y_prob[i])
    bin_size = max(1, len(idx) // bins)
    points: List[Tuple[float, float, int]] = []
    for b in range(bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < bins - 1 else len(idx)
        if start >= len(idx):
            break
        ids = idx[start:end]
        avg_p = sum(y_prob[i] for i in ids) / len(ids)
        obs = sum(y_true[i] for i in ids) / len(ids)
        points.append((avg_p, obs, len(ids)))

    def sx(x: float) -> float:
        return margin + x * plot_w

    def sy(y: float) -> float:
        return margin + (1.0 - y) * plot_h

    svg: List[str] = []
    svg.append(_svg_header(width, height))
    svg.append("<rect width='100%' height='100%' fill='#ffffff' />")
    svg.append(
        f"<text x='{width/2:.1f}' y='28' text-anchor='middle' font-family='Georgia, serif' "
        f"font-size='16' fill='#222'>{title}</text>"
    )

    svg.append(f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#222' />")
    svg.append(f"<line x1='{margin}' y1='{height - margin}' x2='{margin}' y2='{margin}' stroke='#222' />")
    svg.append(
        f"<text x='{width/2:.1f}' y='{height - 20}' text-anchor='middle' font-family='Georgia, serif' font-size='12' fill='#222'>Mean Predicted Probability</text>"
    )
    svg.append(
        f"<text x='18' y='{height/2:.1f}' text-anchor='middle' font-family='Georgia, serif' font-size='12' fill='#222' transform='rotate(-90 18 {height/2:.1f})'>Observed Frequency</text>"
    )

    # Perfect calibration diagonal
    svg.append(f"<line x1='{sx(0):.2f}' y1='{sy(0):.2f}' x2='{sx(1):.2f}' y2='{sy(1):.2f}' stroke='#bbb' stroke-dasharray='4 4' />")

    # Points + connecting line
    if points:
        poly = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y, _n in points)
        svg.append(f"<polyline points='{poly}' fill='none' stroke='#1f4e5f' stroke-width='2' />")
        for x, y, n_bin in points:
            svg.append(f"<circle cx='{sx(x):.2f}' cy='{sy(y):.2f}' r='4' fill='#b6403a' />")
            svg.append(
                f"<text x='{sx(x) + 8:.2f}' y='{sy(y) - 8:.2f}' font-family='Georgia, serif' "
                f"font-size='10' fill='#444'>n={n_bin}</text>"
            )

    svg.append(_svg_footer())
    path.write_text("\n".join(svg), encoding="utf-8")


def save_forest_plot_svg(
    labels: Sequence[str],
    odds_ratios: Sequence[float],
    ci_low: Sequence[float],
    ci_high: Sequence[float],
    path: Path,
    title: str,
    legend_lines: Optional[Sequence[str]] = None,
) -> None:
    if not labels:
        return
    if not (len(labels) == len(odds_ratios) == len(ci_low) == len(ci_high)):
        raise ValueError("Forest plot inputs must have identical lengths.")

    font_family = "Georgia, serif"
    label_font = 11
    value_font = 11
    axis_font = 11

    def fmt_num(x: float) -> str:
        if math.isnan(x):
            return "NA"
        if math.isinf(x):
            return "∞"
        return f"{x:.2f}"

    value_texts = [f"{fmt_num(o)} [{fmt_num(lo)}, {fmt_num(hi)}]" for o, lo, hi in zip(odds_ratios, ci_low, ci_high)]

    # Dynamic sizing to avoid clipped labels / values in typical markdown viewers.
    char_w = 0.60 * max(label_font, value_font)
    max_label_len = max(len(s) for s in labels)
    max_value_len = max(len(s) for s in value_texts + ["OR [95% CI]"])
    label_col_w = max(240, min(520, int(max_label_len * char_w) + 36))
    value_col_w = max(220, min(520, int(max_value_len * char_w) + 36))

    plot_w = 560
    pad_left = 20
    pad_right = 20
    margin_left = pad_left + label_col_w
    margin_right = pad_right + value_col_w
    width = margin_left + plot_w + margin_right

    # Vertical layout
    margin_top = 82
    row_gap = 34
    plot_h = row_gap * (len(labels) - 1) if len(labels) > 1 else row_gap

    legend_lines = list(legend_lines or [])
    extra_legend_h = 16 * len(legend_lines)
    # Bottom space for ticks + x label + legend.
    margin_bottom = 132 + extra_legend_h
    height = int(margin_top + plot_h + margin_bottom)

    def bounded(value: float, cap: float = 1e6) -> float:
        if math.isnan(value):
            return float("nan")
        if math.isinf(value):
            return cap
        if value > cap:
            return cap
        return max(value, 1e-12)

    logs = [math.log(bounded(v)) if math.isfinite(bounded(v)) else float("nan") for v in odds_ratios]
    lows = [math.log(bounded(v)) if math.isfinite(bounded(v)) else float("nan") for v in ci_low]
    highs = [math.log(bounded(v)) if math.isfinite(bounded(v)) else float("nan") for v in ci_high]

    finite_lows = [v for v in lows if math.isfinite(v)]
    finite_highs = [v for v in highs if math.isfinite(v)]
    if not finite_lows or not finite_highs:
        return

    x_min = min(finite_lows + [0.0])
    x_max = max(finite_highs + [0.0])
    span = x_max - x_min
    pad = 0.15 * span if span > 1e-9 else 1.0
    x_min -= pad
    x_max += pad

    def sx(x: float) -> float:
        if x_max == x_min:
            return margin_left + plot_w / 2
        return margin_left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(i: int) -> float:
        return margin_top + i * row_gap

    # Generate log-scale ticks (in OR space) with 1-2-5 per decade.
    def safe_exp(x: float) -> float:
        try:
            return math.exp(x)
        except OverflowError:
            return float("inf")

    min_or = max(1e-12, safe_exp(x_min))
    max_or = safe_exp(x_max)
    if not math.isfinite(max_or):
        max_or = 1e12

    def decade_range(lo: float, hi: float) -> Tuple[int, int]:
        lo_d = int(math.floor(math.log10(lo))) if lo > 0 else -12
        hi_d = int(math.ceil(math.log10(hi))) if hi > 0 else 12
        return lo_d, hi_d

    lo_d, hi_d = decade_range(min_or, max_or)
    tick_candidates: List[float] = [1.0]
    for d in range(lo_d, hi_d + 1):
        base = 10.0 ** d
        for m in (1.0, 2.0, 5.0):
            tick_candidates.append(m * base)
    ticks = sorted({t for t in tick_candidates if min_or <= t <= max_or})
    # Keep the tick count reasonable.
    if len(ticks) > 11:
        major = [t for t in ticks if abs(math.log10(t) - round(math.log10(t))) < 1e-9]  # 1*10^k
        # Always keep 1 plus decade ticks; add a few 2/5 ticks if sparse.
        ticks = sorted(set(([1.0] + major)))
        if len(ticks) < 6:
            ticks = sorted({t for t in tick_candidates if min_or <= t <= max_or and (t in ticks or t / (10 ** round(math.log10(t))) in (2.0, 5.0))})

    def fmt_tick(t: float) -> str:
        if t >= 100:
            return f"{t:.0f}"
        if t >= 10:
            return f"{t:.0f}"
        if t >= 1:
            return f"{t:.1f}".rstrip("0").rstrip(".")
        return f"{t:.2f}".rstrip("0").rstrip(".")

    # SVG assembly
    svg: List[str] = []
    svg.append(_svg_header(int(width), int(height)))
    svg.append("<rect width='100%' height='100%' fill='#ffffff' />")
    svg.append(
        f"<text x='{width/2:.1f}' y='28' text-anchor='middle' font-family='{font_family}' "
        f"font-size='16' fill='#222'>{title}</text>"
    )

    y_axis_bottom = margin_top + plot_h
    x_axis_left = margin_left
    x_axis_right = margin_left + plot_w
    value_x = margin_left + plot_w + 12

    # Column headers
    header_y = margin_top - 16
    svg.append(
        f"<text x='{margin_left - 10}' y='{header_y:.2f}' text-anchor='end' font-family='{font_family}' "
        f"font-size='{axis_font}' fill='#222' font-weight='bold'>Predictor</text>"
    )
    svg.append(
        f"<text x='{value_x}' y='{header_y:.2f}' text-anchor='start' font-family='{font_family}' "
        f"font-size='{axis_font}' fill='#222' font-weight='bold'>OR [95% CI]</text>"
    )

    # Light row bands to help scan.
    for i in range(len(labels)):
        if i % 2 == 1:
            y0 = max(margin_top, sy(i) - row_gap / 2)
            y1 = min(y_axis_bottom, sy(i) + row_gap / 2)
            svg.append(
                f"<rect x='10' y='{y0:.2f}' width='{width - 20:.2f}' height='{max(0.0, y1 - y0):.2f}' fill='#fafafa' />"
            )

    # Vertical grid lines + ticks (OR scale)
    for t in ticks:
        x = sx(math.log(t))
        svg.append(f"<line x1='{x:.2f}' y1='{margin_top:.2f}' x2='{x:.2f}' y2='{y_axis_bottom:.2f}' stroke='#f0f0f0' />")
        svg.append(f"<line x1='{x:.2f}' y1='{y_axis_bottom:.2f}' x2='{x:.2f}' y2='{y_axis_bottom + 6:.2f}' stroke='#666' />")
        svg.append(
            f"<text x='{x:.2f}' y='{y_axis_bottom + 20:.2f}' text-anchor='middle' font-family='{font_family}' "
            f"font-size='{axis_font}' fill='#444'>{fmt_tick(t)}</text>"
        )

    # Axes
    svg.append(
        f"<line x1='{x_axis_left}' y1='{y_axis_bottom}' x2='{x_axis_right}' y2='{y_axis_bottom}' stroke='#222' stroke-width='1' />"
    )
    svg.append(
        f"<line x1='{x_axis_left}' y1='{y_axis_bottom}' x2='{x_axis_left}' y2='{margin_top}' stroke='#222' stroke-width='1' />"
    )
    svg.append(
        f"<text x='{(x_axis_left + x_axis_right)/2:.2f}' y='{y_axis_bottom + 46:.2f}' text-anchor='middle' "
        f"font-family='{font_family}' font-size='12' fill='#222'>Odds ratio (log scale)</text>"
    )

    # OR=1 reference line
    x_ref = sx(0.0)
    svg.append(
        f"<line x1='{x_ref:.2f}' y1='{margin_top:.2f}' x2='{x_ref:.2f}' y2='{y_axis_bottom:.2f}' stroke='#888' stroke-dasharray='4 4' />"
    )

    # Effects
    whisk = 6
    for i, label in enumerate(labels):
        y = sy(i)
        svg.append(
            f"<text x='{margin_left - 10}' y='{y + 4:.2f}' text-anchor='end' font-family='{font_family}' "
            f"font-size='{label_font}' fill='#222'>{label}</text>"
        )

        lo_x = lows[i]
        hi_x = highs[i]
        mid_x = logs[i]
        if not (math.isfinite(lo_x) and math.isfinite(hi_x) and math.isfinite(mid_x)):
            continue

        lo_x = min(max(lo_x, x_min), x_max)
        hi_x = min(max(hi_x, x_min), x_max)
        mid_x = min(max(mid_x, x_min), x_max)

        x1 = sx(lo_x)
        x2 = sx(hi_x)
        xm = sx(mid_x)

        svg.append(f"<line x1='{x1:.2f}' y1='{y:.2f}' x2='{x2:.2f}' y2='{y:.2f}' stroke='#1f4e5f' stroke-width='2' />")
        # CI whiskers
        svg.append(f"<line x1='{x1:.2f}' y1='{y - whisk/2:.2f}' x2='{x1:.2f}' y2='{y + whisk/2:.2f}' stroke='#1f4e5f' stroke-width='2' />")
        svg.append(f"<line x1='{x2:.2f}' y1='{y - whisk/2:.2f}' x2='{x2:.2f}' y2='{y + whisk/2:.2f}' stroke='#1f4e5f' stroke-width='2' />")
        # Point estimate
        svg.append(f"<circle cx='{xm:.2f}' cy='{y:.2f}' r='4' fill='#b6403a' />")

        svg.append(
            f"<text x='{value_x}' y='{y + 4:.2f}' text-anchor='start' font-family='{font_family}' "
            f"font-size='{value_font}' fill='#222'>{value_texts[i]}</text>"
        )

    # Legend (glyph + short text)
    legend_y = y_axis_bottom + 70
    # OR/CI glyph
    g_x0 = margin_left
    g_y1 = legend_y
    svg.append(f"<line x1='{g_x0 + 12:.2f}' y1='{g_y1:.2f}' x2='{g_x0 + 62:.2f}' y2='{g_y1:.2f}' stroke='#1f4e5f' stroke-width='2' />")
    svg.append(f"<circle cx='{g_x0 + 37:.2f}' cy='{g_y1:.2f}' r='4' fill='#b6403a' />")
    svg.append(
        f"<text x='{g_x0 + 78:.2f}' y='{g_y1 + 4:.2f}' font-family='{font_family}' font-size='{axis_font}' fill='#444'>"
        f"Dot = odds ratio; line = 95% CI</text>"
    )
    # OR=1 glyph
    g_y2 = legend_y + 18
    svg.append(f"<line x1='{g_x0 + 37:.2f}' y1='{g_y2 - 8:.2f}' x2='{g_x0 + 37:.2f}' y2='{g_y2 + 8:.2f}' stroke='#888' stroke-dasharray='4 4' />")
    svg.append(
        f"<text x='{g_x0 + 78:.2f}' y='{g_y2 + 4:.2f}' font-family='{font_family}' font-size='{axis_font}' fill='#444'>"
        f"Dashed line = OR=1 (no association)</text>"
    )

    if legend_lines:
        text_y0 = legend_y + 42
        for i, line in enumerate(legend_lines):
            y = text_y0 + i * 16
            svg.append(
                f"<text x='{margin_left:.2f}' y='{y:.2f}' font-family='{font_family}' font-size='{axis_font}' fill='#444'>{line}</text>"
            )

    svg.append(_svg_footer())
    path.write_text("\n".join(svg), encoding="utf-8")


# --- Data plumbing ---


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def compute_platform_groups(rows: List[Dict[str, str]], top_k: int) -> Dict[str, str]:
    counts = Counter(r["Most_Used_Platform"] for r in rows)
    top = {name for name, _ in counts.most_common(top_k)}
    return {platform: (platform if platform in top else "Other") for platform in counts.keys()}


def one_hot_encode(
    rows: List[Dict[str, str]],
    *,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    platform_map: Dict[str, str],
    drop_first: bool = True,
) -> Tuple[List[List[float]], List[str], Dict[str, Tuple[float, float]]]:
    feature_names, standardization, levels = build_one_hot_spec(
        rows,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        platform_map=platform_map,
        drop_first=drop_first,
    )
    x_rows = vectorize_rows(
        rows,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        platform_map=platform_map,
        standardization=standardization,
        levels=levels,
    )
    return x_rows, feature_names, standardization


def build_one_hot_spec(
    rows: List[Dict[str, str]],
    *,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    platform_map: Dict[str, str],
    drop_first: bool = True,
) -> Tuple[List[str], Dict[str, Tuple[float, float]], Dict[str, List[str]]]:
    numeric_vals = {col: [float(r[col]) for r in rows] for col in numeric_cols}
    standardization: Dict[str, Tuple[float, float]] = {}
    for col, vals in numeric_vals.items():
        mu = mean(vals)
        sd = sample_std(vals) or 1.0
        standardization[col] = (mu, sd)

    normalized_rows = normalize_platform_rows(rows, platform_map)

    levels: Dict[str, List[str]] = {}
    for col in categorical_cols:
        unique = sorted({r[col] for r in normalized_rows})
        if drop_first and unique:
            unique = unique[1:]
        levels[col] = unique

    feature_names: List[str] = ["Intercept"]
    feature_names.extend(numeric_cols)
    for col in categorical_cols:
        for level in levels[col]:
            feature_names.append(f"{col}={level}")

    return feature_names, standardization, levels


def vectorize_rows(
    rows: List[Dict[str, str]],
    *,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    platform_map: Dict[str, str],
    standardization: Dict[str, Tuple[float, float]],
    levels: Dict[str, List[str]],
) -> List[List[float]]:
    normalized_rows = normalize_platform_rows(rows, platform_map)
    x_rows: List[List[float]] = []
    for r in normalized_rows:
        vec: List[float] = [1.0]
        for col in numeric_cols:
            mu, sd = standardization[col]
            vec.append((float(r[col]) - mu) / sd)
        for col in categorical_cols:
            for level in levels[col]:
                vec.append(1.0 if r[col] == level else 0.0)
        x_rows.append(vec)
    return x_rows


def normalize_platform_rows(
    rows: List[Dict[str, str]], platform_map: Dict[str, str]
) -> List[Dict[str, str]]:
    normalized_rows: List[Dict[str, str]] = []
    for r in rows:
        rr = dict(r)
        rr["Most_Used_Platform"] = platform_map.get(rr["Most_Used_Platform"], rr["Most_Used_Platform"])
        normalized_rows.append(rr)
    return normalized_rows


def train_test_split_indices(
    n: int, test_frac: float, rng: random.Random
) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    rng.shuffle(indices)
    test_size = int(round(n * test_frac))
    test_size = max(1, min(test_size, n - 1))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return train_idx, test_idx


def k_fold_indices(n: int, k: int, rng: random.Random) -> List[List[int]]:
    k = max(2, min(k, n))
    indices = list(range(n))
    rng.shuffle(indices)
    base = n // k
    remainder = n % k
    folds: List[List[int]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < remainder else 0)
        folds.append(indices[start : start + size])
        start += size
    return folds


def classification_metrics(y_true: Sequence[int], y_score: Sequence[float]) -> Dict[str, float]:
    auc = roc_auc(y_true, y_score)
    tp = sum(1 for yi, pi in zip(y_true, y_score) if yi == 1 and pi >= 0.5)
    fp = sum(1 for yi, pi in zip(y_true, y_score) if yi == 0 and pi >= 0.5)
    tn = sum(1 for yi, pi in zip(y_true, y_score) if yi == 0 and pi < 0.5)
    fn = sum(1 for yi, pi in zip(y_true, y_score) if yi == 1 and pi < 0.5)
    acc = (tp + tn) / len(y_true) if y_true else float("nan")
    return {
        "auc": auc,
        "accuracy@0.5": acc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def contingency_counts_yes_no(
    rows: List[Dict[str, str]],
    *,
    outcome_col: str,
    category_col: str,
    platform_map: Dict[str, str],
) -> Tuple[List[str], List[List[int]]]:
    # Two outcome columns: Yes then No
    if category_col == "Most_Used_Platform":
        categories = sorted(
            {
                platform_map.get(r["Most_Used_Platform"], r["Most_Used_Platform"])
                for r in rows
            }
        )
    else:
        categories = sorted({r[category_col] for r in rows})
    index = {cat: i for i, cat in enumerate(categories)}
    counts = [[0, 0] for _ in categories]
    for r in rows:
        category = r[category_col]
        if category_col == "Most_Used_Platform":
            category = platform_map.get(category, category)
        out_yes = 1 if r[outcome_col] == "Yes" else 0
        counts[index[category]][0 if out_yes else 1] += 1
    return categories, counts


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for r in rows:
            writer.writerow(list(r))


def format_p(p: float) -> str:
    if math.isnan(p):
        return "NA"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="Data/media_addiction.csv")
    parser.add_argument("--out-dir", default="results/academic_impact")
    parser.add_argument("--platform-top-k", type=int, default=6)
    parser.add_argument("--exclude-high-school", action="store_true")
    parser.add_argument("--l2", type=float, default=0.0, help="Optional L2 penalty for logistic regression.")
    parser.add_argument("--test-frac", type=float, default=0.2, help="Holdout test fraction (e.g., 0.2).")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    args = parser.parse_args()

    rows = read_rows(Path(args.csv_path))
    if args.exclude_high_school:
        rows = [r for r in rows if r["Academic_Level"] != "High School"]
    if not rows:
        raise SystemExit("No rows to analyze")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    platform_map = compute_platform_groups(rows, args.platform_top_k)

    # Outcome + prevalence CI
    y = [1 if r["Affects_Academic_Performance"] == "Yes" else 0 for r in rows]
    n = len(y)
    yes = sum(y)
    no = n - yes
    p_hat = yes / n
    p_lo, p_hi = wilson_ci_95(yes, n)

    rng = random.Random(args.seed)
    train_idx, test_idx = train_test_split_indices(n, args.test_frac, rng)
    cv_folds = k_fold_indices(n, args.cv_folds, rng)
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    # Numeric group comparisons (Yes vs No)
    numeric_cols = [
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
        "Addicted_Score",
        "Age",
        "Conflicts_Over_Social_Media",
    ]
    numeric_rows = []
    for col in numeric_cols:
        group_yes = [float(r[col]) for r in rows if r["Affects_Academic_Performance"] == "Yes"]
        group_no = [float(r[col]) for r in rows if r["Affects_Academic_Performance"] == "No"]
        diff, lo, hi = diff_in_means_ci_95(group_yes, group_no)
        numeric_rows.append(
            (
                col,
                len(group_yes),
                mean(group_yes),
                sample_std(group_yes),
                len(group_no),
                mean(group_no),
                sample_std(group_no),
                diff,
                lo,
                hi,
                cohen_d(group_yes, group_no),
            )
        )
    write_csv(
        out_dir / "numeric_group_comparisons.csv",
        [
            "variable",
            "n_yes",
            "mean_yes",
            "sd_yes",
            "n_no",
            "mean_no",
            "sd_no",
            "diff_yes_minus_no",
            "ci95_lo",
            "ci95_hi",
            "cohen_d",
        ],
        numeric_rows,
    )

    # Chi-square tests for categorical variables
    categorical_cols = ["Gender", "Academic_Level", "Relationship_Status", "Most_Used_Platform"]
    chi_summary_rows = []
    chi_details: Dict[str, Tuple[List[str], List[List[int]]]] = {}
    for col in categorical_cols:
        categories, counts = contingency_counts_yes_no(
            rows, outcome_col="Affects_Academic_Performance", category_col=col, platform_map=platform_map
        )
        chi2, df, pval, min_exp = chi_square_independence(counts)
        v = cramer_v(chi2, n, len(categories), 2)
        chi_summary_rows.append((col, chi2, df, pval, v, min_exp))
        chi_details[col] = (categories, counts)
    write_csv(
        out_dir / "categorical_chi_square_summary.csv",
        ["variable", "chi2", "df", "p_value", "cramers_v", "min_expected"],
        chi_summary_rows,
    )

    # Logistic regression models (handle usage/addiction collinearity by fitting 3 variants)
    base_numeric = ["Sleep_Hours_Per_Night", "Mental_Health_Score", "Age", "Conflicts_Over_Social_Media"]
    models = [
        ("full", ["Avg_Daily_Usage_Hours", "Addicted_Score"] + base_numeric),
        ("usage_only", ["Avg_Daily_Usage_Hours"] + base_numeric),
        ("addiction_only", ["Addicted_Score"] + base_numeric),
    ]
    model_fits: Dict[str, LogisticFit] = {}
    model_metrics: Dict[str, Dict[str, float]] = {}
    model_metrics_holdout: Dict[str, Dict[str, float]] = {}
    model_metrics_cv: Dict[str, Dict[str, float]] = {}

    for model_name, model_numeric in models:
        x_rows, feature_names, _standardization = one_hot_encode(
            rows,
            numeric_cols=model_numeric,
            categorical_cols=categorical_cols,
            platform_map=platform_map,
            drop_first=True,
        )
        fit = fit_logistic_regression_irls(
            x_rows, y, feature_names, l2_penalty=args.l2, max_iter=60, tol=1e-8
        )
        model_fits[model_name] = fit

        preds = [fit.predict_proba(xi) for xi in x_rows]
        metrics = classification_metrics(y, preds)
        model_metrics[model_name] = {
            "log_likelihood": fit.log_likelihood,
            **metrics,
        }

        feature_names_tt, standardization_tt, levels_tt = build_one_hot_spec(
            train_rows,
            numeric_cols=model_numeric,
            categorical_cols=categorical_cols,
            platform_map=platform_map,
            drop_first=True,
        )
        x_train = vectorize_rows(
            train_rows,
            numeric_cols=model_numeric,
            categorical_cols=categorical_cols,
            platform_map=platform_map,
            standardization=standardization_tt,
            levels=levels_tt,
        )
        x_test = vectorize_rows(
            test_rows,
            numeric_cols=model_numeric,
            categorical_cols=categorical_cols,
            platform_map=platform_map,
            standardization=standardization_tt,
            levels=levels_tt,
        )
        fit_tt = fit_logistic_regression_irls(
            x_train, y_train, feature_names_tt, l2_penalty=args.l2, max_iter=60, tol=1e-8
        )
        preds_train = [fit_tt.predict_proba(xi) for xi in x_train]
        preds_test = [fit_tt.predict_proba(xi) for xi in x_test]
        model_metrics_holdout[model_name] = {
            "train": classification_metrics(y_train, preds_train),
            "test": classification_metrics(y_test, preds_test),
        }
        if model_name == "full":
            write_csv(
                out_dir / "logistic_full_predictions_holdout.csv",
                ["y_true", "p_hat"],
                [(yi, pi) for yi, pi in zip(y_test, preds_test)],
            )
            save_roc_curve_svg(
                y_test,
                preds_test,
                out_dir / "fig_logistic_roc_holdout.svg",
                title="Logistic Model (Full): Holdout ROC Curve",
            )
            save_calibration_plot_svg(
                y_test,
                preds_test,
                out_dir / "fig_logistic_calibration_holdout.svg",
                title="Logistic Model (Full): Holdout Calibration",
            )

        cv_auc: List[float] = []
        cv_acc: List[float] = []
        for fold in cv_folds:
            val_idx = fold
            val_set = set(val_idx)
            train_idx_cv = [i for i in range(n) if i not in val_set]
            train_rows_cv = [rows[i] for i in train_idx_cv]
            val_rows = [rows[i] for i in val_idx]
            y_train_cv = [y[i] for i in train_idx_cv]
            y_val = [y[i] for i in val_idx]

            feature_names_cv, standardization_cv, levels_cv = build_one_hot_spec(
                train_rows_cv,
                numeric_cols=model_numeric,
                categorical_cols=categorical_cols,
                platform_map=platform_map,
                drop_first=True,
            )
            x_train_cv = vectorize_rows(
                train_rows_cv,
                numeric_cols=model_numeric,
                categorical_cols=categorical_cols,
                platform_map=platform_map,
                standardization=standardization_cv,
                levels=levels_cv,
            )
            x_val = vectorize_rows(
                val_rows,
                numeric_cols=model_numeric,
                categorical_cols=categorical_cols,
                platform_map=platform_map,
                standardization=standardization_cv,
                levels=levels_cv,
            )
            fit_cv = fit_logistic_regression_irls(
                x_train_cv, y_train_cv, feature_names_cv, l2_penalty=args.l2, max_iter=60, tol=1e-8
            )
            preds_val = [fit_cv.predict_proba(xi) for xi in x_val]
            met = classification_metrics(y_val, preds_val)
            cv_auc.append(met["auc"])
            cv_acc.append(met["accuracy@0.5"])

        model_metrics_cv[model_name] = {
            "auc_mean": mean(cv_auc),
            "auc_sd": sample_std(cv_auc),
            "acc_mean": mean(cv_acc),
            "acc_sd": sample_std(cv_acc),
            "folds": len(cv_folds),
        }

        # Save OR table
        se_by_name: Dict[str, float] = {}
        if fit.standard_errors is not None:
            se_by_name = dict(zip(fit.feature_names, fit.standard_errors))

        or_rows = []
        for name, beta in zip(fit.feature_names, fit.coefficients):
            if name == "Intercept":
                continue
            se = se_by_name.get(name, float("nan"))
            if math.isnan(se):
                lo = float("nan")
                hi = float("nan")
            else:
                lo = beta - Z_95 * se
                hi = beta + Z_95 * se
            or_rows.append(
                (
                    name,
                    beta,
                    se,
                    exp_or_inf(beta),
                    exp_or_inf(lo) if not math.isnan(lo) else float("nan"),
                    exp_or_inf(hi) if not math.isnan(hi) else float("nan"),
                )
            )
        write_csv(
            out_dir / f"logistic_{model_name}_odds_ratios.csv",
            ["feature", "coef", "se", "odds_ratio", "or_ci95_lo", "or_ci95_hi"],
            or_rows,
        )

        if model_name == "full":
            # Save prediction diagnostics for the primary model in this run.
            write_csv(
                out_dir / "logistic_full_predictions.csv",
                ["y_true", "p_hat"],
                [(yi, pi) for yi, pi in zip(y, preds)],
            )
            save_roc_curve_svg(
                y, preds, out_dir / "fig_logistic_roc.svg", title="Logistic Model (Full): ROC Curve"
            )
            save_calibration_plot_svg(
                y, preds, out_dir / "fig_logistic_calibration.svg", title="Logistic Model (Full): Calibration"
            )

    # Forest plot for strongest effects (full model)
    full = model_fits["full"]
    if full.standard_errors is not None:
        se_map = dict(zip(full.feature_names, full.standard_errors))
        effects = []
        for name, beta in zip(full.feature_names, full.coefficients):
            if name == "Intercept":
                continue
            effects.append((abs(beta), name, beta, se_map.get(name, float("nan"))))
        effects.sort(reverse=True)
        effects = effects[:12]

        labels: List[str] = []
        ors: List[float] = []
        lo: List[float] = []
        hi: List[float] = []
        for _absb, name, beta, se in effects:
            label = (
                name.replace("Most_Used_Platform=", "Platform=")
                .replace("Academic_Level=", "Level=")
                .replace("Relationship_Status=", "Rel=")
            )
            labels.append(label)
            ors.append(exp_or_inf(beta))
            lo.append(exp_or_inf(beta - Z_95 * se) if not math.isnan(se) else float("nan"))
            hi.append(exp_or_inf(beta + Z_95 * se) if not math.isnan(se) else float("nan"))
        save_forest_plot_svg(
            labels,
            ors,
            lo,
            hi,
            out_dir / "fig_logistic_or_forest.svg",
            title="Academic Impact Model (Full): Strongest Effects (OR, 95% CI)",
            legend_lines=[
                "Numeric predictors are standardized (1 SD); categorical levels are vs a reference level.",
            ],
        )
        save_forest_plot_svg(
            labels,
            ors,
            lo,
            hi,
            out_dir / "fig_logistic_or_forest_legend.svg",
            title="Academic Impact Model (Full): Strongest Effects (OR, 95% CI)",
            legend_lines=[
                "Numeric predictors are standardized (1 SD); categories are vs reference.",
                "OR > 1 indicates higher odds of reporting impact; OR < 1 indicates lower odds.",
            ],
        )

    # Markdown report
    md: List[str] = []
    md.append("# Academic Impact Analysis (CI, Chi-square, Logistic Regression)")
    md.append("")
    md.append("## Sample")
    md.append(f"- Rows analyzed: {n}")
    md.append(f"- Academic impact: Yes={yes} ({p_hat:.3f}), No={no} ({1 - p_hat:.3f})")
    md.append(f"- 95% CI for proportion Yes (Wilson): [{p_lo:.3f}, {p_hi:.3f}]")
    if args.exclude_high_school:
        md.append("- Note: excluded `Academic_Level = High School`.")
    md.append("")

    md.append("## Numeric comparisons (Yes vs No)")
    md.append("Outputs: `numeric_group_comparisons.csv` (mean difference CI + Cohen’s d).")
    md.append("")

    md.append("## Categorical associations (chi-square independence tests)")
    md.append("Summary: `categorical_chi_square_summary.csv` (includes Cramér’s V).")
    md.append("")
    for col in categorical_cols:
        categories, counts = chi_details[col]
        chi2, df, pval, min_exp = chi_square_independence(counts)
        v = cramer_v(chi2, n, len(categories), 2)
        md.append(f"### {col}")
        md.append(f"- Chi-square={chi2:.2f}, df={df}, p={format_p(pval)}, Cramér’s V={v:.3f}, min expected={min_exp:.2f}")
        if min_exp < 5:
            md.append("- Caution: some expected counts are < 5; consider merging rare categories (especially platforms).")
        md.append("")

    md.append("## Logistic regression (binary outcome)")
    md.append("Coefficient tables (odds ratios + 95% CI):")
    md.append("- `logistic_full_odds_ratios.csv`")
    md.append("- `logistic_usage_only_odds_ratios.csv`")
    md.append("- `logistic_addiction_only_odds_ratios.csv`")
    md.append("")

    md.append("### Model evaluation (holdout + cross-validation)")
    md.append(
        f"- Split: train={len(train_idx)}, test={len(test_idx)} (test_frac={args.test_frac:.2f}, seed={args.seed})"
    )
    md.append("| Model | Train AUC | Test AUC | Train Acc@0.5 | Test Acc@0.5 | CV AUC (mean±sd) | CV Acc@0.5 (mean±sd) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for model_name in ["full", "usage_only", "addiction_only"]:
        fit = model_fits[model_name]
        met_holdout = model_metrics_holdout[model_name]
        met_cv = model_metrics_cv[model_name]
        md.append(
            "| {model} | {train_auc:.3f} | {test_auc:.3f} | {train_acc:.3f} | {test_acc:.3f} | {cv_auc:.3f}±{cv_auc_sd:.3f} | {cv_acc:.3f}±{cv_acc_sd:.3f} |".format(
                model=model_name,
                train_auc=met_holdout["train"]["auc"],
                test_auc=met_holdout["test"]["auc"],
                train_acc=met_holdout["train"]["accuracy@0.5"],
                test_acc=met_holdout["test"]["accuracy@0.5"],
                cv_auc=met_cv["auc_mean"],
                cv_auc_sd=met_cv["auc_sd"],
                cv_acc=met_cv["acc_mean"],
                cv_acc_sd=met_cv["acc_sd"],
            )
        )
    md.append("")
    md.append("### Model metrics (in-sample)")
    md.append("| Model | Converged | Iter | LogLik | AUC | Acc@0.5 | TP | FP | TN | FN |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model_name in ["full", "usage_only", "addiction_only"]:
        fit = model_fits[model_name]
        met = model_metrics[model_name]
        md.append(
            "| {model} | {conv} | {it} | {ll:.1f} | {auc:.3f} | {acc:.3f} | {tp} | {fp} | {tn} | {fn} |".format(
                model=model_name,
                conv=str(fit.converged),
                it=fit.iterations,
                ll=met["log_likelihood"],
                auc=met["auc"],
                acc=met["accuracy@0.5"],
                tp=int(met["tp"]),
                fp=int(met["fp"]),
                tn=int(met["tn"]),
                fn=int(met["fn"]),
            )
        )
    md.append("")
    if (out_dir / "fig_logistic_or_forest.svg").exists():
        md.append("### Forest plot (largest effects in full model)")
        md.append("![](fig_logistic_or_forest.svg)")
        md.append("")
    if (out_dir / "fig_logistic_roc_holdout.svg").exists():
        md.append("### Holdout diagnostics (full model)")
        md.append("![](fig_logistic_roc_holdout.svg)")
        if (out_dir / "fig_logistic_calibration_holdout.svg").exists():
            md.append("![](fig_logistic_calibration_holdout.svg)")
        md.append("")

    perfect = any(met["fp"] == 0 and met["fn"] == 0 for met in model_metrics.values())
    perfect_holdout = any(
        met["test"]["fp"] == 0 and met["test"]["fn"] == 0 for met in model_metrics_holdout.values()
    )
    if perfect or perfect_holdout:
        md.append("### Modeling caution")
        if perfect:
            md.append("- The in-sample evaluation achieves perfect classification (FP=0 and FN=0).")
        if perfect_holdout:
            md.append("- The holdout evaluation achieves perfect classification (FP=0 and FN=0).")
        md.append("- This often indicates (near-)complete separation in synthetic/survey-like data; odds ratios/SEs can become unstable.")
        md.append(
            "- If needed, rerun with ridge regularization, e.g. "
            "`python3 scripts/academic_impact_analysis.py --l2 1.0`."
        )
        md.append("")

    md.append("## Interpretation notes")
    md.append("- Observational survey data: interpret as association, not causation.")
    md.append("- Usage and addiction can be highly correlated; compare `full` vs single-predictor models to assess sensitivity.")
    md.append("")

    (out_dir / "academic_impact_report.md").write_text("\n".join(md), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
