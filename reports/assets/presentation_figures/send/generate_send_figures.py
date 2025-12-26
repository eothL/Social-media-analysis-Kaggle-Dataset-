#!/usr/bin/env python3
"""
Regenerate the SVG figures in `reports/assets/presentation_figures/send/`.

This script is intentionally dependency-free (stdlib-only): it reads the CSV
outputs produced by `scripts/academic_impact_analysis.py` and writes SVGs.

Prerequisites (run once):
  python3 scripts/academic_impact_analysis.py --out-dir results/academic_impact
  python3 scripts/academic_impact_analysis.py --l2 1.0 --out-dir results/academic_impact_ridge

Then:
  python3 reports/assets/presentation_figures/send/generate_send_figures.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _read_csv_dict(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_numeric_summary(path: Path) -> Dict[str, Dict[str, float]]:
    rows = _read_csv_dict(path)
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        var = row["variable"]
        out[var] = {
            "mean_yes": float(row["mean_yes"]),
            "mean_no": float(row["mean_no"]),
            "cohen_d": float(row["cohen_d"]),
        }
    return out


def _load_or_table(path: Path) -> List[Dict[str, float]]:
    rows = _read_csv_dict(path)
    out: List[Dict[str, float]] = []
    for row in rows:
        out.append(
            {
                "feature": row["feature"],
                "coef": float(row["coef"]),
                "or": float(row["odds_ratio"]),
                "lo": float(row["or_ci95_lo"]),
                "hi": float(row["or_ci95_hi"]),
            }
        )
    return out


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_group_diff_svg(numeric: Dict[str, Dict[str, float]], out_path: Path) -> None:
    # Figure: Academic Impact Groups: Average Outcomes
    # Variables chosen to match the original story slide.
    vars_ = [
        ("Usage", "Avg_Daily_Usage_Hours", "#2a9d8f"),
        ("Addiction", "Addicted_Score", "#e76f51"),
        ("Mental Health", "Mental_Health_Score", "#264653"),
        ("Sleep", "Sleep_Hours_Per_Night", "#f4a261"),
    ]

    width, height = 760, 420
    x0, x1 = 70, 690
    y_top, y_bottom = 70, 350
    plot_h = y_bottom - y_top
    y_max = 10.0
    scale = plot_h / y_max

    # Bar layout (keeps compatibility with the existing SVG coordinates)
    bar_w = 62.0
    group_no_x = [101.0, 163.0, 225.0, 287.0]
    group_yes_x = [411.0, 473.0, 535.0, 597.0]
    group_centers = {"No": 225.0, "Yes": 535.0}

    ticks = [0, 2, 4, 6, 8, 10]

    svg: List[str] = []
    svg.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>")
    svg.append("")
    svg.append("<rect width='100%' height='100%' fill='#ffffff' />")
    svg.append("<!-- Y-axis ticks + grid (0–10) -->")
    for t in ticks:
        y = y_bottom - t * scale
        svg.append(f"<line x1='{x0}' y1='{y:.0f}' x2='{x1}' y2='{y:.0f}' stroke='#e0e0e0' stroke-width='1' />")
    for t in ticks:
        y = y_bottom - t * scale
        svg.append(f"<line x1='{x0-5:.0f}' y1='{y:.0f}' x2='{x0}' y2='{y:.0f}' stroke='#222' stroke-width='1' />")
        svg.append(
            f"<text x='{x0-10:.0f}' y='{y+4:.0f}' text-anchor='end' font-family='Georgia, serif' "
            f"font-size='11' fill='#444'>{t}</text>"
        )

    # Axes
    svg.append(f"<line x1='{x0}' y1='{y_bottom}' x2='{x1}' y2='{y_bottom}' stroke='#222' stroke-width='1' />")
    svg.append(f"<line x1='{x0}' y1='{y_bottom}' x2='{x0}' y2='{y_top}' stroke='#222' stroke-width='1' />")

    def bar_rect(x: float, value: float, color: str) -> str:
        h = max(0.0, min(y_max, value)) * scale
        y = y_bottom - h
        return f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w:.2f}' height='{h:.2f}' fill='{color}' opacity='0.85' />"

    # Bars (No, then Yes)
    for (label, var, color), x in zip(vars_, group_no_x):
        svg.append(bar_rect(x, numeric[var]["mean_no"], color))
    svg.append(
        f"<text x='{group_centers['No']:.2f}' y='370' text-anchor='middle' font-family='Georgia, serif' "
        f"font-size='11' fill='#222'>No</text>"
    )
    for (label, var, color), x in zip(vars_, group_yes_x):
        svg.append(bar_rect(x, numeric[var]["mean_yes"], color))
    svg.append(
        f"<text x='{group_centers['Yes']:.2f}' y='370' text-anchor='middle' font-family='Georgia, serif' "
        f"font-size='11' fill='#222'>Yes</text>"
    )

    # Legend (boxed, below title)
    svg.append("<!-- Legend (boxed, below title) -->")
    svg.append("<rect x='145' y='38' width='470' height='26' rx='6' ry='6' fill='#ffffff' stroke='#cfcfcf' stroke-width='1' />")
    legend_x = 160
    for name, _var, color in vars_:
        svg.append(f"<rect x='{legend_x}' y='45' width='10' height='10' fill='{color}' />")
        svg.append(f"<text x='{legend_x+16}' y='54' font-family='Georgia, serif' font-size='11' fill='#222'>{name}</text>")
        # crude spacing by label length (keeps legend single-line)
        legend_x += 16 + max(55, 9 * len(name))

    svg.append(
        f"<text x='{width/2:.1f}' y='24' text-anchor='middle' font-family='Georgia, serif' "
        f"font-size='16' fill='#222'>Academic Impact Groups: Average Outcomes</text>"
    )
    svg.append(
        f"<text x='20' y='{height/2:.1f}' text-anchor='middle' font-family='Georgia, serif' "
        f"font-size='12' fill='#222' transform='rotate(-90 20 {height/2:.1f})'>Average value</text>"
    )
    svg.append("</svg>")

    _write_text(out_path, "\n".join(svg) + "\n")


def _build_cohen_d_slide_svg(numeric: Dict[str, Dict[str, float]], out_path: Path) -> None:
    # Figure: slide-style mean comparison + Cohen's d (graph-only)
    rows = [
        ("Addicted Score", "Addicted_Score"),
        ("Avg Daily Usage (Hours)", "Avg_Daily_Usage_Hours"),
        ("Mental Health Score (lower is worse)", "Mental_Health_Score"),
        ("Sleep (Hours/Night)", "Sleep_Hours_Per_Night"),
    ]

    width, height = 1200, 520
    axis_x0, axis_x1 = 260.0, 860.0
    axis_w = axis_x1 - axis_x0
    value_max = 10.0
    px_per_unit = axis_w / value_max

    def x_of(v: float) -> float:
        return axis_x0 + v * px_per_unit

    svg: List[str] = []
    svg.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    svg.append(
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" "
        f"width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    )
    svg.append("  <style>")
    svg.append("    text { font-family: Arial, sans-serif; fill: #222; }")
    svg.append("    .label { font-size: 20px; }")
    svg.append("    .axis-line { stroke: #d0d0d0; stroke-width: 2; }")
    svg.append("    .tick-line { stroke: #d0d0d0; stroke-width: 1; }")
    svg.append("    .tick-label { font-size: 14px; fill: #666; }")
    svg.append("    .value { font-size: 16px; font-weight: bold; }")
    svg.append("    .status { font-size: 14px; }")
    svg.append("    .yes { fill: #b0811a; }")
    svg.append("    .no { fill: #9e9e9e; }")
    svg.append("    .connector { stroke: #c4c4c4; stroke-width: 2; }")
    svg.append("    .evidence { font-size: 16px; fill: #333; }")
    svg.append("  </style>")
    svg.append("")
    svg.append("  <defs>")
    svg.append("    <g id=\"axis\">")
    svg.append(f"      <line class=\"axis-line\" x1=\"{axis_x0:.0f}\" y1=\"0\" x2=\"{axis_x1:.0f}\" y2=\"0\" />")
    for i in range(0, 11):
        x = axis_x0 + i * px_per_unit
        svg.append(f"      <line class=\"tick-line\" x1=\"{x:.0f}\" y1=\"0\" x2=\"{x:.0f}\" y2=\"8\" />")
    for i in range(0, 11, 2):
        x = axis_x0 + i * px_per_unit
        svg.append(f"      <text class=\"tick-label\" x=\"{x:.0f}\" y=\"32\" text-anchor=\"middle\">{i}</text>")
    svg.append("    </g>")
    svg.append("  </defs>")
    svg.append("")

    for idx, (label, var) in enumerate(rows, start=1):
        label_y = 80 + (idx - 1) * 120
        group_y = 110 + (idx - 1) * 120
        mean_no = numeric[var]["mean_no"]
        mean_yes = numeric[var]["mean_yes"]
        d = numeric[var]["cohen_d"]
        x_no = x_of(mean_no)
        x_yes = x_of(mean_yes)
        x_lo, x_hi = (x_no, x_yes) if x_no <= x_yes else (x_yes, x_no)

        svg.append(f"  <!-- Row {idx}: {label} -->")
        svg.append(f"  <text class=\"label\" x=\"60\" y=\"{label_y}\">{label}</text>")
        svg.append(f"  <g transform=\"translate(0,{group_y})\">")
        svg.append("    <use href=\"#axis\" xlink:href=\"#axis\" />")
        svg.append(f"    <line class=\"connector\" x1=\"{x_lo:.1f}\" y1=\"0\" x2=\"{x_hi:.1f}\" y2=\"0\" />")
        svg.append(f"    <circle cx=\"{x_no:.1f}\" cy=\"0\" r=\"8\" class=\"no\" />")
        svg.append(f"    <text class=\"value no\" x=\"{x_no:.1f}\" y=\"-18\" text-anchor=\"middle\">{mean_no:.2f}</text>")
        svg.append(f"    <text class=\"status no\" x=\"{x_no:.1f}\" y=\"20\" text-anchor=\"middle\">No</text>")
        svg.append(f"    <circle cx=\"{x_yes:.1f}\" cy=\"0\" r=\"8\" class=\"yes\" />")
        svg.append(f"    <text class=\"value yes\" x=\"{x_yes:.1f}\" y=\"-18\" text-anchor=\"middle\">{mean_yes:.2f}</text>")
        svg.append(f"    <text class=\"status yes\" x=\"{x_yes:.1f}\" y=\"20\" text-anchor=\"middle\">Yes</text>")
        svg.append(f"    <text class=\"evidence\" x=\"900\" y=\"4\">Strength of Evidence: d = {d:.2f}</text>")
        svg.append("  </g>")
        svg.append("")

    svg.append("</svg>")
    _write_text(out_path, "\n".join(svg) + "\n")


def _svg_header(width: int, height: int) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>"
    )


def _svg_footer() -> str:
    return "</svg>\n"


def _save_forest_plot_svg(
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

    # Dynamic sizing to avoid clipped labels / values.
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

    row_h = 34
    top_margin = 82
    header_h = 22
    base_bottom_margin = 72
    extra_legend_h = 0
    if legend_lines:
        extra_legend_h = 18 * len(legend_lines)
    height = top_margin + header_h + row_h * len(labels) + base_bottom_margin + extra_legend_h

    # Log axis range (fixed ticks for interpretability)
    ticks = [0.01, 0.1, 1, 10, 100, 1000]
    xmin, xmax = ticks[0], ticks[-1]
    log_min = math.log10(xmin)
    log_max = math.log10(xmax)

    def x_map(val: float) -> float:
        if val <= 0 or math.isnan(val):
            return margin_left
        v = min(max(val, xmin), xmax)
        t = (math.log10(v) - log_min) / (log_max - log_min)
        return margin_left + t * plot_w

    svg: List[str] = []
    svg.append(_svg_header(width, height))
    svg.append("<rect width='100%' height='100%' fill='#ffffff' />")
    svg.append(
        f"<text x='{width/2:.1f}' y='28' text-anchor='middle' font-family='{font_family}' "
        f"font-size='16' fill='#222'>{title}</text>"
    )

    # Table headers
    header_y = 66
    svg.append(
        f"<text x='{margin_left - 10}' y='{header_y:.2f}' text-anchor='end' font-family='{font_family}' "
        f"font-size='{label_font}' fill='#222' font-weight='bold'>Predictor</text>"
    )
    value_x = margin_left + plot_w + 12
    svg.append(
        f"<text x='{value_x}' y='{header_y:.2f}' text-anchor='start' font-family='{font_family}' "
        f"font-size='{value_font}' fill='#222' font-weight='bold'>OR [95% CI]</text>"
    )

    # Alternating row stripes
    y0 = top_margin + header_h - 5
    for i in range(len(labels)):
        if i % 2 == 0:
            svg.append(
                f"<rect x='10' y='{y0 + i * row_h:.2f}' width='{width - 20:.2f}' height='{row_h:.2f}' fill='#fafafa' />"
            )

    # Axes box
    y_axis_top = top_margin
    y_axis_bottom = top_margin + row_h * len(labels)
    svg.append(f"<line x1='{margin_left:.2f}' y1='{y_axis_bottom:.2f}' x2='{margin_left + plot_w:.2f}' y2='{y_axis_bottom:.2f}' stroke='#222' stroke-width='1' />")
    svg.append(f"<line x1='{margin_left:.2f}' y1='{y_axis_bottom:.2f}' x2='{margin_left:.2f}' y2='{y_axis_top:.2f}' stroke='#222' stroke-width='1' />")
    svg.append(
        f"<text x='{margin_left + plot_w/2:.2f}' y='{y_axis_bottom + 46:.2f}' text-anchor='middle' "
        f"font-family='{font_family}' font-size='12' fill='#222'>Odds ratio (log scale)</text>"
    )

    # Grid + tick labels
    for t in ticks:
        x = x_map(t)
        svg.append(f"<line x1='{x:.2f}' y1='{y_axis_top:.2f}' x2='{x:.2f}' y2='{y_axis_bottom:.2f}' stroke='#f0f0f0' />")
        svg.append(f"<line x1='{x:.2f}' y1='{y_axis_bottom:.2f}' x2='{x:.2f}' y2='{y_axis_bottom + 6:.2f}' stroke='#666' />")
        svg.append(
            f"<text x='{x:.2f}' y='{y_axis_bottom + 20:.2f}' text-anchor='middle' font-family='{font_family}' "
            f"font-size='{axis_font}' fill='#444'>{t:g}</text>"
        )

    # Reference line at OR=1
    x_ref = x_map(1.0)
    svg.append(f"<line x1='{x_ref:.2f}' y1='{y_axis_top:.2f}' x2='{x_ref:.2f}' y2='{y_axis_bottom:.2f}' stroke='#888' stroke-dasharray='4 4' />")

    # Points + CIs + labels
    for i, (lab, or_, lo_, hi_, val_text) in enumerate(zip(labels, odds_ratios, ci_low, ci_high, value_texts)):
        y = y_axis_top + i * row_h
        y_mid = y + row_h / 2
        svg.append(
            f"<text x='{margin_left - 10}' y='{y_mid + 4:.2f}' text-anchor='end' font-family='{font_family}' "
            f"font-size='{label_font}' fill='#222'>{lab}</text>"
        )

        x_or = x_map(or_)
        x_lo = x_map(lo_) if not math.isnan(lo_) else float("nan")
        x_hi = x_map(hi_) if not math.isnan(hi_) else float("nan")
        if not math.isnan(x_lo) and not math.isnan(x_hi):
            svg.append(f"<line x1='{x_lo:.2f}' y1='{y_mid:.2f}' x2='{x_hi:.2f}' y2='{y_mid:.2f}' stroke='#1f4e5f' stroke-width='2' />")
            svg.append(f"<line x1='{x_lo:.2f}' y1='{y_mid - 3:.2f}' x2='{x_lo:.2f}' y2='{y_mid + 3:.2f}' stroke='#1f4e5f' stroke-width='2' />")
            svg.append(f"<line x1='{x_hi:.2f}' y1='{y_mid - 3:.2f}' x2='{x_hi:.2f}' y2='{y_mid + 3:.2f}' stroke='#1f4e5f' stroke-width='2' />")
        svg.append(f"<circle cx='{x_or:.2f}' cy='{y_mid:.2f}' r='4' fill='#b6403a' />")

        svg.append(
            f"<text x='{value_x}' y='{y_mid + 4:.2f}' text-anchor='start' font-family='{font_family}' "
            f"font-size='{value_font}' fill='#222'>{val_text}</text>"
        )

    # Mini legend
    g_x0 = margin_left + 12
    g_y0 = y_axis_bottom + 70
    svg.append(f"<line x1='{g_x0:.2f}' y1='{g_y0:.2f}' x2='{g_x0 + 50:.2f}' y2='{g_y0:.2f}' stroke='#1f4e5f' stroke-width='2' />")
    svg.append(f"<circle cx='{g_x0 + 25:.2f}' cy='{g_y0:.2f}' r='4' fill='#b6403a' />")
    svg.append(
        f"<text x='{g_x0 + 66:.2f}' y='{g_y0 + 4:.2f}' font-family='{font_family}' font-size='{axis_font}' fill='#444'>"
        f"Dot = odds ratio; line = 95% CI</text>"
    )
    svg.append(f"<line x1='{g_x0 + 25:.2f}' y1='{g_y0 + 10:.2f}' x2='{g_x0 + 25:.2f}' y2='{g_y0 + 26:.2f}' stroke='#888' stroke-dasharray='4 4' />")
    svg.append(
        f"<text x='{g_x0 + 66:.2f}' y='{g_y0 + 22:.2f}' font-family='{font_family}' font-size='{axis_font}' fill='#444'>"
        f"Dashed line = OR=1 (no association)</text>"
    )

    # Extra legend lines
    if legend_lines:
        for j, line in enumerate(legend_lines):
            y = g_y0 + 42 + 18 * j
            svg.append(
                f"<text x='{margin_left:.2f}' y='{y:.2f}' font-family='{font_family}' font-size='{axis_font}' fill='#444'>{line}</text>"
            )

    svg.append(_svg_footer())
    _write_text(path, "\n".join(svg))


def _build_logistic_forest_svg(or_table: List[Dict[str, float]], out_path: Path) -> None:
    # Match the selection logic from `academic_impact_analysis.py`:
    # take the 12 largest |coef| effects from the full model.
    effects = sorted(or_table, key=lambda r: abs(r["coef"]), reverse=True)[:12]

    labels: List[str] = []
    ors: List[float] = []
    lo: List[float] = []
    hi: List[float] = []
    for r in effects:
        label = (
            str(r["feature"])
            .replace("Most_Used_Platform=", "Platform=")
            .replace("Academic_Level=", "Level=")
            .replace("Relationship_Status=", "Rel=")
        )
        labels.append(label)
        ors.append(float(r["or"]))
        lo.append(float(r["lo"]))
        hi.append(float(r["hi"]))

    _save_forest_plot_svg(
        labels,
        ors,
        lo,
        hi,
        out_path,
        title="Academic Impact Model (Full): Strongest Effects (OR, 95% CI)",
        legend_lines=[
            "Numeric predictors are standardized (1 SD); categorical levels are vs a reference level.",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the SVGs in reports/assets/presentation_figures/send/"
    )
    parser.add_argument(
        "--numeric",
        type=Path,
        default=Path("results/academic_impact/numeric_group_comparisons.csv"),
        help="Path to numeric group comparison CSV.",
    )
    parser.add_argument(
        "--or-table",
        type=Path,
        default=Path("results/academic_impact_ridge/logistic_full_odds_ratios.csv"),
        help="Path to ridge logistic odds-ratio table CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/assets/presentation_figures/send"),
        help="Output directory for SVG files.",
    )
    args = parser.parse_args()

    numeric = _load_numeric_summary(args.numeric)
    or_table = _load_or_table(args.or_table)

    out_dir = args.out_dir
    _build_group_diff_svg(numeric, out_dir / "fig_academic_impact_group_diff.svg")
    _build_cohen_d_slide_svg(numeric, out_dir / "cohen_d_slide_graph_only.svg")
    _build_logistic_forest_svg(or_table, out_dir / "fig_logistic_or_forest.svg")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
