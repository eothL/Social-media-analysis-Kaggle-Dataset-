from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    HAS_MPL = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    HAS_MPL = False


def _percent_to_fraction(percent: float) -> float:
    if percent > 1.0:
        return percent / 100.0
    return percent


def _draw_waffle(ax, percent: float, rows: int = 10, cols: int = 10) -> None:
    filled_color = "#c35a3a"
    empty_color = "#b7b7b7"
    total = rows * cols
    filled = int(round(_percent_to_fraction(percent) * total))
    square = 0.8
    gap = 0.1
    for i in range(total):
        row = rows - 1 - (i // cols)
        col = i % cols
        x = col + gap
        y = row + gap
        color = filled_color if i < filled else empty_color
        patch = FancyBboxPatch(
            (x, y),
            square,
            square,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=0,
            facecolor=color,
        )
        ax.add_patch(patch)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_text(
    x: int,
    y: int,
    lines: list[str],
    size: int,
    weight: str,
    color: str,
    line_height: float = 1.25,
    anchor: str = "start",
) -> str:
    parts = [
        (
            f"<text x='{x}' y='{y}' text-anchor='{anchor}' "
            f"font-family='Georgia, serif' font-size='{size}' "
            f"font-weight='{weight}' fill='{color}'>"
        )
    ]
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else int(size * line_height)
        parts.append(
            f"<tspan x='{x}' dy='{dy}'>{_escape_xml(line)}</tspan>"
        )
    parts.append("</text>")
    return "\n".join(parts)


def build_svg(
    percent: float,
    n_students: int,
    ci_low: float,
    ci_high: float,
    output_path: Path,
    graph_only: bool,
    background: str | None,
) -> None:
    if graph_only:
        rows = 10
        cols = 10
        square = 36
        gap = 8
        margin = 20
        width = margin * 2 + cols * square + (cols - 1) * gap
        height = margin * 2 + rows * square + (rows - 1) * gap
        svg: list[str] = [
            "<?xml version='1.0' encoding='UTF-8'?>",
            (
                f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' "
                f"height='{height}' viewBox='0 0 {width} {height}'>"
            ),
        ]
        if background:
            svg.append(f"<rect width='{width}' height='{height}' fill='{background}' />")
        filled_color = "#c35a3a"
        empty_color = "#b7b7b7"
        total = rows * cols
        filled = int(round(_percent_to_fraction(percent) * total))
        for i in range(total):
            row = i // cols
            col = i % cols
            x = margin + col * (square + gap)
            y = margin + row * (square + gap)
            color = filled_color if i < filled else empty_color
            svg.append(
                (
                    f"<rect x='{x}' y='{y}' width='{square}' height='{square}' "
                    f"rx='6' ry='6' fill='{color}' />"
                )
            )
        svg.append("</svg>")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(svg), encoding="utf-8")
        return

    width = 1920
    height = 1080
    svg: list[str] = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        (
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' "
            f"height='{height}' viewBox='0 0 {width} {height}'>"
        ),
    ]
    if background:
        svg.append(f"<rect width='{width}' height='{height}' fill='{background}' />")

    svg.append(
        _svg_text(
            120,
            140,
            ["The Problem is Pervasive and", "Demands Investigation"],
            size=72,
            weight="700",
            color="#1c2b3a",
            line_height=1.15,
        )
    )

    grid_x = 120
    grid_y = 320
    square = 36
    gap = 8
    filled_color = "#c35a3a"
    empty_color = "#b7b7b7"
    total = 100
    filled = int(round(_percent_to_fraction(percent) * total))
    for i in range(total):
        row = i // 10
        col = i % 10
        x = grid_x + col * (square + gap)
        y = grid_y + row * (square + gap)
        color = filled_color if i < filled else empty_color
        svg.append(
            (
                f"<rect x='{x}' y='{y}' width='{square}' height='{square}' "
                f"rx='6' ry='6' fill='{color}' />"
            )
        )

    percent_text = f"{_percent_to_fraction(percent) * 100:.1f}%"
    svg.append(
        _svg_text(
            720,
            500,
            [percent_text],
            size=96,
            weight="700",
            color="#c35a3a",
        )
    )

    body_lines = [
        "Nearly two-thirds of students",
        "report that social media",
        "negatively affects their",
        "academic performance.",
    ]
    svg.append(
        _svg_text(
            720,
            600,
            body_lines,
            size=28,
            weight="400",
            color="#1c2b3a",
            line_height=1.3,
        )
    )

    bullets = [
        f"- Based on a sample of {n_students} students.",
        "- The 95% confidence interval for this",
        f"  proportion is [{ci_low:.1f}%, {ci_high:.1f}%].",
    ]
    svg.append(
        _svg_text(
            720,
            760,
            bullets,
            size=20,
            weight="400",
            color="#6b6b6b",
            line_height=1.3,
        )
    )

    right_lines = [
        "Our investigation",
        "seeks to answer:",
        "What truly separates",
        "the students who",
        "report academic",
        "harm from those who",
        "do not?",
    ]
    svg.append(
        _svg_text(
            1360,
            380,
            right_lines,
            size=32,
            weight="500",
            color="#1c2b3a",
            line_height=1.2,
        )
    )

    svg.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def build_figure(
    percent: float,
    n_students: int,
    ci_low: float,
    ci_high: float,
    output_path: Path,
    graph_only: bool,
    background: str | None,
) -> None:
    if HAS_MPL and output_path.suffix.lower() != ".svg":
        if graph_only:
            rows = 10
            cols = 10
            square = 0.8
            gap = 0.1
            fig = plt.figure(figsize=(4, 4), dpi=200)
            if background:
                fig.patch.set_facecolor(background)
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            ax.axis("off")
            _draw_waffle(ax, percent, rows=rows, cols=cols)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path,
                bbox_inches="tight",
                facecolor=fig.get_facecolor() if background else "none",
            )
            return

        fig = plt.figure(figsize=(13.333, 7.5), dpi=150)
        if background:
            fig.patch.set_facecolor(background)

        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        title = "The Problem is Pervasive and\nDemands Investigation"
        fig.text(
            0.06,
            0.88,
            title,
            ha="left",
            va="top",
            fontsize=32,
            fontweight="bold",
            color="#1c2b3a",
            family="serif",
        )

        waffle_ax = fig.add_axes([0.06, 0.18, 0.32, 0.45])
        _draw_waffle(waffle_ax, percent)

        percent_text = f"{_percent_to_fraction(percent) * 100:.1f}%"
        fig.text(
            0.42,
            0.56,
            percent_text,
            ha="left",
            va="center",
            fontsize=48,
            fontweight="bold",
            color="#c35a3a",
            family="serif",
        )

        body = (
            "Nearly two-thirds of students\n"
            "report that social media\n"
            "negatively affects their\n"
            "academic performance."
        )
        fig.text(
            0.42,
            0.44,
            body,
            ha="left",
            va="top",
            fontsize=14,
            color="#1c2b3a",
            family="serif",
        )

        bullets = (
            f"- Based on a sample of {n_students} students.\n"
            f"- The 95% confidence interval for this\n"
            f"  proportion is [{ci_low:.1f}%, {ci_high:.1f}%]."
        )
        fig.text(
            0.42,
            0.26,
            bullets,
            ha="left",
            va="top",
            fontsize=10,
            color="#6b6b6b",
            family="serif",
        )

        right_text = (
            "Our investigation\n"
            "seeks to answer:\n"
            "What truly separates\n"
            "the students who\n"
            "report academic\n"
            "harm from those who\n"
            "do not?"
        )
        fig.text(
            0.70,
            0.62,
            right_text,
            ha="left",
            va="top",
            fontsize=16,
            color="#1c2b3a",
            family="serif",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            bbox_inches="tight",
            facecolor=fig.get_facecolor() if background else "none",
        )
        return

    if output_path.suffix.lower() != ".svg":
        output_path = output_path.with_suffix(".svg")
    build_svg(
        percent,
        n_students,
        ci_low,
        ci_high,
        output_path,
        graph_only,
        background,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a slide-style waffle chart for academic impact."
    )
    parser.add_argument("--percent", type=float, default=64.3)
    parser.add_argument("--n", type=int, default=705)
    parser.add_argument("--ci-low", type=float, default=60.6)
    parser.add_argument("--ci-high", type=float, default=67.7)
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument("--background", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/assets/presentation_figures/"
            "percentage_students_impacted_by_social_media_generated.png"
        ),
    )
    args = parser.parse_args()
    background = args.background
    if not background and not args.graph_only:
        background = "#f7f3ef"
    build_figure(
        args.percent,
        args.n,
        args.ci_low,
        args.ci_high,
        args.output,
        args.graph_only,
        background,
    )


if __name__ == "__main__":
    main()
