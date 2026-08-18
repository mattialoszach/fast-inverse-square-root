#!/usr/bin/env python3
"""Create separate publication-style figures from benchmark CSV output."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter


REQUIRED_COLUMNS = {
    "method",
    "median_ns_per_value",
    "p10_ns_per_value",
    "p90_ns_per_value",
    "gvalues_per_second",
    "max_error_ppm",
    "mean_error_ppm",
}
NUMBER_COLUMNS = REQUIRED_COLUMNS - {"method"}

# A restrained, color-blind-friendly palette. Related implementations share a
# hue, while step count is distinguished by shade.
PALETTE = {
    "standard": "#334155",
    "quake": "#C2417B",
    "magic": "#7C3AED",
    "neon_scalar_1": "#F59E0B",
    "neon_scalar_2": "#D97706",
    "neon_vector_1": "#06A6A6",
    "neon_vector_2": "#2563EB",
    "fallback": "#64748B",
}

INK = "#172033"
MUTED = "#5F6B7A"
GRID = "#DCE2EA"
PAPER = "#FBFCFE"
WHITE = "#FFFFFF"
FRONTIER = "#E45756"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create separate publication-style benchmark figures."
    )
    parser.add_argument("csv", type=Path, help="CSV produced by bench --csv")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="directory for generated figures (default: results/figures)",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "svg", "png"),
        default="pdf",
        help="figure format (default: pdf)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="raster resolution for PNG output (default: 300)",
    )
    return parser.parse_args()


def load_results(path: Path) -> list[dict[str, str | float]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise ValueError("missing columns: " + ", ".join(sorted(missing)))

            results: list[dict[str, str | float]] = []
            for line_number, row in enumerate(reader, start=2):
                method = (row.get("method") or "").strip()
                if not method:
                    raise ValueError(f"line {line_number}: method is empty")

                item: dict[str, str | float] = {"method": method}
                for column in NUMBER_COLUMNS:
                    try:
                        value = float(row[column])
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"line {line_number}: {column} is not a number"
                        ) from error
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(
                            f"line {line_number}: {column} must be finite and non-negative"
                        )
                    item[column] = value
                results.append(item)
    except OSError as error:
        raise ValueError(f"could not read {path}: {error.strerror}") from error

    if not results:
        raise ValueError("CSV contains no benchmark rows")

    for item in results:
        for column in (
            "median_ns_per_value",
            "p10_ns_per_value",
            "p90_ns_per_value",
            "gvalues_per_second",
        ):
            if item[column] <= 0:
                raise ValueError(f"{item['method']}: {column} must be positive")
        if item["p10_ns_per_value"] > item["median_ns_per_value"]:
            raise ValueError(f"{item['method']}: P10 must not exceed the median")
        if item["median_ns_per_value"] > item["p90_ns_per_value"]:
            raise ValueError(f"{item['method']}: median must not exceed P90")
        if item["mean_error_ppm"] > item["max_error_ppm"]:
            raise ValueError(f"{item['method']}: mean error must not exceed max error")

    if not any(item["max_error_ppm"] > 0 for item in results):
        raise ValueError("at least one maximum error must be positive")

    results.sort(key=lambda item: float(item["median_ns_per_value"]))
    return results


def method_color(method: str) -> str:
    name = method.lower()
    if "std::" in name:
        key = "standard"
    elif "quake" in name:
        key = "quake"
    elif "improved magic" in name:
        key = "magic"
    elif "neon scalar" in name:
        key = "neon_scalar_2" if "2 newton" in name else "neon_scalar_1"
    elif "neon 4-wide" in name:
        key = "neon_vector_2" if "2 newton" in name else "neon_vector_1"
    else:
        key = "fallback"
    return PALETTE[key]


def short_label(method: str) -> str:
    replacements = {
        "std::sqrt + division": "Standard library",
        "Quake III, 1 Newton": "Quake III · 1 step",
        "Improved magic, 2 Newton": "Improved magic · 2 steps",
        "NEON scalar, 1 Newton": "NEON scalar · 1 step",
        "NEON scalar, 2 Newton": "NEON scalar · 2 steps",
        "NEON 4-wide, 1 Newton": "NEON 4-wide · 1 step",
        "NEON 4-wide, 2 Newton": "NEON 4-wide · 2 steps",
    }
    return replacements.get(method, method)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": PAPER,
            "savefig.facecolor": WHITE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#9AA6B2",
            "axes.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def polish_axis(axis: plt.Axes, *, grid_axis: str = "x") -> None:
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.65, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)


def add_title(axis: plt.Axes, title: str, subtitle: str) -> None:
    axis.set_title(title, loc="left", pad=22)
    axis.text(
        0,
        1.035,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )


def make_throughput_figure(
    results: list[dict[str, str | float]],
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.4, 4.4), layout="constrained")
    positions = list(range(len(results)))
    throughput = [float(item["gvalues_per_second"]) for item in results]
    low = [1.0 / float(item["p90_ns_per_value"]) for item in results]
    high = [1.0 / float(item["p10_ns_per_value"]) for item in results]
    errors = [
        [value - lower for value, lower in zip(throughput, low)],
        [upper - value for value, upper in zip(throughput, high)],
    ]
    colors = [method_color(str(item["method"])) for item in results]

    axis.barh(
        positions,
        throughput,
        height=0.54,
        color=colors,
        edgecolor=WHITE,
        linewidth=0.5,
        zorder=2,
    )
    axis.errorbar(
        throughput,
        positions,
        xerr=errors,
        fmt="none",
        ecolor=INK,
        elinewidth=0.85,
        capsize=2.7,
        capthick=0.85,
        zorder=3,
    )
    for y, value in zip(positions, throughput):
        axis.annotate(
            f"{value:.2f}",
            (value, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )

    axis.set_yticks(positions, [short_label(str(item["method"])) for item in results])
    axis.invert_yaxis()
    axis.set_xlim(0, max(high) * 1.12)
    axis.set_xlabel("Throughput (billion values s$^{-1}$)")
    axis.tick_params(axis="y", length=0, pad=7)
    axis.spines["left"].set_visible(False)
    polish_axis(axis)
    add_title(
        axis,
        "Throughput by implementation",
        "Median throughput; intervals show the P10–P90 timing range",
    )
    return figure


def make_error_figure(
    results: list[dict[str, str | float]],
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.4, 4.4), layout="constrained")
    positions = list(range(len(results)))
    positive = [
        float(item[key])
        for item in results
        for key in ("mean_error_ppm", "max_error_ppm")
        if float(item[key]) > 0
    ]
    floor = 10 ** math.floor(math.log10(min(positive))) / 2
    mean_error = [max(float(item["mean_error_ppm"]), floor) for item in results]
    max_error = [max(float(item["max_error_ppm"]), floor) for item in results]

    for y, mean, maximum, item in zip(positions, mean_error, max_error, results):
        color = method_color(str(item["method"]))
        axis.plot([mean, maximum], [y, y], color=color, alpha=0.38, linewidth=2.0, zorder=1)
        axis.scatter(
            mean,
            y,
            s=38,
            marker="o",
            facecolor=PAPER,
            edgecolor=color,
            linewidth=1.5,
            zorder=3,
        )
        axis.scatter(
            maximum,
            y,
            s=38,
            marker="s",
            color=color,
            edgecolor=WHITE,
            linewidth=0.5,
            zorder=3,
        )

    axis.set_xscale("log")
    axis.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_yticks(positions, [short_label(str(item["method"])) for item in results])
    axis.invert_yaxis()
    axis.set_xlabel("Relative error (ppm, logarithmic scale)")
    axis.tick_params(axis="y", length=0, pad=7)
    axis.spines["left"].set_visible(False)
    polish_axis(axis)
    axis.legend(
        handles=[
            Line2D([], [], marker="o", markerfacecolor=PAPER, markeredgecolor=INK, linestyle="none", label="Mean error"),
            Line2D([], [], marker="s", markerfacecolor=INK, markeredgecolor=INK, linestyle="none", label="Maximum error"),
        ],
        loc="lower right",
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    add_title(
        axis,
        "Numerical error by implementation",
        "Relative error against double-precision 1/√x; lower is better",
    )
    return figure


def pareto_frontier(results: list[dict[str, str | float]]) -> list[int]:
    indices = []
    for i, candidate in enumerate(results):
        dominated = any(
            j != i
            and other["gvalues_per_second"] >= candidate["gvalues_per_second"]
            and other["max_error_ppm"] <= candidate["max_error_ppm"]
            and (
                other["gvalues_per_second"] > candidate["gvalues_per_second"]
                or other["max_error_ppm"] < candidate["max_error_ppm"]
            )
            for j, other in enumerate(results)
        )
        if not dominated:
            indices.append(i)
    return sorted(indices, key=lambda i: float(results[i]["gvalues_per_second"]))


def make_tradeoff_figure(
    results: list[dict[str, str | float]],
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.1, 5.0), layout="constrained")
    throughput = [float(item["gvalues_per_second"]) for item in results]
    errors = [float(item["max_error_ppm"]) for item in results]
    positive_errors = [value for value in errors if value > 0]
    floor = 10 ** math.floor(math.log10(min(positive_errors))) / 2
    errors = [max(value, floor) for value in errors]
    colors = [method_color(str(item["method"])) for item in results]
    frontier = pareto_frontier(results)

    if frontier:
        axis.plot(
            [throughput[i] for i in frontier],
            [errors[i] for i in frontier],
            color=FRONTIER,
            linewidth=1.4,
            zorder=1,
        )

    for index, (item, x, y, color) in enumerate(
        zip(results, throughput, errors, colors)
    ):
        on_frontier = index in frontier
        axis.scatter(
            x,
            y,
            s=64 if on_frontier else 52,
            color=color,
            edgecolor=FRONTIER if on_frontier else WHITE,
            linewidth=1.8 if on_frontier else 0.8,
            zorder=3,
        )
        x_offset = 7 if x < max(throughput) * 0.86 else -7
        alignment = "left" if x_offset > 0 else "right"
        y_offset = -10 if "Quake" in str(item["method"]) else 6
        axis.annotate(
            short_label(str(item["method"])),
            (x, y),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=7.6,
            color=INK,
        )

    axis.set_yscale("log")
    axis.set_xlim(left=0)
    axis.set_xlabel("Throughput (billion values s$^{-1}$)")
    axis.set_ylabel("Maximum relative error (ppm)")
    polish_axis(axis, grid_axis="both")
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                color=FRONTIER,
                marker="o",
                markerfacecolor=WHITE,
                markeredgecolor=FRONTIER,
                linewidth=1.4,
                label="Pareto frontier",
            )
        ],
        loc="upper left",
    )
    add_title(
        axis,
        "Speed–accuracy trade-off",
        "The lower-right region combines high throughput with low maximum error",
    )
    return figure


def save_figures(
    results: list[dict[str, str | float]],
    output_dir: Path,
    file_format: str,
    dpi: int,
) -> list[Path]:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "throughput": make_throughput_figure(results),
        "accuracy": make_error_figure(results),
        "speed-accuracy-tradeoff": make_tradeoff_figure(results),
    }
    written = []
    for name, figure in figures.items():
        path = output_dir / f"{name}.{file_format}"
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        written.append(path)
    return written


def main() -> int:
    args = parse_args()
    if args.dpi < 72:
        print("error: --dpi must be at least 72", file=sys.stderr)
        return 1

    try:
        results = load_results(args.csv)
        written = save_figures(
            results, args.output_dir, args.format, args.dpi
        )
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
