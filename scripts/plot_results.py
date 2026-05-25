#!/usr/bin/env python3
"""
plot_results.py
---------------
Visualise benchmark results from benchmark_results_real.csv (or the
synthetic equivalent).

Usage:
    python scripts/plot_results.py                          # uses real CSV
    python scripts/plot_results.py data/results/benchmark_results.csv

Produces:  data/results/plots/benchmark_summary.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ------------------------------------------------------------------ #
# load data
# ------------------------------------------------------------------ #
def _load(path: str) -> dict:
    """Return data as nested dict: data[schedule][density] = list of scores."""
    import csv
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "schedule": row["schedule"],
                "density": float(row["density"]),
                "combined_score": float(row["combined_score"]),
                "pearson_r": float(row["pearson_r"]),
                "rmsd_norm": float(row["rmsd_norm"]),
            })
    return rows


def _group(rows, metric):
    """Return (schedules, densities, mean_matrix, std_matrix)."""
    schedules = sorted(set(r["schedule"] for r in rows))
    densities = sorted(set(r["density"] for r in rows))
    mean = np.zeros((len(schedules), len(densities)))
    std  = np.zeros((len(schedules), len(densities)))
    for i, sch in enumerate(schedules):
        for j, den in enumerate(densities):
            vals = [r[metric] for r in rows
                    if r["schedule"] == sch and r["density"] == den]
            mean[i, j] = np.mean(vals)
            std[i, j]  = np.std(vals)
    return schedules, densities, mean, std


# ------------------------------------------------------------------ #
# colour / style helpers
# ------------------------------------------------------------------ #
PALETTE = ["#4C72B0", "#DD8452", "#55A868"]
SCHEDULE_LABELS = {
    "poisson_gap":     "Poisson-gap",
    "uniform_random":  "Uniform random",
    "quantile_biased": "Quantile-biased",
}


# ------------------------------------------------------------------ #
# individual plots
# ------------------------------------------------------------------ #
def _plot_line(ax, rows, metric, ylabel, title):
    """Line plot: score vs density, one line per schedule."""
    schedules, densities, mean, std = _group(rows, metric)
    for i, sch in enumerate(schedules):
        ax.plot(densities, mean[i], marker="o", color=PALETTE[i],
                label=SCHEDULE_LABELS.get(sch, sch), linewidth=2)
        ax.fill_between(densities,
                        mean[i] - std[i], mean[i] + std[i],
                        alpha=0.15, color=PALETTE[i])
    ax.set_xlabel("Sampling density")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(densities)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def _plot_box(ax, rows, metric, ylabel, title):
    """Box plot: distribution across seeds per schedule."""
    schedules = sorted(set(r["schedule"] for r in rows))
    data = [[r[metric] for r in rows if r["schedule"] == sch]
            for sch in schedules]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(schedules) + 1))
    ax.set_xticklabels(
        [SCHEDULE_LABELS.get(s, s) for s in schedules],
        fontsize=8,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def _plot_heatmap(ax, rows, metric, label, title):
    """Heatmap: schedule × density, colour = score."""
    schedules, densities, mean, _ = _group(rows, metric)
    im = ax.imshow(mean, aspect="auto", cmap="RdYlGn",
                   vmin=0 if "rmsd" not in metric else None)
    ax.set_xticks(range(len(densities)))
    ax.set_xticklabels([f"{d:.0%}" for d in densities])
    ax.set_yticks(range(len(schedules)))
    ax.set_yticklabels([SCHEDULE_LABELS.get(s, s) for s in schedules],
                       fontsize=8)
    ax.set_xlabel("Sampling density")
    ax.set_title(title, fontweight="bold")
    plt.colorbar(im, ax=ax, label=label, shrink=0.8)
    for i in range(len(schedules)):
        for j in range(len(densities)):
            ax.text(j, i, f"{mean[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        repo, "data", "results", "benchmark_results_real.csv"
    )
    if not os.path.exists(csv_path):
        csv_path = os.path.join(repo, "data", "results", "benchmark_results.csv")
    if not os.path.exists(csv_path):
        print("No results CSV found. Run run_benchmark.py or run_benchmark_real.py first.")
        sys.exit(1)

    rows = _load(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    out_dir = os.path.join(repo, "data", "results", "plots")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("NUS Schedule Benchmark", fontsize=14, fontweight="bold", y=1.01)

    # Row 1: combined score
    _plot_line(axes[0, 0], rows, "combined_score",
               "Combined score", "Score vs density")
    _plot_box(axes[0, 1], rows, "combined_score",
              "Combined score", "Score distribution")
    _plot_heatmap(axes[0, 2], rows, "combined_score",
                  "Score", "Score heatmap\n(higher = better)")

    # Row 2: Pearson r and RMSD
    _plot_line(axes[1, 0], rows, "pearson_r",
               "Pearson r", "Peak volume correlation vs density")
    _plot_box(axes[1, 1], rows, "pearson_r",
              "Pearson r", "Correlation distribution")
    _plot_heatmap(axes[1, 2], rows, "rmsd_norm",
                  "RMSD", "RMSD heatmap\n(lower = better)")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "benchmark_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
