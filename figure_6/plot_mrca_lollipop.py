#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


DATA_PATH = Path("ultrametric_tree_mrca.tsv")
GESTATION = 268 / 365
SAMPLES = ["5559", "5657", "5823"]


def point_size(n: float) -> float:
    return float(np.clip(math.sqrt(n) * 35, 26, 210))


def main() -> None:
    df = pd.read_csv(DATA_PATH, sep="\t")
    df = df[(df["is_tip"] == False) & df["top_level_clade"].notna()].copy()
    df = df[df["descendant_cells"] >= 2].copy()
    df["sample_clade"] = df["sample"].astype(str) + "-" + df["top_level_clade"].astype(str)
    if "CH_fraction_plot" not in df.columns:
        df["CH_fraction_plot"] = df["CH_cells"] / df["descendant_cells"]
        df.loc[df["sample"].astype(str) == "5559", "CH_fraction_plot"] = np.nan
    df["is_terminal_mrca"] = (df["direct_leaf_cells"] > 0) & (df["direct_child_clades"] == 0)

    order: list[str] = []
    for sample in SAMPLES:
        sub = df[df["sample"].astype(str) == sample]
        for clade in sorted(sub["top_level_clade"].astype(str).unique()):
            order.append(f"{sample}-{clade}")

    # Radius: keep small gaps between samples, but preserve compact rings.
    r_map: dict[str, float] = {}
    r = 1.25
    for sample in SAMPLES:
        sample_order = [o for o in order if o.startswith(sample + "-")]
        for label in sample_order:
            r_map[label] = r
            r += 0.62
        r += 0.32

    x_min = -GESTATION
    x_max = max(1.30, float(df["time_from_birth"].max()) + 0.06)
    theta_start = np.deg2rad(215)
    theta_span = np.deg2rad(290)

    def theta_from_time(x: float) -> float:
        return theta_start + (float(x) - x_min) / (x_max - x_min) * theta_span

    theta_conception = theta_from_time(-GESTATION)
    theta_birth = theta_from_time(0.0)
    theta_max = theta_from_time(x_max)

    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    fig = plt.figure(figsize=(8.0, 8.0))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("white")

    rng = np.random.default_rng(17)
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, 1)
    r_max = max(r_map.values()) + 0.9

    # Background rings and labels.
    theta_grid = np.linspace(theta_conception, theta_max, 400)
    for label in order:
        rr = r_map[label]
        ax.plot(theta_grid, np.full_like(theta_grid, rr), color="#eeeeee", linewidth=0.9, zorder=0)
        ax.text(
            theta_conception - 0.08,
            rr,
            label,
            ha="right",
            va="center",
            fontsize=9,
            color="#333333",
            rotation=0,
        )

    # Conception/Birth reference lines.
    for theta, label, color, ls, lw in [
        (theta_conception, "Conception", "#8a8a8a", ":", 1.25),
        (theta_birth, "Birth", "#4d4d4d", "--", 1.25),
    ]:
        ax.plot([theta, theta], [0.75, r_max], color=color, linestyle=ls, linewidth=lw, zorder=1)
        ax.text(theta, r_max + 0.28, label, ha="center", va="center", fontsize=10, color=color)

    # Pale lollipop arcs from conception to each inferred MRCA time.
    for _, row in df.iterrows():
        rr = r_map[row["sample_clade"]] + rng.normal(0, 0.025)
        theta1 = theta_from_time(row["time_from_birth"])
        arc = np.linspace(theta_conception, theta1, 80)
        ax.plot(arc, np.full_like(arc, rr), color="#e4e4e4", linewidth=0.85, zorder=1)

    # Points.
    thetas: list[float] = []
    radii: list[float] = []
    sizes: list[float] = []
    facecolors: list[str | tuple[float, float, float, float]] = []
    edgecolors: list[str] = []
    linewidths: list[float] = []
    for _, row in df.iterrows():
        thetas.append(theta_from_time(row["time_from_birth"]))
        radii.append(r_map[row["sample_clade"]] + rng.normal(0, 0.04))
        sizes.append(point_size(row["descendant_cells"]))
        if str(row["sample"]) == "5559" or pd.isna(row["CH_fraction_plot"]):
            facecolors.append("#bdbdbd")
        else:
            facecolors.append(cmap(norm(row["CH_fraction_plot"])))
        if bool(row["is_terminal_mrca"]):
            edgecolors.append("#d62728")
            linewidths.append(0.65)
        else:
            edgecolors.append("#9a9a9a")
            linewidths.append(0.45)

    ax.scatter(
        thetas,
        radii,
        s=sizes,
        c=facecolors,
        edgecolors=edgecolors,
        linewidths=linewidths,
        alpha=0.92,
        zorder=3,
    )

    ax.set_ylim(0.65, r_max + 0.55)
    ax.set_title("MRCA timing", y=1.04, fontweight="bold")

    # A small time guide along the outer arc.
    tick_values = [-GESTATION, -0.5, 0, 0.5, 1.0]
    tick_values = [v for v in tick_values if x_min <= v <= x_max]
    for v in tick_values:
        th = theta_from_time(v)
        ax.plot([th, th], [r_max - 0.06, r_max + 0.06], color="#777777", linewidth=0.7)
        lab = "zygote" if abs(v + GESTATION) < 1e-6 else f"{v:g}"
        ax.text(th, r_max + 0.18, lab, ha="center", va="center", fontsize=8, color="#555555")
    ax.text(theta_from_time(0.55), r_max + 0.48, "Years relative to birth", ha="center", va="center", fontsize=9)

    # Colorbar.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.08, shrink=0.76)
    cbar.set_label("CH fraction", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Legends.
    size_values = [2, 5, 10, 25, 50]
    size_handles = [
        ax.scatter([], [], s=point_size(v), facecolor="white", edgecolor="#333333", linewidth=0.8)
        for v in size_values
    ]
    legend1 = ax.legend(
        size_handles,
        [str(v) for v in size_values],
        title="Descendant\ncells",
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.00),
        labelspacing=0.8,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(legend1)

    edge_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#d62728", markeredgewidth=0.85, markersize=7),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#9a9a9a", markeredgewidth=0.65, markersize=7),
    ]
    ax.legend(
        edge_handles,
        ["terminal MRCA node", "internal MRCA node"],
        title="Node type",
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.00),
        fontsize=8,
        title_fontsize=9,
    )


    fig.tight_layout()
    out_prefix = Path("ultrametric_tree_mrca_lollipop")
    fig.savefig(str(out_prefix) + ".pdf")
    df.to_csv(str(out_prefix) + ".plot_data.tsv", sep="\t", index=False)
    print(f"Wrote {out_prefix}.pdf")


if __name__ == "__main__":
    main()
