#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.font_manager as fm
from pathlib import Path


LTT_PATH = Path("5823.ultrametric_tree.LTT.tsv")
GESTATION = 268 / 365
WEEK = 7 / 365


def step_values_at(times, values, grid):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    idx = np.searchsorted(times, grid, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def smoothed_dlogdt(sub, grid, window_years=0.18):
    sub = sub.sort_values("time_from_birth")
    n_grid = step_values_at(sub["time_from_birth"].to_numpy(), sub["lineages"].to_numpy(), grid)
    logn = np.log(np.clip(n_grid, 1, None))
    rates = np.full_like(logn, np.nan, dtype=float)
    half = window_years / 2
    for i, x in enumerate(grid):
        mask = (grid >= x - half) & (grid <= x + half)
        if mask.sum() < 4:
            continue
        xx = grid[mask]
        yy = logn[mask]
        if np.nanmax(yy) == np.nanmin(yy):
            rates[i] = 0.0
        else:
            rates[i] = np.polyfit(xx, yy, 1)[0]
    return n_grid, logn, np.clip(rates, 0, None)


def main():
    ltt = pd.read_csv(LTT_PATH, sep="\t")

    clade_order = ["a", "b", "c", "d"]
    ltt["clade"] = pd.Categorical(ltt["clade"], categories=clade_order, ordered=True)
    ltt = ltt.sort_values(["clade", "time_from_birth"])

    palette = {
        "a": "#2171b5",
        "b": "#da627d",
        "c": "#783f8e",
        "d": "#d28b26",
    }

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax2 = ax.twinx()
    x_min = -GESTATION - 0.03
    x_max = 1.23
    rate_grid = np.linspace(-GESTATION, x_max, 500)
    rate_rows = []

    for clade in clade_order:
        sub = ltt[ltt["clade"].astype(str) == clade]
        if sub.empty:
            continue
        _, _, rate = smoothed_dlogdt(sub, rate_grid, window_years=0.18)
        ax.step(
            sub["time_from_birth"],
            sub["lineages"],
            where="post",
            color=palette[clade],
            linewidth=1.9,
            label=f"clade {clade}",
        )
        ax2.plot(
            rate_grid,
            rate,
            color=palette[clade],
            linewidth=1.5,
            linestyle="--",
            alpha=0.85,
        )
        for x, r in zip(rate_grid, rate):
            rate_rows.append({"clade": clade, "time_from_birth": x, "smoothed_dlogN_dt": r})

    ax.axvline(-GESTATION, color="#8a8a8a", linestyle=":", linewidth=0.9)
    ax.axvline(0, color="#4d4d4d", linestyle="--", linewidth=0.9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max(ltt["lineages"]) * 1.05)
    xticks = [
        -GESTATION,
        -GESTATION + 12 * WEEK,
        -GESTATION + 28 * WEEK,
        0,
        1.0,
    ]
    xticklabels = [
        "Zygote",
        "12 wpc",
        "28 wpc",
        "Birth",
        "1 yr\nafter birth",
    ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=30, ha="right")
    ax.set_xlabel("Developmental time of branching event")
    ax.set_ylabel("Number of lineages")
    ax2.set_ylabel(r"Smoothed $d\log(N)/dt$", labelpad=12)
    rate_df = pd.DataFrame(rate_rows)
    ymax_rate = np.nanpercentile(rate_df["smoothed_dlogN_dt"], 99)
    ax2.set_ylim(0, max(0.1, ymax_rate * 1.15))
    ax.set_title("5823 lineage-through-time and diversification rate", loc="left", fontsize=12, fontweight="bold")
    ax.legend(title="Top-level clade", frameon=False, loc="center left", bbox_to_anchor=(1.16, 0.58))
    sns.despine(ax=ax)
    ax2.spines["top"].set_visible(False)
    ax.tick_params(
        axis="both",
        which="major",
        bottom=True,
        left=True,
        length=4,
        width=1,
        direction="out",
        color="black",
    )
    ax2.tick_params(
        axis="y",
        which="major",
        right=True,
        length=4,
        width=1,
        direction="out",
        color="black",
    )
    ax.tick_params(axis="both", which="minor", bottom=False, left=False)

    fig.set_size_inches(5.2, 4.0)
    fig.tight_layout()
    pdf = Path("5823.ultrametric_tree.LTT.pdf")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
