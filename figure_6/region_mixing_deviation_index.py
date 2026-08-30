#!/usr/bin/env python3
"""Ridge plots for departure from CH/V mixing within terminal subclades."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


FILL_PALETTE = {
    "clade_a": "#6baed6",
    "clade_b": "#ffa5ab",
    "clade_c": "#bfacc8",
    "clade_d": "#f2b770",
}

LINE_PALETTE = {
    "clade_a": "#2171b5",
    "clade_b": "#da627d",
    "clade_c": "#783f8e",
    "clade_d": "#b15928",
}


def load_plotter(script_path: Path):
    spec = importlib.util.spec_from_file_location("plot_lineage_tree_snv_burden", script_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load {script_path}")
    spec.loader.exec_module(module)
    return module


def collect_subtree_cells(node: str, children: dict[str, list[str]], cells_by_node: dict[str, list[str]]) -> list[str]:
    cells = list(cells_by_node.get(node, []))
    for child in children.get(node, []):
        cells.extend(collect_subtree_cells(child, children, cells_by_node))
    return cells


def build_terminal_subclade_info(sample: str, matrix: pd.DataFrame, cell_regions: dict[str, str], cell_burden: dict[str, float], plotter) -> pd.DataFrame:
    """Partition cells into non-overlapping terminal subclades.

    A terminal subclade is defined by the deepest mutation node to which one or
    more cells are directly attached. Directly attached cells at an internal
    node are treated as one terminal subclade, and descendants under child nodes
    form separate terminal subclades. This produces non-overlapping groups that
    partition the leaves within each top-level clade.
    """
    parent_map, children, attachments = plotter.infer_tree_from_matrix(matrix)
    _, cells_by_node = plotter.assign_leaf_x(list(matrix.columns), attachments)

    root_children = children.get(plotter.PSEUDO_ROOT, [])
    clade_letters = list("abcdefghijklmnopqrstuvwxyz")
    clade_by_root = {root: clade_letters[i] for i, root in enumerate(root_children)}

    rows = []
    for root in root_children:
        clade = clade_by_root[root]
        cluster_i = 1

        def visit(node: str) -> None:
            nonlocal cluster_i
            direct_cells = list(cells_by_node.get(node, []))
            if direct_cells:
                cluster = f"{clade}_{cluster_i}"
                cluster_i += 1
                for cell in direct_cells:
                    rows.append(
                        {
                            "sample": sample,
                            "cell": cell,
                            "clade": clade,
                            "clade_root": root,
                            "cluster": cluster,
                            "cluster_node": node,
                            "region": cell_regions.get(cell, "NA"),
                            "burden": cell_burden.get(cell, np.nan),
                        }
                    )
            for child in children.get(node, []):
                visit(child)

        visit(root)

    return pd.DataFrame(rows)


def _split_cells(value: str) -> list[str]:
    if pd.isna(value) or str(value) == "":
        return []
    return [x for x in str(value).split(";") if x]


def build_terminal_subclade_info_from_collapsed_node_map(
    sample: str,
    collapsed_node_map: Path,
    cell_regions: dict[str, str],
    cell_burden: dict[str, float],
) -> pd.DataFrame:
    """Partition cells into terminal subclades from the collapsed tree node map.

    The collapsed node map has one row per resolved internal lineage node, with
    each node represented by its full descendant cell set. A terminal subclade
    is defined as cells directly attached to an internal node after subtracting
    all child-node descendant cell sets. This keeps subclades mutually exclusive
    and uses the same collapsed-node definition as the final rtreefit tree.
    """
    node_df = pd.read_csv(collapsed_node_map, sep="\t")
    required = {"node_id", "parent_node_id", "cells", "top_level_clade", "top_level_clade_root_mutation"}
    missing = required - set(node_df.columns)
    if missing:
        raise ValueError(f"{collapsed_node_map} missing required columns: {sorted(missing)}")

    node_df = node_df.copy()
    node_df["cell_list"] = node_df["cells"].map(_split_cells)
    cell_sets = {row.node_id: set(row.cell_list) for row in node_df.itertuples(index=False)}

    children: dict[str, list[str]] = {}
    for row in node_df.itertuples(index=False):
        parent = str(row.parent_node_id)
        if parent == "__PSEUDO_ROOT__":
            continue
        children.setdefault(parent, []).append(str(row.node_id))

    first_rank = node_df.set_index("node_id").get("first_mutation_rank")
    if first_rank is not None:
        rank_map = first_rank.to_dict()
        for parent in list(children):
            children[parent].sort(key=lambda x: rank_map.get(x, 10**9))

    rows = []
    cluster_i_by_clade: dict[str, int] = {}
    for row in node_df.sort_values(["top_level_clade", "first_mutation_rank"], na_position="last").itertuples(index=False):
        node = str(row.node_id)
        clade = str(row.top_level_clade)
        if clade == "nan" or clade == "":
            continue
        child_cells = set()
        for child in children.get(node, []):
            child_cells |= cell_sets.get(child, set())
        direct_cells = sorted(cell_sets.get(node, set()) - child_cells)
        if not direct_cells:
            continue
        cluster_i_by_clade[clade] = cluster_i_by_clade.get(clade, 0) + 1
        cluster = f"{clade}_{cluster_i_by_clade[clade]}"
        for cell in direct_cells:
            rows.append(
                {
                    "sample": sample,
                    "cell": cell,
                    "clade": clade,
                    "clade_root": str(row.top_level_clade_root_mutation),
                    "cluster": cluster,
                    "cluster_node": node,
                    "region": cell_regions.get(cell, "NA"),
                    "burden": cell_burden.get(cell, np.nan),
                }
            )

    return pd.DataFrame(rows)


def compute_prop_ch(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, sub_df in df.groupby("cluster", sort=False):
        n_total = len(sub_df)
        n_ch = int((sub_df["region"] == "CH").sum())
        rows.append(
            {
                "cluster": cluster,
                "n_total": n_total,
                "n_ch": n_ch,
                "prop_ch": n_ch / n_total if n_total > 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def compute_mixture_index(cluster_ratio_df: pd.DataFrame, global_ratio: float, global_counts: int, min_cells: int = 3) -> float:
    res = 0.0
    for _, row in cluster_ratio_df.iterrows():
        counts = row["n_total"]
        if counts >= min_cells:
            res += (counts / global_counts) * abs(row["prop_ch"] - global_ratio)
    return float(res)


def random_shuffle_test(clade_df: pd.DataFrame, global_ratio: float, n_iterations: int, min_cells: int, rng: np.random.Generator) -> list[float]:
    n_cells = len(clade_df)
    regions = clade_df["region"].to_numpy(copy=True)
    mix_indices = []
    for _ in range(n_iterations):
        shuffled_df = clade_df.copy()
        shuffled_df["region"] = rng.permutation(regions)
        ratio_df = compute_prop_ch(shuffled_df)
        mix_indices.append(compute_mixture_index(ratio_df, global_ratio, n_cells, min_cells=min_cells))
    return mix_indices


def run_permutation(clade_info: pd.DataFrame, n_iterations: int, min_cells: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    all_perm_rows = []
    summary_rows = []

    for clade, clade_df in clade_info.groupby("clade", sort=False):
        clade_df = clade_df.reset_index(drop=True)
        total = len(clade_df)
        if total == 0:
            continue
        global_ratio = float((clade_df["region"] == "CH").sum() / total)
        ratio_df = compute_prop_ch(clade_df)
        observed = compute_mixture_index(ratio_df, global_ratio, total, min_cells=min_cells)
        perm_values = random_shuffle_test(clade_df, global_ratio, n_iterations=n_iterations, min_cells=min_cells, rng=rng)
        p_value = float(sum(1 for value in perm_values if value >= observed) / len(perm_values))

        sample_name = f"clade_{clade}"
        for value in perm_values:
            all_perm_rows.append({"sample": sample_name, "clade": clade, "nullmetric": value})
        summary_rows.append(
            {
                "clade": clade,
                "ridge_sample": sample_name,
                "n_cells": total,
                "n_terminal_subclades": ratio_df.shape[0],
                "n_terminal_subclades_min_cells": int((ratio_df["n_total"] >= min_cells).sum()),
                "global_CH_ratio": global_ratio,
                "observed_mixture_index": observed,
                "p_value": p_value,
                "n_iterations": n_iterations,
                "min_cells": min_cells,
            }
        )

    return pd.DataFrame(all_perm_rows), pd.DataFrame(summary_rows)


def plot_ridges(sample: str, perm_df: pd.DataFrame, summary_df: pd.DataFrame, output_prefix: Path) -> None:
    categories = [f"clade_{clade}" for clade in summary_df["clade"].tolist()]
    perm_df = perm_df.copy()
    perm_df["sample"] = pd.Categorical(perm_df["sample"], categories=categories, ordered=True)

    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    height = 1.05
    aspect = 4.2
    g = sns.FacetGrid(
        perm_df,
        row="sample",
        hue="sample",
        aspect=aspect,
        height=height,
        palette={k: FILL_PALETTE.get(k, "#cccccc") for k in categories},
    )
    g.map(
        sns.kdeplot,
        "nullmetric",
        bw_adjust=1.2,
        clip_on=False,
        fill=True,
        alpha=0.80,
        linewidth=1,
    )

    max_x = max(perm_df["nullmetric"].max(), summary_df["observed_mixture_index"].max()) * 1.15
    max_x = max(max_x, 0.05)
    for ax, ridge_sample in zip(g.axes.flat, categories):
        row = summary_df.loc[summary_df["ridge_sample"] == ridge_sample].iloc[0]
        obs = row["observed_mixture_index"]
        p_value = row["p_value"]
        color = LINE_PALETTE.get(ridge_sample, "#333333")
        ax.axvline(obs, color=color, lw=2, ls="--")
        ax.hlines(y=0, xmin=0, xmax=max_x, color="gray", linewidth=1)
        ax.text(
            max_x * 0.98,
            0.55,
            rf"$p = {p_value:.3g}$",
            ha="right",
            va="center",
            transform=ax.get_xaxis_transform(),
            fontsize=10,
        )
        ax.text(
            -0.02 * max_x,
            0.50,
            f"Clade {row['clade']}",
            ha="right",
            va="center",
            transform=ax.get_xaxis_transform(),
            fontsize=11,
        )

    g.figure.subplots_adjust(hspace=-0.40)
    g.set_titles("")
    g.set(yticks=[], ylabel="", xlabel="Deviation of CH/V composition from expected mixing")
    g.set(xlim=(0, max_x))
    g.despine(bottom=True, left=True)
    g.figure.suptitle(sample, x=0.08, y=1.02, ha="left", fontsize=12, weight="bold")
    g.figure.tight_layout()
    g.figure.savefig(output_prefix.parent / f"{output_prefix.name}.pdf", bbox_inches="tight")
    plt.close(g.figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--snv-burden", type=Path, required=True)
    parser.add_argument("--plotter", type=Path, required=True)
    parser.add_argument("--collapsed-node-map", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-iterations", type=int, default=1000)
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    plotter = load_plotter(args.plotter)
    cell_regions = plotter.read_cell_region(args.snv_burden)
    cell_burden = plotter.read_cell_burden(args.snv_burden)

    if args.collapsed_node_map is not None:
        clade_info = build_terminal_subclade_info_from_collapsed_node_map(
            args.sample,
            args.collapsed_node_map,
            cell_regions,
            cell_burden,
        )
    else:
        matrix = plotter.read_matrix(args.matrix)
        clade_info = build_terminal_subclade_info(args.sample, matrix, cell_regions, cell_burden, plotter)
    perm_df, summary_df = run_permutation(clade_info, n_iterations=args.n_iterations, min_cells=args.min_cells, seed=args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = args.output_suffix
    clade_info_path = args.outdir / f"{args.sample}.terminal_subclade_info{suffix}.tsv"
    perm_path = args.outdir / f"{args.sample}.terminal_subclade_mixing_permutations{suffix}.tsv"
    summary_path = args.outdir / f"{args.sample}.terminal_subclade_mixing_summary{suffix}.tsv"
    clade_info.to_csv(clade_info_path, sep="\t", index=False)
    perm_df.to_csv(perm_path, sep="\t", index=False)
    summary_df.to_csv(summary_path, sep="\t", index=False)

    output_prefix = args.outdir / f"{args.sample}.terminal_subclade_mixing_ridges{suffix}"
    plot_ridges(args.sample, perm_df, summary_df, output_prefix)

    print(f"Wrote: {output_prefix}.pdf")
    print(f"Wrote: {clade_info_path}")
    print(f"Wrote: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
