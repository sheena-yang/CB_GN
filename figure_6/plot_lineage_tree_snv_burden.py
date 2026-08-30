#!/usr/bin/env python3
"""
Add corrected SNV burden columns to a refined denoised genotype matrix and
plot mutation lineage trees with corrected burden as branch lengths.

The script expects:
  - rows = mutation sites
  - columns = cells
  - genotype values are 0/1 (NA tolerated but treated as non-mutant)
  - tree topology is inferred directly from the denoised matrix
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


PSEUDO_ROOT = "__PSEUDO_ROOT__"
REGION_COLOR_DICT = {"CH": "#706993", "V": "#9bc1bc"}


def read_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix


def read_sensitivity(path: Path) -> Dict[str, float]:
    table = pd.read_csv(path)
    if table.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns")
    cell_col = table.columns[0]
    sens_col = table.columns[-1]
    sens = pd.to_numeric(table[sens_col], errors="coerce")
    out = {}
    for cell, value in zip(table[cell_col].astype(str), sens):
        if pd.notna(value) and value > 0:
            out[cell] = float(value)
    return out


def read_cell_burden(path: Path) -> Dict[str, float]:
    table = pd.read_csv(path)
    if table.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns")
    cell_col = table.columns[0]
    if "burden" in table.columns:
        burden_col = "burden"
    else:
        raise ValueError(f"{path} must contain a 'burden' column for cell annotation")
    burden = pd.to_numeric(table[burden_col], errors="coerce")
    out = {}
    for cell, value in zip(table[cell_col].astype(str), burden):
        if pd.notna(value):
            out[cell] = float(value)
    return out


def read_cell_region(path: Path) -> Dict[str, str]:
    table = pd.read_csv(path)
    if table.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns")
    cell_col = table.columns[0]
    if "region" not in table.columns:
        raise ValueError(f"{path} must contain a 'region' column for leaf coloring")
    return {
        str(cell): str(region)
        for cell, region in zip(table[cell_col].astype(str), table["region"].astype(str))
        if pd.notna(region)
    }


def read_misclassification(path: Path) -> Dict[Tuple[str, str], float]:
    table = pd.read_csv(path)
    if table.shape[1] < 3:
        raise ValueError(f"{path} must contain at least three columns")
    c1, c2, mcol = table.columns[:3]
    rates: Dict[Tuple[str, str], float] = {}
    for a, b, m in zip(table[c1].astype(str), table[c2].astype(str), pd.to_numeric(table[mcol], errors="coerce")):
        if pd.isna(m):
            continue
        m = float(m)
        if m >= 1:
            continue
        if m < 0:
            continue
        rates[(a, b)] = m
        rates[(b, a)] = m
    return rates


def pair_rate(rates: Dict[Tuple[str, str], float], a: str, b: str) -> Optional[float]:
    if a == b:
        return None
    return rates.get((a, b))


def compute_corrected_burden(
    matrix: pd.DataFrame,
    rates: Dict[Tuple[str, str], float],
    sensitivity: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return matrix with snv_num/corrected_snv_burden and a long QC table."""
    genotype_cols = [c for c in matrix.columns if c not in {"snv_num", "corrected_snv_burden"}]
    numeric = matrix[genotype_cols].apply(pd.to_numeric, errors="coerce")

    burdens: List[float] = []
    qc_rows: List[dict] = []
    for mut, row in numeric.iterrows():
        mutant_cells = [c for c in genotype_cols if row.get(c) == 1]
        cell_burdens = []
        for cell in mutant_cells:
            correction_terms = []
            missing_pairs = 0
            for other in mutant_cells:
                if other == cell:
                    continue
                m = pair_rate(rates, cell, other)
                if m is None:
                    missing_pairs += 1
                    continue
                correction_terms.append(1.0 / (1.0 - m))
            if correction_terms:
                corrected_mut_num = float(np.mean(correction_terms))
            else:
                corrected_mut_num = 1.0

            sens = sensitivity.get(cell)
            if sens is None or sens <= 0:
                burden = np.nan
            else:
                burden = corrected_mut_num / sens
                cell_burdens.append(burden)
            qc_rows.append(
                {
                    "mutation": mut,
                    "cell": cell,
                    "n_mutant_cells": len(mutant_cells),
                    "n_pair_rates_used": len(correction_terms),
                    "n_pair_rates_missing": missing_pairs,
                    "corrected_mutation_num": corrected_mut_num,
                    "sensitivity": sens,
                    "cell_corrected_snv_burden": burden,
                }
            )
        burdens.append(float(np.nanmean(cell_burdens)) if cell_burdens else np.nan)

    out = matrix.copy()
    out["snv_num"] = 1
    out["corrected_snv_burden"] = burdens
    qc = pd.DataFrame(qc_rows)
    return out, qc


def infer_tree_from_matrix(matrix: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, str]]:
    """Infer a perfect-phylogeny clade tree directly from an ordered genotype matrix.

    The matrix is assumed to be already denoised and ordered. For each mutation,
    the set of mutant cells defines a clade. A mutation's parent is the nearest
    previous mutation whose mutant-cell set is a superset. Identical clades are
    chained in row order, representing multiple mutations on the same branch.
    If no previous superset exists, the mutation is attached to the pseudo-root.

    Returns parent_map, children, and cell attachments to the deepest observed
    mutation in each cell.
    """
    genotype_cols = [c for c in matrix.columns if c not in {"snv_num", "corrected_snv_burden"}]
    numeric = matrix[genotype_cols].apply(pd.to_numeric, errors="coerce")
    mutations = list(numeric.index.astype(str))
    mut_sets: Dict[str, frozenset[str]] = {
        mut: frozenset(c for c in genotype_cols if numeric.at[mut, c] == 1)
        for mut in mutations
    }

    parent_map: Dict[str, str] = {}
    depths: Dict[str, int] = {PSEUDO_ROOT: 0}
    for i, mut in enumerate(mutations):
        current = mut_sets[mut]
        candidates = []
        for j in range(i):
            cand = mutations[j]
            cand_set = mut_sets[cand]
            if current and current.issubset(cand_set):
                candidates.append(cand)
            elif not current and not cand_set:
                candidates.append(cand)
        if candidates:
            # Prefer the smallest enclosing clade; for identical sets or ties,
            # use the closest previous mutation to preserve matrix row order.
            parent = min(candidates, key=lambda c: (len(mut_sets[c]), -mutations.index(c)))
        else:
            parent = PSEUDO_ROOT
        parent_map[mut] = parent
        depths[mut] = depths.get(parent, 0) + 1

    children: Dict[str, List[str]] = defaultdict(list)
    row_rank = {m: i for i, m in enumerate(mutations)}
    for child, parent in parent_map.items():
        children[parent].append(child)
    for parent in list(children):
        children[parent].sort(key=lambda m: row_rank[m])

    attachments: Dict[str, str] = {}
    for cell in genotype_cols:
        obs = [mut for mut in mutations if numeric.at[mut, cell] == 1]
        attachments[cell] = max(obs, key=lambda m: depths.get(m, 0)) if obs else PSEUDO_ROOT
    return parent_map, children, attachments


def compute_node_depths(parent_map: Dict[str, str]) -> Dict[str, int]:
    depths = {PSEUDO_ROOT: 0}
    pending = set(parent_map)
    changed = True
    while pending and changed:
        changed = False
        for node in list(pending):
            parent = parent_map.get(node, PSEUDO_ROOT)
            if parent in depths:
                depths[node] = depths[parent] + 1
                pending.remove(node)
                changed = True
    for node in pending:
        depths[node] = 1
    return depths


def descendants_by_parent(children: Dict[str, List[str]], root: str = PSEUDO_ROOT) -> List[str]:
    order = []
    stack = deque(children.get(root, []))
    while stack:
        node = stack.popleft()
        order.append(node)
        stack.extend(children.get(node, []))
    return order


def compute_cumulative_y(
    parent_map: Dict[str, str],
    weights: Dict[str, float],
) -> Dict[str, float]:
    y = {PSEUDO_ROOT: 0.0}
    pending = set(parent_map)
    changed = True
    while pending and changed:
        changed = False
        for node in list(pending):
            parent = parent_map.get(node, PSEUDO_ROOT)
            if parent in y:
                w = weights.get(node, 1.0)
                if not math.isfinite(w):
                    w = 1.0
                y[node] = y[parent] + max(float(w), 0.0)
                pending.remove(node)
                changed = True
    for node in pending:
        y[node] = weights.get(node, 1.0)
    return y


def sanitize_newick_label(label: str) -> str:
    text = str(label)
    # Quote labels so chromosome coordinates and cell names containing ":" or
    # punctuation are valid Newick labels rather than being parsed as lengths.
    text = text.replace("'", "''")
    return f"'{text}'"


def format_branch_length(value: float) -> str:
    if value is None or not math.isfinite(value):
        value = 0.0
    return f"{float(value):.10g}"


def write_newick(
    parent_map: Dict[str, str],
    children: Dict[str, List[str]],
    cells: Sequence[str],
    attachments: Dict[str, str],
    weights: Dict[str, float],
    output: Path,
) -> None:
    """Write the matrix-derived cell-leaf lineage tree in Newick format."""
    _, cells_by_node = assign_leaf_x(cells, attachments)

    def node_to_newick(node: str) -> str:
        parts: List[str] = []
        for child in children.get(node, []):
            parts.append(node_to_newick(child))
        for cell in cells_by_node.get(node, []):
            parts.append(f"{sanitize_newick_label(cell)}:0")

        label = "" if node == PSEUDO_ROOT else sanitize_newick_label(node)
        if parts:
            subtree = f"({','.join(parts)}){label}"
        else:
            subtree = label

        if node == PSEUDO_ROOT:
            return subtree
        return f"{subtree}:{format_branch_length(weights.get(node, 1.0))}"

    output.write_text(node_to_newick(PSEUDO_ROOT) + ";\n")


def assign_leaf_x(cells: Sequence[str], attachments: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    cell_x = {cell: i for i, cell in enumerate(cells)}
    cells_by_node: Dict[str, List[str]] = defaultdict(list)
    for cell in cells:
        cells_by_node[attachments.get(cell, PSEUDO_ROOT)].append(cell)
    return cell_x, cells_by_node


def compute_node_x(
    node: str,
    children: Dict[str, List[str]],
    cells_by_node: Dict[str, List[str]],
    cell_x: Dict[str, float],
    cache: Dict[str, float],
) -> float:
    if node in cache:
        return cache[node]
    xs = [cell_x[c] for c in cells_by_node.get(node, []) if c in cell_x]
    for child in children.get(node, []):
        xs.append(compute_node_x(child, children, cells_by_node, cell_x, cache))
    cache[node] = float(np.mean(xs)) if xs else 0.0
    return cache[node]


def sort_children_by_x(children: Dict[str, List[str]], node_x: Dict[str, float]) -> Dict[str, List[str]]:
    return {p: sorted(cs, key=lambda c: node_x.get(c, 0.0)) for p, cs in children.items()}


def plot_tree(
    parent_map: Dict[str, str],
    children: Dict[str, List[str]],
    y: Dict[str, float],
    cells: Sequence[str],
    attachments: Dict[str, str],
    output: Path,
    title: str,
    y_label: str,
) -> None:
    cell_x, cells_by_node = assign_leaf_x(cells, attachments)
    node_x_cache: Dict[str, float] = {}
    compute_node_x(PSEUDO_ROOT, children, cells_by_node, cell_x, node_x_cache)
    for node in parent_map:
        compute_node_x(node, children, cells_by_node, cell_x, node_x_cache)
    children = sort_children_by_x(children, node_x_cache)

    max_y = max(y.values()) if y else 1.0
    max_y = max(max_y, 1.0)
    leaf_y = max_y * 1.04

    width = max(8.0, min(28.0, 0.08 * max(len(cells), 1)))
    height = 7.5
    fig, ax = plt.subplots(figsize=(width, height))

    def draw_edges(parent: str) -> None:
        px = node_x_cache.get(parent, 0.0)
        py = y.get(parent, 0.0)
        child_nodes = children.get(parent, [])
        if child_nodes:
            child_xs = [node_x_cache[c] for c in child_nodes]
            ax.hlines(py, min(child_xs + [px]), max(child_xs + [px]), color="black", linewidth=0.9)
        for child in child_nodes:
            cx = node_x_cache[child]
            cy = y.get(child, py + 1)
            ax.vlines(cx, py, cy, color="black", linewidth=0.9)
            draw_edges(child)

    draw_edges(PSEUDO_ROOT)

    for cell in cells:
        node = attachments.get(cell, PSEUDO_ROOT)
        x = cell_x[cell]
        start_y = y.get(node, 0.0)
        ax.vlines(x, start_y, leaf_y, color="black", linewidth=0.35)

    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    ax.set_ylabel(y_label)
    ax.set_xlabel("cells")
    ax.set_xlim(-1, max(len(cells), 1))
    ax.set_ylim(-0.5, leaf_y * 1.02)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def collect_subtree_cells(
    node: str,
    children: Dict[str, List[str]],
    cells_by_node: Dict[str, List[str]],
    memo: Dict[str, List[str]],
) -> List[str]:
    if node in memo:
        return memo[node]
    cells = list(cells_by_node.get(node, []))
    for child in children.get(node, []):
        cells.extend(collect_subtree_cells(child, children, cells_by_node, memo))
    memo[node] = cells
    return cells


def hierarchical_cell_order(
    root: str,
    children: Dict[str, List[str]],
    cells_by_node: Dict[str, List[str]],
    original_rank: Dict[str, int],
) -> List[str]:
    """Return a leaf order in which every mutation subtree is contiguous.

    Each mutation node is treated as a clade. Children and directly attached
    cells are ordered by their earliest cell position in the original order,
    but the final layout enforces clade contiguity.
    """
    subtree_memo: Dict[str, List[str]] = {}

    def item_rank(item: Tuple[str, str]) -> int:
        kind, value = item
        if kind == "cell":
            return original_rank.get(value, 10**9)
        cells = collect_subtree_cells(value, children, cells_by_node, subtree_memo)
        return min([original_rank.get(c, 10**9) for c in cells] or [10**9])

    def visit(node: str) -> List[str]:
        items: List[Tuple[str, str]] = []
        items.extend(("child", child) for child in children.get(node, []))
        items.extend(("cell", cell) for cell in cells_by_node.get(node, []))
        ordered_cells: List[str] = []
        for kind, value in sorted(items, key=item_rank):
            if kind == "cell":
                ordered_cells.append(value)
            else:
                ordered_cells.extend(visit(value))
        return ordered_cells

    return visit(root)


def plot_clade_tree(
    parent_map: Dict[str, str],
    children: Dict[str, List[str]],
    y: Dict[str, float],
    cells: Sequence[str],
    attachments: Dict[str, str],
    output: Path,
    title: str,
    y_label: str,
    cell_annotation_values: Optional[Dict[str, float]] = None,
    cell_annotation_label: str = "cell SNV burden",
    cell_regions: Optional[Dict[str, str]] = None,
    base_linewidth: float = 2.0,
    leaf_linewidth: float = 1.5,
) -> None:
    """Draw a cell-leaf tree where every mutation clade occupies contiguous width.

    The pseudo-root is only used as the top split; no root node or vertical root
    trunk is drawn. Each split from one cell set into sub-clades/cells is drawn
    as one horizontal segment with vertical descendants.
    """
    _, cells_by_node = assign_leaf_x(cells, attachments)
    original_rank = {cell: i for i, cell in enumerate(cells)}
    ordered_cells = hierarchical_cell_order(PSEUDO_ROOT, children, cells_by_node, original_rank)
    seen = set()
    ordered_cells = [c for c in ordered_cells if not (c in seen or seen.add(c))]
    ordered_cells += [c for c in cells if c not in seen]
    cell_x = {cell: i for i, cell in enumerate(ordered_cells)}

    subtree_memo: Dict[str, List[str]] = {}
    node_x: Dict[str, float] = {}
    subtree_cell_counts: Dict[str, int] = {}
    for node in [PSEUDO_ROOT] + list(parent_map):
        subtree_cells = collect_subtree_cells(node, children, cells_by_node, subtree_memo)
        subtree_cell_counts[node] = len(subtree_cells)
        xs = [cell_x[c] for c in subtree_cells if c in cell_x]
        node_x[node] = float(np.mean(xs)) if xs else 0.0

    child_counts = [subtree_cell_counts.get(node, 0) for node in parent_map]
    finite_child_counts = [count for count in child_counts if count > 0]
    branch_cmap = plt.colormaps["OrRd"]
    if finite_child_counts:
        branch_norm = Normalize(vmin=min(finite_child_counts), vmax=max(finite_child_counts))
    else:
        branch_norm = Normalize(vmin=0, vmax=1)

    def branch_style(node: str) -> Tuple[Tuple[float, float, float, float], float]:
        count = subtree_cell_counts.get(node, 0)
        if finite_child_counts and max(finite_child_counts) > min(finite_child_counts):
            scaled = branch_norm(count)
            color = branch_cmap(0.25 + 0.75 * scaled)
            linewidth = 3.0 + 3.0 * scaled
        else:
            color = branch_cmap(0.75)
            linewidth = 3.0
        return color, linewidth

    def leaf_color(cell: str) -> str:
        if cell_regions is None:
            return "black"
        return REGION_COLOR_DICT.get(cell_regions.get(cell, ""), "black")

    max_y = max(y.values()) if y else 1.0
    max_y = max(max_y, 1.0)
    leaf_y = max_y + 0.8
    annotation_gap = max_y * 0.025 + 0.15
    annotation_height = max_y * 0.025 + 0.20
    annotation_y = leaf_y + annotation_gap
    cell_label_gap = max_y * 0.015 + 0.12
    cell_label_y = annotation_y + annotation_height + cell_label_gap

    width = max(8.0, min(30.0, 0.09 * max(len(ordered_cells), 1)))
    fig, ax = plt.subplots(figsize=(width, 8.0))

    def direct_cell_items(node: str) -> List[Tuple[str, float]]:
        return [(cell, cell_x[cell]) for cell in cells_by_node.get(node, []) if cell in cell_x]

    def draw_node(node: str) -> None:
        py = y.get(node, 0.0)
        child_nodes = children.get(node, [])
        child_items = [(child, node_x.get(child, 0.0)) for child in child_nodes]
        cell_items = direct_cell_items(node)
        item_xs = [x for _, x in child_items] + [x for _, x in cell_items]

        if len(item_xs) >= 2:
            ax.hlines(py, min(item_xs), max(item_xs), color="black", linewidth=base_linewidth)

        for child, cx in child_items:
            cy = y.get(child, py + 1.0)
            color, linewidth = branch_style(child)
            ax.vlines(cx, py, cy, color=color, linewidth=linewidth)
            draw_node(child)

        for cell, cx in cell_items:
            # A cell is a terminal leaf attached to the current mutation path.
            ax.vlines(cx, py, leaf_y, color=leaf_color(cell), linewidth=leaf_linewidth, alpha=0.95)

    # Pseudo-root: draw only the first split at y=0, not a root stem/node.
    draw_node(PSEUDO_ROOT)

    if cell_annotation_values:
        finite_values = [
            cell_annotation_values[cell]
            for cell in ordered_cells
            if cell in cell_annotation_values and math.isfinite(cell_annotation_values[cell])
        ]
        if finite_values:
            cmap = plt.get_cmap("YlOrBr")
            norm = Normalize(vmin=min(finite_values), vmax=max(finite_values))
            for cell in ordered_cells:
                value = cell_annotation_values.get(cell)
                if value is None or not math.isfinite(value):
                    facecolor = "white"
                else:
                    facecolor = cmap(norm(value))
                ax.add_patch(
                    Rectangle(
                        (cell_x[cell] - 0.28, annotation_y),
                        0.56,
                        annotation_height,
                        facecolor=facecolor,
                        edgecolor="none",
                        linewidth=0,
                    )
                )
            ax.text(
                -0.8,
                annotation_y + annotation_height / 2,
                cell_annotation_label,
                ha="right",
                va="center",
                fontsize=8,
            )
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(
                sm,
                ax=ax,
                orientation="horizontal",
                fraction=0.035,
                pad=0.025,
                aspect=35,
            )
            cbar.set_label(cell_annotation_label, fontsize=8)
            cbar.ax.tick_params(labelsize=7)

    for cell in ordered_cells:
        ax.text(
            cell_x[cell],
            cell_label_y,
            cell,
            rotation=90,
            ha="center",
            va="top",
            fontsize=4.5,
            color="black",
        )

    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    ax.set_ylabel(y_label)
    ax.set_xlabel("cells")
    ax.set_xlim(-1, max(len(ordered_cells), 1))
    bottom_y = cell_label_y + max_y * 0.16 + 0.8 if ordered_cells else annotation_y + annotation_height
    ax.set_ylim(-0.5, bottom_y * 1.04)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.30)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--misclassification", type=Path, required=True)
    parser.add_argument("--snv-burden", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    matrix = read_matrix(args.matrix)
    rates = read_misclassification(args.misclassification)
    sensitivity = read_sensitivity(args.snv_burden)
    cell_burden = read_cell_burden(args.snv_burden)
    cell_regions = read_cell_region(args.snv_burden)
    with_burden, qc = compute_corrected_burden(matrix, rates, sensitivity)

    burden_path = args.outdir / "denoised_matrix_refine.with_burden.csv"
    qc_path = args.outdir / "corrected_snv_burden.cell_qc.tsv"
    with_burden.to_csv(burden_path)
    qc.to_csv(qc_path, sep="\t", index=False)

    genotype_cols = [c for c in matrix.columns if c not in {"snv_num", "corrected_snv_burden"}]
    cells = genotype_cols
    parent_map, children, attachments = infer_tree_from_matrix(matrix)

    inferred_edges = pd.DataFrame(
        [{"parent": parent, "child": child} for child, parent in parent_map.items()]
    )
    inferred_edges.to_csv(args.outdir / "tree_edges_used_for_plotting.tsv", sep="\t", index=False)

    tree_summary_rows = []
    for weight_col, ylabel, suffix in [
        ("snv_num", "SNV burden", "snv_num"),
        ("corrected_snv_burden", "corrected SNV burden", "corrected_snv_burden"),
    ]:
        weights = pd.to_numeric(with_burden[weight_col], errors="coerce").to_dict()
        y = compute_cumulative_y(parent_map, weights)
        tree_summary_rows.append(
            {
                "weight": weight_col,
                "n_mutations": len(parent_map),
                "n_cells": len(cells),
                "max_tree_depth": max(y.values()) if y else 0,
            }
        )
        plot_tree(
            parent_map=parent_map,
            children=children,
            y=y,
            cells=cells,
            attachments=attachments,
            output=args.outdir / f"{args.sample}.lineage_tree.{suffix}.pdf",
            title=f"{args.sample} lineage tree ({weight_col})",
            y_label=ylabel,
        )
        plot_clade_tree(
            parent_map=parent_map,
            children=children,
            y=y,
            cells=cells,
            attachments=attachments,
            output=args.outdir / f"{args.sample}.lineage_tree.{suffix}.clade_layout.pdf",
            title=f"{args.sample} lineage tree ({weight_col}, clade layout)",
            y_label=ylabel,
            cell_annotation_values=cell_burden,
            cell_annotation_label="cell SNV burden",
            cell_regions=cell_regions,
            base_linewidth=2.0,
            leaf_linewidth=1.5,
        )
        write_newick(
            parent_map=parent_map,
            children=children,
            cells=cells,
            attachments=attachments,
            weights=weights,
            output=args.outdir / f"{args.sample}.lineage_tree.{suffix}.newick",
        )

    pd.DataFrame(tree_summary_rows).to_csv(args.outdir / "lineage_tree_plot_summary.tsv", sep="\t", index=False)
    print(f"Wrote {burden_path}")
    print(f"Wrote {qc_path}")
    print(f"Wrote {args.outdir / (args.sample + '.lineage_tree.snv_num.pdf')}")
    print(f"Wrote {args.outdir / (args.sample + '.lineage_tree.corrected_snv_burden.pdf')}")


if __name__ == "__main__":
    main()
