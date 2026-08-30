#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


GESTATION = 268 / 365
SAMPLE_AGES = {
    "5559": 19.8,
    "5657": 82.2,
    "5823": 82.7,
}


@dataclass
class Node:
    label: Optional[str] = None
    length: Optional[float] = None
    children: list["Node"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class NewickParser:
    def __init__(self, text: str):
        self.text = text.strip()
        self.i = 0

    def peek(self) -> str:
        return self.text[self.i] if self.i < len(self.text) else ""

    def consume(self, expected: Optional[str] = None) -> str:
        ch = self.peek()
        if expected is not None and ch != expected:
            raise ValueError(f"Expected {expected!r} at {self.i}, got {ch!r}")
        self.i += 1
        return ch

    def parse_label(self) -> str:
        if self.peek() == "'":
            self.consume("'")
            chars = []
            while self.peek() and self.peek() != "'":
                chars.append(self.consume())
            self.consume("'")
            return "".join(chars)
        chars = []
        while self.peek() and self.peek() not in ":,();":
            chars.append(self.consume())
        return "".join(chars).strip()

    def parse_length(self) -> Optional[float]:
        if self.peek() != ":":
            return None
        self.consume(":")
        chars = []
        while self.peek() and self.peek() not in ",();":
            chars.append(self.consume())
        raw = "".join(chars).strip()
        return float(raw) if raw else None

    def parse_subtree(self) -> Node:
        if self.peek() == "(":
            self.consume("(")
            children = [self.parse_subtree()]
            while self.peek() == ",":
                self.consume(",")
                children.append(self.parse_subtree())
            self.consume(")")
            label = self.parse_label() or None
            length = self.parse_length()
            return Node(label=label, length=length, children=children)
        label = self.parse_label()
        length = self.parse_length()
        return Node(label=label, length=length)

    def parse(self) -> Node:
        root = self.parse_subtree()
        if self.peek() == ";":
            self.consume(";")
        return root


def leaf_order(root: Node) -> list[str]:
    out = []

    def walk(node: Node) -> None:
        if node.is_leaf:
            if node.label is not None:
                out.append(node.label)
            return
        for child in node.children:
            walk(child)

    walk(root)
    return out


def load_node_times(path: Path) -> dict[str, float]:
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["node_label"].astype(str), df["time_from_conception"].astype(float)))


def find_node_by_label(root: Node, label: str) -> Optional[Node]:
    if root.label == label:
        return root
    for child in root.children:
        hit = find_node_by_label(child, label)
        if hit is not None:
            return hit
    return None


def highlight_nodes_for_snvs(node_map_path: Path, snvs: Optional[list[str]]) -> list[tuple[str, str]]:
    if not snvs:
        return []
    node_map = pd.read_csv(node_map_path, sep="\t")
    out = []
    for snv in snvs:
        found = None
        for row in node_map.itertuples(index=False):
            muts = str(row.mutations).split(";")
            if snv in muts:
                found = str(row.node_id)
                break
        if found is None:
            raise ValueError(f"SNV {snv!r} was not found in {node_map_path}")
        out.append((snv, found))
    return out


def draw_tree(
    root: Node,
    node_times: dict[str, float],
    output_pdf: Path,
    sample: str,
    age_from_birth: float,
    highlights: Optional[list[tuple[str, str]]] = None,
) -> None:
    leaves = leaf_order(root)
    x = {leaf: i for i, leaf in enumerate(leaves)}
    sample_age = age_from_birth + GESTATION

    display_cut_from_birth = 10.0
    display_cut = GESTATION + display_cut_from_birth
    terminal_extension = 1.0
    display_sample_age = display_cut + terminal_extension

    def y_raw(node: Node) -> float:
        if node.is_leaf:
            return sample_age
        if node.label and node.label in node_times:
            return float(node_times[node.label])
        # The root in the collapsed input Newick is unlabeled; it is the pseudo-root.
        return 0.0

    def y_display(t: float) -> float:
        if t <= display_cut:
            return t
        frac = (t - display_cut) / (sample_age - display_cut)
        return display_cut + frac * terminal_extension

    def node_x(node: Node) -> float:
        if node.is_leaf:
            return float(x[node.label])
        xs = [node_x(child) for child in node.children]
        return float(np.mean(xs))

    width = max(8.0, min(30.0, 0.09 * max(len(leaves), 1)))
    fig, ax = plt.subplots(figsize=(width, 7.5))

    def draw(node: Node) -> None:
        y0 = y_display(y_raw(node))
        child_xs = [node_x(child) for child in node.children]
        if len(child_xs) >= 2:
            ax.hlines(y0, min(child_xs), max(child_xs), color="black", linewidth=1.0)
        for child in node.children:
            cx = node_x(child)
            y1 = y_display(y_raw(child))
            ax.vlines(cx, y0, y1, color="black", linewidth=0.65)
            draw(child)

    draw(root)

    if highlights:
        placed = {}
        for idx, (label, node_label) in enumerate(highlights):
            target = find_node_by_label(root, node_label)
            if target is None:
                raise ValueError(f"Highlight node {node_label!r} was not found in the Newick tree")
            hx = node_x(target)
            hy = y_display(y_raw(target))
            ax.scatter([hx], [hy], s=32, facecolor="#e34a33", edgecolor="white", linewidth=0.6, zorder=5)
            x_offset = max(8.0, len(leaves) * 0.04)
            # Spread labels if multiple highlights are close on the same horizontal layer.
            near_count = sum(abs(hy - y) < 0.25 for y in placed.values())
            placed[label] = hy
            y_direction = -1 if idx % 2 == 0 else 1
            y_offset = y_direction * (0.45 + 0.22 * near_count)
            ax.annotate(
                label,
                xy=(hx, hy),
                xytext=(min(hx + x_offset, len(leaves) - 3), max(hy + y_offset, 0.03)),
                arrowprops=dict(arrowstyle="->", color="#e34a33", linewidth=1.2),
                color="#e34a33",
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="center",
                zorder=6,
            )

    ax.set_title(f"{sample} ultrametric lineage tree", loc="left", fontsize=12, weight="bold")
    ax.set_xlim(-1, len(leaves))
    ax.set_ylim(-0.05 * display_sample_age, display_sample_age * 1.03)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_ylabel("Age")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    tick_positions = [
        0.0,
        GESTATION,
        GESTATION + 1.0,
        GESTATION + 5.0,
        GESTATION + 10.0,
        display_sample_age,
    ]
    tick_labels = ["zygote", "birth", "1 yr", "5 yr", "10 yr", f"{age_from_birth:.1f} yr"]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    ax.axhline(GESTATION, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    for yline in [GESTATION + 1.0, GESTATION + 5.0, GESTATION + 10.0]:
        ax.axhline(yline, color="gray", linestyle=":", linewidth=0.55, alpha=0.45, zorder=0)
    ax.text(len(leaves), GESTATION, "birth", va="bottom", ha="right", fontsize=8, color="gray")

    ybreak = display_cut + 0.12
    trans = ax.get_yaxis_transform()
    ax.plot([1.006, 1.024], [ybreak - 0.08, ybreak + 0.02],
            transform=trans, color="black", linewidth=0.9, clip_on=False)
    ax.plot([1.006, 1.024], [ybreak + 0.06, ybreak + 0.16],
            transform=trans, color="black", linewidth=0.9, clip_on=False)

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_one(sample: str, highlight_snvs: Optional[list[str]] = None) -> None:
    if sample not in SAMPLE_AGES:
        raise ValueError(f"Unknown sample {sample!r}; expected one of {sorted(SAMPLE_AGES)}")
    newick = Path(f"{sample}.ultrametric_tree.newick")
    node_times = Path(f"{sample}.rtreefit_collapsed.poisson_tree.eg1.sens0p99.niter5000.rounded.node_times.tsv")
    node_map = Path(f"{sample}.rtreefit_collapsed_input.node_mutation_cell_map.tsv")
    suffix = ".highlightSNV" if highlight_snvs else ""
    pdf_out = Path(f"{sample}.ultrametric.corrected_snv_burden{suffix}.pdf")
    root = NewickParser(newick.read_text()).parse()
    times = load_node_times(node_times)
    highlights = highlight_nodes_for_snvs(node_map, highlight_snvs)
    for snv, node in highlights:
        print(f"{sample}: {snv} maps to {node}")
    draw_tree(
        root,
        times,
        pdf_out,
        sample=sample,
        age_from_birth=SAMPLE_AGES[sample],
        highlights=highlights,
    )
    print(f"Wrote {pdf_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot truncated ultrametric trees from collapsed Newick topology.")
    parser.add_argument("--sample", action="append", choices=sorted(SAMPLE_AGES), help="Sample to plot; repeat for multiple. Defaults to all samples.")
    parser.add_argument("--highlight-snv", action="append", help="Optional SNV label to mark with an arrow; repeat for multiple.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = args.sample or sorted(SAMPLE_AGES)
    for sample in samples:
        plot_one(sample, highlight_snvs=args.highlight_snv)


if __name__ == "__main__":
    main()
