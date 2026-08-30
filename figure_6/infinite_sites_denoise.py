#!/usr/bin/env python3
"""Denoise a 0/1/NA genotype matrix under an infinite-sites tree model.

Rows are mutation sites and columns are cells.  The algorithm follows a
frequency-ordered, error-tolerant mutation-tree heuristic and writes a complete
audit trail for every iteration and every corrected matrix entry.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="CSV/TSV matrix; first column is mutation ID")
    p.add_argument("annotation", type=Path, nargs="?",
                   help="Optional mutation annotation CSV/TSV with Site_ID and filter columns; only filter=='final' is used for tree building")
    p.add_argument("--output-dir", type=Path, default=Path("infinite_sites_output"))
    p.add_argument("--annotation-id-column", default="Site_ID")
    p.add_argument("--annotation-filter-column", default="filter")
    p.add_argument("--annotation-final-label", default="final")
    p.add_argument("--min-mutated-cells", type=int, default=2)
    p.add_argument("--small-mutation-threshold", type=int, default=10)
    p.add_argument("--max-parent-conflict-count", type=float, default=3.0,
                   help="Maximum weighted child=1/parent!=1 count for mutations in >10 cells")
    p.add_argument("--max-parent-conflict-ratio", type=float, default=0.50,
                   help="Maximum weighted child=1/parent!=1 ratio for mutations in <=10 cells")
    p.add_argument("--max-branch-overlap-ratio", type=float, default=0.30,
                   help="Mutual-exclusion ratio cutoff for mutations in >10 cells")
    p.add_argument("--small-branch-overlap-ratio", type=float, default=0.50,
                   help="Mutual-exclusion ratio cutoff for mutations in <=10 cells")
    p.add_argument("--min-branch-support-cells", type=int, default=2,
                   help="Minimum attached cells in a mutation subtree; lower values are branch-private")
    p.add_argument("--max-branch-support-mutation-count", type=int, default=10,
                   help="Apply branch-private support filtering only to mutations with this many 1 calls or fewer")
    p.add_argument("--max-branch-support-iterations", type=int, default=3,
                   help="Maximum rebuilds after removing branch-private mutations")
    p.add_argument("--max-final-unsupported-pruning", type=int, default=10,
                   help="Maximum final unsupported-subtree pruning rounds after tree refinement")
    p.add_argument("--enable-final-split-pruning", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Prune low-frequency retained mutations whose observed 1 cells are split inside and outside their assigned subtree")
    p.add_argument("--enable-anchor-pruning", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Prune low-frequency root-level anchors whose descendants fit another branch cleanly")
    p.add_argument("--anchor-pruning-max-frequency", type=int, default=10,
                   help="Only root-level anchors with this many 1 calls or fewer are considered")
    p.add_argument("--anchor-pruning-min-descendant-overlap", type=int, default=3,
                   help="Minimum 1/1 overlap between an external branch and the rescued descendant")
    p.add_argument("--anchor-pruning-max-descendant-conflict", type=float, default=0.0,
                   help="Maximum definite conflicts allowed for external branch -> descendant")
    p.add_argument("--anchor-pruning-min-anchor-conflict", type=float, default=2.0,
                   help="Minimum definite conflicts required for external branch -> suspect anchor")
    p.add_argument("--anchor-pruning-min-overlap-gain", type=int, default=2,
                   help="Descendant must gain this many overlaps over the suspect anchor")
    p.add_argument("--allow-equal-frequency-parent", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Allow already assigned equal-frequency mutations as parents when penalties pass")
    p.add_argument("--min-compatible-ancestor-rerank-cells", type=int, default=5,
                   help="Minimum mutation count for preferring candidates that include high-level compatible ancestors")
    p.add_argument("--ancestor-promotion-max-conflict-ratio-excess", type=float, default=0.02,
                   help="Allow NA-rich lower-raw-count ancestors when their directed conflict ratio is not worse than the reverse by more than this")
    p.add_argument("--ancestor-promotion-min-missing-rate-excess", type=float, default=0.10,
                   help="Minimum missing-rate excess required for tie-like NA-aware ancestor promotion")
    p.add_argument("--cell-false-positive-penalty", type=float, default=1.0,
                   help="Cell attachment penalty for observed mutation=1 outside a candidate path")
    p.add_argument("--cell-false-negative-penalty", type=float, default=1.0,
                   help="Cell attachment penalty for observed mutation=0 inside a candidate path")
    p.add_argument("--na-penalty", type=float, default=0.5)
    p.add_argument("--max-correction-fraction", type=float, default=0.30)
    p.add_argument("--cell-order-observed-support", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Sort cells by expected-or-observed support profile before attachment tie-breakers")
    p.add_argument("--cell-order-empty-last", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Place cells with no retained observed mutations, or pseudo-root attachments, after informative cells")
    p.add_argument("--require-observed-mutation-for-cell-attachment",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Attach cells with no retained observed mutation=1 to the pseudo-root instead of imputing branch mutations from NA-only support")
    p.add_argument("--block-covered-parent-missingness-promotion",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Do not promote a NA-rich mutation above a parent that fully covers its observed 1-calls when the direction is only a missingness-supported tie")
    p.add_argument("--prefer-observed-parent-over-na-only",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Prefer parent candidates with real 1/1 support over frequency-closer candidates that are compatible only through parent NA calls")
    p.add_argument("--conservative-denoise", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Preserve observed 0/1 calls and impute NA minimally from observed mutation ancestry instead of projecting every cell to its attachment path")
    p.add_argument("--smooth-single-cell-dropouts", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Optional experimental display-oriented smoothing; disabled by default")
    p.add_argument("--smooth-min-observed-ones", type=int, default=3,
                   help="Only apply optional local smoothing to mutations with at least this many observed 1 calls")
    p.add_argument("--row-order-mode", choices=["depth_first", "branch_headers", "parent_child_blocks"],
                   default="depth_first",
                   help="Mutation row ordering: classic DFS or direct branch headers before deeper descendants")
    p.add_argument("--pseudo-root-id", default="__PSEUDO_ROOT__",
                   help="Internal root assumed mutated in all cells; excluded from biological outputs")
    return p.parse_args()


def load_matrix(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    raw = pd.read_csv(path, sep=sep, index_col=0, dtype=str, keep_default_na=False)
    if raw.index.has_duplicates:
        raise ValueError("Duplicate mutation IDs found")
    if raw.columns.has_duplicates:
        raise ValueError("Duplicate cell IDs found")
    normalized = raw.apply(lambda c: c.str.strip().str.upper())
    missing = {"", "NA", "NAN", "N/A", ".", "-"}
    valid = {"0", "1"} | missing
    bad = sorted(set(np.ravel(normalized.to_numpy())).difference(valid))
    if bad:
        raise ValueError(f"Invalid genotype values: {bad[:20]}")
    numeric = normalized.replace(list(missing), np.nan).astype(float)
    numeric.index.name = raw.index.name or "mutation"
    return numeric


def load_annotation(path: Path | None, args: argparse.Namespace,
                    matrix_index: pd.Index) -> pd.Series:
    if path is None:
        return pd.Series("final", index=matrix_index, dtype=object)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    anno = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    if args.annotation_id_column not in anno.columns:
        raise ValueError(f"Annotation ID column {args.annotation_id_column!r} not found")
    if args.annotation_filter_column not in anno.columns:
        raise ValueError(f"Annotation filter column {args.annotation_filter_column!r} not found")
    anno = anno[[args.annotation_id_column, args.annotation_filter_column]].copy()
    anno[args.annotation_id_column] = anno[args.annotation_id_column].astype(str).str.strip()
    anno[args.annotation_filter_column] = (
        anno[args.annotation_filter_column].astype(str).str.strip().replace({"": "unannotated"})
    )
    if anno[args.annotation_id_column].duplicated().any():
        duplicated = anno.loc[anno[args.annotation_id_column].duplicated(), args.annotation_id_column].head(10)
        raise ValueError(f"Duplicate mutation IDs in annotation: {duplicated.tolist()}")
    mapping = anno.set_index(args.annotation_id_column)[args.annotation_filter_column]
    return pd.Series(index=matrix_index, data=[mapping.get(m, "unannotated") for m in matrix_index],
                     dtype=object)


def passes_cutoff(count: float, ratio: float, n1: int, count_cut: float,
                  ratio_cut: float, small_threshold: int) -> bool:
    return ratio <= ratio_cut if n1 <= small_threshold else count <= count_cut


def containment(parent: np.ndarray, child: np.ndarray, na_penalty: float) -> tuple[float, float]:
    child_one = child == 1
    n = int(child_one.sum())
    if n == 0:
        return math.inf, math.inf
    count = float(((parent == 0) & child_one).sum())
    count += na_penalty * float((np.isnan(parent) & child_one).sum())
    return count, count / n


def definite_containment(parent: np.ndarray, child: np.ndarray) -> tuple[float, float, int]:
    """Observed evidence against parent -> child, excluding unknown parent calls."""
    child_one_parent_known = (child == 1) & ~np.isnan(parent)
    comparable = int(child_one_parent_known.sum())
    count = float(((child == 1) & (parent == 0)).sum())
    return count, count / comparable if comparable else 0.0, comparable


def wilson_lower(successes: float, trials: int, z: float = 1.96) -> float:
    """Lower Wilson bound used to avoid rejecting ancestry from sparse errors."""
    if trials <= 0:
        return 0.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = p + z * z / (2.0 * trials)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials)
    return max(0.0, (centre - margin) / denom)


def branch_overlap(a: np.ndarray, b: np.ndarray, na_penalty: float) -> tuple[float, float]:
    """Return definite cross-branch overlap.

    Only observed 1/1 pairs are evidence against mutual exclusion.  A 1/NA
    pair is unknown rather than contradictory, so ``na_penalty`` is accepted
    for API compatibility but intentionally not applied here.
    """
    a1, b1 = a == 1, b == 1
    count = float((a1 & b1).sum())
    denom = min(int(a1.sum()), int(b1.sum()))
    return count, count / denom if denom else 0.0


def lineage_compatible(a: np.ndarray, b: np.ndarray, args: argparse.Namespace) -> bool:
    """Whether overlapping mutations can plausibly be ordered on one branch.

    A pair is protected from a mutual-exclusion penalty only when it has enough
    definite 1/1 overlap to violate the applicable branch cutoff and at least
    one directed containment test is statistically compatible with the parent
    error cutoff.  NA calls never count as definite conflicts.
    """
    _, overlap_ratio = branch_overlap(a, b, args.na_penalty)
    smaller = min(int((a == 1).sum()), int((b == 1).sum()))
    branch_cutoff = (args.small_branch_overlap_ratio
                     if smaller <= args.small_mutation_threshold
                     else args.max_branch_overlap_ratio)
    if overlap_ratio <= branch_cutoff:
        return False
    for possible_parent, possible_child in ((a, b), (b, a)):
        dc, _, comparable = definite_containment(possible_parent, possible_child)
        if comparable <= args.small_mutation_threshold:
            if (dc / comparable if comparable else 0.0) <= args.max_parent_conflict_ratio:
                return True
        elif dc <= args.max_parent_conflict_count:
            return True
    return False


def effective_prevalence(values: np.ndarray) -> float:
    known = int(((values == 0) | (values == 1)).sum())
    return float((values == 1).sum()) / known if known else 0.0


def missing_rate(values: np.ndarray) -> float:
    return float(np.isnan(values).sum()) / len(values) if len(values) else 0.0


def direction_plausible(parent_values: np.ndarray, child_values: np.ndarray,
                        args: argparse.Namespace) -> tuple[bool, float, float, int]:
    """Test a directed lineage relation using only definite calls."""
    count, ratio, comparable = definite_containment(parent_values, child_values)
    if comparable <= args.small_mutation_threshold:
        ok = ratio <= args.max_parent_conflict_ratio
    else:
        ok = count <= args.max_parent_conflict_count
    return ok, count, ratio, comparable


def reparent_missingness_inversions(
    child: str,
    assigned_parent: str,
    matrix: pd.DataFrame,
    freq: pd.Series,
    parent: dict[str, str | None],
    decisions: list[dict],
    args: argparse.Namespace,
) -> list[dict]:
    """Move a higher-raw-count sibling below a lower-coverage ancestor.

    The operation is deliberately local: only an existing direct child of the
    newly selected parent can be moved.  It requires a raw-count inversion, a
    higher call-rate-corrected prevalence for the new ancestor, substantial
    1/1 overlap, and better directed containment in the proposed direction.
    """
    events: list[dict] = []
    child_values = matrix.loc[child].to_numpy(float)
    child_eff = effective_prevalence(child_values)
    for other, old_parent in list(parent.items()):
        if other == child or old_parent != assigned_parent:
            continue
        if int(freq[other]) <= int(freq[child]):
            continue
        other_values = matrix.loc[other].to_numpy(float)
        if child_eff <= effective_prevalence(other_values):
            continue
        if not lineage_compatible(child_values, other_values, args):
            continue
        forward_ok, forward_count, forward_ratio, forward_n = direction_plausible(
            child_values, other_values, args)
        _, reverse_count, reverse_ratio, reverse_n = direction_plausible(
            other_values, child_values, args)
        child_missing = missing_rate(child_values)
        other_missing = missing_rate(other_values)
        strictly_better_direction = forward_ratio < reverse_ratio
        missing_supported_tie = (
            forward_ratio <= reverse_ratio + args.ancestor_promotion_max_conflict_ratio_excess
            and child_missing >= other_missing + args.ancestor_promotion_min_missing_rate_excess
        )
        if not forward_ok or not (strictly_better_direction or missing_supported_tie):
            continue
        parent[other] = child
        for record in decisions:
            if record.get("mutation") == other:
                record["parent"] = child
                record["reason"] = "reparented_missingness_frequency_inversion"
                record["reparented_from"] = assigned_parent
                break
        events.append({
            "ancestor": child, "descendant": other, "previous_parent": assigned_parent,
            "ancestor_raw_ones": int(freq[child]), "descendant_raw_ones": int(freq[other]),
            "ancestor_effective_prevalence": child_eff,
            "descendant_effective_prevalence": effective_prevalence(other_values),
            "ancestor_missing_rate": child_missing,
            "descendant_missing_rate": other_missing,
            "forward_definite_conflicts": forward_count,
            "forward_conflict_ratio": forward_ratio, "forward_comparable": forward_n,
            "reverse_definite_conflicts": reverse_count,
            "reverse_conflict_ratio": reverse_ratio, "reverse_comparable": reverse_n,
            "promotion_rule": ("strictly_better_direction" if strictly_better_direction
                               else "missing_supported_tie"),
        })
    return events


def promotion_allowed(proposed_ancestor_values: np.ndarray,
                      proposed_descendant_values: np.ndarray,
                      args: argparse.Namespace) -> tuple[bool, dict]:
    """Whether a lower-coverage mutation can be placed above another node."""
    forward_ok, forward_count, forward_ratio, forward_n = direction_plausible(
        proposed_ancestor_values, proposed_descendant_values, args)
    _, reverse_count, reverse_ratio, reverse_n = direction_plausible(
        proposed_descendant_values, proposed_ancestor_values, args)
    ancestor_missing = missing_rate(proposed_ancestor_values)
    descendant_missing = missing_rate(proposed_descendant_values)
    strictly_better_direction = forward_ratio < reverse_ratio
    missing_supported_tie = (
        forward_ratio <= reverse_ratio + args.ancestor_promotion_max_conflict_ratio_excess
        and ancestor_missing >= descendant_missing + args.ancestor_promotion_min_missing_rate_excess
    )
    ok = bool(forward_ok and (strictly_better_direction or missing_supported_tie))
    return ok, {
        "forward_definite_conflicts": forward_count,
        "forward_conflict_ratio": forward_ratio,
        "forward_comparable": forward_n,
        "reverse_definite_conflicts": reverse_count,
        "reverse_conflict_ratio": reverse_ratio,
        "reverse_comparable": reverse_n,
        "ancestor_missing_rate": ancestor_missing,
        "descendant_missing_rate": descendant_missing,
        "promotion_rule": ("strictly_better_direction" if strictly_better_direction
                           else "missing_supported_tie" if missing_supported_tie
                           else "not_promoted"),
    }


def direct_children(parent: dict[str, str | None], node: str) -> list[str]:
    return [child for child, p in parent.items() if p == node]


def reparent_compatible_children_below_new_ancestor(
    ancestor: str,
    old_parent: str,
    matrix: pd.DataFrame,
    freq: pd.Series,
    parent: dict[str, str | None],
    decisions: list[dict],
    args: argparse.Namespace,
) -> list[dict]:
    """Move existing direct children of old_parent below a newly added ancestor."""
    events: list[dict] = []
    ancestor_values = matrix.loc[ancestor].to_numpy(float)
    for other in list(direct_children(parent, old_parent)):
        if other == ancestor:
            continue
        other_values = matrix.loc[other].to_numpy(float)
        ok, stats = promotion_allowed(ancestor_values, other_values, args)
        if not ok:
            continue
        if effective_prevalence(ancestor_values) < effective_prevalence(other_values):
            continue
        parent[other] = ancestor
        for record in decisions:
            if record.get("mutation") == other:
                record["parent"] = ancestor
                record["reason"] = "reparented_below_compatible_new_ancestor"
                record["reparented_from"] = old_parent
                break
        events.append({
            "ancestor": ancestor, "descendant": other, "previous_parent": old_parent,
            "ancestor_raw_ones": int(freq[ancestor]),
            "descendant_raw_ones": int(freq[other]),
            "ancestor_effective_prevalence": effective_prevalence(ancestor_values),
            "descendant_effective_prevalence": effective_prevalence(other_values),
            **stats,
        })
    return events


def promote_above_compatible_parents(
    child: str,
    root: str,
    matrix: pd.DataFrame,
    freq: pd.Series,
    parent: dict[str, str | None],
    decisions: list[dict],
    args: argparse.Namespace,
) -> list[dict]:
    """Insert a NA-rich child above same-branch parents when direction is inverted."""
    events: list[dict] = []
    child_values = matrix.loc[child].to_numpy(float)
    while parent.get(child) not in {None, root}:
        current_parent = parent[child]
        if current_parent is None:
            break
        parent_values = matrix.loc[current_parent].to_numpy(float)
        if effective_prevalence(child_values) <= effective_prevalence(parent_values):
            break
        ok, stats = promotion_allowed(child_values, parent_values, args)
        if not ok:
            break
        if (
            args.block_covered_parent_missingness_promotion
            and stats.get("promotion_rule") == "missing_supported_tie"
            and int(freq.get(current_parent, 0)) >= int(freq.get(child, 0))
            and fully_covers_observed_ones(parent_values, child_values)
        ):
            break
        grandparent = parent.get(current_parent)
        parent[child] = grandparent
        parent[current_parent] = child
        for record in decisions:
            if record.get("mutation") == child:
                record["parent"] = grandparent if grandparent is not None else ""
                record["reason"] = "promoted_above_missingness_inverted_parent"
                record["reparented_from"] = current_parent
                break
        events.append({
            "ancestor": child, "descendant": current_parent,
            "previous_parent": grandparent if grandparent is not None else "",
            "ancestor_raw_ones": int(freq[child]),
            "descendant_raw_ones": int(freq[current_parent]),
            "ancestor_effective_prevalence": effective_prevalence(child_values),
            "descendant_effective_prevalence": effective_prevalence(parent_values),
            **stats,
        })
    return events


def support_from_parent(parent_values: np.ndarray,
                        child_values: np.ndarray) -> tuple[int, int, int]:
    child_one = child_values == 1
    overlap = int(((parent_values == 1) & child_one).sum())
    hard = int(((parent_values == 0) & child_one).sum())
    missing = int((np.isnan(parent_values) & child_one).sum())
    return overlap, hard, missing


def fully_covers_observed_ones(parent_values: np.ndarray,
                               child_values: np.ndarray) -> bool:
    """Whether every observed child=1 is also an observed parent=1."""
    overlap, hard, missing = support_from_parent(parent_values, child_values)
    child_ones = int((child_values == 1).sum())
    return bool(child_ones > 0 and overlap == child_ones and hard == 0 and missing == 0)


def descendants_of(node: str, parent: dict[str, str | None]) -> set[str]:
    children: dict[str, list[str]] = defaultdict(list)
    for child, p in parent.items():
        if p is not None:
            children[p].append(child)
    out: set[str] = set()
    stack = list(children.get(node, []))
    while stack:
        x = stack.pop()
        out.add(x)
        stack.extend(children.get(x, []))
    return out


def sink_weak_siblings_below_stronger_descendants(
    node: str,
    matrix: pd.DataFrame,
    freq: pd.Series,
    parent: dict[str, str | None],
    decisions: list[dict],
    args: argparse.Namespace,
) -> list[dict]:
    """Within one branch, put weaker same-frequency siblings below stronger descendants."""
    events: list[dict] = []
    if node not in matrix.index:
        return events
    node_values = matrix.loc[node].to_numpy(float)
    changed = True
    while changed:
        changed = False
        for weak in list(direct_children(parent, node)):
            if weak not in matrix.index:
                continue
            weak_values = matrix.loc[weak].to_numpy(float)
            weak_support = support_from_parent(node_values, weak_values)
            best_candidate = ""
            best_support: tuple[int, int, int] | None = None
            for sibling in direct_children(parent, node):
                if sibling == weak:
                    continue
                for candidate in [sibling] + sorted(descendants_of(sibling, parent),
                                                    key=lambda x: int(freq.get(x, 0))):
                    if candidate not in matrix.index or candidate == weak:
                        continue
                    if is_ancestor(weak, candidate, parent):
                        continue
                    candidate_values = matrix.loc[candidate].to_numpy(float)
                    candidate_support = support_from_parent(node_values, candidate_values)
                    if candidate_support[0] <= weak_support[0]:
                        continue
                    if candidate_support[1] > weak_support[1]:
                        continue
                    ok, _, _, _ = direction_plausible(candidate_values, weak_values, args)
                    if not ok:
                        continue
                    candidate_to_weak = support_from_parent(candidate_values, weak_values)
                    if (
                        args.prefer_observed_parent_over_na_only
                        and candidate_to_weak[0] < min(args.min_branch_support_cells,
                                                       int(freq.get(weak, 0)))
                        and candidate_to_weak[2] > 0
                    ):
                        continue
                    if (best_support is None
                            or candidate_support[0] > best_support[0]
                            or (candidate_support[0] == best_support[0]
                                and int(freq[candidate]) >= int(freq.get(best_candidate, 0)))):
                        best_candidate = candidate
                        best_support = candidate_support
            if best_candidate:
                previous_parent = parent[weak]
                parent[weak] = best_candidate
                for record in decisions:
                    if record.get("mutation") == weak:
                        record["parent"] = best_candidate
                        record["reason"] = "sunk_below_stronger_compatible_descendant"
                        record["reparented_from"] = previous_parent
                        break
                events.append({
                    "ancestor": best_candidate, "descendant": weak,
                    "previous_parent": previous_parent,
                    "ancestor_raw_ones": int(freq[best_candidate]),
                    "descendant_raw_ones": int(freq[weak]),
                    "ancestor_effective_prevalence": effective_prevalence(
                        matrix.loc[best_candidate].to_numpy(float)),
                    "descendant_effective_prevalence": effective_prevalence(weak_values),
                    "ancestor_missing_rate": missing_rate(
                        matrix.loc[best_candidate].to_numpy(float)),
                    "descendant_missing_rate": missing_rate(weak_values),
                    "forward_definite_conflicts": np.nan,
                    "forward_conflict_ratio": np.nan,
                    "forward_comparable": np.nan,
                    "reverse_definite_conflicts": np.nan,
                    "reverse_conflict_ratio": np.nan,
                    "reverse_comparable": np.nan,
                    "promotion_rule": "sibling_support_order",
                })
                changed = True
                break
    return events


def ancestors(node: str, parent: dict[str, str | None]) -> list[str]:
    out: list[str] = []
    while node in parent and parent[node] is not None:
        node = parent[node]  # type: ignore[assignment]
        out.append(node)
    return out


def is_ancestor(a: str, b: str, parent: dict[str, str | None]) -> bool:
    return a == b or a in ancestors(b, parent)


def dfs_order(root: str, parent: dict[str, str | None], freq: pd.Series,
              mode: str = "depth_first", matrix: pd.DataFrame | None = None) -> list[str]:
    children: dict[str, list[str]] = defaultdict(list)
    for child, p in parent.items():
        if p is not None:
            children[p].append(child)
    def child_sort_key(p: str, child: str) -> tuple:
        if matrix is None or child not in matrix.index or p not in matrix.index:
            return (-int(freq[child]), child)
        parent_values = matrix.loc[p].to_numpy(float)
        child_values = matrix.loc[child].to_numpy(float)
        overlap = int(((parent_values == 1) & (child_values == 1)).sum())
        conflict = int(((parent_values == 0) & (child_values == 1)).sum())
        comparable = int((~np.isnan(parent_values) & ~np.isnan(child_values)).sum())
        conflict_ratio = conflict / max(comparable, 1)
        # For siblings on the same branch, show the best-supported continuation
        # first.  This only changes display/traversal order, not the inferred
        # parent-child structure.
        return (-overlap, conflict_ratio, conflict, -int(freq[child]), child)

    for p in children:
        children[p].sort(key=lambda x, p=p: child_sort_key(p, x))
    order: list[str] = []

    def visit_depth_first(node: str) -> None:
        order.append(node)
        for child in children.get(node, []):
            visit_depth_first(child)

    def visit_branch_headers(node: str, include_self: bool = True) -> None:
        if include_self:
            order.append(node)
        direct_children = children.get(node, [])
        order.extend(direct_children)
        for child in direct_children:
            visit_branch_headers(child, include_self=False)

    def visit_parent_child_blocks(node: str, include_self: bool = True) -> None:
        if include_self:
            order.append(node)
        direct_children = children.get(node, [])
        order.extend(direct_children)
        for child in direct_children:
            for grandchild in children.get(child, []):
                visit_parent_child_blocks(grandchild, include_self=True)

    if mode == "parent_child_blocks":
        visit_parent_child_blocks(root)
    elif mode == "branch_headers":
        visit_branch_headers(root)
    else:
        visit_depth_first(root)
    return order


def real_mutation_order(root: str, parent: dict[str, str | None], freq: pd.Series,
                        mode: str = "depth_first",
                        matrix: pd.DataFrame | None = None) -> list[str]:
    """Return tree order excluding the internal pseudo root."""
    return [m for m in dfs_order(root, parent, freq, mode, matrix) if m != root]


def hide_pseudo_root(df: pd.DataFrame, root: str) -> pd.DataFrame:
    """Remove the internal pseudo-root label from user-facing tables."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].replace(root, "")
    return out


def evaluate_candidate(
    child: str,
    candidate: str,
    root: str,
    matrix: pd.DataFrame,
    freq: pd.Series,
    parent: dict[str, str | None],
    args: argparse.Namespace,
) -> dict:
    c = matrix.loc[child].to_numpy(float)
    n1 = int(freq[child])
    if candidate == root:
        pc, pr, dc, dr, comparable, dlower = 0.0, 0.0, 0.0, 0.0, n1, 0.0
        parent_ok, frequency_direction_ok = True, True
    else:
        p = matrix.loc[candidate].to_numpy(float)
        pc, pr = containment(p, c, args.na_penalty)
        dc, dr, comparable = definite_containment(p, c)
        parent_overlap_count, parent_hard_conflict_count, parent_missing_child_ones = support_from_parent(p, c)
        dlower = wilson_lower(dc, comparable)
        parent_ok = passes_cutoff(pc, pr, n1, args.max_parent_conflict_count,
                                  args.max_parent_conflict_ratio,
                                  args.small_mutation_threshold)
        frequency_direction_ok = True
        # Raw 1 counts can be inverted by different NA rates.  When the child
        # has higher call-rate-corrected prevalence and the reverse direction
        # is both plausible and has fewer definite conflicts, do not force the
        # higher raw-count mutation to be its parent.
        if effective_prevalence(c) > effective_prevalence(p):
            reverse_ok, _, reverse_ratio, _ = direction_plausible(c, p, args)
            if reverse_ok and reverse_ratio < dr:
                frequency_direction_ok = False
                parent_ok = False
    if candidate == root:
        parent_overlap_count, parent_hard_conflict_count, parent_missing_child_ones = n1, 0, 0

    # Competing branches include (a) the originally requested mutations with
    # frequencies between child and candidate and (b) sibling branch roots
    # encountered along the candidate-to-root path.  Rule (b) prevents a child
    # from entering one root branch while substantially overlapping a sibling
    # root branch whose mutation happens to be more frequent than its parent.
    competing: set[str] = set()
    for other in parent:
        if other in {root, candidate}:
            continue
        # Only higher-frequency mutations between child and candidate are relevant.
        if not (int(freq[child]) < int(freq[other]) < int(freq[candidate])):
            continue
        # Mutations on the candidate's ancestral path belong to the same branch.
        if is_ancestor(other, candidate, parent):
            continue
        if lineage_compatible(matrix.loc[other].to_numpy(float), c, args):
            continue
        competing.add(other)

    # Apply an extra sibling-root guard along the candidate-to-root path.  This
    # prevents a low-frequency terminal branch from being placed under one
    # sibling while most of its observed 1 calls actually sit inside another
    # sibling branch.  This is now safe for small mutations because
    # branch_overlap counts only real 1/1 evidence; NA no longer contributes.
    path_node = candidate
    while parent.get(path_node) is not None:
        path_parent = parent[path_node]
        for other, other_parent in parent.items():
            if other != path_node and other_parent == path_parent:
                if lineage_compatible(matrix.loc[other].to_numpy(float), c, args):
                    continue
                competing.add(other)
        path_node = path_parent  # type: ignore[assignment]

    conflicts: list[tuple[str, float, float]] = []
    for other in competing:
        oc, oratio = branch_overlap(matrix.loc[other].to_numpy(float), c, args.na_penalty)
        conflicts.append((other, oc, oratio))

    if conflicts:
        worst = max(conflicts, key=lambda z: (z[2], z[1]))
        bc, br, worst_other = worst[1], worst[2], worst[0]
        branch_cutoff = (args.small_branch_overlap_ratio
                         if n1 <= args.small_mutation_threshold
                         else args.max_branch_overlap_ratio)
        branch_ok = br <= branch_cutoff
    else:
        bc, br, worst_other, branch_ok = 0.0, 0.0, "", True

    missing_compatible: list[tuple[str, float]] = []
    if n1 >= args.min_compatible_ancestor_rerank_cells:
        candidate_path = set(ancestors(candidate, parent) + [candidate])
        for other in parent:
            if other in candidate_path or other == child:
                continue
            if int(freq[other]) <= int(freq[child]):
                continue
            other_values = matrix.loc[other].to_numpy(float)
            overlap_count, overlap_ratio = branch_overlap(other_values, c, args.na_penalty)
            smaller = min(int(freq[other]), int(freq[child]))
            compatible_cutoff = (args.small_branch_overlap_ratio
                                 if smaller <= args.small_mutation_threshold
                                 else args.max_branch_overlap_ratio)
            if overlap_ratio <= compatible_cutoff:
                continue
            direction_ok, _, _, _ = direction_plausible(other_values, c, args)
            if direction_ok:
                missing_compatible.append((other, overlap_ratio))

    missing_compatible.sort(key=lambda z: (-z[1], -int(freq[z[0]]), z[0]))
    missing_compatible_count = len(missing_compatible)
    best_missing_compatible = missing_compatible[0][0] if missing_compatible else ""
    best_missing_compatible_overlap = missing_compatible[0][1] if missing_compatible else 0.0
    return {
        "candidate": candidate,
        "frequency_difference": int(freq[candidate] - freq[child]),
        "parent_conflict_count": pc,
        "parent_conflict_ratio": pr,
        "parent_overlap_count": parent_overlap_count,
        "parent_hard_conflict_count": parent_hard_conflict_count,
        "parent_missing_child_ones": parent_missing_child_ones,
        "parent_definite_conflict_count": dc,
        "parent_definite_conflict_ratio": dr,
        "parent_comparable_child_ones": comparable,
        "parent_conflict_wilson_lower": dlower,
        "frequency_direction_ok": frequency_direction_ok,
        "parent_ok": parent_ok,
        "branch_overlap_count": bc,
        "branch_overlap_ratio": br,
        "worst_other_branch": worst_other,
        "branch_ok": branch_ok,
        "missing_compatible_ancestor_count": missing_compatible_count,
        "best_missing_compatible_ancestor": best_missing_compatible,
        "best_missing_compatible_overlap": best_missing_compatible_overlap,
        "valid": bool(parent_ok and branch_ok),
    }


def build_tree(matrix: pd.DataFrame, mutations: list[str], args: argparse.Namespace) -> dict:
    freq = matrix.loc[mutations].eq(1).sum(axis=1).astype(int)
    if args.pseudo_root_id in matrix.index:
        raise ValueError(f"Pseudo-root ID {args.pseudo_root_id!r} already exists as a mutation ID")
    original_position = {m: i for i, m in enumerate(matrix.index)}
    ordered = sorted(mutations, key=lambda m: (-int(freq[m]), original_position[m]))
    root = args.pseudo_root_id
    freq.loc[root] = matrix.shape[1]
    parent: dict[str, str | None] = {root: None}
    decisions: list[dict] = [{"mutation": root, "status": "pseudo_root", "parent": ""}]
    deferred: list[str] = []
    reparenting_events: list[dict] = []

    def valid_candidate_key(child: str, record: dict) -> tuple:
        root_penalty = 1 if record["candidate"] == root else 0
        n1 = int(freq[child])
        observed_parent_support = int(record.get("parent_overlap_count", 0))
        na_only_parent_penalty = int(
            args.prefer_observed_parent_over_na_only
            and record["candidate"] != root
            and n1 > 0
            and observed_parent_support < min(args.min_branch_support_cells, n1)
            and int(record.get("parent_missing_child_ones", 0)) > 0
        )
        return (
            root_penalty,
            record["missing_compatible_ancestor_count"],
            na_only_parent_penalty,
            -observed_parent_support,
            record["frequency_difference"],
            record["parent_conflict_ratio"],
            record["branch_overlap_ratio"],
            record["candidate"],
        )

    for child in ordered:
        # Higher raw frequency mutations are parent candidates.  Equal-frequency
        # candidates can also be valid when two sites are nearly coextensive but
        # one has a few missing/error calls; this keeps same-branch mutations
        # adjacent instead of forcing them to become sibling branches.
        if args.allow_equal_frequency_parent:
            candidates = [m for m in parent if int(freq[m]) >= int(freq[child])]
        else:
            candidates = [m for m in parent if int(freq[m]) > int(freq[child])]
        if root not in candidates:
            candidates.append(root)
        evaluated = [evaluate_candidate(child, p, root, matrix, freq, parent, args)
                     for p in candidates]
        valid = [x for x in evaluated if x["valid"]]
        if valid:
            best = min(valid, key=lambda x: valid_candidate_key(child, x))
            parent[child] = best["candidate"]
            decisions.append({"mutation": child, "status": "assigned",
                              "parent": best["candidate"], **best})
            reparenting_events.extend(promote_above_compatible_parents(
                child, root, matrix, freq, parent, decisions, args))
            assigned_parent = parent.get(child)
            if assigned_parent is not None:
                reparenting_events.extend(reparent_compatible_children_below_new_ancestor(
                    child, assigned_parent, matrix, freq, parent, decisions, args))
                reparenting_events.extend(sink_weak_siblings_below_stronger_descendants(
                    assigned_parent, matrix, freq, parent, decisions, args))
            reparenting_events.extend(sink_weak_siblings_below_stronger_descendants(
                child, matrix, freq, parent, decisions, args))
            reparenting_events.extend(reparent_missingness_inversions(
                child, best["candidate"], matrix, freq, parent, decisions, args))
        else:
            deferred.append(child)
            if evaluated:
                best = min(evaluated, key=lambda x: (
                    (0 if x["valid"] else 1),
                    x["parent_conflict_ratio"] + x["branch_overlap_ratio"],
                    x["frequency_difference"], x["candidate"]))
                reason = []
                if not best["parent_ok"]:
                    reason.append("parent_conflict")
                if not best["branch_ok"]:
                    reason.append("branch_overlap")
                decisions.append({"mutation": child, "status": "deferred_possible_noise",
                                  "parent": "", "reason": "+".join(reason), **best})
            else:
                decisions.append({"mutation": child, "status": "deferred_possible_noise",
                                  "parent": "", "reason": "no_higher_frequency_candidate"})
    return {"root": root, "parent": parent, "deferred": deferred,
            "decisions": pd.DataFrame(decisions), "frequency": freq,
            "reparenting_events": reparenting_events}


def refine_descendant_parent_placements(matrix: pd.DataFrame,
                                        freq: pd.Series,
                                        parent: dict[str, str | None],
                                        root: str,
                                        args: argparse.Namespace,
                                        max_rounds: int = 5,
                                        allow_lower_frequency: bool = False,
                                        prefer_deepest: bool = False) -> list[dict]:
    """Sink assigned mutations from broad ancestors to compatible descendant parents.

    The initial tree is built greedily, so a mutation can be placed under a broad
    ancestor before a more specific same-branch parent has appeared.  This final
    refinement does not introduce new cutoffs: a candidate descendant parent must
    pass the same directed containment and mutual-exclusion checks used during
    tree construction.  It only considers moves within an already assigned
    branch, and it never moves pseudo-root children into another root branch.
    """
    events: list[dict] = []
    for refinement_round in range(1, max_rounds + 1):
        moved = False
        nodes = [m for m in matrix.index if m in parent and m != root]
        # Low-frequency terminal nodes are the most sensitive to early greedy
        # placement, so consider them before broad backbone mutations.
        nodes.sort(key=lambda m: (int(freq.get(m, 0)), m))
        for child in nodes:
            current_parent = parent.get(child)
            if current_parent is None or current_parent == root:
                continue
            current_record = evaluate_candidate(
                child, current_parent, root, matrix, freq, parent, args)
            child_n1 = int(freq.get(child, 0))
            subtree = descendants_of(current_parent, parent)
            child_subtree = descendants_of(child, parent) | {child}
            best: dict | None = None
            for candidate in sorted(subtree - child_subtree,
                                    key=lambda m: (int(freq.get(m, 0)), m)):
                if candidate not in matrix.index:
                    continue
                if int(freq.get(candidate, 0)) < child_n1 and not allow_lower_frequency:
                    continue
                record = evaluate_candidate(child, candidate, root, matrix, freq, parent, args)
                compatible = bool(record["valid"] or (
                    allow_lower_frequency
                    and record["parent_ok"]
                    and record["branch_ok"]
                ))
                if not compatible:
                    continue
                if int(record.get("parent_overlap_count", 0)) < min(
                        args.min_branch_support_cells, child_n1):
                    continue
                # Only accept a genuinely more specific placement.  The new
                # parent should be closer in observed frequency than the current
                # broad ancestor; ties are decided by stronger observed support
                # and lower branch-overlap penalty.
                if (record["frequency_difference"] > current_record["frequency_difference"]
                        and not allow_lower_frequency):
                    continue
                if prefer_deepest:
                    score = (
                        -len(ancestors(candidate, parent)),
                        record["branch_overlap_ratio"],
                        record["parent_conflict_ratio"],
                        -int(record.get("parent_overlap_count", 0)),
                        abs(record["frequency_difference"]),
                        candidate,
                    )
                    current_score = (
                        -len(ancestors(current_parent, parent)),
                        current_record["branch_overlap_ratio"],
                        current_record["parent_conflict_ratio"],
                        -int(current_record.get("parent_overlap_count", 0)),
                        abs(current_record["frequency_difference"]),
                        current_parent,
                    )
                else:
                    score = (
                        record["frequency_difference"],
                        record["branch_overlap_ratio"],
                        record["parent_conflict_ratio"],
                        -int(record.get("parent_overlap_count", 0)),
                        candidate,
                    )
                    current_score = (
                        current_record["frequency_difference"],
                        current_record["branch_overlap_ratio"],
                        current_record["parent_conflict_ratio"],
                        -int(current_record.get("parent_overlap_count", 0)),
                        current_parent,
                    )
                if score >= current_score:
                    continue
                if best is None or score < best["score"]:
                    best = {"score": score, **record}
            if best is None:
                continue
            new_parent = str(best["candidate"])
            previous_parent = current_parent
            parent[child] = new_parent
            events.append({
                "round": refinement_round,
                "mutation": child,
                "previous_parent": previous_parent,
                "new_parent": new_parent,
                "mutation_frequency": child_n1,
                "previous_frequency_difference": current_record["frequency_difference"],
                "new_frequency_difference": best["frequency_difference"],
                "previous_parent_overlap_count": current_record["parent_overlap_count"],
                "new_parent_overlap_count": best["parent_overlap_count"],
                "previous_parent_conflict_ratio": current_record["parent_conflict_ratio"],
                "new_parent_conflict_ratio": best["parent_conflict_ratio"],
                "previous_branch_overlap_ratio": current_record["branch_overlap_ratio"],
                "new_branch_overlap_ratio": best["branch_overlap_ratio"],
                "new_worst_other_branch": best["worst_other_branch"],
                "refinement_rule": "sink_to_more_specific_compatible_descendant",
            })
            moved = True
        if not moved:
            break
    return events


def refine_noisy_parent_child_direction(matrix: pd.DataFrame,
                                        freq: pd.Series,
                                        parent: dict[str, str | None],
                                        root: str,
                                        args: argparse.Namespace,
                                        max_rounds: int = 5) -> list[dict]:
    """Reverse direct edges where a noisy same-branch parent sits above a cleaner child.

    Some same-frequency or near-coextensive mutations are oriented incorrectly
    when the noisier mutation has extra observed 1 calls outside the branch.  We
    use the current cell attachments as a branch-consistency audit: if a direct
    child has more observed 1 calls inside its subtree and fewer outside-subtree
    1 calls than its parent, and the reverse direction passes the existing
    containment rules, the cleaner child is promoted above the noisy parent.
    """
    events: list[dict] = []

    def children_map() -> dict[str | None, list[str]]:
        children: dict[str | None, list[str]] = defaultdict(list)
        for child, p in parent.items():
            children[p].append(child)
        return children

    for refinement_round in range(1, max_rounds + 1):
        moved = False
        order = real_mutation_order(root, parent, freq, args.row_order_mode, matrix)
        _, attachment = expected_and_cell_order(matrix, order, parent, root, args)
        children = children_map()

        def subtree_nodes(node: str) -> set[str]:
            out = {node}
            stack = list(children.get(node, []))
            while stack:
                x = stack.pop()
                out.add(x)
                stack.extend(children.get(x, []))
            return out

        subtree_cells: dict[str, set[str]] = {}
        for node in order:
            nodes = subtree_nodes(node)
            subtree_cells[node] = set(
                attachment.loc[
                    attachment["attachment_node"].astype(str).isin(nodes), "cell"
                ].astype(str)
            )

        for noisy_parent in order:
            if noisy_parent == root or noisy_parent not in matrix.index:
                continue
            for clean_child in list(children.get(noisy_parent, [])):
                if clean_child not in matrix.index:
                    continue
                parent_obs = set(matrix.columns[matrix.loc[noisy_parent].eq(1)])
                child_obs = set(matrix.columns[matrix.loc[clean_child].eq(1)])
                parent_in = len(parent_obs & subtree_cells.get(noisy_parent, set()))
                parent_out = len(parent_obs - subtree_cells.get(noisy_parent, set()))
                child_in = len(child_obs & subtree_cells.get(clean_child, set()))
                child_out = len(child_obs - subtree_cells.get(clean_child, set()))
                if child_in < args.min_branch_support_cells:
                    continue
                if child_in <= parent_in:
                    continue
                if child_out >= parent_out:
                    continue
                if parent_out == 0:
                    continue
                ok, conflict_count, conflict_ratio, comparable = direction_plausible(
                    matrix.loc[clean_child].to_numpy(float),
                    matrix.loc[noisy_parent].to_numpy(float),
                    args,
                )
                if not ok:
                    continue
                reverse_record = evaluate_candidate(
                    noisy_parent, clean_child, root, matrix, freq, parent, args)
                if not reverse_record["valid"]:
                    continue
                previous_grandparent = parent.get(noisy_parent)
                parent[clean_child] = previous_grandparent
                parent[noisy_parent] = clean_child
                events.append({
                    "round": refinement_round,
                    "promoted_parent": clean_child,
                    "demoted_child": noisy_parent,
                    "previous_grandparent": previous_grandparent,
                    "promoted_frequency": int(freq.get(clean_child, 0)),
                    "demoted_frequency": int(freq.get(noisy_parent, 0)),
                    "promoted_in_subtree_ones": child_in,
                    "promoted_out_subtree_ones": child_out,
                    "demoted_in_subtree_ones": parent_in,
                    "demoted_out_subtree_ones": parent_out,
                    "reverse_conflict_count": conflict_count,
                    "reverse_conflict_ratio": conflict_ratio,
                    "reverse_comparable": comparable,
                    "refinement_rule": "promote_clean_child_above_noisy_parent",
                })
                moved = True
                break
            if moved:
                break
        if not moved:
            break
    return events


def prune_unsupported_subtree_mutations(matrix: pd.DataFrame,
                                        freq: pd.Series,
                                        parent: dict[str, str | None],
                                        root: str,
                                        args: argparse.Namespace,
                                        max_rounds: int = 10) -> list[dict]:
    """Remove final-tree mutations lacking observed support in their own subtree.

    Earlier branch-private pruning is necessarily based on a still-changing
    greedy tree.  After final parent refinement and cell attachment, we perform
    the same support audit once more: a retained mutation must be supported by
    at least ``min_branch_support_cells`` observed mutant cells assigned to its
    own subtree.  Unsupported nodes are pruned from the final tree; their
    children are reattached to the pruned node's parent and the audit repeats.
    """
    events: list[dict] = []
    for pruning_round in range(1, max_rounds + 1):
        order = real_mutation_order(root, parent, freq, args.row_order_mode, matrix)
        _, attachment = expected_and_cell_order(matrix, order, parent, root, args)
        children: dict[str | None, list[str]] = defaultdict(list)
        for child, p in parent.items():
            children[p].append(child)

        def subtree_nodes(node: str) -> set[str]:
            out = {node}
            stack = list(children.get(node, []))
            while stack:
                x = stack.pop()
                out.add(x)
                stack.extend(children.get(x, []))
            return out

        rows = []
        for mutation in order:
            if mutation not in matrix.index:
                continue
            subtree = subtree_nodes(mutation)
            subtree_cells = set(
                attachment.loc[
                    attachment["attachment_node"].astype(str).isin(subtree), "cell"
                ].astype(str)
            )
            obs_cells = set(matrix.columns[matrix.loc[mutation].eq(1)])
            obs_in_subtree = len(obs_cells & subtree_cells)
            rows.append({
                "mutation": mutation,
                "frequency": int(freq.get(mutation, 0)),
                "subtree_cells": len(subtree_cells),
                "observed_ones_in_subtree": obs_in_subtree,
                "observed_ones_outside_subtree": len(obs_cells - subtree_cells),
                "children": len(children.get(mutation, [])),
            })
        unsupported = [
            r for r in rows
            if r["observed_ones_in_subtree"] < args.min_branch_support_cells
        ]
        if not unsupported:
            break
        # Prune the shallowest unsupported node first so child support is
        # recalculated after reattachment.
        unsupported.sort(key=lambda r: (
            len(ancestors(r["mutation"], parent)),
            -r["frequency"],
            r["mutation"],
        ))
        row = unsupported[0]
        mutation = row["mutation"]
        previous_parent = parent.get(mutation)
        for child, p in list(parent.items()):
            if p == mutation:
                parent[child] = previous_parent
        del parent[mutation]
        events.append({
            "round": pruning_round,
            "mutation": mutation,
            "previous_parent": previous_parent,
            "frequency": row["frequency"],
            "subtree_cells": row["subtree_cells"],
            "observed_ones_in_subtree": row["observed_ones_in_subtree"],
            "observed_ones_outside_subtree": row["observed_ones_outside_subtree"],
            "children_reattached": row["children"],
            "pruning_rule": "final_observed_support_in_subtree_below_min",
        })
    return events


def prune_split_subtree_support_mutations(matrix: pd.DataFrame,
                                          freq: pd.Series,
                                          parent: dict[str, str | None],
                                          root: str,
                                          args: argparse.Namespace,
                                          max_rounds: int = 10) -> list[dict]:
    """Prune low-frequency mutations whose observed support is split off-branch.

    A retained mutation should have most of its observed mutant cells inside its
    assigned subtree.  If a low-frequency mutation has at least as many observed
    mutant cells outside its subtree as inside it, and the outside count reaches
    the existing branch-support minimum, the site is treated as an unstable
    split/off-branch signal rather than a shared lineage marker.
    """
    events: list[dict] = []
    for pruning_round in range(1, max_rounds + 1):
        order = real_mutation_order(root, parent, freq, args.row_order_mode, matrix)
        _, attachment = expected_and_cell_order(matrix, order, parent, root, args)
        children: dict[str | None, list[str]] = defaultdict(list)
        for child, p in parent.items():
            children[p].append(child)

        def subtree_nodes(node: str) -> set[str]:
            out = {node}
            stack = list(children.get(node, []))
            while stack:
                x = stack.pop()
                out.add(x)
                stack.extend(children.get(x, []))
            return out

        candidates = []
        for mutation in order:
            if mutation not in matrix.index:
                continue
            mutation_freq = int(freq.get(mutation, 0))
            if mutation_freq > args.small_mutation_threshold:
                continue
            subtree = subtree_nodes(mutation)
            subtree_cells = set(
                attachment.loc[
                    attachment["attachment_node"].astype(str).isin(subtree), "cell"
                ].astype(str)
            )
            obs_cells = set(matrix.columns[matrix.loc[mutation].eq(1)])
            inside = len(obs_cells & subtree_cells)
            outside = len(obs_cells - subtree_cells)
            if outside < args.min_branch_support_cells:
                continue
            if inside < args.min_branch_support_cells:
                continue
            if outside < inside:
                continue
            candidates.append({
                "mutation": mutation,
                "frequency": mutation_freq,
                "subtree_cells": len(subtree_cells),
                "observed_ones_in_subtree": inside,
                "observed_ones_outside_subtree": outside,
                "children": len(children.get(mutation, [])),
            })
        if not candidates:
            break
        candidates.sort(key=lambda r: (
            len(ancestors(r["mutation"], parent)),
            -(r["observed_ones_outside_subtree"] - r["observed_ones_in_subtree"]),
            -r["frequency"],
            r["mutation"],
        ))
        row = candidates[0]
        mutation = row["mutation"]
        previous_parent = parent.get(mutation)
        for child, p in list(parent.items()):
            if p == mutation:
                parent[child] = previous_parent
        del parent[mutation]
        events.append({
            "round": pruning_round,
            "mutation": mutation,
            "previous_parent": previous_parent,
            "frequency": row["frequency"],
            "subtree_cells": row["subtree_cells"],
            "observed_ones_in_subtree": row["observed_ones_in_subtree"],
            "observed_ones_outside_subtree": row["observed_ones_outside_subtree"],
            "children_reattached": row["children"],
            "pruning_rule": "final_observed_support_split_outside_subtree",
        })
    return events


def branch_support_table(parent: dict[str, str | None], order: list[str],
                         attachment: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    children: dict[str, list[str]] = defaultdict(list)
    for child, p in parent.items():
        if p is not None:
            children[p].append(child)

    def descendants(node: str) -> set[str]:
        out: set[str] = set()
        stack = [node]
        while stack:
            x = stack.pop()
            out.add(x)
            stack.extend(children.get(x, []))
        return out

    rows = []
    for node in order:
        subtree = descendants(node)
        subtree_cell_ids = set(attachment.loc[attachment.attachment_node.isin(subtree), "cell"])
        subtree_cells = len(subtree_cell_ids)
        direct_cells = int((attachment.attachment_node == node).sum())
        observed_one_cells = set(matrix.columns[matrix.loc[node].eq(1)])
        mutated_cells_in_subtree = len(observed_one_cells & subtree_cell_ids)
        rows.append({"mutation": node, "direct_cell_count": direct_cells,
                     "subtree_cell_count": subtree_cells,
                     "mutated_cells_in_subtree": mutated_cells_in_subtree})
    return pd.DataFrame(rows)


def lineage_neighbor_support(node: str, parent: dict[str, str | None],
                             matrix: pd.DataFrame, args: argparse.Namespace) -> dict:
    """Observed 1/1 support with adjacent tree nodes, used to protect shared sites."""
    if node not in matrix.index:
        return {"lineage_neighbor_support": 0, "lineage_neighbor": "",
                "lineage_neighbor_direction": "", "lineage_backbone_support": 0,
                "lineage_backbone_ancestor": "", "lineage_neighbor_ratio": 0.0,
                "lineage_backbone_ratio": 0.0}
    values = matrix.loc[node].to_numpy(float)
    n1 = max(int((values == 1).sum()), 1)
    neighbors: list[tuple[str, str]] = []
    p = parent.get(node)
    if p is not None and p in matrix.index:
        neighbors.append((p, "parent"))
    for child, child_parent in parent.items():
        if child_parent == node and child in matrix.index:
            neighbors.append((child, "child"))
    best = {"lineage_neighbor_support": 0, "lineage_neighbor": "",
            "lineage_neighbor_direction": ""}
    for other, direction in neighbors:
        other_values = matrix.loc[other].to_numpy(float)
        overlap = int(((values == 1) & (other_values == 1)).sum())
        if direction == "parent":
            ok, _, _, _ = direction_plausible(other_values, values, args)
        else:
            ok, _, _, _ = direction_plausible(values, other_values, args)
        if ok and overlap > best["lineage_neighbor_support"]:
            best = {"lineage_neighbor_support": overlap, "lineage_neighbor": other,
                    "lineage_neighbor_direction": direction}
    backbone_best = 0
    backbone_ancestor = ""
    node_ancestors = ancestors(node, parent)
    for anc in node_ancestors[1:]:
        if anc not in matrix.index:
            continue
        anc_values = matrix.loc[anc].to_numpy(float)
        overlap = int(((values == 1) & (anc_values == 1)).sum())
        ok, _, _, _ = direction_plausible(anc_values, values, args)
        if ok and overlap > backbone_best:
            backbone_best = overlap
            backbone_ancestor = anc
    best["lineage_backbone_support"] = backbone_best
    best["lineage_backbone_ancestor"] = backbone_ancestor
    best["lineage_neighbor_ratio"] = best["lineage_neighbor_support"] / n1
    best["lineage_backbone_ratio"] = backbone_best / n1
    return best


def reliable_support_nodes(support: pd.DataFrame, args: argparse.Namespace) -> set[str]:
    """Nodes stable enough to serve as rescue parents for branch-private orphans."""
    reliable = support[
        (support["frequency"] > args.max_branch_support_mutation_count)
        | (support["mutated_cells_in_subtree"] >= args.min_branch_support_cells)
        | (
            (support["lineage_neighbor_support"] >= args.min_branch_support_cells)
            & (support["lineage_backbone_support"] >= args.min_branch_support_cells)
        )
    ]
    return set(reliable["mutation"].astype(str))


def rescue_branch_private_orphans(branch_private: pd.DataFrame,
                                  support: pd.DataFrame,
                                  matrix: pd.DataFrame,
                                  freq: pd.Series,
                                  parent: dict[str, str | None],
                                  root: str,
                                  args: argparse.Namespace) -> list[dict]:
    """Move branch-private candidates under reliable parents when evidence fits.

    This is a narrow rescue step: it only considers mutations that would
    otherwise be removed as branch-private and only uses already supported tree
    nodes as alternative parents.  The same directed containment cutoff is used
    for parent-child compatibility, with an added observed 1/1 overlap count.
    """
    doomed = set(branch_private["mutation"].astype(str))
    reliable = reliable_support_nodes(support, args) - doomed - {root}
    events: list[dict] = []
    for orphan in branch_private["mutation"].astype(str):
        if orphan not in matrix.index:
            continue
        orphan_values = matrix.loc[orphan].to_numpy(float)
        orphan_n1 = int((orphan_values == 1).sum())
        best: dict | None = None
        for candidate in reliable:
            if candidate not in matrix.index or candidate == orphan:
                continue
            if is_ancestor(orphan, candidate, parent):
                continue
            candidate_freq = int(freq.get(candidate, 0))
            frequency_difference = candidate_freq - orphan_n1
            if frequency_difference < 0:
                continue
            if frequency_difference >= args.small_mutation_threshold:
                continue
            candidate_values = matrix.loc[candidate].to_numpy(float)
            overlap = int(((candidate_values == 1) & (orphan_values == 1)).sum())
            if overlap < args.min_branch_support_cells:
                continue
            ok, conflict_count, conflict_ratio, comparable = direction_plausible(
                candidate_values, orphan_values, args)
            if not ok:
                continue
            candidate_support = support.loc[support["mutation"].astype(str) == candidate]
            subtree_cells = (int(candidate_support["subtree_cell_count"].iloc[0])
                             if not candidate_support.empty else 0)
            score = (
                frequency_difference,
                -overlap,
                conflict_ratio,
                -subtree_cells,
                candidate,
            )
            record = {
                "mutation": orphan,
                "rescued_parent": candidate,
                "previous_parent": parent.get(orphan, ""),
                "rescue_overlap": overlap,
                "rescue_conflict_count": conflict_count,
                "rescue_conflict_ratio": conflict_ratio,
                "rescue_comparable": comparable,
                "rescue_parent_frequency": candidate_freq,
                "mutation_frequency": orphan_n1,
                "score": score,
            }
            if best is None or score < best["score"]:
                best = record
        if best is not None:
            parent[orphan] = best["rescued_parent"]
            best.pop("score", None)
            events.append(best)
    return events


def path_from_ancestor_to_descendant(ancestor: str,
                                     descendant: str,
                                     parent: dict[str, str | None]) -> list[str]:
    """Return ancestor..descendant path, or [] when ancestor is not on the path."""
    path = [descendant]
    node = descendant
    while parent.get(node) is not None:
        node = parent[node]  # type: ignore[assignment]
        path.append(node)
        if node == ancestor:
            return list(reversed(path))
    return []


def identify_incompatible_anchor_pruning(parent: dict[str, str | None],
                                         support: pd.DataFrame,
                                         matrix: pd.DataFrame,
                                         freq: pd.Series,
                                         root: str,
                                         args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict]]:
    """Find low-frequency root anchors that split a compatible downstream branch.

    This intentionally narrow refinement targets cases where a low-frequency
    root-level mutation is a poor parent for the global branch structure, but a
    descendant below it is cleanly compatible with another established branch.
    Only the incompatible anchor-to-descendant prefix is removed; the compatible
    descendant remains active and can be reattached in the next build.
    """
    if not args.enable_anchor_pruning:
        return pd.DataFrame(), []
    support_by_mut = support.set_index("mutation", drop=False)
    rows: list[dict] = []
    events: list[dict] = []
    scheduled: set[str] = set()
    root_children = [m for m, p in parent.items()
                     if p == root and m in matrix.index
                     and int(freq.get(m, 0)) <= args.anchor_pruning_max_frequency]
    for anchor in root_children:
        if anchor in scheduled:
            continue
        anchor_descendants = descendants_of(anchor, parent)
        if not anchor_descendants:
            continue
        anchor_values = matrix.loc[anchor].to_numpy(float)
        best: dict | None = None
        for descendant in anchor_descendants:
            if descendant not in matrix.index:
                continue
            descendant_values = matrix.loc[descendant].to_numpy(float)
            path = path_from_ancestor_to_descendant(anchor, descendant, parent)
            if len(path) < 2:
                continue
            for external in parent:
                if external in {root, anchor, descendant} or external not in matrix.index:
                    continue
                if external in anchor_descendants:
                    continue
                if int(freq.get(external, 0)) < int(freq.get(descendant, 0)):
                    continue
                external_values = matrix.loc[external].to_numpy(float)
                desc_overlap, _, _ = support_from_parent(external_values, descendant_values)
                desc_ok, desc_conflict, desc_ratio, desc_comparable = direction_plausible(
                    external_values, descendant_values, args)
                if not desc_ok:
                    continue
                if desc_overlap < args.anchor_pruning_min_descendant_overlap:
                    continue
                if desc_conflict > args.anchor_pruning_max_descendant_conflict:
                    continue
                anchor_overlap, anchor_conflict, _ = support_from_parent(
                    external_values, anchor_values)
                if anchor_conflict < args.anchor_pruning_min_anchor_conflict:
                    continue
                if desc_overlap - anchor_overlap < args.anchor_pruning_min_overlap_gain:
                    continue

                prune_path = []
                for node in path[:-1]:
                    if node not in matrix.index:
                        continue
                    node_values = matrix.loc[node].to_numpy(float)
                    node_ok, node_conflict, _, _ = direction_plausible(
                        external_values, node_values, args)
                    node_overlap, _, _ = support_from_parent(external_values, node_values)
                    if node_ok and node_overlap >= args.anchor_pruning_min_descendant_overlap:
                        break
                    prune_path.append(node)
                if not prune_path:
                    continue
                score = (
                    -desc_overlap,
                    desc_conflict,
                    -int(freq.get(external, 0)),
                    len(prune_path),
                    external,
                    descendant,
                )
                record = {
                    "anchor": anchor,
                    "external_branch": external,
                    "compatible_descendant": descendant,
                    "prune_path": prune_path,
                    "anchor_frequency": int(freq.get(anchor, 0)),
                    "external_frequency": int(freq.get(external, 0)),
                    "descendant_frequency": int(freq.get(descendant, 0)),
                    "anchor_external_overlap": anchor_overlap,
                    "anchor_external_conflict": anchor_conflict,
                    "descendant_external_overlap": desc_overlap,
                    "descendant_external_conflict": desc_conflict,
                    "descendant_external_conflict_ratio": desc_ratio,
                    "descendant_external_comparable": desc_comparable,
                    "score": score,
                }
                if best is None or score < best["score"]:
                    best = record
        if best is None:
            continue
        best.pop("score", None)
        events.append({
            **{k: v for k, v in best.items() if k != "prune_path"},
            "pruned_mutations": ",".join(best["prune_path"]),
        })
        for node in best["prune_path"]:
            if node in scheduled:
                continue
            scheduled.add(node)
            hit = support_by_mut.loc[node].to_dict() if node in support_by_mut.index else {}
            rows.append({
                "mutation": node,
                "reason": "incompatible_low_frequency_anchor",
                "subtree_cell_count": int(hit.get("subtree_cell_count", 0)),
                "direct_cell_count": int(hit.get("direct_cell_count", 0)),
                "mutated_cells_in_subtree": int(hit.get("mutated_cells_in_subtree", 0)),
                "lineage_neighbor_support": int(hit.get("lineage_neighbor_support", 0)),
                "lineage_neighbor": hit.get("lineage_neighbor", ""),
                "lineage_backbone_support": int(hit.get("lineage_backbone_support", 0)),
                "lineage_backbone_ancestor": hit.get("lineage_backbone_ancestor", ""),
                "lineage_neighbor_ratio": float(hit.get("lineage_neighbor_ratio", 0.0)),
                "lineage_backbone_ratio": float(hit.get("lineage_backbone_ratio", 0.0)),
                "anchor_external_branch": best["external_branch"],
                "anchor_compatible_descendant": best["compatible_descendant"],
                "anchor_external_overlap": best["anchor_external_overlap"],
                "anchor_external_conflict": best["anchor_external_conflict"],
                "descendant_external_overlap": best["descendant_external_overlap"],
                "descendant_external_conflict": best["descendant_external_conflict"],
            })
    return pd.DataFrame(rows), events


def depth_map(parent: dict[str, str | None], root: str) -> dict[str, int]:
    depth = {root: 0}
    for node in parent:
        if node == root:
            continue
        depth[node] = len(ancestors(node, parent))
    return depth


def expected_and_cell_order(matrix: pd.DataFrame, order: list[str], parent: dict[str, str | None],
                            root: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = list(matrix.columns)
    depths = depth_map(parent, root)
    tree_pos = {root: -1, **{m: i for i, m in enumerate(order)}}
    candidate_nodes = [root] + [m for m in order if m != root]
    attachments: list[dict] = []
    expected_by_cell: dict[str, np.ndarray] = {}
    for cell in cells:
        obs = matrix.loc[order, cell].to_numpy(float)
        if args.require_observed_mutation_for_cell_attachment and int((obs == 1).sum()) == 0:
            exp = np.zeros(len(order), dtype=float)
            expected_by_cell[cell] = exp
            covered = ~np.isnan(obs)
            attachments.append({"cell": cell, "attachment_node": root,
                                "depth": 0, "covered_sites": int(covered.sum()),
                                "mismatch_count": 0.0, "mismatch_ratio": 0.0})
            continue
        choices = []
        for node in candidate_nodes:
            path = set(ancestors(node, parent) + [node])
            exp = np.array([1.0 if m in path else 0.0 for m in order])
            covered = ~np.isnan(obs)
            false_positive = (obs == 1) & (exp == 0) & covered
            false_negative = (obs == 0) & (exp == 1) & covered
            mismatch = (
                args.cell_false_positive_penalty * float(false_positive.sum())
                + args.cell_false_negative_penalty * float(false_negative.sum())
            )
            # NA is deliberately neutral for choosing placement; record it separately.
            choices.append((mismatch, -depths[node], tree_pos[node], node, exp))
        mismatch, neg_depth, _, node, exp = min(choices)
        expected_by_cell[cell] = exp
        attachments.append({"cell": cell, "attachment_node": node,
                            "depth": -neg_depth, "covered_sites": int((~np.isnan(obs)).sum()),
                            "mismatch_count": mismatch,
                            "mismatch_ratio": mismatch / max(int((~np.isnan(obs)).sum()), 1)})
    attach = pd.DataFrame(attachments).set_index("cell")
    def cell_sort_key(c: str) -> tuple:
        obs_col = matrix.loc[order, c].to_numpy(float)
        observed_ones = int((obs_col == 1).sum())
        empty_rank = int(
            args.cell_order_empty_last
            and (observed_ones == 0 or attach.loc[c, "attachment_node"] == root)
        )
        # A waterfall plot should keep the observed 1-class of an early branch
        # together when possible.  The attachment node remains a tie-breaker,
        # but the primary key uses the union of expected and observed mutation
        # support so that a high-level observed 1 is not visually split merely
        # because the cell has additional evidence for a sibling branch.
        if not args.cell_order_observed_support:
            return (empty_rank,
                    tree_pos[attach.loc[c, "attachment_node"]],
                    float(attach.loc[c, "mismatch_ratio"]), c)
        exp_col = expected_by_cell[c]
        support_profile = tuple(
            0 if (exp_col[i] == 1 or obs_col[i] == 1) else 1
            for i in range(len(order))
        )
        return (empty_rank, support_profile, tree_pos[attach.loc[c, "attachment_node"]],
                float(attach.loc[c, "mismatch_ratio"]), c)

    cell_order = sorted(cells, key=cell_sort_key)
    expected = pd.DataFrame({c: expected_by_cell[c] for c in cell_order}, index=order)
    return expected, attach.loc[cell_order].reset_index()


def conservative_denoised_matrix(observed: pd.DataFrame,
                                 expected: pd.DataFrame,
                                 parent: dict[str, str | None],
                                 order: list[str]) -> pd.DataFrame:
    """Minimally correct calls using both observed ancestry and attachment.

    A site is set to 1 when it is an ancestor of an observed mutation=1 on the
    selected attachment path in the same cell, allowing supported 0/NA->1
    corrections.  Locally supported off-path observed 1 calls are preserved, but
    they are not allowed to force high-level ancestor filling.  Other off-path
    observed 1 calls are treated as false positives and set to 0, and remaining
    NA values are set to 0.
    """
    out = observed.copy()
    order_set = set(order)
    children: dict[str, list[str]] = defaultdict(list)
    for child, p in parent.items():
        if p is not None:
            children[p].append(child)

    def has_nearby_observed_lineage_support(mutation: str, cell: str,
                                            max_distance: int = 1) -> bool:
        node = mutation
        for _ in range(max_distance):
            p = parent.get(node)
            if p is None or p not in order_set:
                break
            if observed.at[p, cell] == 1:
                return True
            node = p
        frontier = [(mutation, 0)]
        seen = {mutation}
        while frontier:
            node, dist = frontier.pop(0)
            if dist >= max_distance:
                continue
            for child in children.get(node, []):
                if child in seen or child not in order_set:
                    continue
                if observed.at[child, cell] == 1:
                    return True
                seen.add(child)
                frontier.append((child, dist + 1))
        return False

    for cell in observed.columns:
        path_observed_ones = [
            m for m in order
            if observed.at[m, cell] == 1
            and int(expected.at[m, cell]) == 1
        ]
        local_supported_ones = [
            m for m in order
            if observed.at[m, cell] == 1
            and int(expected.at[m, cell]) == 0
            and has_nearby_observed_lineage_support(m, cell)
        ]
        required = set(path_observed_ones)
        for mutation in path_observed_ones:
            required.update(a for a in ancestors(mutation, parent) if a in order_set)
        required.update(local_supported_ones)
        for mutation in order:
            old = observed.at[mutation, cell]
            if mutation in required:
                out.at[mutation, cell] = 1
            elif old == 1 and int(expected.at[mutation, cell]) == 0:
                out.at[mutation, cell] = 0
            elif pd.isna(old):
                out.at[mutation, cell] = 1 if mutation in required else 0
            else:
                out.at[mutation, cell] = int(old)
    return out.astype(int)


def smooth_single_cell_dropouts(denoised: pd.DataFrame,
                                observed: pd.DataFrame,
                                min_observed_ones: int) -> pd.DataFrame:
    """Correct isolated 0 calls flanked by 1/1 in the ordered waterfall."""
    out = denoised.copy()
    cells = list(out.columns)
    for mutation in out.index:
        if int((observed.loc[mutation] == 1).sum()) < min_observed_ones:
            continue
        for i in range(1, len(cells) - 1):
            cell = cells[i]
            if not (observed.at[mutation, cell] == 0 or pd.isna(observed.at[mutation, cell])):
                continue
            if (int(out.at[mutation, cells[i - 1]]) == 1
                    and int(out.at[mutation, cells[i]]) == 0
                    and int(out.at[mutation, cells[i + 1]]) == 1):
                out.at[mutation, cell] = 1
    return out.astype(int)


def smooth_vertical_lineage_dropouts(denoised: pd.DataFrame,
                                     observed: pd.DataFrame,
                                     min_observed_ones: int) -> pd.DataFrame:
    """Correct small vertical breaks inside a cell's ordered mutation block."""
    out = denoised.copy()
    mutations = list(out.index)
    for cell in out.columns:
        # First, preserve an observed 1 that sits next to an existing local
        # block of denoised 1s.  This rescues off-path child calls that are
        # visually part of the same vertical branch block, while isolated
        # off-path 1s remain corrected to 0.
        for i, mutation in enumerate(mutations):
            if int((observed.loc[mutation] == 1).sum()) < min_observed_ones:
                continue
            if observed.at[mutation, cell] != 1 or int(out.at[mutation, cell]) == 1:
                continue
            before = any(int(out.at[mutations[j], cell]) == 1
                         for j in range(max(0, i - 2), i))
            after = any(int(out.at[mutations[j], cell]) == 1
                        for j in range(i + 1, min(len(mutations), i + 3)))
            if before and after:
                out.at[mutation, cell] = 1

        # Then fill a single observed 0 gap between local 1 blocks.
        for i, mutation in enumerate(mutations):
            if int((observed.loc[mutation] == 1).sum()) < min_observed_ones:
                continue
            if observed.at[mutation, cell] != 0 or int(out.at[mutation, cell]) == 1:
                continue
            before = any(int(out.at[mutations[j], cell]) == 1
                         for j in range(max(0, i - 2), i))
            after = any(int(out.at[mutations[j], cell]) == 1
                        for j in range(i + 1, min(len(mutations), i + 3)))
            if before and after:
                out.at[mutation, cell] = 1
    return out.astype(int)


def to_newick(root: str, parent: dict[str, str | None], freq: pd.Series) -> str:
    children: dict[str, list[str]] = defaultdict(list)
    for node, p in parent.items():
        if p is not None:
            children[p].append(node)
    for p in children:
        children[p].sort(key=lambda x: (-int(freq[x]), x))

    def safe(s: str) -> str:
        return "'" + s.replace("'", "''") + "'"

    def rec(node: str) -> str:
        inside = ""
        if children.get(node):
            inside = "(" + ",".join(rec(x) for x in children[node]) + ")"
        if node == root:
            return inside
        return inside + safe(node) + ("" if node == root else ":1")
    return rec(root) + ";\n"


def save_heatmap(observed: pd.DataFrame, expected: pd.DataFrame, out: Path,
                 skipped_observed: pd.DataFrame | None = None,
                 nonfinal_observed: pd.DataFrame | None = None,
                 row_annotation: pd.Series | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        skipped_observed = (skipped_observed if skipped_observed is not None
                            else pd.DataFrame(columns=observed.columns))
        nonfinal_observed = (nonfinal_observed if nonfinal_observed is not None
                             else pd.DataFrame(columns=observed.columns))
        row_annotation = (row_annotation if row_annotation is not None
                          else pd.Series("final", index=observed.index, dtype=object))

        pieces = [observed]
        labels = list(observed.index)
        row_group = ["retained"] * observed.shape[0]
        annotation_values = [str(row_annotation.get(m, "final")) for m in observed.index]
        section_lines: list[int] = []

        def add_section(section: pd.DataFrame, group: str, label_prefix: str) -> None:
            nonlocal pieces, labels, row_group, annotation_values, section_lines
            if section.empty:
                return
            gap_n = 2
            gap = pd.DataFrame(np.nan, index=[f"__gap_{group}_{i + 1}__" for i in range(gap_n)],
                               columns=observed.columns)
            section_lines.append(sum(x.shape[0] for x in pieces))
            pieces.extend([gap, section])
            labels.extend([""] * gap_n + [
                f"[{label_prefix}] {m}" for m in section.index
            ])
            row_group.extend(["gap"] * gap_n + [group] * section.shape[0])
            annotation_values.extend(["gap"] * gap_n + [
                str(row_annotation.get(m, group)) for m in section.index
            ])

        add_section(skipped_observed, "skipped", "skipped")
        add_section(nonfinal_observed, "nonfinal", "non-final")
        observed_for_plot = pd.concat(pieces, axis=0)
        has_extra = len(pieces) > 1

        arr = observed_for_plot.to_numpy(float)
        shown = np.where(np.isnan(arr), 0, np.where(arr == 0, 1, 2))
        if has_extra:
            gap_rows = np.array([g == "gap" for g in row_group])
            shown[gap_rows, :] = 3
        cmap = ListedColormap(["#bdbdbd", "#ffffff", "#3182bd", "#111111"])

        annotation_palette = {
            "final": "#4daf4a",
            "germ": "#984ea3",
            "vaf": "#ff7f00",
            "batch": "#e41a1c",
            "unannotated": "#999999",
            "skipped": "#377eb8",
            "nonfinal": "#999999",
            "gap": "#111111",
        }
        extra_categories = sorted(set(annotation_values).difference(annotation_palette))
        fallback_colors = ["#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"]
        for i, cat in enumerate(extra_categories):
            annotation_palette[cat] = fallback_colors[i % len(fallback_colors)]
        annotation_categories = [c for c in annotation_palette if c in set(annotation_values)]
        ann_code = {cat: i for i, cat in enumerate(annotation_categories)}
        ann_arr = np.array([[ann_code[v]] for v in annotation_values])
        ann_cmap = ListedColormap([annotation_palette[c] for c in annotation_categories])

        # Large custom page: individual mutation labels and subtle cell/site gaps
        # remain readable in the vector PDF even for ~180 columns and ~100 rows.
        width = max(24, observed_for_plot.shape[1] * 0.15)
        observed_height = max(10, observed_for_plot.shape[0] * 0.135)
        expected_height = max(10, expected.shape[0] * 0.135)
        fig, axes = plt.subplots(
            2, 1, figsize=(width, observed_height + expected_height + 2),
            dpi=160, gridspec_kw={"height_ratios": [observed_height, expected_height]}
        )
        x_edges = np.arange(observed_for_plot.shape[1] + 1)
        y_edges = np.arange(observed_for_plot.shape[0] + 1)
        ann_x_edges = np.array([-1.2, -0.2])
        axes[0].pcolormesh(x_edges, y_edges, shown, cmap=cmap, vmin=0, vmax=3,
                           shading="flat", edgecolors="#f7f7f7", linewidth=0.16)
        axes[0].pcolormesh(ann_x_edges, y_edges, ann_arr, cmap=ann_cmap, vmin=0,
                           vmax=max(len(annotation_categories) - 1, 1),
                           shading="flat", edgecolors="#f7f7f7", linewidth=0.16)
        axes[0].set_title(
            "Ordered observed matrix with skipped and non-final mutations "
            "(gray=NA, white=WT, blue=mutation)"
            if has_extra else
            "Ordered observed matrix (gray=NA, white=WT, blue=mutation)"
        )
        for y in section_lines:
            axes[0].axhline(y, color="black", linewidth=1.2)
            axes[0].axhline(y + 2, color="black", linewidth=1.2)
        if annotation_categories:
            handles = [Patch(facecolor=annotation_palette[c], edgecolor="none", label=c)
                       for c in annotation_categories if c != "gap"]
            axes[0].legend(handles=handles, title="Mutation annotation", loc="upper left",
                           bbox_to_anchor=(1.005, 1.0), fontsize=7, title_fontsize=8,
                           borderaxespad=0)

        x_edges_expected = np.arange(expected.shape[1] + 1)
        y_edges_expected = np.arange(expected.shape[0] + 1)
        axes[1].pcolormesh(x_edges_expected, y_edges_expected, expected.to_numpy(float),
                           cmap=ListedColormap(["#ffffff", "#3182bd"]), vmin=0, vmax=1,
                           shading="flat", edgecolors="#f7f7f7", linewidth=0.16)
        axes[1].set_title("Infinite-sites denoised matrix (retained mutations only)")

        axes[0].set_xlim(-1.2, observed_for_plot.shape[1])
        axes[0].set_ylim(observed_for_plot.shape[0], 0)
        axes[0].set_yticks(np.arange(observed_for_plot.shape[0]) + 0.5)
        axes[0].set_yticklabels(labels, fontsize=5.5)

        axes[1].set_xlim(0, expected.shape[1])
        axes[1].set_ylim(expected.shape[0], 0)
        axes[1].set_yticks(np.arange(expected.shape[0]) + 0.5)
        axes[1].set_yticklabels(expected.index, fontsize=5.5)

        for ax in axes:
            ax.set_aspect("auto")
            ax.set_xticks([])
            ax.tick_params(axis="y", length=0, pad=3)
            ax.set_ylabel("Mutation site", fontsize=9)
            ax.set_xlabel(f"Cells (n={observed_for_plot.shape[1]})", fontsize=9)
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # plotting is optional, tabular outputs are primary
        print(f"Warning: heatmap was not generated: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    matrix = load_matrix(args.input)
    mutation_annotation = load_annotation(args.annotation, args, matrix.index)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    freq_all = matrix.eq(1).sum(axis=1).astype(int)
    final_label = args.annotation_final_label.strip().lower()
    final_annotated = [
        m for m in matrix.index
        if str(mutation_annotation.get(m, "unannotated")).strip().lower() == final_label
    ]
    nonfinal_annotated = [m for m in matrix.index if m not in set(final_annotated)]
    if args.annotation is not None:
        pd.DataFrame({
            "mutation": matrix.index,
            "annotation": [mutation_annotation.get(m, "unannotated") for m in matrix.index],
            "frequency": [int(freq_all[m]) for m in matrix.index],
            "used_for_tree": [m in set(final_annotated) for m in matrix.index],
        }).to_csv(out / "mutation_annotation_used.tsv", sep="\t", index=False)
    private = list(freq_all.loc[final_annotated][freq_all.loc[final_annotated] < args.min_mutated_cells].index)
    eligible = [m for m in final_annotated if m not in private]
    if not eligible:
        raise ValueError("No final-annotated mutations remain after private-mutation filtering")
    pd.DataFrame({"mutation": private, "mutated_cell_count": [int(freq_all[m]) for m in private],
                  "reason": "private_mutation"}).to_csv(out / "removed_private_mutations.tsv",
                                                          sep="\t", index=False)

    active = eligible.copy()
    history: list[pd.DataFrame] = []
    removed_noise: list[dict] = []
    final_result = None
    stop_reason = "single_pass_step2_deferred_no_post_tree_deletion"

    final_iteration = 0
    max_builds = args.max_branch_support_iterations + 1
    for iteration in range(1, max_builds + 1):
        idir = out / f"iteration_{iteration}"
        idir.mkdir(exist_ok=True)
        result = build_tree(matrix, active, args)
        decisions = result["decisions"].copy()
        decisions.insert(0, "iteration", iteration)
        decisions_to_write = decisions[decisions["mutation"] != result["root"]].copy()
        decisions_to_write = hide_pseudo_root(decisions_to_write, result["root"])
        decisions_to_write.to_csv(idir / "mutation_decisions.tsv", sep="\t", index=False)
        pd.DataFrame(result["reparenting_events"]).to_csv(
            idir / "missingness_reparenting_events.tsv", sep="\t", index=False)
        assigned = decisions_to_write[decisions_to_write.status == "assigned"]
        deferred = decisions_to_write[decisions_to_write.status == "deferred_possible_noise"]
        assigned.to_csv(idir / "assigned_mutations.tsv", sep="\t", index=False)
        deferred.to_csv(idir / "deferred_mutations.tsv", sep="\t", index=False)
        def compute_order_and_support() -> tuple[list[str], pd.DataFrame]:
            current_order = real_mutation_order(result["root"], result["parent"],
                                                result["frequency"], args.row_order_mode, matrix)
            _, current_attachment = expected_and_cell_order(
                matrix, current_order, result["parent"], result["root"], args)
            current_support = branch_support_table(
                result["parent"], current_order, current_attachment, matrix)
            current_support = current_support.merge(
                pd.DataFrame({"mutation": current_order,
                              "frequency": [int(result["frequency"][m]) for m in current_order]}),
                on="mutation", how="left")
            current_lineage_support = pd.DataFrame([
                {"mutation": m, **lineage_neighbor_support(m, result["parent"], matrix, args)}
                for m in current_order
            ])
            current_support = current_support.merge(current_lineage_support, on="mutation", how="left")
            return current_order, current_support

        def branch_private_from_support(current_support: pd.DataFrame) -> pd.DataFrame:
            return current_support[
                (current_support["mutation"] != result["root"])
                & (current_support["frequency"] <= args.max_branch_support_mutation_count)
                & (current_support["mutated_cells_in_subtree"] < args.min_branch_support_cells)
                & (
                    (current_support["lineage_neighbor_support"] < args.min_branch_support_cells)
                    | (current_support["lineage_backbone_support"] < args.min_branch_support_cells)
                )
            ].copy()

        order, support = compute_order_and_support()
        branch_private = branch_private_from_support(support)
        anchor_pruning_events: list[dict] = []
        anchor_private = pd.DataFrame()
        rescue_events = []
        if not branch_private.empty and iteration <= args.max_branch_support_iterations:
            rescue_events = rescue_branch_private_orphans(
                branch_private, support, matrix, result["frequency"],
                result["parent"], result["root"], args)
            if rescue_events:
                rescue_parent = {x["mutation"]: x["rescued_parent"] for x in rescue_events}
                for idx, row in decisions.iterrows():
                    mutation = str(row.get("mutation", ""))
                    if mutation in rescue_parent:
                        decisions.loc[idx, "parent"] = rescue_parent[mutation]
                        decisions.loc[idx, "reason"] = "rescued_branch_private_orphan"
                        decisions.loc[idx, "reparented_from"] = next(
                            x["previous_parent"] for x in rescue_events if x["mutation"] == mutation)
                decisions_to_write = decisions[decisions["mutation"] != result["root"]].copy()
                decisions_to_write = hide_pseudo_root(decisions_to_write, result["root"])
                assigned = decisions_to_write[decisions_to_write.status == "assigned"]
                deferred = decisions_to_write[decisions_to_write.status == "deferred_possible_noise"]
                decisions_to_write.to_csv(idir / "mutation_decisions.tsv", sep="\t", index=False)
                assigned.to_csv(idir / "assigned_mutations.tsv", sep="\t", index=False)
                deferred.to_csv(idir / "deferred_mutations.tsv", sep="\t", index=False)
                order, support = compute_order_and_support()
                branch_private = branch_private_from_support(support)

        if iteration <= args.max_branch_support_iterations:
            anchor_private, anchor_pruning_events = identify_incompatible_anchor_pruning(
                result["parent"], support, matrix, result["frequency"],
                result["root"], args)
            if not anchor_private.empty:
                existing = set(branch_private["mutation"].astype(str)) if not branch_private.empty else set()
                anchor_private = anchor_private[
                    ~anchor_private["mutation"].astype(str).isin(existing)
                ].copy()
                if not anchor_private.empty:
                    branch_private = pd.concat([branch_private, anchor_private],
                                               ignore_index=True, sort=False)

        pd.DataFrame(rescue_events).to_csv(idir / "orphan_rescue_events.tsv", sep="\t", index=False)
        pd.DataFrame(anchor_pruning_events).to_csv(
            idir / "anchor_pruning_events.tsv", sep="\t", index=False)
        edges = pd.DataFrame([{"parent": "" if p == result["root"] else p, "child": c}
                              for c, p in result["parent"].items()
                              if p is not None and c != result["root"]])
        edges.to_csv(idir / "mutation_tree_edges.tsv", sep="\t", index=False)
        pd.DataFrame({"dfs_order": range(1, len(order) + 1), "mutation": order,
                      "frequency": [int(result["frequency"][m]) for m in order]}).to_csv(
                          idir / "mutation_order.tsv", sep="\t", index=False)
        support.to_csv(idir / "branch_support.tsv", sep="\t", index=False)
        noise = None
        removed_this_round = []
        if not branch_private.empty and iteration <= args.max_branch_support_iterations:
            for row in branch_private.to_dict("records"):
                removed_this_round.append({
                    "mutation": row["mutation"],
                    "reason": "branch_private_within_assigned_subtree",
                    "iteration": iteration,
                    "subtree_cell_count": int(row["subtree_cell_count"]),
                    "direct_cell_count": int(row["direct_cell_count"]),
                    "mutated_cells_in_subtree": int(row["mutated_cells_in_subtree"]),
                    "lineage_neighbor_support": int(row.get("lineage_neighbor_support", 0)),
                    "lineage_neighbor": row.get("lineage_neighbor", ""),
                    "lineage_backbone_support": int(row.get("lineage_backbone_support", 0)),
                    "lineage_backbone_ancestor": row.get("lineage_backbone_ancestor", ""),
                    "lineage_neighbor_ratio": float(row.get("lineage_neighbor_ratio", 0.0)),
                    "lineage_backbone_ratio": float(row.get("lineage_backbone_ratio", 0.0)),
                })
            removed_noise.extend(removed_this_round)
            active = [m for m in active if m not in set(branch_private["mutation"])]
            noise = ",".join(branch_private["mutation"].astype(str))
        pd.DataFrame(removed_this_round, columns=[
            "mutation", "reason", "iteration", "subtree_cell_count",
            "direct_cell_count", "mutated_cells_in_subtree",
            "lineage_neighbor_support", "lineage_neighbor",
            "lineage_backbone_support", "lineage_backbone_ancestor",
            "lineage_neighbor_ratio", "lineage_backbone_ratio"
        ]).to_csv(idir / "removed_mutation.tsv", sep="\t", index=False)
        summary = {
            "iteration": iteration, "input_mutations": len(active), "root": result["root"],
            "assigned_mutations": len(result["parent"]) - 1,
            "deferred_mutations": len(result["deferred"]),
            "deferred_mutation_ids": result["deferred"], "removed_mutation": noise,
            "removed_mutation_reason": "branch_private_within_assigned_subtree" if noise else "",
        }
        (idir / "iteration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Iteration {iteration}: pseudo_root={result['root']}; assigned={len(result['parent']) - 1}; "
              f"deferred={len(result['deferred'])}; removed={noise or 'none'}")
        for row in deferred.to_dict("records"):
            print(f"  Skipped mutation {row['mutation']}: {row.get('reason', 'no_valid_parent')}; "
                  f"best candidate={row.get('candidate', 'none')}")
        history.append(decisions)
        if removed_this_round and iteration <= args.max_branch_support_iterations:
            for row in removed_this_round:
                print(f"  Removed branch-private mutation {row['mutation']}: "
                      f"subtree_cell_count={row['subtree_cell_count']}; "
                      f"mutated_cells_in_subtree={row['mutated_cells_in_subtree']}; "
                      f"direct_cell_count={row['direct_cell_count']}")
            continue
        final_result = result
        final_iteration = iteration
        break

    assert final_result is not None
    final_refinement_events = refine_descendant_parent_placements(
        matrix, final_result["frequency"], final_result["parent"],
        final_result["root"], args)
    final_direction_events = refine_noisy_parent_child_direction(
        matrix, final_result["frequency"], final_result["parent"],
        final_result["root"], args)
    final_split_pruning_events = (
        prune_split_subtree_support_mutations(
            matrix, final_result["frequency"], final_result["parent"],
            final_result["root"], args)
        if args.enable_final_split_pruning else []
    )
    final_unsupported_pruning_events = prune_unsupported_subtree_mutations(
        matrix, final_result["frequency"], final_result["parent"],
        final_result["root"], args, max_rounds=args.max_final_unsupported_pruning)
    final_post_prune_refinement_events = refine_descendant_parent_placements(
        matrix, final_result["frequency"], final_result["parent"],
        final_result["root"], args,
        allow_lower_frequency=True, prefer_deepest=True)
    if final_refinement_events:
        final_result["reparenting_events"].extend(final_refinement_events)
    if final_direction_events:
        final_result["reparenting_events"].extend(final_direction_events)
    if final_split_pruning_events:
        final_result["reparenting_events"].extend(final_split_pruning_events)
    if final_unsupported_pruning_events:
        final_result["reparenting_events"].extend(final_unsupported_pruning_events)
    if final_post_prune_refinement_events:
        final_result["reparenting_events"].extend(final_post_prune_refinement_events)
    pd.DataFrame(final_refinement_events).to_csv(
        out / "final_parent_refinement_events.tsv", sep="\t", index=False)
    pd.DataFrame(final_direction_events).to_csv(
        out / "final_direction_refinement_events.tsv", sep="\t", index=False)
    pd.DataFrame(final_split_pruning_events).to_csv(
        out / "final_split_subtree_pruning_events.tsv", sep="\t", index=False)
    pd.DataFrame(final_unsupported_pruning_events).to_csv(
        out / "final_unsupported_subtree_pruning_events.tsv", sep="\t", index=False)
    pd.DataFrame(final_post_prune_refinement_events).to_csv(
        out / "final_post_prune_parent_refinement_events.tsv", sep="\t", index=False)

    history_to_write = pd.concat(history, ignore_index=True)
    history_to_write = history_to_write[history_to_write["mutation"] != final_result["root"]]
    history_to_write = hide_pseudo_root(history_to_write, final_result["root"])
    history_to_write.to_csv(out / "iteration_history.tsv", sep="\t", index=False)
    pd.DataFrame(removed_noise).to_csv(out / "removed_noise_mutations.tsv", sep="\t", index=False)
    unresolved = final_result["decisions"]
    unresolved = unresolved[unresolved.status == "deferred_possible_noise"].copy()
    removed_noise_ids = {x["mutation"] for x in removed_noise}
    unresolved = unresolved[~unresolved.mutation.isin(removed_noise_ids)]
    if not unresolved.empty:
        unresolved["final_status"] = "unresolved_possible_noise"
    hide_pseudo_root(unresolved, final_result["root"]).to_csv(
        out / "unresolved_mutations.tsv", sep="\t", index=False)
    pd.DataFrame(final_result["reparenting_events"]).to_csv(
        out / "missingness_reparenting_events.tsv", sep="\t", index=False)

    # A single audit table for every mutation excluded from the final tree.
    # Counts and ratios are kept side by side because the active containment
    # metric depends on mutation frequency, while mutual exclusion is always
    # judged by its ratio in the current implementation.
    filtered_parts: list[pd.DataFrame] = []
    if not unresolved.empty:
        deferred_report = unresolved.copy()
        deferred_report["filter_action"] = "deferred_unresolved"
        deferred_report["filter_iteration"] = 1
        filtered_parts.append(deferred_report)
    if removed_noise:
        history_all = pd.concat(history, ignore_index=True)
        removed_rows: list[pd.DataFrame] = []
        for item in removed_noise:
            hit = history_all[(history_all["iteration"] == item["iteration"])
                              & (history_all["mutation"] == item["mutation"])].copy()
            if not hit.empty:
                hit = hit.tail(1)
                hit["filter_action"] = item.get("reason", "removed_as_noise")
                hit["filter_iteration"] = int(item["iteration"])
                hit["subtree_cell_count"] = item.get("subtree_cell_count", np.nan)
                hit["direct_cell_count"] = item.get("direct_cell_count", np.nan)
                hit["mutated_cells_in_subtree"] = item.get("mutated_cells_in_subtree", np.nan)
                removed_rows.append(hit)
        if removed_rows:
            filtered_parts.append(pd.concat(removed_rows, ignore_index=True))
    filtered = (pd.concat(filtered_parts, ignore_index=True, sort=False)
                if filtered_parts else pd.DataFrame())
    if not filtered.empty:
        filtered["mutated_cell_count"] = filtered["mutation"].map(freq_all).astype(int)
        filtered["parent_metric_used"] = np.where(
            filtered["mutated_cell_count"] <= args.small_mutation_threshold,
            "weighted_ratio", "weighted_count")
        filtered["parent_cutoff_used"] = np.where(
            filtered["mutated_cell_count"] <= args.small_mutation_threshold,
            args.max_parent_conflict_ratio, args.max_parent_conflict_count)
        filtered["parent_score_used"] = np.where(
            filtered["parent_metric_used"] == "weighted_ratio",
            filtered["parent_conflict_ratio"], filtered["parent_conflict_count"])
        filtered["parent_excess_over_cutoff"] = (
            filtered["parent_score_used"] - filtered["parent_cutoff_used"])
        filtered["branch_metric_used"] = "ratio"
        filtered["branch_cutoff_used"] = np.where(
            filtered["mutated_cell_count"] <= args.small_mutation_threshold,
            args.small_branch_overlap_ratio, args.max_branch_overlap_ratio)
        filtered["branch_score_used"] = filtered["branch_overlap_ratio"]
        filtered["branch_excess_over_cutoff"] = (
            filtered["branch_score_used"] - filtered["branch_cutoff_used"])
        filtered["diagnostic_ratio_sum"] = (
            filtered["parent_conflict_ratio"] + filtered["branch_overlap_ratio"])
        report_columns = [
            "mutation", "mutated_cell_count", "filter_action", "filter_iteration",
            "reason", "candidate", "frequency_difference",
            "parent_conflict_count", "parent_conflict_ratio",
            "parent_definite_conflict_count", "parent_definite_conflict_ratio",
            "parent_comparable_child_ones", "parent_conflict_wilson_lower",
            "parent_metric_used",
            "parent_score_used", "parent_cutoff_used", "parent_excess_over_cutoff",
            "branch_overlap_count", "branch_overlap_ratio", "branch_metric_used",
            "branch_score_used", "branch_cutoff_used", "branch_excess_over_cutoff",
            "worst_other_branch", "subtree_cell_count", "direct_cell_count",
            "mutated_cells_in_subtree",
            "diagnostic_ratio_sum"]
        filtered = filtered.reindex(columns=report_columns).sort_values(
            ["filter_action", "reason", "mutated_cell_count", "mutation"],
            ascending=[True, True, False, True])
        filtered = hide_pseudo_root(filtered, final_result["root"])
    filtered.to_csv(out / "filtered_mutation_penalty_scores.tsv", sep="\t", index=False)

    root = final_result["root"]
    parent = final_result["parent"]
    freq = final_result["frequency"]
    order = real_mutation_order(root, parent, freq, args.row_order_mode, matrix)
    expected, attachment = expected_and_cell_order(matrix, order, parent, root, args)
    cells = list(expected.columns)
    observed = matrix.loc[order, cells]
    denoised = (conservative_denoised_matrix(observed, expected, parent, order)
                if args.conservative_denoise else expected.astype(int))
    if args.smooth_single_cell_dropouts:
        denoised = smooth_vertical_lineage_dropouts(
            denoised, observed, args.smooth_min_observed_ones)
        denoised = smooth_single_cell_dropouts(
            denoised, observed, args.smooth_min_observed_ones)
    original_position = {m: i for i, m in enumerate(matrix.index)}
    skipped_for_heatmap = [m for m in final_annotated if m not in set(order)]
    skipped_for_heatmap = sorted(
        skipped_for_heatmap,
        key=lambda m: (-int(freq_all.get(m, 0)), original_position.get(m, 10**9), m)
    )
    nonfinal_for_heatmap = sorted(
        nonfinal_annotated,
        key=lambda m: (-int(freq_all.get(m, 0)), original_position.get(m, 10**9), m)
    )
    skipped_observed = matrix.loc[skipped_for_heatmap, cells] if skipped_for_heatmap else pd.DataFrame(columns=cells)
    nonfinal_observed = (matrix.loc[nonfinal_for_heatmap, cells]
                         if nonfinal_for_heatmap else pd.DataFrame(columns=cells))
    skipped_reasons: dict[str, str] = {m: "not_in_final_tree" for m in skipped_for_heatmap}
    for m in private:
        if m in skipped_reasons:
            skipped_reasons[m] = "private_mutation"
    for m in unresolved["mutation"].tolist() if not unresolved.empty else []:
        if m in skipped_reasons:
            skipped_reasons[m] = "deferred_unresolved"
    for item in removed_noise:
        m = item.get("mutation")
        if m in skipped_reasons:
            skipped_reasons[m] = item.get("reason", "removed_as_noise")
    pd.DataFrame({
        "mutation_order": range(1, len(skipped_for_heatmap) + 1),
        "mutation": skipped_for_heatmap,
        "frequency": [int(freq_all.get(m, 0)) for m in skipped_for_heatmap],
        "annotation": [mutation_annotation.get(m, "unannotated") for m in skipped_for_heatmap],
        "reason": [skipped_reasons.get(m, "not_in_final_tree") for m in skipped_for_heatmap],
    }).to_csv(out / "skipped_mutations_for_heatmap.tsv", sep="\t", index=False)
    pd.DataFrame({
        "mutation_order": range(1, len(nonfinal_for_heatmap) + 1),
        "mutation": nonfinal_for_heatmap,
        "frequency": [int(freq_all.get(m, 0)) for m in nonfinal_for_heatmap],
        "annotation": [mutation_annotation.get(m, "unannotated") for m in nonfinal_for_heatmap],
    }).to_csv(out / "nonfinal_mutations_for_heatmap.tsv", sep="\t", index=False)
    expected.index.name = matrix.index.name
    denoised.index.name = matrix.index.name
    observed.to_csv(out / "ordered_observed_matrix.tsv", sep="\t", na_rep="NA")
    denoised.astype(int).to_csv(out / "denoised_matrix.tsv", sep="\t")
    denoised.astype(int).to_csv(out / "denoised_matrix.csv")
    attachment.to_csv(out / "cell_attachments.tsv", sep="\t", index=False)
    pd.DataFrame({"cell_order": range(1, len(cells) + 1), "cell": cells}).to_csv(
        out / "cell_order.tsv", sep="\t", index=False)
    pd.DataFrame({"mutation_order": range(1, len(order) + 1), "mutation": order,
                  "frequency": [int(freq[m]) for m in order]}).to_csv(
                      out / "mutation_order.tsv", sep="\t", index=False)

    corrections: list[dict] = []
    for mutation in order:
        for cell in cells:
            old = observed.at[mutation, cell]
            new = int(denoised.at[mutation, cell])
            if pd.isna(old) or int(old) != new:
                if pd.isna(old):
                    ctype, reason = f"NA_to_{new}", "tree_expected_state"
                else:
                    ctype, reason = f"{int(old)}_to_{new}", "tree_conflict_correction"
                corrections.append({"mutation": mutation, "cell": cell,
                                    "original_value": "NA" if pd.isna(old) else int(old),
                                    "corrected_value": new, "correction_type": ctype,
                                    "reason": reason})
    correction_df = pd.DataFrame(corrections)
    correction_df.to_csv(out / "corrections.tsv", sep="\t", index=False)
    correction_counts = correction_df.groupby("mutation").size() if not correction_df.empty else pd.Series(dtype=int)
    qc = pd.DataFrame({"mutation": order,
                       "frequency_observed": [int(freq[m]) for m in order],
                       "correction_count": [int(correction_counts.get(m, 0)) for m in order]})
    qc["correction_fraction"] = qc.correction_count / matrix.shape[1]
    qc["low_confidence"] = qc.correction_fraction > args.max_correction_fraction
    qc.to_csv(out / "mutation_correction_qc.tsv", sep="\t", index=False)

    edges = pd.DataFrame([{"parent": "" if p == root else p, "child": c, "branch_length": 1}
                          for c, p in parent.items() if p is not None and c != root])
    edges.to_csv(out / "mutation_tree_edges.tsv", sep="\t", index=False)
    (out / "mutation_tree.newick").write_text(to_newick(root, parent, freq), encoding="utf-8")
    raw_depths = depth_map(parent, root)
    depths = {node: raw_depths[node] - 1 for node in order}
    max_tree_depth = max(depths.values()) if depths else 0
    deepest_mutations = sorted([node for node, depth in depths.items() if depth == max_tree_depth])
    (out / "max_tree_depth.txt").write_text(
        f"max_tree_depth\t{max_tree_depth}\n"
        f"deepest_mutations\t{','.join(deepest_mutations)}\n",
        encoding="utf-8")
    direct_counts = attachment.attachment_node.value_counts()
    node_table = []
    for node in order:
        subtree = [n for n in order if is_ancestor(node, n, parent)]
        subtree_cells = int(attachment.attachment_node.isin(subtree).sum())
        node_parent = parent[node]
        node_table.append({"node_id": node, "parent_id": "" if node_parent == root else (node_parent or ""),
                           "depth": depths[node], "observed_mutated_cells": int(freq[node]),
                           "direct_cell_count": int(direct_counts.get(node, 0)),
                           "subtree_cell_count": subtree_cells, "branch_mutations": 1})
    pd.DataFrame(node_table).to_csv(out / "tree_nodes_for_plotting.tsv", sep="\t", index=False)
    heatmap_annotation = mutation_annotation.copy()
    heatmap_annotation.loc[order] = [mutation_annotation.get(m, "final") for m in order]
    save_heatmap(observed, denoised, out / "ordered_and_denoised_heatmaps.pdf",
                 skipped_observed, nonfinal_observed, heatmap_annotation)

    summary = {
        "input": str(args.input), "input_mutations": matrix.shape[0], "cells": matrix.shape[1],
        "annotation": str(args.annotation) if args.annotation is not None else "",
        "annotation_final_label": args.annotation_final_label,
        "annotated_final_mutations_in_matrix": len(final_annotated),
        "annotated_nonfinal_mutations_in_matrix": len(nonfinal_annotated),
        "private_mutations_removed": len(private), "iterative_noise_mutations_removed": len(removed_noise),
        "final_tree_mutations": len(order), "final_unresolved_mutations": len(unresolved),
        "root_mutation": "", "pseudo_root_used": True, "pseudo_root_id": root,
        "root_children": sorted([c for c, p in parent.items() if p == root]),
        "total_corrections": len(correction_df), "stop_reason": stop_reason,
        "final_iteration": final_iteration, "max_tree_depth": max_tree_depth,
        "deepest_mutations": deepest_mutations,
        "parameters": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
            if k not in {"input", "output_dir"}
        },
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
