"""
CuPy multi-level demo focusing on interface refinement.

Scenario:
  1. Build a coarse mask (rectangular regions) and a coarse per-cell field.
  2. Prolong geometry and field to a finer grid (ratio configurable).
  3. Add a fine-only circular feature representing a refined interface region.
  4. Project the refined mask back to the coarse level to see the extra coverage.
  5. Restrict the fine field to coarse (aligned case) and visualise the delta.

Usage:
    python -m subsetix_cupy.demo_multilevel --coarse 64 --ratio 2 [--plot]
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from . import (
    build_interval_set,
    create_interval_field,
    evaluate,
    make_difference,
    make_input,
    make_union,
    prolong_field,
    prolong_set,
    restrict_field,
    restrict_set,
)
from .demo_multilevel_layout import build_cell_rectangles, plot_cell_layout_from_sets


def _field_to_dense(field, width: int) -> np.ndarray:
    offsets = cp.asnumpy(field.interval_set.row_offsets)
    begin = cp.asnumpy(field.interval_set.begin)
    end = cp.asnumpy(field.interval_set.end)
    cell_offsets = cp.asnumpy(field.interval_cell_offsets)
    values = cp.asnumpy(field.values)
    rows = offsets.size - 1
    dense = np.zeros((rows, width), dtype=values.dtype)
    for row in range(rows):
        start = offsets[row]
        stop = offsets[row + 1]
        for interval in range(start, stop):
            cell_start = cell_offsets[interval]
            cell_stop = cell_offsets[interval + 1]
            dense[row, begin[interval]:end[interval]] = values[cell_start:cell_stop]
    return dense


def _build_coarse_set(width: int, height: int):
    begin: list[int] = []
    end: list[int] = []
    offsets = [0]
    for y in range(height):
        if 5 < y < height // 2:
            begin.append(4)
            end.append(width // 3)
        if height // 3 < y < height - 5:
            begin.append(width // 2)
            end.append(width - 4)
        offsets.append(len(begin))
    return build_interval_set(row_offsets=offsets, begin=begin, end=end)


def _make_circle_set(width: int, height: int, center: Tuple[int, int], radius: int):
    cx, cy = center
    begin: list[int] = []
    end: list[int] = []
    offsets = [0]
    for y in range(height):
        dy = y - cy
        if abs(dy) > radius:
            offsets.append(len(begin))
            continue
        span = int(math.sqrt(max(0, radius * radius - dy * dy)))
        x0 = max(0, cx - span)
        x1 = min(width, cx + span + 1)
        if x0 < x1:
            begin.append(x0)
            end.append(x1)
        offsets.append(len(begin))
    return build_interval_set(row_offsets=offsets, begin=begin, end=end)


def _cell_collection(interval_set, base_dim: int, ratio: int, facecolor: str, offset_x: float = 0.0):
    patches = build_cell_rectangles(interval_set, base_dim, max(1, ratio), offset_x=offset_x)
    return PatchCollection(patches, facecolor=facecolor, edgecolor="k", linewidth=0.3)


def _field_collection_from_dense(interval_set, dense, base_dim: int, ratio: int, cmap, offset_x: float = 0.0):
    if interval_set is None:
        return PatchCollection([])
    row_offsets = cp.asnumpy(interval_set.row_offsets)
    begin = cp.asnumpy(interval_set.begin)
    end = cp.asnumpy(interval_set.end)
    row_count = interval_set.row_count
    target_ratio = max(1, ratio)
    cell_size = 1.0 / target_ratio
    patches = []
    values = []
    for row in range(row_count):
        y = row / target_ratio
        start = row_offsets[row]
        stop = row_offsets[row + 1]
        for idx in range(start, stop):
            for x in range(begin[idx], end[idx]):
                rect = Rectangle((x / target_ratio + offset_x, y), cell_size, cell_size)
                patches.append(rect)
                values.append(dense[row, x])
    collection = PatchCollection(patches, edgecolor="k", linewidth=0.15, cmap=cmap)
    collection.set_array(np.array(values))
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-level interface refinement demo.")
    parser.add_argument("--coarse", type=int, default=64, help="Coarse resolution (square grid).")
    parser.add_argument("--ratio", type=int, default=2, help="Refinement ratio coarse -> fine.")
    parser.add_argument("--plot", action="store_true", help="Display matplotlib figures.")
    parser.add_argument("--cells", action="store_true", help="Show per-cell layout viewer.")
    args = parser.parse_args()

    width = height = args.coarse
    ratio = args.ratio
    fine_width = width * ratio
    fine_height = height * ratio

    # --- Geometry ---
    coarse_set = _build_coarse_set(width, height)
    fine_base_set = prolong_set(coarse_set, ratio)
    circle = _make_circle_set(fine_width, fine_height, (fine_width // 2, fine_height // 2), radius=max(4, width // 4 * ratio))
    fine_union_set = evaluate(make_union(make_input(fine_base_set), make_input(circle)))

    coarse_cover = restrict_set(fine_union_set, ratio)
    coarse_delta = evaluate(make_difference(make_input(coarse_cover), make_input(coarse_set)))

    # --- Fields (aligned geometry only) ---
    coarse_field = create_interval_field(coarse_set, fill_value=0.0, dtype=cp.float32)
    coarse_field.values[:] = cp.linspace(0.0, 1.0, coarse_field.values.size, dtype=cp.float32)
    fine_field = prolong_field(coarse_field, ratio)
    coarse_back = restrict_field(fine_field, ratio, reducer="mean")

    coarse_dense = _field_to_dense(coarse_field, width)
    fine_dense = _field_to_dense(fine_field, fine_width)
    coarse_back_dense = _field_to_dense(coarse_back, width)
    field_delta = coarse_back_dense - coarse_dense

    # --- Stats ---
    print("Coarse active cells:", int(coarse_field.interval_cell_offsets[-1].item()))
    print("Fine active cells (aligned):", int(fine_field.interval_cell_offsets[-1].item()))
    print("Additional coarse coverage cells:", int(coarse_delta.row_offsets[-1]))
    print("Coarse field min/max:", float(coarse_dense.min()), float(coarse_dense.max()))
    print("Fine field min/max:", float(fine_dense.min()), float(fine_dense.max()))

    if args.plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.ravel()
        gap = width * 0.25
        coarse_extent = (0, width, 0, height)
        fine_extent = (width + gap, width + gap + width, 0, height)

        coarse_coll = _cell_collection(coarse_set, width, 1, "#bdbdbd")
        axes[0].add_collection(coarse_coll)
        axes[0].set_xlim(*coarse_extent[:2])
        axes[0].set_ylim(0, height)
        axes[0].set_aspect("equal")
        axes[0].set_title("Coarse mask")
        axes[0].axis("off")

        fine_coll = _cell_collection(fine_union_set, width, ratio, "#ff6961", offset_x=width + gap)
        axes[1].add_collection(fine_coll)
        axes[1].set_xlim(*fine_extent[:2])
        axes[1].set_ylim(0, height)
        axes[1].set_aspect("equal")
        axes[1].set_title("Fine mask")
        axes[1].axis("off")

        coarse_cover_coll = _cell_collection(coarse_cover, width, 1, "#8fd694")
        axes[2].add_collection(coarse_cover_coll)
        axes[2].set_xlim(*coarse_extent[:2])
        axes[2].set_ylim(0, height)
        axes[2].set_aspect("equal")
        axes[2].set_title("Coarse cover")
        axes[2].axis("off")

        coarse_delta_coll = _cell_collection(coarse_delta, width, 1, "#7fa2ff")
        axes[3].add_collection(coarse_delta_coll)
        axes[3].set_xlim(*coarse_extent[:2])
        axes[3].set_ylim(0, height)
        axes[3].set_aspect("equal")
        axes[3].set_title("New coarse coverage")
        axes[3].axis("off")

        coarse_field_coll = _field_collection_from_dense(coarse_set, coarse_dense, width, 1, plt.get_cmap("viridis"))
        axes[4].add_collection(coarse_field_coll)
        axes[4].set_xlim(*coarse_extent[:2])
        axes[4].set_ylim(0, height)
        axes[4].set_aspect("equal")
        axes[4].set_title("Coarse field")
        fig.colorbar(coarse_field_coll, ax=axes[4])

        delta_colormap = plt.get_cmap("coolwarm")
        delta_collection = _field_collection_from_dense(coarse_set, field_delta, width, 1, delta_colormap)
        axes[5].add_collection(delta_collection)
        axes[5].set_xlim(*coarse_extent[:2])
        axes[5].set_ylim(0, height)
        axes[5].set_aspect("equal")
        axes[5].set_title("Field delta")
        fig.colorbar(delta_collection, ax=axes[5])

        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        plt.show()

        if args.cells:
            plot_cell_layout_from_sets(
                [coarse_set, fine_union_set],
                [1, ratio],
                width,
                labels=["Coarse", "Fine"],
            )


if __name__ == "__main__":
    main()
