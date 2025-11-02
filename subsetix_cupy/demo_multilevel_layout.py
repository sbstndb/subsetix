"""
Multi-level coverage layout demo.

Shows how to describe a square AMR domain with multiple refinement levels using
interval sets. Each level refines a subset of the previous one (nested squares).
The script reports coverage stats and can render either a coarse heatmap or a
cell-level view at the finest resolution using rectangular primitives.

Usage:
    python -m subsetix_cupy.demo_multilevel_layout --coarse 64 --levels 3 --plot --mode cells
"""

from __future__ import annotations

import argparse
from typing import Tuple

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons

from . import MultiLevel2D, build_interval_set, prolong_set


def _square_patch(width: int, height: int, x0: int, y0: int, size: int):
    """Create an IntervalSet covering a square [x0, x0+size) × [y0, y0+size)."""
    begin: list[int] = []
    end: list[int] = []
    offsets = [0]
    x1 = min(width, x0 + size)
    y1 = min(height, y0 + size)
    for y in range(height):
        if y0 <= y < y1:
            begin.append(x0)
            end.append(x1)
        offsets.append(len(begin))
    return build_interval_set(row_offsets=offsets, begin=begin, end=end)


def _intervals_to_mask(interval_set, width: int) -> np.ndarray:
    offsets = cp.asnumpy(interval_set.row_offsets)
    begin = cp.asnumpy(interval_set.begin)
    end = cp.asnumpy(interval_set.end)
    rows = offsets.size - 1
    mask = np.zeros((rows, width), dtype=np.uint8)
    for row in range(rows):
        start = offsets[row]
        stop = offsets[row + 1]
        for idx in range(start, stop):
            mask[row, begin[idx]:end[idx]] = 1
    return mask


def build_layout_levels(width: int, height: int, n_levels: int) -> MultiLevel2D:
    if n_levels < 1:
        raise ValueError("Must request at least one level")

    ml = MultiLevel2D.create(n_levels, base_ratio=2)

    # Level 0: full domain coverage
    lvl0 = _square_patch(width, height, 0, 0, width)
    ml.set_level(0, lvl0)

    # Level 1..n: nested squares towards the centre
    size = width
    x0 = 0
    y0 = 0
    for level in range(1, n_levels):
        size //= 2
        x0 += size // 2
        y0 += size // 2
        lvl = _square_patch(width, height, x0, y0, max(4, size))
        ml.set_level(level, lvl)

    return ml


def summarise_levels(ml: MultiLevel2D) -> None:
    for level, interval_set in enumerate(ml.levels):
        if interval_set is None:
            print(f"Level {level}: empty")
            continue
        cell_count = int(cp.sum(interval_set.end - interval_set.begin).item())
        row_count = interval_set.row_count
        print(f"Level {level}: {cell_count} active cells across {row_count} rows")


def make_level_map(ml: MultiLevel2D, width: int) -> np.ndarray:
    """Highest level per coarse cell."""
    height = ml.levels[0].row_count if ml.levels[0] is not None else 0
    level_map = np.zeros((height, width), dtype=np.int32)
    for level, interval_set in enumerate(ml.levels):
        if interval_set is None:
            continue
        mask = _intervals_to_mask(interval_set, width)
        level_map[mask == 1] = level
    return level_map


def build_cell_rectangles(interval_set, base_dim: int, target_ratio: int, offset_x: float = 0.0):
    if interval_set is None:
        return []
    row_count = interval_set.row_count
    if row_count == 0:
        return []
    actual_ratio = max(1, row_count // base_dim)
    if target_ratio % actual_ratio != 0:
        raise ValueError("Ratios must divide the target ratio")
    scale = target_ratio // actual_ratio
    refined = prolong_set(interval_set, scale) if scale > 1 else interval_set
    begin = cp.asnumpy(refined.begin)
    end = cp.asnumpy(refined.end)
    offsets = cp.asnumpy(refined.row_offsets)
    cell_size = 1.0 / target_ratio
    patches = []
    for row in range(refined.row_count):
        y = row / target_ratio
        start = offsets[row]
        stop = offsets[row + 1]
        for idx in range(start, stop):
            for x in range(begin[idx], end[idx]):
                patches.append(Rectangle((x / target_ratio + offset_x, y), cell_size, cell_size))
    return patches


def plot_layout_heatmap(ml: MultiLevel2D, width: int) -> None:
    level_map = make_level_map(ml, width)
    fig, ax = plt.subplots(figsize=(6, 6))
    base_colors = ["#bdbdbd", "#ffb347", "#ff6961", "#8fd694", "#7fa2ff", "#c18aff"]
    cmap = ListedColormap(base_colors[: len(ml.levels)])
    extent = (-0.5, width - 0.5, -0.5, width - 0.5)
    im = ax.imshow(
        level_map,
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=len(ml.levels) - 1,
        interpolation="nearest",
        extent=extent,
    )
    ax.set_title("Refinement levels (coarse view)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    step = max(1, width // 8)
    ax.set_xticks(np.arange(0, width, step))
    ax.set_yticks(np.arange(0, width, step))
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, width, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.4)
    ax.tick_params(which="minor", length=0)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, width - 0.5)

    cbar = fig.colorbar(im, ax=ax, ticks=range(len(ml.levels)))
    cbar.ax.set_yticklabels([f"Level {lvl}" for lvl in range(len(ml.levels))])
    plt.tight_layout()
    plt.show()


def plot_cell_layout_from_sets(level_sets, ratios, base_width, labels=None) -> None:
    base_colors = ["#bdbdbd", "#ffb347", "#ff6961", "#8fd694", "#7fa2ff", "#c18aff"]
    if labels is None:
        labels = [f"Level {idx}" for idx in range(len(level_sets))]

    finest_ratio = max(max(1, r) for r in ratios)

    fig, ax = plt.subplots(figsize=(7, 7))
    collections = []
    for level, (interval_set, ratio) in enumerate(zip(level_sets, ratios)):
        patches = build_cell_rectangles(interval_set, base_width, finest_ratio, offset_x=0.0)
        collection = PatchCollection(
            patches,
            facecolor=base_colors[level % len(base_colors)],
            edgecolor="k",
            linewidth=0.3,
        )
        collection.set_visible(True)
        collections.append(collection)
        ax.add_collection(collection)

    ax.set_xlim(0, base_width)
    ax.set_ylim(0, base_width)
    ax.set_aspect("equal")
    ax.set_title("Refinement levels (per-cell view)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(np.arange(0, base_width + 1, max(1, base_width // 8)))
    ax.set_yticks(np.arange(0, base_width + 1, max(1, base_width // 8)))

    fig.subplots_adjust(bottom=0.12)
    w = 0.08
    h = 0.03
    checkbox_axes = []
    for idx, label in enumerate(labels):
        x = 0.05 + idx * (w + 0.02)
        y = 0.04
        cax = plt.axes([x, y, w, h], facecolor="lightgoldenrodyellow")
        checkbox_axes.append(cax)
    checkboxes = [
        CheckButtons(cax, [labels[idx]], [collections[idx].get_visible()])
        for idx, cax in enumerate(checkbox_axes)
    ]

    def make_callback(index: int):
        def _cb(_label: str) -> None:
            coll = collections[index]
            coll.set_visible(not coll.get_visible())
            fig.canvas.draw_idle()

        return _cb

    for idx, cb in enumerate(checkboxes):
        cb.on_clicked(make_callback(idx))

    plt.show()


def plot_layout_cells(ml: MultiLevel2D, width: int) -> None:
    plot_cell_layout_from_sets(list(ml.levels), list(ml.ratios), width)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-level layout demo")
    parser.add_argument("--coarse", type=int, default=64, help="Square domain resolution")
    parser.add_argument("--levels", type=int, default=3, help="Number of refinement levels")
    parser.add_argument("--plot", action="store_true", help="Visualise the layout")
    parser.add_argument(
        "--mode",
        choices=("heatmap", "cells"),
        default="heatmap",
        help="heatmap: coarse cell view; cells: finest resolution rectangles",
    )
    args = parser.parse_args()

    width = height = args.coarse
    ml = build_layout_levels(width, height, args.levels)
    summarise_levels(ml)

    total_cells = width * height
    coverage = []
    for level, interval_set in enumerate(ml.levels):
        if interval_set is None:
            coverage.append(0.0)
            continue
        cells = int(cp.sum(interval_set.end - interval_set.begin).item())
        coverage.append(100.0 * cells / total_cells)

    print(
        "Coverage by level (% of domain):",
        ", ".join(f"L{idx}: {cov:.1f}%" for idx, cov in enumerate(coverage)),
    )

    if args.plot:
        if args.mode == "heatmap":
            plot_layout_heatmap(ml, width)
        else:
            plot_layout_cells(ml, width)


if __name__ == "__main__":
    main()
