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
from matplotlib.colors import ListedColormap

from . import MultiLevel2D, build_interval_set
from .plot_utils import intervals_to_mask, plot_cell_layout_from_sets


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
        mask = intervals_to_mask(interval_set, width)
        level_map[mask == 1] = level
    return level_map


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
