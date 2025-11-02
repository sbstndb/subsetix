"""
Demonstrate ghost-zone generation around interval sets.

Usage
-----
    python -m subsetix_cupy.demo_ghosts --resolution 256 --halo-x 3 --halo-y 2 --bc clamp --plot
"""

from __future__ import annotations

import argparse
import math
from typing import List, Sequence, Tuple

import cupy as cp
import matplotlib.pyplot as plt

from . import (
    CuPyWorkspace,
    build_interval_set,
    evaluate,
    make_input,
    make_union,
)
from .morphology import dilate_interval_set, ghost_zones
from .plot_utils import make_cell_collection, setup_cell_axes

Rectangle = Tuple[int, int, int, int]
Circle = Tuple[int, int, int]


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if s < e]


def _accumulate_rows(height: int) -> List[List[Tuple[int, int]]]:
    return [[] for _ in range(height)]


def _rasterize_rectangles(
    rectangles: Sequence[Rectangle], width: int, height: int
) -> List[List[Tuple[int, int]]]:
    rows = _accumulate_rows(height)
    for x0, x1, y0, y1 in rectangles:
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x0 >= x1 or y0 >= y1:
            continue
        for y in range(y0, y1):
            rows[y].append((x0, x1))
    return [_merge_intervals(row) for row in rows]


def _rasterize_circles(
    circles: Sequence[Circle], width: int, height: int
) -> List[List[Tuple[int, int]]]:
    rows = _accumulate_rows(height)
    for cx, cy, radius in circles:
        if radius <= 0:
            continue
        y_min = max(0, cy - radius)
        y_max = min(height - 1, cy + radius)
        for y in range(y_min, y_max + 1):
            dy = y - cy
            span = int(math.floor(math.sqrt(max(0, radius * radius - dy * dy))))
            x0 = max(0, cx - span)
            x1 = min(width, cx + span + 1)
            if x0 < x1:
                rows[y].append((x0, x1))
    return [_merge_intervals(row) for row in rows]


def _rows_to_interval_set(rows: Sequence[Sequence[Tuple[int, int]]]) -> "object":
    begin: List[int] = []
    end: List[int] = []
    row_offsets = [0]
    for intervals in rows:
        for x0, x1 in intervals:
            begin.append(x0)
            end.append(x1)
        row_offsets.append(len(begin))
    return build_interval_set(row_offsets=row_offsets, begin=begin, end=end)


def _generate_rectangles(width: int, height: int, rows: int, cols: int, strips: int) -> List[Rectangle]:
    rectangles: List[Rectangle] = []
    cell_w = width / cols
    cell_h = height / rows
    for gy in range(rows):
        y0 = int(gy * cell_h)
        y1 = int(min(height, (gy + 1) * cell_h))
        if y1 <= y0:
            continue
        for gx in range(cols):
            base_x = gx * cell_w
            strip_step = cell_w / strips
            strip_width = max(1, int(strip_step * 0.4))
            for s in range(strips):
                x0 = int(base_x + s * strip_step)
                x1 = int(min(width, x0 + strip_width))
                if x1 > x0:
                    rectangles.append((x0, x1, y0, y1))
    return rectangles


def _generate_circles(width: int, height: int, rows: int, cols: int, count: int, scale: float) -> List[Circle]:
    circles: List[Circle] = []
    cell_w = width / cols
    cell_h = height / rows
    per_side = max(1, int(math.ceil(math.sqrt(count))))
    for gy in range(rows):
        for gx in range(cols):
            base_x = gx * cell_w
            base_y = gy * cell_h
            for idx in range(count):
                sub_x = idx % per_side
                sub_y = idx // per_side
                cx = int(base_x + (sub_x + 0.5) * cell_w / per_side)
                cy = int(base_y + (sub_y + 0.5) * cell_h / per_side)
                radius = int(scale * min(cell_w / per_side, cell_h / per_side))
                if radius > 0:
                    circles.append((cx, cy, radius))
    return circles


def _build_multishape_union(
    width: int,
    height: int,
    grid_rows: int,
    grid_cols: int,
) -> Tuple["object", "object", "object"]:
    rectangles = _generate_rectangles(width, height, grid_rows, grid_cols, strips=2)
    circles = _generate_circles(width, height, grid_rows, grid_cols, count=2, scale=0.45)
    rect_rows = _rasterize_rectangles(rectangles, width, height)
    circle_rows = _rasterize_circles(circles, width, height)
    rect_set = _rows_to_interval_set(rect_rows)
    circle_set = _rows_to_interval_set(circle_rows)
    workspace = CuPyWorkspace()
    base_expr = make_union(make_input(rect_set), make_input(circle_set))
    base_set = evaluate(base_expr, workspace)
    return base_set, rect_set, circle_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Ghost-zone demo on interval grids.")
    parser.add_argument("--resolution", type=int, default=256, help="Square grid resolution.")
    parser.add_argument("--rows", type=int, default=4, help="Procedural rectangle rows.")
    parser.add_argument("--cols", type=int, default=4, help="Procedural rectangle columns.")
    parser.add_argument("--halo-x", type=int, default=3, help="Ghost radius along x.")
    parser.add_argument("--halo-y", type=int, default=2, help="Ghost radius along y.")
    parser.add_argument("--bc", choices=("clamp", "wrap"), default="clamp", help="Boundary condition for halo generation.")
    parser.add_argument("--plot", action="store_true", help="Display matplotlib figures.")
    args = parser.parse_args()

    width = height = args.resolution

    base_set, rect_set, circle_set = _build_multishape_union(width, height, args.rows, args.cols)

    dilated = dilate_interval_set(
        base_set,
        halo_x=args.halo_x,
        halo_y=args.halo_y,
        width=width,
        height=height,
        bc=args.bc,
    )
    ghosts = ghost_zones(
        base_set,
        halo_x=args.halo_x,
        halo_y=args.halo_y,
        width=width,
        height=height,
        bc=args.bc,
    )

    cp_mod = cp
    base_cells = int(cp_mod.sum(base_set.end - base_set.begin).item())
    dilated_cells = int(cp_mod.sum(dilated.end - dilated.begin).item())
    ghost_cells = int(cp_mod.sum(ghosts.end - ghosts.begin).item())

    print(f"Boundary condition: {args.bc}")
    print(f"Base cells: {base_cells}")
    print(f"Dilated cells: {dilated_cells} (+{dilated_cells - base_cells})")
    print(f"Ghost cells: {ghost_cells} ({100.0 * ghost_cells / max(1, base_cells):.2f}% of base)")

    if args.plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        panels = [
            (base_set, "#8fd694", "Base union"),
            (dilated, "#ffb347", f"Dilated (halo={args.halo_x}×{args.halo_y})"),
            (ghosts, "#7fa2ff", "Ghost zones"),
        ]
        for ax, (iset, color, title) in zip(axes, panels):
            coll = make_cell_collection(iset, height, 1, facecolor=color, edgecolor="k", linewidth=0.2)
            ax.add_collection(coll)
            setup_cell_axes(ax, width, height, title=title)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
