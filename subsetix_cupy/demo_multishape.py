"""
CuPy multi-shape demo for unions, intersections, and differences.

The script procedurally generates a grid of rectangles and circles, runs the
GPU set-operations through `subsetix_cupy`, and visualises the results.

Usage:
    python -m subsetix_cupy.demo_multishape [options]

Example (higher resolution):
    python -m subsetix_cupy.demo_multishape --resolution 1024 --rows 32 --cols 32
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable, List, Sequence, Tuple

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from . import (
    CuPyWorkspace,
    IntervalSet,
    build_interval_set,
    evaluate,
    make_difference,
    make_input,
    make_intersection,
    make_union,
)
from .plot_utils import (
    intervals_to_mask as _mask_from_intervals,
    make_cell_collection,
    setup_cell_axes,
)

Rectangle = Tuple[int, int, int, int]  # (x0, x1, y0, y1)
Circle = Tuple[int, int, int]  # (cx, cy, radius)


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


def _rows_to_interval_set(rows: Sequence[Sequence[Tuple[int, int]]]) -> IntervalSet:
    begin: List[int] = []
    end: List[int] = []
    row_offsets = [0]
    for intervals in rows:
        for x0, x1 in intervals:
            begin.append(x0)
            end.append(x1)
        row_offsets.append(len(begin))
    return build_interval_set(row_offsets=row_offsets, begin=begin, end=end)


def _intervals_to_mask(interval_set: IntervalSet, width: int) -> Tuple[np.ndarray, np.ndarray]:
    mask = _mask_from_intervals(interval_set, width)
    offsets = cp.asnumpy(interval_set.row_offsets)
    return mask, offsets


def _plot_cell_sets(
    sets: Sequence[Tuple[IntervalSet, str, str]],
    width: int,
    height: int,
) -> None:
    cols = len(sets)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]
    for ax, (interval_set, title, color) in zip(axes, sets):
        collection = make_cell_collection(
            interval_set,
            height,
            1,
            facecolor=color,
            edgecolor="k",
            linewidth=0.2,
        )
        ax.add_collection(collection)
        setup_cell_axes(ax, width, height, title=title)
    fig.tight_layout()
    plt.show()


def _generate_rectangles(
    width: int,
    height: int,
    grid_rows: int,
    grid_cols: int,
    strips_per_cell: int,
) -> List[Rectangle]:
    rectangles: List[Rectangle] = []
    cell_w = width / grid_cols
    cell_h = height / grid_rows
    for gy in range(grid_rows):
        y0 = int(gy * cell_h)
        y1 = int(min(height, (gy + 1) * cell_h))
        if y1 <= y0:
            continue
        for gx in range(grid_cols):
            base_x = gx * cell_w
            strip_step = cell_w / strips_per_cell
            strip_width = max(1, int(strip_step * 0.4))
            for s in range(strips_per_cell):
                x0 = int(base_x + s * strip_step)
                x1 = int(min(width, x0 + strip_width))
                if x1 > x0:
                    rectangles.append((x0, x1, y0, y1))
    return rectangles


def _generate_circles(
    width: int,
    height: int,
    grid_rows: int,
    grid_cols: int,
    circles_per_cell: int,
    radius_scale: float,
) -> List[Circle]:
    circles: List[Circle] = []
    cell_w = width / grid_cols
    cell_h = height / grid_rows
    per_side = max(1, int(math.ceil(math.sqrt(circles_per_cell))))
    for gy in range(grid_rows):
        for gx in range(grid_cols):
            base_x = gx * cell_w
            base_y = gy * cell_h
            for idx in range(circles_per_cell):
                sub_x = idx % per_side
                sub_y = idx // per_side
                cx = int(base_x + (sub_x + 0.5) * cell_w / per_side)
                cy = int(base_y + (sub_y + 0.5) * cell_h / per_side)
                radius = int(
                    radius_scale * min(cell_w / per_side, cell_h / per_side)
                )
                if radius > 0:
                    circles.append((cx, cy, radius))
    return circles


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CuPy multi-shape set-ops demo.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Square raster resolution (default: 512).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=9,
        help="Grid rows used to place shapes (default: 9).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="Grid columns used to place shapes (default: 9).",
    )
    parser.add_argument(
        "--rect-strips",
        type=int,
        default=1,
        help="Number of rectangle strips per grid cell (default: 1).",
    )
    parser.add_argument(
        "--circles-per-cell",
        type=int,
        default=1,
        help="Number of circles per grid cell (default: 1).",
    )
    parser.add_argument(
        "--circle-radius-scale",
        type=float,
        default=0.35,
        help="Radius scale relative to cell size (default: 0.35).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable matplotlib visualisation (useful for headless runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    width = height = args.resolution

    rectangles = _generate_rectangles(
        width, height, args.rows, args.cols, args.rect_strips
    )
    circles = _generate_circles(
        width,
        height,
        args.rows,
        args.cols,
        args.circles_per_cell,
        args.circle_radius_scale,
    )

    rect_rows = _rasterize_rectangles(rectangles, width, height)
    circle_rows = _rasterize_circles(circles, width, height)

    rect_set = _rows_to_interval_set(rect_rows)
    circle_set = _rows_to_interval_set(circle_rows)

    workspace = CuPyWorkspace()
    rect_expr = make_input(rect_set)
    circle_expr = make_input(circle_set)

    rect_mask, rect_offsets = _intervals_to_mask(rect_set, width)
    circle_mask, circle_offsets = _intervals_to_mask(circle_set, width)

    union_set = evaluate(make_union(rect_expr, circle_expr), workspace)
    union_mask, union_offsets = _intervals_to_mask(union_set, width)
    intersection_set = evaluate(make_intersection(rect_expr, circle_expr), workspace)
    intersection_mask, intersection_offsets = _intervals_to_mask(intersection_set, width)
    difference_set = evaluate(make_difference(rect_expr, circle_expr), workspace)
    difference_mask, difference_offsets = _intervals_to_mask(difference_set, width)

    print("Resolution:", width, "x", height)
    print("Rectangles count:", len(rectangles))
    print("Circles count:", len(circles))
    print("Rectangles intervals:", int(rect_offsets[-1]))
    print("Circles intervals:", int(circle_offsets[-1]))
    print("Union intervals:", int(union_offsets[-1]))
    print("Intersection intervals:", int(intersection_offsets[-1]))
    print("Difference intervals:", int(difference_offsets[-1]))

    if not args.no_plot:
        _plot_cell_sets(
            [
                (rect_set, "Rectangles", "#8fd694"),
                (circle_set, "Circles", "#ffb347"),
                (union_set, "Union", "#7fa2ff"),
                (intersection_set, "Intersection", "#ff6961"),
                (difference_set, "Rectangles \\ Circles", "#c18aff"),
            ],
            width,
            height,
        )


if __name__ == "__main__":
    main()
