"""
Benchmark loop for the CuPy multi-shape set-operations demo.

Runs the union / intersection / difference sequence repeatedly without plotting
in order to stress-test the GPU kernels.

Usage:
    python -m subsetix_cupy.benchmark_multishape --iterations 100 --no-warmup
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Iterable, List, Sequence, Tuple

import cupy as cp

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
    parser = argparse.ArgumentParser(
        description="Benchmark repeated multi-shape set operations with CuPy."
    )
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
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warm-up iterations before timing (default: 5).",
    )
    return parser.parse_args()


def _build_interval_sets(
    resolution: int,
    rows: int,
    cols: int,
    rect_strips: int,
    circles_per_cell: int,
    circle_radius_scale: float,
):
    width = height = resolution
    rectangles = _generate_rectangles(width, height, rows, cols, rect_strips)
    circles = _generate_circles(
        width, height, rows, cols, circles_per_cell, circle_radius_scale
    )

    rect_rows = _rasterize_rectangles(rectangles, width, height)
    circle_rows = _rasterize_circles(circles, width, height)

    rect_set = _rows_to_interval_set(rect_rows)
    circle_set = _rows_to_interval_set(circle_rows)
    return rect_set, circle_set, rectangles, circles


def main() -> None:
    args = _parse_args()

    rect_set, circle_set, rectangles, circles = _build_interval_sets(
        args.resolution,
        args.rows,
        args.cols,
        args.rect_strips,
        args.circles_per_cell,
        args.circle_radius_scale,
    )

    workspace = CuPyWorkspace()
    rect_expr = make_input(rect_set)
    circle_expr = make_input(circle_set)

    rect_total = int(rect_set.row_offsets[-1].get())
    circle_total = int(circle_set.row_offsets[-1].get())

    print("Resolution:", args.resolution, "x", args.resolution)
    print("Rectangles count:", len(rectangles))
    print("Circles count:", len(circles))
    print("Rectangles intervals:", rect_total)
    print("Circles intervals:", circle_total)
    print(
        f"Benchmarking {args.iterations} iterations "
        f"(warmup={args.warmup}) without plotting..."
    )

    union_expr = make_union(rect_expr, circle_expr)
    intersection_expr = make_intersection(rect_expr, circle_expr)
    difference_expr = make_difference(rect_expr, circle_expr)

    union_total = inter_total = diff_total = 0

    # Warm-up
    for _ in range(args.warmup):
        union_total = int(evaluate(union_expr, workspace).row_offsets[-1].get())
        inter_total = int(evaluate(intersection_expr, workspace).row_offsets[-1].get())
        diff_total = int(evaluate(difference_expr, workspace).row_offsets[-1].get())

    cp.cuda.runtime.deviceSynchronize()

    start = time.perf_counter()
    for _ in range(args.iterations):
        union_total = int(evaluate(union_expr, workspace).row_offsets[-1].get())
        inter_total = int(evaluate(intersection_expr, workspace).row_offsets[-1].get())
        diff_total = int(evaluate(difference_expr, workspace).row_offsets[-1].get())
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - start

    iter_ms = (elapsed / args.iterations) * 1e3
    total_ops = union_total + inter_total + diff_total

    print("Union intervals:", union_total)
    print("Intersection intervals:", inter_total)
    print("Difference intervals:", diff_total)
    print(f"Elapsed: {elapsed:.3f} s for {args.iterations} iterations")
    print(f"Per iteration: {iter_ms:.3f} ms")
    if total_ops > 0:
        print(f"Per produced interval: {iter_ms * 1e6 / total_ops:.2f} ns")


if __name__ == "__main__":
    main()
