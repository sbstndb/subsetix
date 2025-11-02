"""
Benchmark loop for the CuPy multi-shape set-operations demo.

Runs the union / intersection / difference sequence repeatedly without plotting
in order to stress-test the GPU kernels.

Usage:
    python -m subsetix_cupy.benchmark_multishape --iterations 100 --no-warmup
"""

from __future__ import annotations

import argparse
import time

import cupy as cp

from . import (
    CuPyWorkspace,
    build_interval_set,
    evaluate,
    make_difference,
    make_input,
    make_intersection,
    make_union,
)
from .demo_multishape import (
    _generate_circles,
    _generate_rectangles,
    _rasterize_circles,
    _rasterize_rectangles,
    _rows_to_interval_set,
)


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
