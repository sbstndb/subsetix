"""
Micro-benchmark for enforce_two_level_grading_set and related morphology ops.

Example:
    python -m subsetix_amr2.bench_grading --size 3200 --iterations 5 --mode von_neumann
"""

from __future__ import annotations

import argparse
import time
from statistics import mean, stdev

from subsetix_amr2.regrid import (
    _require_cupy,
    enforce_two_level_grading_set,
    gradient_tag_threshold_set,
)
from subsetix_cupy import evaluate, make_input, make_union
from subsetix_cupy.morphology import dilate_interval_set


def _format_stats(values):
    if not values:
        return "   n/a"
    if len(values) == 1:
        return f"{values[0] * 1e3:8.3f} ms"
    return f"{mean(values) * 1e3:8.3f} ms ± {stdev(values) * 1e3:6.3f} ms"


def _timed_call(fn):
    cp = _require_cupy()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    elapsed = cp.cuda.get_elapsed_time(start, end) / 1e3
    return result, elapsed


def run_benchmark(
    size: int,
    iterations: int,
    threshold: float,
    padding: int,
    mode: str,
    seed: int | None,
):
    cp = _require_cupy()
    if seed is not None:
        cp.random.seed(seed)

    gradients = cp.random.random((size, size), dtype=cp.float32)
    refine_set = gradient_tag_threshold_set(gradients, threshold)
    cp.cuda.runtime.deviceSynchronize()

    enforce_times = []
    horiz_times = []
    vert_times = []
    merge_times = []
    union_times = []
    dilate_times = []

    enforce_two_level_grading_set(
        refine_set, padding=padding, mode=mode, width=size, height=size
    )
    cp.cuda.runtime.deviceSynchronize()

    for _ in range(iterations):
        t0 = time.perf_counter()
        enforce_two_level_grading_set(
            refine_set, padding=padding, mode=mode, width=size, height=size
        )
        cp.cuda.runtime.deviceSynchronize()
        enforce_times.append(time.perf_counter() - t0)

        if mode == "moore":
            expanded, dilate_time = _timed_call(
                lambda: dilate_interval_set(
                    refine_set,
                    halo_x=padding,
                    halo_y=padding,
                    width=size,
                    height=size,
                    bc="clamp",
                )
            )
            dilate_times.append(dilate_time)
            _, union_time = _timed_call(
                lambda: evaluate(
                    make_union(make_input(refine_set), make_input(expanded))
                )
            )
            union_times.append(union_time)
        else:
            horiz, horiz_time = _timed_call(
                lambda: dilate_interval_set(
                    refine_set,
                    halo_x=padding,
                    halo_y=0,
                    width=size,
                    height=size,
                    bc="clamp",
                )
            )
            horiz_times.append(horiz_time)
            vert, vert_time = _timed_call(
                lambda: dilate_interval_set(
                    refine_set,
                    halo_x=0,
                    halo_y=padding,
                    width=size,
                    height=size,
                    bc="clamp",
                )
            )
            vert_times.append(vert_time)
            expanded, merge_time = _timed_call(
                lambda: evaluate(
                    make_union(make_input(horiz), make_input(vert))
                )
            )
            merge_times.append(merge_time)
            _, union_time = _timed_call(
                lambda: evaluate(
                    make_union(make_input(refine_set), make_input(expanded))
                )
            )
            union_times.append(union_time)

    print(f"Grid: {size} x {size}, iterations: {iterations}, mode: {mode}")
    print(f"enforce_two_level_grading : {_format_stats(enforce_times)}")
    if mode == "moore":
        print(f"dilate (moore)           : {_format_stats(dilate_times)}")
    else:
        print(f"dilate horizontal        : {_format_stats(horiz_times)}")
        print(f"dilate vertical          : {_format_stats(vert_times)}")
        print(f"merge expanded union     : {_format_stats(merge_times)}")
    print(f"union with refine        : {_format_stats(union_times)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark grading dilation steps.")
    parser.add_argument("--size", type=int, default=3200, help="Grid resolution.")
    parser.add_argument("--iterations", type=int, default=5, help="Benchmark iterations.")
    parser.add_argument("--threshold", type=float, default=0.05, help="Gradient threshold.")
    parser.add_argument("--padding", type=int, default=1, help="Grading halo size.")
    parser.add_argument(
        "--mode",
        choices=["von_neumann", "moore"],
        default="von_neumann",
        help="Grading mode.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(
        size=args.size,
        iterations=args.iterations,
        threshold=args.threshold,
        padding=args.padding,
        mode=args.mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
