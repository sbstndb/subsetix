"""
Micro-benchmark for synchronize_two_level.

Example:
    python -m subsetix_amr2.bench_sync --size 3200 --ratio 2 --iterations 20
"""

from __future__ import annotations

import argparse
from statistics import mean, stdev

import cupy as cp

from subsetix_amr2.fields import ActionField, synchronize_two_level
from subsetix_cupy.expressions import IntervalSet


def _central_refine(cp_mod, height: int, width: int, margin: float = 0.25) -> IntervalSet:
    y0 = int(height * margin)
    y1 = int(height * (1.0 - margin))
    x0 = int(width * margin)
    x1 = int(width * (1.0 - margin))
    begins = []
    ends = []
    row_offsets = [0]
    for row in range(height):
        if y0 <= row < y1:
            begins.append(x0)
            ends.append(x1)
        row_offsets.append(len(begins))
    return IntervalSet(
        begin=cp_mod.asarray(begins, dtype=cp_mod.int32),
        end=cp_mod.asarray(ends, dtype=cp_mod.int32),
        row_offsets=cp_mod.asarray(row_offsets, dtype=cp_mod.int32),
    )


def _format(values):
    if len(values) == 1:
        return f"{values[0] * 1e3:8.3f} ms"
    return f"{mean(values) * 1e3:8.3f} ms ± {stdev(values) * 1e3:6.3f} ms"


def run_benchmark(size: int, ratio: int, iterations: int, seed: int | None, copy: bool):
    if seed is not None:
        cp.random.seed(seed)

    coarse = cp.random.random((size, size), dtype=cp.float32)
    fine = cp.random.random((size * ratio, size * ratio), dtype=cp.float32)
    refine = _central_refine(cp, size, size)

    actions = ActionField.full_grid(size, size, ratio)
    actions.set_from_interval_set(refine)

    def call():
        return synchronize_two_level(
            coarse,
            fine,
            actions,
            ratio=ratio,
            reducer="mean",
            fill_fine_outside=True,
            copy=copy,
        )

    for _ in range(3):
        call()
    cp.cuda.runtime.deviceSynchronize()

    times = []
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    for _ in range(iterations):
        start.record()
        call()
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end) / 1e3)

    print(f"Grid {size}x{size}, ratio={ratio}, iterations={iterations}")
    mode = "copy" if copy else "in-place"
    print(f"synchronize_two_level ({mode}) : {_format(times)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark synchronize_two_level.")
    parser.add_argument("--size", type=int, default=1024, help="Coarse grid size.")
    parser.add_argument("--ratio", type=int, default=2, help="Refinement ratio.")
    parser.add_argument("--iterations", type=int, default=20, help="Iterations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Apply updates in-place (copy=False).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(args.size, args.ratio, args.iterations, args.seed, copy=not args.in_place)


if __name__ == "__main__":
    main()
