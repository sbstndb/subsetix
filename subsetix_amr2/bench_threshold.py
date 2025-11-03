"""
Micro-benchmark for the gradient tagging path in subsetix_amr2.regrid.

This script isolates the cost of `_intervals_above_threshold` by measuring:
  * the combined duration of the count + scan + host `.item()` synchronisation,
  * the GPU-only portion of the kernels (with per-kernel timing),
  * the standalone cost of the host `.item()` once the stream is idle,
  * the write-back of the interval bounds.

Running the script requires a working CuPy installation with CUDA support.

Example:
    python -m subsetix_amr2.bench_threshold --size 3200 --iterations 5
"""

from __future__ import annotations

import argparse
import time
from statistics import mean, stdev

from subsetix_amr2.regrid import (
    _COUNT_INTERVALS,
    _WRITE_INTERVALS,
    _require_cupy,
    gradient_tag_threshold_set,
)


def _format_stats(values):
    if len(values) == 1:
        return f"{values[0] * 1e3:8.3f} ms"
    return f"{mean(values) * 1e3:8.3f} ms ± {stdev(values) * 1e3:6.3f} ms"


def _time_gpu(call):
    cp = _require_cupy()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    call()
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / 1e3


def measure_components(data, threshold: float):
    cp = _require_cupy()
    rows, width = data.shape
    rows32 = cp.int32(rows)
    width32 = cp.int32(width)
    thresh32 = cp.float32(threshold)

    # Combined pass (mirrors _intervals_above_threshold).
    counts = cp.zeros(rows, dtype=cp.int32)
    row_offsets = cp.empty(rows + 1, dtype=cp.int32)
    row_offsets[0] = 0
    t0 = time.perf_counter()
    _COUNT_INTERVALS((rows,), (1,), (data, rows32, width32, thresh32, counts))
    cp.cumsum(counts, dtype=cp.int32, out=row_offsets[1:])
    total_combined = int(row_offsets[-1].item())
    combined_time = time.perf_counter() - t0

    # GPU-only timings via CUDA events.
    counts2 = cp.zeros(rows, dtype=cp.int32)
    row_offsets2 = cp.empty(rows + 1, dtype=cp.int32)
    row_offsets2[0] = 0
    count_kernel_time = _time_gpu(
        lambda: _COUNT_INTERVALS((rows,), (1,), (data, rows32, width32, thresh32, counts2))
    )
    scan_time = _time_gpu(lambda: cp.cumsum(counts2, dtype=cp.int32, out=row_offsets2[1:]))
    cp.cuda.runtime.deviceSynchronize()
    ops_time = count_kernel_time + scan_time

    t_item = time.perf_counter()
    total_synced = int(row_offsets2[-1].item())
    item_time = time.perf_counter() - t_item

    begin = cp.empty(int(total_synced), dtype=cp.int32)
    end = cp.empty_like(begin)
    if int(total_synced) > 0:
        write_time = _time_gpu(
            lambda: _WRITE_INTERVALS((rows,), (1,), (data, rows32, width32, thresh32, row_offsets2, begin, end))
        )
        cp.cuda.runtime.deviceSynchronize()
    else:
        write_time = 0.0

    return {
        "combined": combined_time,
        "ops": ops_time,
        "count_kernel": count_kernel_time,
        "scan": scan_time,
        "item": item_time,
        "write": write_time,
        "total_estimate": total_combined,
    }


def run_benchmark(size: int, iterations: int, threshold: float, seed: int | None):
    cp = _require_cupy()
    if seed is not None:
        cp.random.seed(seed)

    data = cp.random.random((size, size), dtype=cp.float32)
    # Warm-up to avoid measuring initial JIT/kernel setup.
    gradient_tag_threshold_set(data, threshold)
    cp.cuda.runtime.deviceSynchronize()

    combined_times = []
    ops_times = []
    item_times = []
    write_times = []
    tag_times = []
    count_kernel_times = []
    scan_times = []

    for _ in range(iterations):
        t = time.perf_counter()
        gradient_tag_threshold_set(data, threshold)
        cp.cuda.runtime.deviceSynchronize()
        tag_times.append(time.perf_counter() - t)

        metrics = measure_components(data, threshold)
        combined_times.append(metrics["combined"])
        ops_times.append(metrics["ops"])
        item_times.append(metrics["item"])
        write_times.append(metrics["write"])
        count_kernel_times.append(metrics["count_kernel"])
        scan_times.append(metrics["scan"])

    print(f"Grid: {size} x {size}, iterations: {iterations}, threshold: {threshold}")
    print(f"tag_threshold_set      : {_format_stats(tag_times)}")
    print(f"count+scan+item (as-is): {_format_stats(combined_times)}")
    print(f"count kernel           : {_format_stats(count_kernel_times)}")
    print(f"scan (cumsum)          : {_format_stats(scan_times)}")
    print(f"count+scan (gpu only)  : {_format_stats(ops_times)}")
    print(f"item() after sync      : {_format_stats(item_times)}")
    print(f"write intervals        : {_format_stats(write_times)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark gradient tagging internals.")
    parser.add_argument("--size", type=int, default=3200, help="Grid dimension (size x size).")
    parser.add_argument("--iterations", type=int, default=5, help="Benchmark iterations.")
    parser.add_argument("--threshold", type=float, default=0.05, help="Gradient threshold.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(args.size, args.iterations, args.threshold, args.seed)


if __name__ == "__main__":
    main()
