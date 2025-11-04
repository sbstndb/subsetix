"""
Micro-benchmark for the upwind advection step with zero exterior values.

Compares the existing slice-based implementation with a RawKernel that
computes upwind differences without allocating neighbour arrays. This is
intended to validate the potential gains before integrating the kernel into
the main simulation code.

Example:
    python -m subsetix_amr2.bench_upwind --size 3200 --ratio 2 --iterations 20
"""

from __future__ import annotations

import argparse
from statistics import mean, stdev

import cupy as cp

from subsetix_cupy.interval_field import create_interval_field
from subsetix_cupy.interval_stencil import step_upwind_interval
from subsetix_cupy.morphology import full_interval_set


def _format(values):
    if len(values) == 1:
        return f"{values[0] * 1e3:8.3f} ms"
    return f"{mean(values) * 1e3:8.3f} ms ± {stdev(values) * 1e3:6.3f} ms"


_UPWIND_KERNEL_SRC = r"""
extern "C" __global__
void upwind_zero(const float* __restrict__ u,
                  float* __restrict__ out,
                  int width,
                  int height,
                  float a,
                  float b,
                  float dt,
                  float dx,
                  float dy)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;
    float center = u[idx];

    int x_left = x == 0 ? 0 : x - 1;
    int x_right = x == width - 1 ? width - 1 : x + 1;
    int y_down = y == 0 ? 0 : y - 1;
    int y_up = y == height - 1 ? height - 1 : y + 1;

    float left = u[y * width + x_left];
    float right = u[y * width + x_right];
    float down = u[y_down * width + x];
    float up = u[y_up * width + x];

    float du_dx = (a >= 0.0f)
        ? (center - left) / dx
        : (right - center) / dx;
    float du_dy = (b >= 0.0f)
        ? (center - down) / dy
        : (up - center) / dy;

    out[idx] = center - dt * (a * du_dx + b * du_dy);
}
"""


def _compile_kernel():
    return cp.RawKernel(_UPWIND_KERNEL_SRC, "upwind_zero", options=("--std=c++11",))


def _run_kernel(kernel, field, a, b, dt, dx, dy):
    height, width = field.shape
    out = cp.empty_like(field)
    block = (32, 8)
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])
    kernel(grid, block, (field, out, width, height, a, b, dt, dx, dy))
    return out


def benchmark(size: int, ratio: int, iterations: int, seed: int | None):
    if seed is not None:
        cp.random.seed(seed)

    coarse = cp.random.random((size, size), dtype=cp.float32)
    fine = cp.random.random((size * ratio, size * ratio), dtype=cp.float32)

    coarse_interval = full_interval_set(size, size)
    coarse_field = create_interval_field(coarse_interval, fill_value=0.0, dtype=cp.float32)
    coarse_field.values[...] = coarse.ravel()

    fine_interval = full_interval_set(size * ratio, size * ratio)
    fine_field = create_interval_field(fine_interval, fill_value=0.0, dtype=cp.float32)
    fine_field.values[...] = fine.ravel()
    coarse_out = cp.empty_like(coarse_field.values)
    fine_out = cp.empty_like(fine_field.values)

    a = 1.0
    b = 1.0
    dx_coarse = 1.0 / size
    dy_coarse = dx_coarse
    dx_fine = dx_coarse / ratio
    dy_fine = dy_coarse / ratio
    dt = 0.5 * dx_coarse

    kernel = _compile_kernel()

    for _ in range(3):
        step_upwind_interval(
            coarse_field,
            width=size,
            height=size,
            a=a,
            b=b,
            dt=dt,
            dx=dx_coarse,
            dy=dy_coarse,
            out=coarse_out,
        )
        _run_kernel(kernel, coarse, a, b, dt, dx_coarse, dy_coarse)
    for _ in range(3):
        step_upwind_interval(
            fine_field,
            width=size * ratio,
            height=size * ratio,
            a=a,
            b=b,
            dt=dt,
            dx=dx_fine,
            dy=dy_fine,
            out=fine_out,
        )
        _run_kernel(kernel, fine, a, b, dt, dx_fine, dy_fine)
    cp.cuda.runtime.deviceSynchronize()

    def time_call(fn):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        return result, cp.cuda.get_elapsed_time(start, end) / 1e3

    ref_times = []
    ker_times = []

    def interval_step(field, out_buf, width, height, dx, dy):
        step_upwind_interval(
            field,
            width=width,
            height=height,
            a=a,
            b=b,
            dt=dt,
            dx=dx,
            dy=dy,
            out=out_buf,
        )
        return out_buf

    for _ in range(iterations):
        _, t_ref = time_call(
            lambda: interval_step(coarse_field, coarse_out, size, size, dx_coarse, dy_coarse)
        )
        ref_times.append(t_ref)
        _, t_ker = time_call(
            lambda: _run_kernel(kernel, coarse, a, b, dt, dx_coarse, dy_coarse)
        )
        ker_times.append(t_ker)

    print(f"Coarse grid {size}x{size}")
    print(f"  slice-based : {_format(ref_times)}")
    print(f"  raw kernel  : {_format(ker_times)}")

    ref_times.clear()
    ker_times.clear()

    for _ in range(iterations):
        _, t_ref = time_call(
            lambda: interval_step(
                fine_field,
                fine_out,
                size * ratio,
                size * ratio,
                dx_fine,
                dy_fine,
            )
        )
        ref_times.append(t_ref)
        _, t_ker = time_call(
            lambda: _run_kernel(kernel, fine, a, b, dt, dx_fine, dy_fine)
        )
        ker_times.append(t_ker)

    size_fine = size * ratio
    print(f"Fine grid {size_fine}x{size_fine}")
    print(f"  slice-based : {_format(ref_times)}")
    print(f"  raw kernel  : {_format(ker_times)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark upwind implementations.")
    parser.add_argument("--size", type=int, default=1024, help="Coarse grid resolution.")
    parser.add_argument("--ratio", type=int, default=2, help="Refinement ratio.")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark(args.size, args.ratio, args.iterations, args.seed)


if __name__ == "__main__":
    main()
