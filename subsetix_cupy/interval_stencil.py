"""
Stencil operators for IntervalField geometries.
"""

from __future__ import annotations

from typing import Any

from .expressions import _require_cupy
from .interval_field import IntervalField

_INTERVAL_UPWIND_KERNEL = None
_DENSE_UPWIND_ZERO_KERNEL = None
_DENSE_UPWIND_ACTIVE_KERNEL = None

_INTERVAL_UPWIND_SRC = r"""
extern "C" __device__ __forceinline__
float sample_interval_cell(int row,
                           int x,
                           int height,
                           int width,
                           const int* __restrict__ row_offsets,
                           const int* __restrict__ begin,
                           const int* __restrict__ end,
                           const int* __restrict__ cell_offsets,
                           const float* __restrict__ values)
{
    if (row < 0 || row >= height) {
        return 0.0f;
    }
    if (x < 0 || x >= width) {
        return 0.0f;
    }
    int start = row_offsets[row];
    int stop = row_offsets[row + 1];
    int lo = start;
    int hi = stop - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int b = begin[mid];
        int e = end[mid];
        if (x < b) {
            hi = mid - 1;
        } else if (x >= e) {
            lo = mid + 1;
        } else {
            int base = cell_offsets[mid];
            return values[base + (x - b)];
        }
    }
    return 0.0f;
}

extern "C" __global__
void interval_upwind(const int* __restrict__ row_ids,
                     const int* __restrict__ begin,
                     const int* __restrict__ end,
                     const int* __restrict__ row_offsets,
                     const int* __restrict__ cell_offsets,
                     const float* __restrict__ values,
                     float* __restrict__ out,
                     int interval_count,
                     int width,
                     int height,
                     float a,
                     float b,
                     float dt,
                     float dx,
                     float dy)
{
    int interval = blockIdx.x;
    if (interval >= interval_count) {
        return;
    }
    int row = row_ids[interval];
    int start = begin[interval];
    int stop = end[interval];
    if (stop <= start) {
        return;
    }
    int base = cell_offsets[interval];
    int length = cell_offsets[interval + 1] - base;

    for (int idx = threadIdx.x; idx < length; idx += blockDim.x) {
        int x = start + idx;
        int value_index = base + idx;
        float center = values[value_index];
        float left = sample_interval_cell(row, x - 1, height, width, row_offsets, begin, end, cell_offsets, values);
        float right = sample_interval_cell(row, x + 1, height, width, row_offsets, begin, end, cell_offsets, values);
        float down = sample_interval_cell(row - 1, x, height, width, row_offsets, begin, end, cell_offsets, values);
        float up = sample_interval_cell(row + 1, x, height, width, row_offsets, begin, end, cell_offsets, values);

        float du_dx = (a >= 0.0f)
            ? (center - left) / dx
            : (right - center) / dx;
        float du_dy = (b >= 0.0f)
            ? (center - down) / dy
            : (up - center) / dy;
        out[value_index] = center - dt * (a * du_dx + b * du_dy);
    }
}
"""

_DENSE_UPWIND_ZERO_SRC = r"""
extern "C" __global__
void dense_upwind_zero(const float* __restrict__ u,
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
    float left = (x == 0) ? 0.0f : u[idx - 1];
    float right = (x == width - 1) ? 0.0f : u[idx + 1];
    float down = (y == 0) ? 0.0f : u[idx - width];
    float up = (y == height - 1) ? 0.0f : u[idx + width];

    float du_dx = (a >= 0.0f)
        ? (center - left) / dx
        : (right - center) / dx;
    float du_dy = (b >= 0.0f)
        ? (center - down) / dy
        : (up - center) / dy;
    out[idx] = center - dt * (a * du_dx + b * du_dy);
}
"""

_DENSE_UPWIND_ACTIVE_SRC = r"""
extern "C" __global__
void dense_upwind_active(const float* __restrict__ u,
                         float* __restrict__ out,
                         const int* __restrict__ row_ids,
                         const int* __restrict__ begin,
                         const int* __restrict__ end,
                         int interval_count,
                         int width,
                         int height,
                         float a,
                         float b,
                         float dt,
                         float dx,
                         float dy)
{
    int interval = blockIdx.x;
    if (interval >= interval_count) {
        return;
    }
    int row = row_ids[interval];
    int start = begin[interval];
    int stop = end[interval];
    if (stop <= start) {
        return;
    }
    int length = stop - start;
    int row_offset = row * width;
    for (int idx = threadIdx.x; idx < length; idx += blockDim.x) {
        int x = start + idx;
        int center_idx = row_offset + x;
        float center = u[center_idx];
        float left = (x == 0) ? 0.0f : u[center_idx - 1];
        float right = (x == width - 1) ? 0.0f : u[center_idx + 1];
        float down = (row == 0) ? 0.0f : u[center_idx - width];
        float up = (row == height - 1) ? 0.0f : u[center_idx + width];
        float du_dx = (a >= 0.0f)
            ? (center - left) / dx
            : (right - center) / dx;
        float du_dy = (b >= 0.0f)
            ? (center - down) / dy
            : (up - center) / dy;
        out[center_idx] = center - dt * (a * du_dx + b * du_dy);
    }
}
"""


def _get_interval_upwind_kernel():
    global _INTERVAL_UPWIND_KERNEL
    if _INTERVAL_UPWIND_KERNEL is None:
        cp = _require_cupy()
        _INTERVAL_UPWIND_KERNEL = cp.RawKernel(
            _INTERVAL_UPWIND_SRC, "interval_upwind", options=("--std=c++11",)
        )
    return _INTERVAL_UPWIND_KERNEL


def _get_dense_upwind_zero_kernel():
    global _DENSE_UPWIND_ZERO_KERNEL
    if _DENSE_UPWIND_ZERO_KERNEL is None:
        cp = _require_cupy()
        _DENSE_UPWIND_ZERO_KERNEL = cp.RawKernel(
            _DENSE_UPWIND_ZERO_SRC, "dense_upwind_zero", options=("--std=c++11",)
        )
    return _DENSE_UPWIND_ZERO_KERNEL


def _get_dense_upwind_active_kernel():
    global _DENSE_UPWIND_ACTIVE_KERNEL
    if _DENSE_UPWIND_ACTIVE_KERNEL is None:
        cp = _require_cupy()
        _DENSE_UPWIND_ACTIVE_KERNEL = cp.RawKernel(
            _DENSE_UPWIND_ACTIVE_SRC, "dense_upwind_active", options=("--std=c++11",)
        )
    return _DENSE_UPWIND_ACTIVE_KERNEL


def step_upwind_interval(
    field: IntervalField,
    *,
    width: int,
    height: int,
    a: float,
    b: float,
    dt: float,
    dx: float,
    dy: float,
    out: Any | None = None,
    row_ids: Any | None = None,
):
    """
    Apply an upwind stencil over an IntervalField with zero exterior values.
    """

    cp = _require_cupy()
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    interval_set = field.interval_set
    begin = cp.asarray(interval_set.begin, dtype=cp.int32)
    end = cp.asarray(interval_set.end, dtype=cp.int32)
    row_offsets = cp.asarray(interval_set.row_offsets, dtype=cp.int32)
    cell_offsets = cp.asarray(field.interval_cell_offsets, dtype=cp.int32)
    values = cp.asarray(field.values)
    if values.dtype != cp.float32:
        values = values.astype(cp.float32, copy=False)
    if row_offsets.size != height + 1:
        raise ValueError("row_offsets length must be height + 1")

    interval_count = int(begin.size)
    if cell_offsets.size != interval_count + 1:
        raise ValueError("interval_cell_offsets must have length interval_count + 1")

    if out is None:
        out_arr = cp.empty(values.shape, dtype=cp.float32)
    else:
        out_arr = cp.asarray(out)
        if out_arr.dtype != cp.float32:
            raise TypeError("out must be float32")
        if out_arr.shape != values.shape:
            raise ValueError("out must match field values shape")

    if interval_count == 0:
        out_arr.fill(0.0)
        return out_arr

    if row_ids is None:
        row_ids_arr = interval_set.interval_rows()
    else:
        row_ids_arr = cp.asarray(row_ids)
    row_ids_arr = row_ids_arr.astype(cp.int32, copy=False)
    if row_ids_arr.size != interval_count:
        raise ValueError("row_ids must have one entry per interval")

    kernel = _get_interval_upwind_kernel()
    block = 128
    grid = (interval_count,)
    kernel(
        grid,
        (block,),
        (
            row_ids_arr,
            begin,
            end,
            row_offsets,
            cell_offsets,
            values,
            out_arr,
            cp.int32(interval_count),
            cp.int32(width),
            cp.int32(height),
            float(a),
            float(b),
            float(dt),
            float(dx),
            float(dy),
        ),
    )
    return out_arr


def step_upwind_interval_field(
    field: IntervalField,
    *,
    width: int,
    height: int,
    a: float,
    b: float,
    dt: float,
    dx: float,
    dy: float,
    row_ids: Any | None = None,
):
    """
    Wrapper returning a new IntervalField with updated values.
    """

    values = step_upwind_interval(
        field,
        width=width,
        height=height,
        a=a,
        b=b,
        dt=dt,
        dx=dx,
        dy=dy,
        row_ids=row_ids,
    )
    return IntervalField(
        interval_set=field.interval_set,
        values=values,
        interval_cell_offsets=field.interval_cell_offsets,
    )


def step_upwind_dense_zero(
    u,
    *,
    a: float,
    b: float,
    dt: float,
    dx: float,
    dy: float,
    out: Any | None = None,
):
    """
    Dense upwind stencil with zero-value boundary conditions.
    """

    cp = _require_cupy()
    arr = cp.asarray(u)
    if arr.dtype != cp.float32:
        arr = arr.astype(cp.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError("u must be a 2D array")
    height, width = arr.shape
    kernel = _get_dense_upwind_zero_kernel()
    if out is None:
        out_arr = cp.empty_like(arr)
    else:
        out_arr = cp.asarray(out)
        if out_arr.dtype != cp.float32:
            raise TypeError("out must be float32")
        if out_arr.shape != arr.shape:
            raise ValueError("out must match u shape")

    block = (32, 8)
    grid = (
        (width + block[0] - 1) // block[0],
        (height + block[1] - 1) // block[1],
    )
    kernel(
        grid,
        block,
        (
            arr,
            out_arr,
            cp.int32(width),
            cp.int32(height),
            float(a),
            float(b),
            float(dt),
            float(dx),
            float(dy),
        ),
    )
    return out_arr


def step_upwind_dense_active(
    u,
    *,
    row_ids,
    begin,
    end,
    width: int,
    height: int,
    a: float,
    b: float,
    dt: float,
    dx: float,
    dy: float,
    out: Any | None = None,
):
    """
    Dense upwind stencil restricted to the active cells described by intervals.
    """

    cp = _require_cupy()
    arr = cp.asarray(u)
    if arr.dtype != cp.float32:
        arr = arr.astype(cp.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError("u must be a 2D array")
    if arr.shape != (height, width):
        raise ValueError("u shape must match (height, width)")

    row_ids_arr = cp.asarray(row_ids).astype(cp.int32, copy=False)
    begin_arr = cp.asarray(begin).astype(cp.int32, copy=False)
    end_arr = cp.asarray(end).astype(cp.int32, copy=False)
    interval_count = begin_arr.size
    if interval_count != end_arr.size or interval_count != row_ids_arr.size:
        raise ValueError("row_ids, begin, end must share the same length")

    kernel = _get_dense_upwind_active_kernel()
    if out is None:
        out_arr = cp.zeros_like(arr)
    else:
        out_arr = cp.asarray(out)
        if out_arr.dtype != cp.float32:
            raise TypeError("out must be float32")
        if out_arr.shape != arr.shape:
            raise ValueError("out must match u shape")
        out_arr.fill(0.0)

    if interval_count == 0:
        return out_arr

    block = 128
    grid = (interval_count,)
    kernel(
        grid,
        (block,),
        (
            arr.ravel(),
            out_arr.ravel(),
            row_ids_arr,
            begin_arr,
            end_arr,
            cp.int32(interval_count),
            cp.int32(width),
            cp.int32(height),
            float(a),
            float(b),
            float(dt),
            float(dx),
            float(dy),
        ),
    )
    return out_arr


__all__ = [
    "step_upwind_interval",
    "step_upwind_interval_field",
    "step_upwind_dense_zero",
    "step_upwind_dense_active",
]
