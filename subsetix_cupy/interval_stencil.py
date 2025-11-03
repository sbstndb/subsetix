"""
Stencil operators for IntervalField geometries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .expressions import _require_cupy
from .interval_field import IntervalField

_INTERVAL_UPWIND_KERNEL = None

_INTERVAL_UPWIND_SRC = r"""
extern "C" __device__ __forceinline__
float sample_interval_cell(int row_actual,
                           int x,
                           int height,
                           int width,
                           const int* __restrict__ row_lookup,
                           const int* __restrict__ row_offsets,
                           const int* __restrict__ begin,
                           const int* __restrict__ end,
                           const int* __restrict__ cell_offsets,
                           const float* __restrict__ values)
{
    if (row_actual < 0 || row_actual >= height) {
        return 0.0f;
    }
    if (x < 0 || x >= width) {
        return 0.0f;
    }
    int ordinal = row_lookup[row_actual];
    if (ordinal < 0) {
        return 0.0f;
    }
    int start = row_offsets[ordinal];
    int stop = row_offsets[ordinal + 1];
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
                     const int* __restrict__ row_ordinals,
                     const int* __restrict__ row_lookup,
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
    int row_ord = row_ordinals[interval];
    if (row_ord < 0) {
        return;
    }
    int interval_begin = begin[interval];
    int interval_end = end[interval];
    if (interval_end <= interval_begin) {
        return;
    }
    int base = cell_offsets[interval];
    int length = interval_end - interval_begin;

    for (int idx = threadIdx.x; idx < length; idx += blockDim.x) {
        int x = interval_begin + idx;
        int value_index = base + idx;
        float center = values[value_index];

        float left = sample_interval_cell(row, x - 1, height, width, row_lookup, row_offsets, begin, end, cell_offsets, values);
        float right = sample_interval_cell(row, x + 1, height, width, row_lookup, row_offsets, begin, end, cell_offsets, values);
        float down = sample_interval_cell(row - 1, x, height, width, row_lookup, row_offsets, begin, end, cell_offsets, values);
        float up = sample_interval_cell(row + 1, x, height, width, row_lookup, row_offsets, begin, end, cell_offsets, values);

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

def _get_interval_upwind_kernel():
    global _INTERVAL_UPWIND_KERNEL
    if _INTERVAL_UPWIND_KERNEL is None:
        cp = _require_cupy()
        _INTERVAL_UPWIND_KERNEL = cp.RawKernel(
            _INTERVAL_UPWIND_SRC, "interval_upwind", options=("--std=c++11",)
        )
    return _INTERVAL_UPWIND_KERNEL


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

    rows_index = interval_set.rows_index()
    if rows_index.size == 0:
        row_lookup = cp.full((height,), -1, dtype=cp.int32)
    else:
        row_lookup = cp.full((height,), -1, dtype=cp.int32)
        row_lookup[rows_index] = cp.arange(rows_index.size, dtype=cp.int32)
    row_ordinals = row_lookup[row_ids_arr]
    if int(cp.min(row_ordinals).item()) < 0:
        raise ValueError("interval_set rows exceed provided height")

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
            row_ordinals,
            row_lookup,
            values,
            out_arr,
            np.int32(interval_count),
            np.int32(width),
            np.int32(height),
            np.float32(a),
            np.float32(b),
            np.float32(dt),
            np.float32(dx),
            np.float32(dy),
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
    """Wrapper returning a new IntervalField with updated values."""

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


__all__ = [
    "step_upwind_interval",
    "step_upwind_interval_field",
]
