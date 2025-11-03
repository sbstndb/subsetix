from __future__ import annotations

from typing import Optional

import cupy as cp

from subsetix_cupy import (
    evaluate,
    make_input,
    make_union,
    make_intersection,
    dilate_interval_set,
    clip_interval_set,
)
from subsetix_cupy.expressions import IntervalSet, _require_cupy



_COUNT_INTERVALS = cp.RawKernel(
    r"""
    extern "C" __global__
    void subsetix_count_threshold_intervals(const float* __restrict__ data,
                                            int rows,
                                            int width,
                                            float threshold,
                                            int* __restrict__ counts)
    {
        int row = blockIdx.x;
        if (row >= rows) {
            return;
        }
        const int WARP = 32;
        int lane = threadIdx.x & (WARP - 1);
        const float* row_ptr = data + row * width;
        unsigned int prev = 0u;
        int count = 0;
        for (int base = 0; base < width; base += WARP) {
            int col = base + lane;
            bool ge = false;
            if (col < width) {
                ge = row_ptr[col] >= threshold;
            }
            unsigned int mask = __ballot_sync(0xffffffff, ge);
            if (lane == 0) {
                unsigned int prev_mask = (mask << 1) | (prev & 1u);
                unsigned int starts = mask & ~prev_mask;
                count += __popc(starts);
                if (width - base >= WARP) {
                    prev = (mask >> (WARP - 1)) & 1u;
                } else {
                    int last_col = width - base - 1;
                    prev = (last_col >= 0) ? ((mask >> last_col) & 1u) : 0u;
                }
            }
        }
        if (lane == 0) {
            counts[row] = count;
        }
    }
    """,
    "subsetix_count_threshold_intervals",
    options=("--std=c++11",),
)

_WRITE_INTERVALS = cp.RawKernel(
    r"""
    extern "C" __global__
    void subsetix_write_threshold_intervals(const float* __restrict__ data,
                                            int rows,
                                            int width,
                                            float threshold,
                                            const int* __restrict__ row_offsets,
                                            int* __restrict__ begin,
                                            int* __restrict__ end)
    {
        int row = blockIdx.x;
        if (row >= rows) {
            return;
        }
        const int WARP = 32;
        int lane = threadIdx.x & (WARP - 1);
        const float* row_ptr = data + row * width;
        int base_offset = row_offsets[row];
        int begin_idx = 0;
        int end_idx = 0;
        unsigned int prev = 0u;
        for (int base = 0; base < width; base += WARP) {
            int col = base + lane;
            bool ge = false;
            if (col < width) {
                ge = row_ptr[col] >= threshold;
            }
            unsigned int mask = __ballot_sync(0xffffffff, ge);
            if (lane == 0) {
                unsigned int prev_mask = (mask << 1) | (prev & 1u);
                unsigned int starts = mask & ~prev_mask;
                unsigned int stops = prev_mask & ~mask;
                while (starts) {
                    unsigned int bit = __ffs(starts) - 1u;
                    begin[base_offset + begin_idx++] = base + bit;
                    starts &= (starts - 1u);
                }
                while (stops) {
                    unsigned int bit = __ffs(stops) - 1u;
                    end[base_offset + end_idx++] = base + bit;
                    stops &= (stops - 1u);
                }
                if (width - base >= WARP) {
                    prev = (mask >> (WARP - 1)) & 1u;
                } else {
                    int last_col = width - base - 1;
                    prev = (last_col >= 0) ? ((mask >> last_col) & 1u) : 0u;
                }
            }
        }
        if (lane == 0 && prev) {
            end[base_offset + end_idx++] = width;
        }
    }
    """,
    "subsetix_write_threshold_intervals",
    options=("--std=c++11",),
)

_GRADIENT_MAGNITUDE_KERNEL = cp.RawKernel(
    r"""
    extern "C" __global__
    void subsetix_gradient_magnitude(const float* __restrict__ data,
                                     int rows,
                                     int width,
                                     float* __restrict__ output)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= width || y >= rows) {
            return;
        }
        int row_offset = y * width;
        int idx = row_offset + x;
        float center = data[idx];

        float gx = 0.0f;
        if (width > 1) {
            if (x == 0) {
                gx = data[row_offset + 1] - center;
            } else if (x == width - 1) {
                gx = center - data[row_offset + x - 1];
            } else {
                gx = (data[row_offset + x + 1] - data[row_offset + x - 1]) * 0.5f;
            }
        }

        float gy = 0.0f;
        if (rows > 1) {
            if (y == 0) {
                gy = data[(y + 1) * width + x] - center;
            } else if (y == rows - 1) {
                gy = center - data[(y - 1) * width + x];
            } else {
                gy = (data[(y + 1) * width + x] - data[(y - 1) * width + x]) * 0.5f;
            }
        }

        output[idx] = sqrtf(gx * gx + gy * gy);
    }
    """,
    "subsetix_gradient_magnitude",
    options=("--std=c++11",),
)

def _empty_interval_set(rows: int) -> IntervalSet:
    cp_mod = _require_cupy()
    zero = cp_mod.zeros(0, dtype=cp_mod.int32)
    offsets = cp_mod.zeros(rows + 1, dtype=cp_mod.int32)
    return IntervalSet(begin=zero, end=zero, row_offsets=offsets)


def _intervals_above_threshold(
    data: cp.ndarray,
    threshold: float,
) -> IntervalSet:
    cp_mod = _require_cupy()
    if data.ndim != 2:
        raise ValueError("data must be 2D")
    rows, width = data.shape
    if rows == 0 or width == 0:
        return _empty_interval_set(rows)

    counts = cp_mod.zeros(rows, dtype=cp_mod.int32)
    _COUNT_INTERVALS(
        (rows,),
        (32,),
        (
            data,
            cp_mod.int32(rows),
            cp_mod.int32(width),
            cp_mod.float32(threshold),
            counts,
        ),
    )
    row_offsets = cp_mod.empty(rows + 1, dtype=cp_mod.int32)
    row_offsets[0] = 0
    cp_mod.cumsum(counts, dtype=cp_mod.int32, out=row_offsets[1:])
    total = int(row_offsets[-1].item()) if rows > 0 else 0
    begin = cp_mod.empty(total, dtype=cp_mod.int32)
    end = cp_mod.empty_like(begin)
    if total > 0:
        _WRITE_INTERVALS(
            (rows,),
            (32,),
            (
                data,
                cp_mod.int32(rows),
                cp_mod.int32(width),
                cp_mod.float32(threshold),
                row_offsets,
                begin,
                end,
            ),
        )
    return IntervalSet(begin=begin, end=end, row_offsets=row_offsets)


def gradient_magnitude(field: cp.ndarray) -> cp.ndarray:
    """
    Return sqrt((du/dx)^2 + (du/dy)^2) using first-order differences.
    """

    cp_mod = _require_cupy()
    data = cp_mod.asarray(field, dtype=cp_mod.float32)
    if data.ndim != 2:
        raise ValueError("field must be a 2D array")
    rows, width = data.shape
    if rows == 0 or width == 0:
        return cp_mod.zeros_like(data)
    out = cp_mod.empty_like(data)
    block = (32, 8)
    grid = (
        (width + block[0] - 1) // block[0],
        (rows + block[1] - 1) // block[1],
    )
    _GRADIENT_MAGNITUDE_KERNEL(
        grid,
        block,
        (
            data,
            cp_mod.int32(rows),
            cp_mod.int32(width),
            out,
        ),
    )
    return out


def gradient_tag_set(
    values: cp.ndarray,
    frac_high: float,
    *,
    epsilon: float = 1e-8,
) -> IntervalSet:
    """Tag the top ``frac_high`` fraction of gradients as an IntervalSet."""

    cp_mod = _require_cupy()
    data = cp_mod.asarray(values, dtype=cp_mod.float32)
    if data.ndim != 2:
        raise ValueError("values must be a 2D array")
    rows = data.shape[0]
    frac_high = float(max(0.0, min(1.0, frac_high)))
    if data.size == 0 or frac_high <= 0.0:
        return _empty_interval_set(rows)

    flat = data.ravel()
    positive = flat[flat > float(epsilon)]
    if positive.size == 0:
        return _empty_interval_set(rows)

    count = int(cp_mod.ceil(positive.size * frac_high))
    count = max(1, min(int(positive.size), count))
    idx = int(positive.size) - count
    part = cp_mod.partition(positive, idx)
    threshold = max(float(part[idx]), float(epsilon))
    return _intervals_above_threshold(data, threshold)


def gradient_tag_threshold_set(
    values: cp.ndarray,
    threshold: float,
    *,
    epsilon: float = 1e-8,
) -> IntervalSet:
    """Tag gradients above ``threshold`` as an IntervalSet."""

    cp_mod = _require_cupy()
    data = cp_mod.asarray(values, dtype=cp_mod.float32)
    if data.ndim != 2:
        raise ValueError("values must be a 2D array")
    rows = data.shape[0]
    if data.size == 0:
        return _empty_interval_set(rows)
    thresh = max(float(threshold), float(epsilon))
    return _intervals_above_threshold(data, thresh)


def enforce_two_level_grading_set(
    refine_set: IntervalSet,
    *,
    padding: int = 1,
    mode: str = "von_neumann",
    width: int,
    height: int,
) -> IntervalSet:
    """IntervalSet variant of the two-level grading dilation."""

    padding = int(padding)
    if padding <= 0:
        return refine_set
    if mode not in {"von_neumann", "moore"}:
        raise ValueError("mode must be 'von_neumann' or 'moore'")

    if mode == "moore":
        expanded = dilate_interval_set(
            refine_set,
            halo_x=padding,
            halo_y=padding,
        )
    else:
        horiz = dilate_interval_set(
            refine_set,
            halo_x=padding,
            halo_y=0,
        )
        vert = dilate_interval_set(
            refine_set,
            halo_x=0,
            halo_y=padding,
        )
        expanded = evaluate(make_union(make_input(horiz), make_input(vert)))

    expanded = clip_interval_set(expanded, width=width, height=height)
    base = clip_interval_set(refine_set, width=width, height=height)
    union = evaluate(make_union(make_input(base), make_input(expanded)))
    return clip_interval_set(union, width=width, height=height)
