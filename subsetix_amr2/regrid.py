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
from subsetix_cupy.interval_field import IntervalField



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

_GRADIENT_MAGNITUDE_FLAT_KERNEL: cp.RawKernel | None = None


def _get_gradient_flat_kernel() -> cp.RawKernel:
    global _GRADIENT_MAGNITUDE_FLAT_KERNEL
    if _GRADIENT_MAGNITUDE_FLAT_KERNEL is None:
        _GRADIENT_MAGNITUDE_FLAT_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__
            void subsetix_gradient_magnitude_flat(const float* __restrict__ data,
                                                  int width,
                                                  int height,
                                                  float* __restrict__ output)
            {
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                int total = width * height;
                if (idx >= total) {
                    return;
                }
                int x = idx % width;
                int y = idx / width;
                float center = data[idx];

                float gx = 0.0f;
                if (width > 1) {
                    if (x == 0) {
                        gx = data[idx + 1] - center;
                    } else if (x == width - 1) {
                        gx = center - data[idx - 1];
                    } else {
                        gx = (data[idx + 1] - data[idx - 1]) * 0.5f;
                    }
                }

                float gy = 0.0f;
                if (height > 1) {
                    if (y == 0) {
                        gy = data[idx + width] - center;
                    } else if (y == height - 1) {
                        gy = center - data[idx - width];
                    } else {
                        gy = (data[idx + width] - data[idx - width]) * 0.5f;
                    }
                }

                output[idx] = sqrtf(gx * gx + gy * gy);
            }
            """,
            "subsetix_gradient_magnitude_flat",
            options=("--std=c++11",),
        )
    return _GRADIENT_MAGNITUDE_FLAT_KERNEL

def _empty_interval_set(rows: int) -> IntervalSet:
    cp_mod = _require_cupy()
    zero = cp_mod.zeros(0, dtype=cp_mod.int32)
    offsets = cp_mod.zeros(rows + 1, dtype=cp_mod.int32)
    rows_arr = cp_mod.arange(rows, dtype=cp_mod.int32) if rows > 0 else zero
    return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=rows_arr)


def gradient_magnitude_interval_field(
    field: IntervalField,
    *,
    width: int,
    height: int,
) -> IntervalField:
    """Compute gradient magnitude for a rectangular IntervalField."""

    cp_mod = _require_cupy()
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    expected_cells = width * height
    if int(field.values.size) != expected_cells:
        raise ValueError("field cell count mismatch with width/height")
    if field.interval_set.row_count != height:
        raise ValueError("interval_set row count mismatch with height")
    if field.values.dtype != cp_mod.float32:
        data = cp_mod.asarray(field.values, dtype=cp_mod.float32)
    else:
        data = field.values

    gradient_values = cp_mod.empty(expected_cells, dtype=cp_mod.float32)
    kernel = _get_gradient_flat_kernel()
    block = 256
    grid = (max(1, (expected_cells + block - 1) // block),)
    kernel(
        grid,
        (block,),
        (
            data,
            cp_mod.int32(width),
            cp_mod.int32(height),
            gradient_values,
        ),
    )
    return IntervalField(
        interval_set=field.interval_set,
        values=gradient_values,
        interval_cell_offsets=field.interval_cell_offsets,
    )


def intervals_above_threshold_interval_field(
    field: IntervalField,
    *,
    width: int,
    height: int,
    threshold: float,
) -> IntervalSet:
    """Return intervals where ``field`` exceeds ``threshold`` without densifying."""

    cp_mod = _require_cupy()
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    expected_cells = width * height
    if int(field.values.size) != expected_cells:
        raise ValueError("field cell count mismatch with width/height")
    if field.values.dtype != cp_mod.float32:
        data = cp_mod.asarray(field.values, dtype=cp_mod.float32)
    else:
        data = field.values

    counts = cp_mod.zeros(height, dtype=cp_mod.int32)
    _COUNT_INTERVALS(
        (height,),
        (32,),
        (
            data,
            cp_mod.int32(height),
            cp_mod.int32(width),
            cp_mod.float32(threshold),
            counts,
        ),
    )
    row_offsets = cp_mod.empty(height + 1, dtype=cp_mod.int32)
    row_offsets[0] = 0
    if height > 0:
        cp_mod.cumsum(counts, dtype=cp_mod.int32, out=row_offsets[1:])
    total = int(row_offsets[-1].item()) if height > 0 else 0
    begin = cp_mod.empty(total, dtype=cp_mod.int32)
    end = cp_mod.empty_like(begin)
    if total > 0:
        _WRITE_INTERVALS(
            (height,),
            (32,),
            (
                data,
                cp_mod.int32(height),
                cp_mod.int32(width),
                cp_mod.float32(threshold),
                row_offsets,
                begin,
                end,
            ),
        )
    rows = cp_mod.arange(height, dtype=cp_mod.int32) if height > 0 else cp_mod.zeros(0, dtype=cp_mod.int32)
    return IntervalSet(begin=begin, end=end, row_offsets=row_offsets, rows=rows)


def gradient_tag_threshold_interval_field(
    field: IntervalField,
    *,
    width: int,
    height: int,
    threshold: float,
    epsilon: float = 1e-8,
) -> IntervalSet:
    gradient_field = gradient_magnitude_interval_field(field, width=width, height=height)
    return intervals_above_threshold_interval_field(
        gradient_field,
        width=width,
        height=height,
        threshold=max(float(threshold), float(epsilon)),
    )


def gradient_tag_set(
    field: IntervalField,
    *,
    width: int,
    height: int,
    frac_high: float,
    epsilon: float = 1e-8,
) -> IntervalSet:
    """Tag the top ``frac_high`` fraction of scalar values as an IntervalSet."""

    cp_mod = _require_cupy()
    frac_high = float(max(0.0, min(1.0, frac_high)))
    if frac_high <= 0.0:
        return _empty_interval_set(height)

    values = field.values.astype(cp_mod.float32, copy=False)
    positive = values[values > cp_mod.float32(epsilon)]
    if positive.size == 0:
        return _empty_interval_set(height)

    count = int(cp_mod.ceil(positive.size * frac_high))
    count = max(1, min(int(positive.size), count))
    idx = int(positive.size) - count
    part = cp_mod.partition(positive, idx)
    threshold = float(part[idx])
    return intervals_above_threshold_interval_field(
        field,
        width=width,
        height=height,
        threshold=max(threshold, float(epsilon)),
    )


def gradient_tag_threshold_set(
    field: IntervalField,
    *,
    width: int,
    height: int,
    threshold: float,
    epsilon: float = 1e-8,
) -> IntervalSet:
    """Tag values above ``threshold`` as an IntervalSet."""

    return intervals_above_threshold_interval_field(
        field,
        width=width,
        height=height,
        threshold=max(float(threshold), float(epsilon)),
    )


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
