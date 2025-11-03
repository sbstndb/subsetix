from __future__ import annotations

from typing import Optional

import cupy as cp

from subsetix_cupy import (
    evaluate,
    make_input,
    make_union,
    dilate_interval_set,
)
from subsetix_cupy.expressions import IntervalSet, _require_cupy

from .geometry import interval_set_to_mask, mask_to_interval_set

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
        const float* row_ptr = data + row * width;
        bool inside = false;
        int count = 0;
        for (int col = 0; col < width; ++col) {
            float v = row_ptr[col];
            bool ge = v >= threshold;
            if (!inside && ge) {
                inside = true;
                count += 1;
            } else if (inside && !ge) {
                inside = false;
            }
        }
        counts[row] = count;
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
        int offset = row_offsets[row];
        int local_index = 0;
        const float* row_ptr = data + row * width;
        bool inside = false;
        for (int col = 0; col < width; ++col) {
            float v = row_ptr[col];
            bool ge = v >= threshold;
            if (!inside && ge) {
                inside = true;
                begin[offset + local_index] = col;
            } else if (inside && !ge) {
                inside = false;
                end[offset + local_index] = col;
                local_index += 1;
            }
        }
        if (inside) {
            end[offset + local_index] = width;
        }
    }
    """,
    "subsetix_write_threshold_intervals",
    options=("--std=c++11",),
)

def _empty_interval_set(rows: int) -> IntervalSet:
    cp_mod = _require_cupy()
    zero = cp_mod.zeros(0, dtype=cp_mod.int32)
    offsets = cp_mod.zeros(rows + 1, dtype=cp_mod.int32)
    return IntervalSet(begin=zero, end=zero, row_offsets=offsets)


def gradient_magnitude(field: cp.ndarray) -> cp.ndarray:
    """
    Return sqrt((du/dx)^2 + (du/dy)^2) using first-order differences.
    """

    gx, gy = cp.gradient(field)
    return cp.sqrt(gx * gx + gy * gy)


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
    width = data.shape[1]
    counts = cp_mod.zeros(rows, dtype=cp_mod.int32)
    if rows > 0:
        _COUNT_INTERVALS(
            (rows,),
            (1,),
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
    if rows > 0:
        cp_mod.cumsum(counts, dtype=cp_mod.int32, out=row_offsets[1:])
    total = int(row_offsets[-1].item()) if rows > 0 else 0
    begin = cp_mod.empty(total, dtype=cp_mod.int32)
    end = cp_mod.empty_like(begin)
    if total > 0:
        _WRITE_INTERVALS(
            (rows,),
            (1,),
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


def gradient_tag(
    values: cp.ndarray,
    frac_high: float,
    *,
    epsilon: float = 1e-8,
) -> cp.ndarray:
    """Wrapper returning a dense mask for backwards compatibility."""

    interval = gradient_tag_set(values, frac_high, epsilon=epsilon)
    return interval_set_to_mask(interval, values.shape[1])


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
            width=width,
            height=height,
            bc="clamp",
        )
    else:
        horiz = dilate_interval_set(
            refine_set,
            halo_x=padding,
            halo_y=0,
            width=width,
            height=height,
            bc="clamp",
        )
        vert = dilate_interval_set(
            refine_set,
            halo_x=0,
            halo_y=padding,
            width=width,
            height=height,
            bc="clamp",
        )
        expanded = evaluate(make_union(make_input(horiz), make_input(vert)))

    union = evaluate(make_union(make_input(refine_set), make_input(expanded)))
    return union


def enforce_two_level_grading(
    refine_mask: cp.ndarray,
    *,
    padding: int = 1,
    mode: str = "von_neumann",
) -> cp.ndarray:
    """
    Ensure a refined cell carries its immediate parents by dilating the mask.
    """

    cp_mod = _require_cupy()
    interval = mask_to_interval_set(refine_mask)
    graded = enforce_two_level_grading_set(
        interval,
        padding=padding,
        mode=mode,
        width=refine_mask.shape[1],
        height=refine_mask.shape[0],
    )
    mask = interval_set_to_mask(graded, refine_mask.shape[1])
    return cp_mod.asarray(mask, dtype=cp_mod.bool_)
