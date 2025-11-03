from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Dict

import cupy as cp
import numpy as np

from subsetix_cupy.interval_field import IntervalField, create_interval_field
from subsetix_cupy.expressions import (
    IntervalSet,
    _require_cupy,
    evaluate,
    make_difference,
    make_input,
)
from subsetix_cupy import prolong_set

_PROLONG_NEAREST_KERNEL = cp.ElementwiseKernel(
    "raw T coarse, int32 coarse_w, int32 ratio",
    "T out",
    """
    const int fine_w = coarse_w * ratio;
    const int fy = i / fine_w;
    const int fx = i - fy * fine_w;
    const int cy = fy / ratio;
    const int cx = fx / ratio;
    out = coarse[cy * coarse_w + cx];
    """,
    "subsetix_amr2_prolong_nearest",
)

_COPY_KERNEL_CACHE: Dict[str, cp.RawKernel] = {}
_FILL_KERNEL_CACHE: Dict[str, cp.RawKernel] = {}

_ACTION_COUNT_KERNEL = cp.RawKernel(
    r"""
    extern "C" __global__
    void subsetix_count_value_intervals(const signed char* __restrict__ data,
                                        int rows,
                                        int width,
                                        signed char target,
                                        int* __restrict__ counts)
    {
        int row = blockIdx.x;
        if (row >= rows) {
            return;
        }
        const signed char* row_ptr = data + row * width;
        bool inside = false;
        int count = 0;
        for (int col = 0; col < width; ++col) {
            signed char v = row_ptr[col];
            bool match = v == target;
            if (!inside && match) {
                inside = true;
                count += 1;
            } else if (inside && !match) {
                inside = false;
            }
        }
        counts[row] = count;
    }
    """,
    "subsetix_count_value_intervals",
    options=("--std=c++11",),
)

_ACTION_WRITE_KERNEL = cp.RawKernel(
    r"""
    extern "C" __global__
    void subsetix_write_value_intervals(const signed char* __restrict__ data,
                                        int rows,
                                        int width,
                                        signed char target,
                                        const int* __restrict__ row_offsets,
                                        int* __restrict__ begin,
                                        int* __restrict__ end)
    {
        int row = blockIdx.x;
        if (row >= rows) {
            return;
        }
        const signed char* row_ptr = data + row * width;
        int offset = row_offsets[row];
        int local_index = 0;
        bool inside = false;
        for (int col = 0; col < width; ++col) {
            signed char v = row_ptr[col];
            bool match = v == target;
            if (!inside && match) {
                inside = true;
                begin[offset + local_index] = col;
            } else if (inside && !match) {
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
    "subsetix_write_value_intervals",
    options=("--std=c++11",),
)


def _intervals_for_value(grid: cp.ndarray, target: int) -> IntervalSet:
    """
    Build an IntervalSet describing the cells whose value equals ``target``.
    """

    cp_mod = _require_cupy()
    if grid.ndim != 2:
        raise ValueError("grid must be a 2D array")
    if grid.dtype != cp_mod.int8:
        raise TypeError("grid dtype must be int8 for action extraction")
    rows, width = grid.shape
    counts = cp_mod.zeros(rows, dtype=cp_mod.int32)
    if rows > 0:
        _ACTION_COUNT_KERNEL(
            (rows,),
            (1,),
            (
                grid,
                cp_mod.int32(rows),
                cp_mod.int32(width),
                cp_mod.int8(target),
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
        _ACTION_WRITE_KERNEL(
            (rows,),
            (1,),
            (
                grid,
                cp_mod.int32(rows),
                cp_mod.int32(width),
                cp_mod.int8(target),
                row_offsets,
                begin,
                end,
            ),
        )
    return IntervalSet(begin=begin, end=end, row_offsets=row_offsets)


def _interval_row_ids(interval_set: IntervalSet) -> cp.ndarray:
    row_offsets = interval_set.row_offsets.astype(cp.int32, copy=False)
    interval_count = int(interval_set.begin.size)
    if interval_count == 0:
        return cp.zeros(0, dtype=cp.int32)
    idx = cp.arange(interval_count, dtype=cp.int32)
    return cp.searchsorted(row_offsets[1:], idx, side="right").astype(cp.int32, copy=False)


def _get_copy_intervals_kernel(dtype: cp.dtype) -> cp.RawKernel:
    key = cp.dtype(dtype).str
    kernel = _COPY_KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel
    if dtype == cp.float32:
        type_name = "float"
    elif dtype == cp.int32:
        type_name = "int"
    elif dtype == cp.int64:
        type_name = "long long"
    elif dtype == cp.bool_:
        type_name = "bool"
    else:
        raise TypeError("interval copy expects float32/int/bool data")
    code = f"""
    extern "C" __global__
    void copy_intervals_2d(const int* __restrict__ row_ids,
                           const int* __restrict__ begin,
                           const int* __restrict__ end,
                           const {type_name}* __restrict__ src,
                           {type_name}* __restrict__ dst,
                           int width)
    {{
        int interval = blockIdx.x;
        int row = row_ids[interval];
        int start = begin[interval];
        int stop = end[interval];
        if (stop <= start) {{
            return;
        }}
        int base = row * width;
        for (int col = start + threadIdx.x; col < stop; col += blockDim.x) {{
            dst[base + col] = src[base + col];
        }}
    }}
    """
    kernel = cp.RawKernel(code, "copy_intervals_2d", options=("--std=c++11",))
    _COPY_KERNEL_CACHE[key] = kernel
    return kernel


def _copy_intervals_into(dst: cp.ndarray, src: cp.ndarray, interval_set: IntervalSet) -> None:
    if dst.ndim != 2 or src.ndim != 2:
        raise ValueError("dst and src must be 2D arrays")
    if dst.shape != src.shape:
        raise ValueError("dst and src must have the same shape")
    if dst.dtype != src.dtype:
        raise TypeError("dst and src must have matching dtypes")
    interval_count = int(interval_set.begin.size)
    if interval_count == 0:
        return
    row_ids = _interval_row_ids(interval_set)
    kernel = _get_copy_intervals_kernel(dst.dtype)
    block = 128
    grid = (interval_count,)
    width = int(dst.shape[1])
    kernel(grid, (block,), (row_ids, interval_set.begin, interval_set.end, src, dst, width))


def _get_fill_intervals_kernel(dtype: cp.dtype) -> cp.RawKernel:
    key = cp.dtype(dtype).str
    kernel = _FILL_KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel
    if dtype == cp.int8:
        type_name = "signed char"
    elif dtype == cp.int32:
        type_name = "int"
    elif dtype == cp.int64:
        type_name = "long long"
    elif dtype == cp.float32:
        type_name = "float"
    else:
        raise TypeError("interval fill expects float32/int data")
    code = f"""
    extern "C" __global__
    void fill_intervals_2d(const int* __restrict__ row_ids,
                           const int* __restrict__ begin,
                           const int* __restrict__ end,
                           {type_name} value,
                           {type_name}* __restrict__ dst,
                           int width)
    {{
        int interval = blockIdx.x;
        int row = row_ids[interval];
        int start = begin[interval];
        int stop = end[interval];
        if (stop <= start) {{
            return;
        }}
        int base = row * width;
        for (int col = start + threadIdx.x; col < stop; col += blockDim.x) {{
            dst[base + col] = value;
        }}
    }}
    """
    kernel = cp.RawKernel(code, "fill_intervals_2d", options=("--std=c++11",))
    _FILL_KERNEL_CACHE[key] = kernel
    return kernel


def _fill_intervals(array: cp.ndarray, interval_set: IntervalSet, value) -> None:
    if array.ndim != 2:
        raise ValueError("array must be 2D")
    interval_count = int(interval_set.begin.size)
    if interval_count == 0:
        return
    row_ids = _interval_row_ids(interval_set)
    kernel = _get_fill_intervals_kernel(array.dtype)
    block = 128
    grid = (interval_count,)
    width = int(array.shape[1])
    kernel(grid, (block,), (row_ids, interval_set.begin, interval_set.end, array.dtype.type(value), array, width))


_PROLONG_BLOCK_KERNEL: cp.RawKernel | None = None
_RESTRICT_BLOCK_KERNEL: cp.RawKernel | None = None


def _get_prolong_block_kernel() -> cp.RawKernel:
    global _PROLONG_BLOCK_KERNEL
    if _PROLONG_BLOCK_KERNEL is None:
        _PROLONG_BLOCK_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__
            void prolong_blocks_from_coarse(const int* __restrict__ row_ids,
                                            const int* __restrict__ begin,
                                            const int* __restrict__ end,
                                            const float* __restrict__ coarse,
                                            int coarse_width,
                                            float* __restrict__ fine,
                                            int fine_width,
                                            int ratio)
            {
                int interval = blockIdx.x;
                int row = row_ids[interval];
                int start = begin[interval];
                int stop = end[interval];
                if (stop <= start) {
                    return;
                }
                int coarse_row_offset = row * coarse_width;
                int fine_row_base = row * ratio;
                for (int cx = start + threadIdx.x; cx < stop; cx += blockDim.x) {
                    float value = coarse[coarse_row_offset + cx];
                    int fine_col0 = cx * ratio;
                    for (int dy = 0; dy < ratio; ++dy) {
                        int fine_row = (fine_row_base + dy) * fine_width;
                        int dst = fine_row + fine_col0;
                        for (int dx = 0; dx < ratio; ++dx) {
                            fine[dst + dx] = value;
                        }
                    }
                }
            }
            """,
            "prolong_blocks_from_coarse",
            options=("--std=c++11",),
        )
    return _PROLONG_BLOCK_KERNEL


def _get_restrict_block_kernel() -> cp.RawKernel:
    global _RESTRICT_BLOCK_KERNEL
    if _RESTRICT_BLOCK_KERNEL is None:
        _RESTRICT_BLOCK_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__
            void restrict_blocks_to_coarse(const int* __restrict__ row_ids,
                                           const int* __restrict__ begin,
                                           const int* __restrict__ end,
                                           const float* __restrict__ fine,
                                           int fine_width,
                                           float* __restrict__ coarse,
                                           int coarse_width,
                                           int ratio,
                                           int mode)
            {
                int interval = blockIdx.x;
                int row = row_ids[interval];
                int start = begin[interval];
                int stop = end[interval];
                if (stop <= start) {
                    return;
                }
                int coarse_row_offset = row * coarse_width;
                int fine_row_base = row * ratio;
                int block_area = ratio * ratio;
                for (int cx = start + threadIdx.x; cx < stop; cx += blockDim.x) {
                    int fine_col0 = cx * ratio;
                    float sum = 0.0f;
                    float max_val = 0.0f;
                    float min_val = 0.0f;
                    bool first = true;
                    for (int dy = 0; dy < ratio; ++dy) {
                        int fine_row = (fine_row_base + dy) * fine_width;
                        int src = fine_row + fine_col0;
                        for (int dx = 0; dx < ratio; ++dx) {
                            float v = fine[src + dx];
                            if (mode == 0 || mode == 1) {
                                sum += v;
                            }
                            if (mode == 2) {
                                if (first || v > max_val) {
                                    max_val = v;
                                }
                            } else if (mode == 3) {
                                if (first || v < min_val) {
                                    min_val = v;
                                }
                            }
                            if (first) {
                                min_val = v;
                                max_val = v;
                            }
                            first = false;
                        }
                    }
                    float result = sum;
                    if (mode == 0) {
                        result = sum / (float)block_area;
                    } else if (mode == 1) {
                        result = sum;
                    } else if (mode == 2) {
                        result = max_val;
                    } else if (mode == 3) {
                        result = min_val;
                    }
                    coarse[coarse_row_offset + cx] = result;
                }
            }
            """,
            "restrict_blocks_to_coarse",
            options=("--std=c++11",),
        )
    return _RESTRICT_BLOCK_KERNEL


def _prolong_blocks_from_coarse(
    coarse: cp.ndarray,
    fine: cp.ndarray,
    intervals: IntervalSet,
    ratio: int,
) -> None:
    interval_count = int(intervals.begin.size)
    if interval_count == 0:
        return
    row_ids = _interval_row_ids(intervals)
    kernel = _get_prolong_block_kernel()
    block = 128
    grid = (interval_count,)
    kernel(
        grid,
        (block,),
        (
            row_ids,
            intervals.begin,
            intervals.end,
            coarse,
            np.int32(coarse.shape[1]),
            fine,
            np.int32(fine.shape[1]),
            np.int32(ratio),
        ),
    )


def _restrict_blocks_to_coarse(
    fine: cp.ndarray,
    coarse: cp.ndarray,
    intervals: IntervalSet,
    ratio: int,
    reducer: str,
) -> None:
    interval_count = int(intervals.begin.size)
    if interval_count == 0:
        return
    mode_map = {"mean": 0, "sum": 1, "max": 2, "min": 3}
    if reducer not in mode_map:
        raise ValueError(f"unsupported reducer '{reducer}'")
    mode = mode_map[reducer]
    row_ids = _interval_row_ids(intervals)
    kernel = _get_restrict_block_kernel()
    block = 128
    grid = (interval_count,)
    kernel(
        grid,
        (block,),
        (
            row_ids,
            intervals.begin,
            intervals.end,
            fine,
            np.int32(fine.shape[1]),
            coarse,
            np.int32(coarse.shape[1]),
            np.int32(ratio),
            np.int32(mode),
        ),
    )


class Action(IntEnum):
    COARSEN = -1
    KEEP = 0
    REFINE = 1


@dataclass
class ActionField:
    """
    Interval-backed field encoding AMR actions per coarse cell.

    The field stores a value per coarse cell using an IntervalField covering the
    entire grid. Cached dense masks and interval sets are refreshed only when
    the field is updated.
    """

    field: IntervalField
    ratio: int
    width: int
    height: int
    _refine_set_cache: IntervalSet | None = None
    _fine_set_cache: IntervalSet | None = None
    _coarse_only_cache: IntervalSet | None = None

    @classmethod
    def full_grid(cls, height: int, width: int, ratio: int, *, default: Action = Action.KEEP) -> "ActionField":
        cp_mod = _require_cupy()
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        begin = cp_mod.zeros(height, dtype=cp_mod.int32)
        end = cp_mod.full(height, width, dtype=cp_mod.int32)
        row_offsets = cp_mod.arange(height + 1, dtype=cp_mod.int32)
        coarse_set = IntervalSet(begin=begin, end=end, row_offsets=row_offsets)
        interval_field = create_interval_field(coarse_set, fill_value=int(default), dtype=cp_mod.int8)
        return cls(field=interval_field, ratio=int(ratio), width=width, height=height)

    def _grid_view(self) -> cp.ndarray:
        return self.field.values.reshape(self.height, self.width)

    def _refresh_from_values(self) -> None:
        refine = _intervals_for_value(self._grid_view(), int(Action.REFINE))
        self._refine_set_cache = refine
        self._fine_set_cache = None
        self._coarse_only_cache = None

    def set_from_interval_set(self, refine_set: IntervalSet) -> None:
        """Populate the action field directly from an IntervalSet."""

        grid = self._grid_view()
        grid[...] = int(Action.KEEP)
        if refine_set.begin.size != 0:
            _fill_intervals(grid, refine_set, int(Action.REFINE))
        self._refine_set_cache = refine_set
        self._fine_set_cache = None
        self._coarse_only_cache = None

    def refine_set(self) -> IntervalSet:
        if self._refine_set_cache is None:
            self._refresh_from_values()
        return self._refine_set_cache

    def fine_set(self) -> IntervalSet:
        if self._fine_set_cache is None:
            self._fine_set_cache = prolong_set(self.refine_set(), int(self.ratio))
        return self._fine_set_cache

    def coarse_only_set(self) -> IntervalSet:
        if self._coarse_only_cache is None:
            coarse_expr = make_input(self.field.interval_set)
            refine_expr = make_input(self.refine_set())
            diff_expr = make_difference(coarse_expr, refine_expr)
            self._coarse_only_cache = evaluate(diff_expr)
        return self._coarse_only_cache

    def coarse_interval_set(self) -> IntervalSet:
        return self.field.interval_set

    def values(self) -> cp.ndarray:
        return self.field.values


def prolong_coarse_to_fine(
    coarse: cp.ndarray,
    ratio: int,
    *,
    out: Optional[cp.ndarray] = None,
    mask: Optional[IntervalSet] = None,
) -> cp.ndarray:
    """
    Repeat a coarse field onto the fine grid (nearest-neighbour prolongation).
    """

    if ratio < 1:
        raise ValueError("ratio must be >= 1")
    if coarse.ndim != 2:
        raise ValueError("coarse must be a 2D array")
    coarse_h, coarse_w = coarse.shape
    fine_shape = (coarse_h * ratio, coarse_w * ratio)
    upsampled = _PROLONG_NEAREST_KERNEL(
        coarse.ravel(), cp.int32(coarse_w), cp.int32(ratio), size=fine_shape[0] * fine_shape[1]
    ).reshape(fine_shape)
    if out is None:
        if mask is not None:
            out = cp.zeros_like(upsampled)
        else:
            return upsampled
    elif out.shape != upsampled.shape:
        raise ValueError("out must match the fine grid shape")
    if mask is None:
        cp.copyto(out, upsampled)
    elif isinstance(mask, IntervalSet):
        if out is None:
            raise ValueError("out array required when using IntervalSet mask")
        _copy_intervals_into(out, upsampled, mask)
    else:
        raise TypeError("mask must be an IntervalSet or None")
    return out


def restrict_fine_to_coarse(
    fine: cp.ndarray,
    ratio: int,
    *,
    reducer: str = "mean",
) -> cp.ndarray:
    """
    Collapse a fine grid onto the coarse resolution using block reducers.
    """

    ratio = int(ratio)
    if ratio < 1:
        raise ValueError("ratio must be >= 1")
    Hf, Wf = fine.shape
    if Hf % ratio != 0 or Wf % ratio != 0:
        raise ValueError("fine shape must be divisible by ratio")
    reshaped = fine.reshape(Hf // ratio, ratio, Wf // ratio, ratio)
    if reducer == "mean":
        return reshaped.mean(axis=(1, 3))
    if reducer == "sum":
        return reshaped.sum(axis=(1, 3))
    if reducer == "max":
        return reshaped.max(axis=(1, 3))
    if reducer == "min":
        return reshaped.min(axis=(1, 3))
    raise ValueError(f"unsupported reducer '{reducer}'")


def synchronize_two_level(
    coarse: cp.ndarray,
    fine: cp.ndarray,
    refine_mask,
    *,
    ratio: int,
    reducer: str = "mean",
    fill_fine_outside: bool = True,
    copy: bool = True,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Perform a coarse<->fine synchronisation round:

    1. Restrict fine values onto the coarse grid inside the refine mask.
    2. Optionally refill fine values outside the refine region from the coarse grid.

    Parameters
    ----------
    copy : bool, optional
        When True (default) the function returns new arrays and leaves the inputs
        untouched. When False the updates are applied in-place to the provided
        `coarse` and `fine` arrays and the same objects are returned.
    """

    if isinstance(refine_mask, IntervalSet):
        rows = refine_mask.row_offsets.size - 1
        if rows != coarse.shape[0]:
            raise ValueError("refine IntervalSet height must match coarse grid")
        action_field = ActionField.full_grid(coarse.shape[0], coarse.shape[1], ratio)
        action_field.set_from_interval_set(refine_mask)
    elif isinstance(refine_mask, ActionField):
        action_field = refine_mask
    else:
        raise TypeError("refine_mask must be an IntervalSet or ActionField")

    if action_field.height != coarse.shape[0] or action_field.width != coarse.shape[1]:
        raise ValueError("ActionField dimensions must match coarse grid shape")
    if int(action_field.ratio) != int(ratio):
        raise ValueError("ratio argument must match ActionField ratio")

    ratio = int(ratio)
    if coarse.dtype != cp.float32 or fine.dtype != cp.float32:
        raise TypeError("coarse and fine grids must be float32")
    if fine.shape != (coarse.shape[0] * ratio, coarse.shape[1] * ratio):
        raise ValueError("fine grid must have shape (height*ratio, width*ratio)")

    coarse_target = cp.array(coarse, copy=True) if copy else coarse
    refine_set = action_field.refine_set()
    _restrict_blocks_to_coarse(fine, coarse_target, refine_set, ratio, reducer)

    if not fill_fine_outside:
        fine_result = cp.array(fine, copy=True) if copy else fine
        return coarse_target, fine_result

    fine_target = cp.array(fine, copy=True) if copy else fine
    coarse_only = action_field.coarse_only_set()
    _prolong_blocks_from_coarse(coarse_target, fine_target, coarse_only, ratio)
    return coarse_target, fine_target
