from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict

import cupy as cp
import numpy as np

from subsetix_cupy.interval_field import IntervalField, create_interval_field
from subsetix_cupy.expressions import (
    IntervalSet,
    _require_cupy,
    evaluate,
    make_difference,
    make_input,
    make_intersection,
)
from subsetix_cupy import prolong_set
from subsetix_cupy.multilevel import prolong_field, restrict_field
from subsetix_cupy.morphology import full_interval_set

_SUBSET_KERNEL_CACHE: Dict[tuple[str, str], cp.RawKernel] = {}


def _interval_row_ids(interval_set: IntervalSet) -> cp.ndarray:
    return interval_set.interval_rows().astype(cp.int32, copy=False)


def _empty_interval_like(interval_set: IntervalSet) -> IntervalSet:
    if not isinstance(interval_set, IntervalSet):
        raise TypeError("interval_set must be an IntervalSet")
    cp_mod = _require_cupy()
    row_count = interval_set.row_count
    zero = cp_mod.zeros(0, dtype=cp_mod.int32)
    row_offsets = cp_mod.zeros(row_count + 1, dtype=cp_mod.int32)
    rows_copy = cp_mod.array(interval_set.rows, dtype=cp_mod.int32, copy=True)
    return IntervalSet(begin=zero, end=zero, row_offsets=row_offsets, rows=rows_copy)


def _interval_field_value_intervals(field: IntervalField, target) -> IntervalSet:
    cp_mod = _require_cupy()
    values = cp_mod.asarray(field.values)
    total_cells = int(values.size)
    interval_set = field.interval_set
    if total_cells == 0:
        return _empty_interval_like(interval_set)

    dtype_type = values.dtype.type
    matches = values == dtype_type(target)
    if not bool(cp_mod.any(matches)):
        return _empty_interval_like(interval_set)

    cell_indices = cp_mod.arange(total_cells, dtype=cp_mod.int32)
    offsets = field.interval_cell_offsets.astype(cp_mod.int32, copy=False)
    interval_indices = cp_mod.searchsorted(offsets[1:], cell_indices, side="right")
    local_pos = cell_indices - offsets[interval_indices]
    begin = interval_set.begin.astype(cp_mod.int32, copy=False)
    cols = begin[interval_indices] + local_pos
    interval_rows = interval_set.interval_rows().astype(cp_mod.int32, copy=False)
    rows_for_cells = interval_rows[interval_indices]

    prev_matches = cp_mod.zeros_like(matches, dtype=cp_mod.bool_)
    if total_cells > 1:
        prev_matches[1:] = matches[:-1]
    if offsets.size > 0:
        prev_matches[offsets[:-1]] = False
    start_mask = matches & (~prev_matches)

    next_matches = cp_mod.zeros_like(matches, dtype=cp_mod.bool_)
    if total_cells > 1:
        next_matches[:-1] = matches[1:]
    if offsets.size > 1:
        end_indices = offsets[1:] - 1
        next_matches[end_indices] = False
    end_mask = matches & (~next_matches)

    start_idx = cp_mod.where(start_mask)[0]
    if start_idx.size == 0:
        return _empty_interval_like(interval_set)
    end_idx = cp_mod.where(end_mask)[0]
    if end_idx.size != start_idx.size:
        raise RuntimeError("mismatched start/end counts when extracting value intervals")

    rows_index = interval_set.rows_index().astype(cp_mod.int32, copy=False)
    row_count = rows_index.size
    if row_count == 0:
        return _empty_interval_like(interval_set)
    row_positions = cp_mod.searchsorted(rows_index, rows_for_cells[start_idx], side="left")
    row_counts = cp_mod.bincount(row_positions, minlength=row_count).astype(cp_mod.int32, copy=False)
    row_offsets = cp_mod.empty(row_count + 1, dtype=cp_mod.int32)
    row_offsets[0] = 0
    if row_count > 0:
        cp_mod.cumsum(row_counts, dtype=cp_mod.int32, out=row_offsets[1:])

    begin_out = cols[start_idx].astype(cp_mod.int32, copy=False)
    end_out = (cols[end_idx] + 1).astype(cp_mod.int32, copy=False)
    rows_copy = cp_mod.array(interval_set.rows, dtype=cp_mod.int32, copy=True)

    return IntervalSet(begin=begin_out, end=end_out, row_offsets=row_offsets, rows=rows_copy)


def _get_interval_subset_kernel(mode: str, dtype: cp.dtype) -> cp.RawKernel:
    cp_mod = _require_cupy()
    dtype_obj = cp_mod.dtype(dtype)
    if dtype_obj == cp_mod.float32:
        type_name = "float"
    elif dtype_obj == cp_mod.float64:
        type_name = "double"
    elif dtype_obj == cp_mod.int8:
        type_name = "signed char"
    elif dtype_obj == cp_mod.int32:
        type_name = "int"
    elif dtype_obj == cp_mod.int64:
        type_name = "long long"
    elif dtype_obj == cp_mod.bool_:
        type_name = "bool"
    else:
        raise TypeError(f"interval subset kernels do not support dtype {dtype_obj}")
    key = (mode, dtype_obj.str)
    kernel = _SUBSET_KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel
    if mode not in {"gather", "scatter"}:
        raise ValueError("mode must be 'gather' or 'scatter'")

    if mode == "gather":
        func_name = "subsetix_interval_subset_gather"
        template = r"""
        extern "C" __global__
        void subsetix_interval_subset_gather(const int* __restrict__ subset_rows,
                                             const int* __restrict__ subset_begin,
                                             const int* __restrict__ subset_end,
                                             const int* __restrict__ subset_cell_offsets,
                                             const int* __restrict__ super_row_offsets,
                                             const int* __restrict__ super_begin,
                                             const int* __restrict__ super_end,
                                             const int* __restrict__ super_cell_offsets,
                                             const @TYPE@* __restrict__ super_values,
                                             @TYPE@* __restrict__ subset_values,
                                             int super_row_count)
        {
            int interval = blockIdx.x;
            int row = subset_rows[interval];
            if (row < 0 || row >= super_row_count) {
                return;
            }
            int start = subset_begin[interval];
            int stop = subset_end[interval];
            if (stop <= start) {
                return;
            }
            int dst_base = subset_cell_offsets[interval];
            int dst_offset = 0;

            int row_start = super_row_offsets[row];
            int row_stop = super_row_offsets[row + 1];
            if (row_start >= row_stop) {
                return;
            }
            int lo = row_start;
            int hi = row_stop - 1;
            int idx = row_start;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                int b = super_begin[mid];
                int e = super_end[mid];
                if (start < b) {
                    hi = mid - 1;
                } else if (start >= e) {
                    lo = mid + 1;
                } else {
                    idx = mid;
                    break;
                }
            }
            if (lo > hi) {
                idx = lo;
            }
            if (idx < row_start) {
                idx = row_start;
            }

            int current = start;
            while (current < stop && idx < row_stop) {
                int b = super_begin[idx];
                int e = super_end[idx];
                if (current < b) {
                    current = b;
                    continue;
                }
                int copy_end = stop < e ? stop : e;
                int length = copy_end - current;
                if (length > 0) {
                    int src_base = super_cell_offsets[idx] + (current - b);
                    for (int t = threadIdx.x; t < length; t += blockDim.x) {
                        subset_values[dst_base + dst_offset + t] = super_values[src_base + t];
                    }
                    dst_offset += length;
                    current = copy_end;
                }
                if (current >= e) {
                    idx += 1;
                }
                if (length == 0 && current < stop) {
                    current = current + 1;
                }
            }
        }
        """
    else:
        func_name = "subsetix_interval_subset_scatter"
        template = r"""
        extern "C" __global__
        void subsetix_interval_subset_scatter(const int* __restrict__ subset_rows,
                                              const int* __restrict__ subset_begin,
                                              const int* __restrict__ subset_end,
                                              const int* __restrict__ subset_cell_offsets,
                                              const int* __restrict__ super_row_offsets,
                                              const int* __restrict__ super_begin,
                                              const int* __restrict__ super_end,
                                              const int* __restrict__ super_cell_offsets,
                                              const @TYPE@* __restrict__ subset_values,
                                              @TYPE@* __restrict__ super_values,
                                              int super_row_count)
        {
            int interval = blockIdx.x;
            int row = subset_rows[interval];
            if (row < 0 || row >= super_row_count) {
                return;
            }
            int start = subset_begin[interval];
            int stop = subset_end[interval];
            if (stop <= start) {
                return;
            }
            int src_base = subset_cell_offsets[interval];
            int src_offset = 0;

            int row_start = super_row_offsets[row];
            int row_stop = super_row_offsets[row + 1];
            if (row_start >= row_stop) {
                return;
            }
            int lo = row_start;
            int hi = row_stop - 1;
            int idx = row_start;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                int b = super_begin[mid];
                int e = super_end[mid];
                if (start < b) {
                    hi = mid - 1;
                } else if (start >= e) {
                    lo = mid + 1;
                } else {
                    idx = mid;
                    break;
                }
            }
            if (lo > hi) {
                idx = lo;
            }
            if (idx < row_start) {
                idx = row_start;
            }

            int current = start;
            while (current < stop && idx < row_stop) {
                int b = super_begin[idx];
                int e = super_end[idx];
                if (current < b) {
                    current = b;
                    continue;
                }
                int copy_end = stop < e ? stop : e;
                int length = copy_end - current;
                if (length > 0) {
                    int dst_base = super_cell_offsets[idx] + (current - b);
                    for (int t = threadIdx.x; t < length; t += blockDim.x) {
                        super_values[dst_base + t] = subset_values[src_base + src_offset + t];
                    }
                    src_offset += length;
                    current = copy_end;
                }
                if (current >= e) {
                    idx += 1;
                }
                if (length == 0 && current < stop) {
                    current = current + 1;
                }
            }
        }
        """
    code = template.replace("@TYPE@", type_name)

    kernel = cp_mod.RawKernel(code, func_name, options=("--std=c++11",))
    _SUBSET_KERNEL_CACHE[key] = kernel
    return kernel


def _gather_subset_into(field: IntervalField, subset_field: IntervalField) -> None:
    cp_mod = _require_cupy()
    interval_count = int(subset_field.interval_set.begin.size)
    if interval_count == 0:
        return
    kernel = _get_interval_subset_kernel("gather", subset_field.values.dtype)
    block = 128
    grid = (interval_count,)
    subset_rows_actual = subset_field.interval_set.interval_rows().astype(cp_mod.int32, copy=False)
    if subset_rows_actual.size > 0:
        super_rows = field.interval_set.rows_index().astype(cp_mod.int32, copy=False)
        positions = cp_mod.searchsorted(super_rows, subset_rows_actual, side="left")
        if int(cp_mod.any(super_rows.take(positions) != subset_rows_actual).item()):
            raise ValueError("subset rows must be contained within the source field")
        subset_rows = positions.astype(cp_mod.int32, copy=False)
    else:
        subset_rows = subset_rows_actual
    subset_begin = subset_field.interval_set.begin.astype(cp_mod.int32, copy=False)
    subset_end = subset_field.interval_set.end.astype(cp_mod.int32, copy=False)
    subset_offsets = subset_field.interval_cell_offsets.astype(cp_mod.int32, copy=False)
    super_row_offsets = field.interval_set.row_offsets.astype(cp_mod.int32, copy=False)
    super_begin = field.interval_set.begin.astype(cp_mod.int32, copy=False)
    super_end = field.interval_set.end.astype(cp_mod.int32, copy=False)
    super_offsets = field.interval_cell_offsets.astype(cp_mod.int32, copy=False)
    kernel(
        grid,
        (block,),
        (
            subset_rows,
            subset_begin,
            subset_end,
            subset_offsets,
            super_row_offsets,
            super_begin,
            super_end,
            super_offsets,
            field.values,
            subset_field.values,
            np.int32(field.interval_set.row_count),
        ),
    )


def _scatter_subset_from(subset_field: IntervalField, field: IntervalField) -> None:
    cp_mod = _require_cupy()
    interval_count = int(subset_field.interval_set.begin.size)
    if interval_count == 0:
        return
    kernel = _get_interval_subset_kernel("scatter", subset_field.values.dtype)
    block = 128
    grid = (interval_count,)
    subset_rows_actual = subset_field.interval_set.interval_rows().astype(cp_mod.int32, copy=False)
    if subset_rows_actual.size > 0:
        super_rows = field.interval_set.rows_index().astype(cp_mod.int32, copy=False)
        positions = cp_mod.searchsorted(super_rows, subset_rows_actual, side="left")
        if int(cp_mod.any(super_rows.take(positions) != subset_rows_actual).item()):
            raise ValueError("subset rows must be contained within the destination field")
        subset_rows = positions.astype(cp_mod.int32, copy=False)
    else:
        subset_rows = subset_rows_actual
    subset_begin = subset_field.interval_set.begin.astype(cp_mod.int32, copy=False)
    subset_end = subset_field.interval_set.end.astype(cp_mod.int32, copy=False)
    subset_offsets = subset_field.interval_cell_offsets.astype(cp_mod.int32, copy=False)
    super_row_offsets = field.interval_set.row_offsets.astype(cp_mod.int32, copy=False)
    super_begin = field.interval_set.begin.astype(cp_mod.int32, copy=False)
    super_end = field.interval_set.end.astype(cp_mod.int32, copy=False)
    super_offsets = field.interval_cell_offsets.astype(cp_mod.int32, copy=False)
    kernel(
        grid,
        (block,),
        (
            subset_rows,
            subset_begin,
            subset_end,
            subset_offsets,
            super_row_offsets,
            super_begin,
            super_end,
            super_offsets,
            subset_field.values,
            field.values,
            np.int32(field.interval_set.row_count),
        ),
    )


def gather_interval_subset(field: IntervalField, subset: IntervalSet) -> IntervalField:
    """
    Extract a subset of ``field`` described by ``subset`` into a new IntervalField.
    """

    if not isinstance(field, IntervalField):
        raise TypeError("field must be an IntervalField")
    if not isinstance(subset, IntervalSet):
        raise TypeError("subset must be an IntervalSet")
    subset_field = create_interval_field(subset, fill_value=0.0, dtype=field.values.dtype)
    _gather_subset_into(field, subset_field)
    return subset_field


def scatter_interval_subset(field: IntervalField, subset_field: IntervalField) -> None:
    """
    Scatter values from ``subset_field`` back into ``field`` at matching coordinates.
    """

    if not isinstance(field, IntervalField):
        raise TypeError("field must be an IntervalField")
    if not isinstance(subset_field, IntervalField):
        raise TypeError("subset_field must be an IntervalField")
    _scatter_subset_from(subset_field, field)


def clone_interval_field(field: IntervalField) -> IntervalField:
    if not isinstance(field, IntervalField):
        raise TypeError("field must be an IntervalField")
    cp_mod = _require_cupy()
    return IntervalField(
        interval_set=field.interval_set,
        values=cp_mod.array(field.values, copy=True),
        interval_cell_offsets=field.interval_cell_offsets,
    )


class Action(IntEnum):
    COARSEN = -1
    KEEP = 0
    REFINE = 1


_FULL_ACTION_INTERVAL_CACHE: Dict[tuple[int, int], IntervalSet] = {}


def _full_action_interval_set(width: int, height: int) -> IntervalSet:
    key = (height, width)
    cached = _FULL_ACTION_INTERVAL_CACHE.get(key)
    if cached is None:
        cached = full_interval_set(width, height)
        _FULL_ACTION_INTERVAL_CACHE[key] = cached
    return cached


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
    def from_interval_set(
        cls,
        interval_set: IntervalSet,
        *,
        width: int,
        height: int,
        ratio: int,
        default: Action = Action.KEEP,
    ) -> "ActionField":
        if not isinstance(interval_set, IntervalSet):
            raise TypeError("interval_set must be an IntervalSet")
        cp_mod = _require_cupy()
        width_int = int(width)
        height_int = int(height)
        if height_int <= 0 or width_int <= 0:
            raise ValueError("width and height must be positive")
        if interval_set.row_count != height_int:
            raise ValueError("interval_set row count must match height")
        row_ids = interval_set.rows_index()
        if row_ids.size != height_int:
            raise ValueError("interval_set rows must cover the dense range [0, height)")
        expected_rows = cp_mod.arange(height_int, dtype=cp_mod.int32)
        if row_ids.size and not bool(cp_mod.all(row_ids == expected_rows)):
            raise ValueError("interval_set rows must be the dense range [0, height)")
        interval_field = create_interval_field(interval_set, fill_value=int(default), dtype=cp_mod.int8)
        expected_cells = width_int * height_int
        if int(interval_field.values.size) != expected_cells:
            raise ValueError("interval_set must cover exactly width * height cells")
        action = cls(field=interval_field, ratio=int(ratio), width=width_int, height=height_int)
        action._refine_set_cache = _empty_interval_like(interval_set)
        return action

    @classmethod
    def full_grid(cls, height: int, width: int, ratio: int, *, default: Action = Action.KEEP) -> "ActionField":
        height_int = int(height)
        width_int = int(width)
        if height_int <= 0 or width_int <= 0:
            raise ValueError("height and width must be positive")
        interval_set = _full_action_interval_set(width_int, height_int)
        return cls.from_interval_set(
            interval_set,
            width=width_int,
            height=height_int,
            ratio=ratio,
            default=default,
        )

    def _refresh_from_values(self) -> None:
        refine = _interval_field_value_intervals(self.field, int(Action.REFINE))
        self._refine_set_cache = refine
        self._fine_set_cache = None
        self._coarse_only_cache = None

    def set_from_interval_set(self, refine_set: IntervalSet) -> None:
        """Populate the action field directly from an IntervalSet."""

        if not isinstance(refine_set, IntervalSet):
            raise TypeError("refine_set must be an IntervalSet")
        domain = self.field.interval_set
        if refine_set.begin.size == 0:
            keep_value = self.field.values.dtype.type(int(Action.KEEP))
            self.field.values.fill(keep_value)
            self._refine_set_cache = _empty_interval_like(domain)
            self._fine_set_cache = None
            self._coarse_only_cache = None
            return

        domain_expr = make_input(domain)
        refine_expr = make_input(refine_set)
        outside = evaluate(make_difference(refine_expr, domain_expr))
        if outside.begin.size != 0:
            raise ValueError("refine_set must be contained within the action domain")
        aligned = evaluate(make_intersection(refine_expr, domain_expr))

        keep_value = self.field.values.dtype.type(int(Action.KEEP))
        self.field.values.fill(keep_value)
        subset_field = create_interval_field(aligned, fill_value=int(Action.REFINE), dtype=self.field.values.dtype)
        scatter_interval_subset(self.field, subset_field)

        self._refine_set_cache = aligned
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
def synchronize_interval_fields(
    coarse: IntervalField,
    fine: IntervalField,
    actions: ActionField,
    *,
    ratio: int,
    reducer: str = "mean",
    fill_fine_outside: bool = True,
    copy: bool = True,
) -> tuple[IntervalField, IntervalField]:
    """Synchronise interval-backed coarse/fine fields based on AMR actions."""

    if not isinstance(coarse, IntervalField):
        raise TypeError("coarse must be an IntervalField")
    if not isinstance(fine, IntervalField):
        raise TypeError("fine must be an IntervalField")
    if not isinstance(actions, ActionField):
        raise TypeError("actions must be an ActionField")
    if int(actions.ratio) != int(ratio):
        raise ValueError("ratio argument must match ActionField ratio")

    cp_mod = _require_cupy()
    if coarse.values.dtype != cp_mod.float32 or fine.values.dtype != cp_mod.float32:
        raise TypeError("coarse and fine interval fields must use float32 values")

    coarse_target = clone_interval_field(coarse) if copy else coarse
    fine_target = clone_interval_field(fine) if copy else fine

    restricted = restrict_field(fine, ratio, reducer=reducer)
    refine_set = actions.refine_set()
    if refine_set.begin.size != 0:
        restricted_subset = gather_interval_subset(restricted, refine_set)
        scatter_interval_subset(coarse_target, restricted_subset)

    if fill_fine_outside:
        coarse_only = actions.coarse_only_set()
        if coarse_only.begin.size != 0:
            coarse_subset = gather_interval_subset(coarse_target, coarse_only)
            fine_subset = prolong_field(coarse_subset, ratio)
            if fine_subset.values.size != 0:
                scatter_interval_subset(fine_target, fine_subset)

    return coarse_target, fine_target
