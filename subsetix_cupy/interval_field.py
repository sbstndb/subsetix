"""
Interval-aligned scalar fields for 2D interval sets.

An IntervalField stores one value per active cell described by an IntervalSet.
It keeps a flat value array alongside prefix offsets so callers can look up or
modify individual cells without expanding to a dense grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .expressions import IntervalSet, _require_cupy


@dataclass(frozen=True)
class IntervalField:
    """
    Scalar values attached to the cells described by an IntervalSet.

    Attributes
    ----------
    interval_set:
        Geometry describing the active cells.
    values:
        CuPy array containing one scalar per active cell, stored row by row.
    interval_cell_offsets:
        CuPy int32 array of length interval_count + 1 giving the start index of
        each interval inside ``values``.
    """

    interval_set: IntervalSet
    values: Any
    interval_cell_offsets: Any

    @property
    def cell_count(self) -> int:
        if self.interval_cell_offsets.size == 0:
            return 0
        return int(self.interval_cell_offsets[-1].item())


def create_interval_field(
    interval_set: IntervalSet,
    fill_value: float = 0.0,
    *,
    dtype: Any | None = None,
) -> IntervalField:
    """
    Allocate an IntervalField initialised with ``fill_value``.

    The field remains tied to ``interval_set``; callers should rebuild a field
    whenever the geometry changes (e.g., after a union/intersection).
    """

    cp = _require_cupy()
    begin = interval_set.begin
    end = interval_set.end
    interval_count = begin.size

    if interval_count != end.size:
        raise ValueError("IntervalSet begin/end size mismatch")

    lengths = end - begin
    if interval_count == 0:
        cell_offsets = cp.zeros(1, dtype=cp.int32)
    else:
        cell_offsets = cp.empty(interval_count + 1, dtype=cp.int32)
        cell_offsets[0] = 0
        cp.cumsum(lengths.astype(cp.int32, copy=False), dtype=cp.int32, out=cell_offsets[1:])

    total_cells = int(cell_offsets[-1].astype(cp.int64).item())

    if dtype is None:
        inferred = cp.asarray(fill_value)
        dtype = inferred.dtype
    else:
        dtype = cp.dtype(dtype)

    values = cp.full((total_cells,), fill_value, dtype=dtype)

    return IntervalField(
        interval_set=interval_set,
        values=values,
        interval_cell_offsets=cell_offsets,
    )


def _locate_interval(interval_set: IntervalSet, row: int, x: int) -> int | None:
    cp = _require_cupy()
    row_value = int(row)
    rows_index = interval_set.rows_index().astype(cp.int32, copy=False)
    if rows_index.size == 0:
        raise IndexError("row out of range")
    row_value_arr = cp.asarray([row_value], dtype=cp.int32)
    pos = int(cp.searchsorted(rows_index, row_value_arr, side="left").item())
    if pos >= rows_index.size or int(rows_index[pos].item()) != row_value:
        raise IndexError("row out of range")

    row_offsets = interval_set.row_offsets.astype(cp.int32, copy=False)

    start = int(row_offsets[pos].item())
    stop = int(row_offsets[pos + 1].item())
    if start == stop:
        return None

    begin = interval_set.begin[start:stop]
    end = interval_set.end[start:stop]

    matches = cp.nonzero((x >= begin) & (x < end))[0]
    if matches.size == 0:
        return None

    return start + int(matches[0].item())


def set_cell(field: IntervalField, row: int, x: int, value: float) -> bool:
    """
    Assign ``value`` to the cell at (row, x). Returns True if the cell exists.
    """

    interval_index = _locate_interval(field.interval_set, row, x)
    if interval_index is None:
        return False

    begin = field.interval_set.begin
    offsets = field.interval_cell_offsets
    cell_base = int(offsets[interval_index].item())
    local_index = x - int(begin[interval_index].item())
    value_index = cell_base + local_index
    if value_index < 0 or value_index >= field.values.size:
        return False
    field.values[value_index] = field.values.dtype.type(value)
    return True


def get_cell(field: IntervalField, row: int, x: int):
    """
    Retrieve the cell value at (row, x). Returns None if the cell is inactive.
    """

    interval_index = _locate_interval(field.interval_set, row, x)
    if interval_index is None:
        return None

    begin = field.interval_set.begin
    offsets = field.interval_cell_offsets
    cell_base = int(offsets[interval_index].item())
    local_index = x - int(begin[interval_index].item())
    value_index = cell_base + local_index
    if value_index < 0 or value_index >= field.values.size:
        return None
    return field.values[value_index]


_LOCATE_KERNEL = None


def _get_locate_kernel():
    global _LOCATE_KERNEL
    if _LOCATE_KERNEL is not None:
        return _LOCATE_KERNEL
    cp = _require_cupy()
    code = r"""
    extern "C" __global__
    void locate_cells(
        const int* __restrict__ row_ids,
        const int* __restrict__ row_offsets,
        const int* __restrict__ begin,
        const int* __restrict__ end,
        const int row_count,
        const int* __restrict__ query_rows,
        const int* __restrict__ query_cols,
        const int query_count,
        int* __restrict__ out_indices
    )
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= query_count) {
            return;
        }

        if (row_count <= 0) {
            out_indices[idx] = -1;
            return;
        }

        int q_row = query_rows[idx];
        int q_col = query_cols[idx];

        int left = 0;
        int right = row_count - 1;
        int row_pos = -1;
        while (left <= right) {
            int mid = (left + right) >> 1;
            int row_val = row_ids[mid];
            if (row_val == q_row) {
                row_pos = mid;
                break;
            }
            if (row_val < q_row) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        if (row_pos < 0) {
            out_indices[idx] = -1;
            return;
        }

        int start = row_offsets[row_pos];
        int stop = row_offsets[row_pos + 1];
        if (start >= stop) {
            out_indices[idx] = -1;
            return;
        }

        int lo = start;
        int hi = stop - 1;
        int found = -1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            int b = begin[mid];
            int e = end[mid];
            if (q_col < b) {
                hi = mid - 1;
            } else if (q_col >= e) {
                lo = mid + 1;
            } else {
                found = mid;
                break;
            }
        }

        out_indices[idx] = found;
    }
    """
    _LOCATE_KERNEL = cp.RawKernel(code, "locate_cells")
    return _LOCATE_KERNEL


def locate_interval_cells(field: IntervalField, rows: Any, cols: Any, *, out: Any | None = None):
    """
    Locate the global interval index for each (row, col) query.

    Returns an int32 CuPy array where -1 indicates an inactive cell.
    """

    cp = _require_cupy()
    rows_arr = cp.asarray(rows, dtype=cp.int32)
    cols_arr = cp.asarray(cols, dtype=cp.int32)
    if rows_arr.shape != cols_arr.shape:
        raise ValueError("rows and cols must share the same shape")

    flat_rows = rows_arr.reshape(-1)
    flat_cols = cols_arr.reshape(-1)
    query_count = int(flat_rows.size)

    if out is None:
        out_arr = cp.empty_like(flat_rows)
    else:
        out_arr = cp.asarray(out, dtype=cp.int32)
        if out_arr.size != query_count:
            raise ValueError("out array must match query size")
    if query_count == 0:
        return out_arr.reshape(rows_arr.shape)

    kernel = _get_locate_kernel()
    row_ids = field.interval_set.rows_index()
    row_offsets = field.interval_set.row_offsets
    begin = field.interval_set.begin
    end = field.interval_set.end
    row_count = int(row_ids.size)

    block = 128
    grid = (query_count + block - 1) // block
    kernel(
        (grid,),
        (block,),
        (
            row_ids,
            row_offsets,
            begin,
            end,
            np.int32(row_count),
            flat_rows,
            flat_cols,
            np.int32(query_count),
            out_arr,
        ),
    )
    return out_arr.reshape(rows_arr.shape)
