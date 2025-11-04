"""
Interval-aligned scalar fields for 2D interval sets.

An IntervalField stores one value per active cell described by an IntervalSet.
It keeps a flat value array alongside prefix offsets so callers can look up or
modify individual cells without expanding to a dense grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
