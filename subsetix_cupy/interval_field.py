"""
Interval-aligned scalar fields for 2D interval sets.

An IntervalField stores one value per active cell described by an IntervalSet.
It keeps a flat value array alongside prefix offsets so callers can look up or
modify individual cells without expanding to a dense grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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


_SCATTER_KERNEL_CACHE: Dict[str, Any] = {}


def _get_scatter_kernel(dtype: Any):
    cp = _require_cupy()
    key = cp.dtype(dtype).str
    kernel = _SCATTER_KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel
    if dtype == cp.float32:
        type_name = "float"
    elif dtype == cp.float64:
        type_name = "double"
    elif dtype == cp.int32:
        type_name = "int"
    elif dtype == cp.int64:
        type_name = "long long"
    else:
        raise TypeError(f"unsupported dtype {dtype} for interval_field_to_dense")
    code = f"""
    extern "C" __global__
    void scatter_interval_field(const int* __restrict__ row_ids,
                                const int* __restrict__ begin,
                                const int* __restrict__ end,
                                const int* __restrict__ cell_offsets,
                                const {type_name}* __restrict__ values,
                                {type_name}* __restrict__ out,
                                int width)
    {{
        int interval = blockIdx.x;
        int row = row_ids[interval];
        int start = begin[interval];
        int stop = end[interval];
        if (stop <= start) {{
            return;
        }}
        int base = cell_offsets[interval];
        int length = cell_offsets[interval + 1] - base;
        int row_offset = row * width;
        for (int idx = threadIdx.x; idx < length; idx += blockDim.x) {{
            out[row_offset + start + idx] = values[base + idx];
        }}
    }}
    """
    kernel = cp.RawKernel(code, "scatter_interval_field", options=("--std=c++11",))
    _SCATTER_KERNEL_CACHE[key] = kernel
    return kernel


def interval_field_to_dense(
    field: IntervalField,
    *,
    width: int,
    height: int,
    fill_value: float = 0.0,
    out: Any | None = None,
):
    """
    Materialise an IntervalField into a dense 2D array of shape (height, width).
    """

    cp = _require_cupy()
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    interval_set = field.interval_set
    begin = cp.asarray(interval_set.begin, dtype=cp.int32)
    end = cp.asarray(interval_set.end, dtype=cp.int32)
    cell_offsets = cp.asarray(field.interval_cell_offsets, dtype=cp.int32)
    values = cp.asarray(field.values, dtype=field.values.dtype)

    interval_count = int(begin.size)
    if cell_offsets.size != interval_count + 1:
        raise ValueError("interval_cell_offsets length mismatch")

    if out is None:
        out = cp.full((height, width), fill_value, dtype=values.dtype)
    else:
        if out.shape != (height, width):
            raise ValueError("out must have shape (height, width)")
        if out.dtype != values.dtype:
            raise TypeError("out dtype must match field values dtype")
        out.fill(values.dtype.type(fill_value))

    if interval_count == 0:
        return out

    row_ids = interval_set.interval_rows().astype(cp.int32, copy=False)
    if row_ids.size:
        max_row = int(row_ids.max().item())
        if max_row >= height:
            raise ValueError("height must exceed the largest active row index")
    if row_ids.size != interval_count:
        raise ValueError("rows/interval mismatch")

    kernel = _get_scatter_kernel(values.dtype)
    block = 128
    grid = (interval_count,)
    kernel(
        grid,
        (block,),
        (row_ids, begin, end, cell_offsets, values, out, cp.int32(width)),
    )
    return out


def _locate_interval(interval_set: IntervalSet, row: int, x: int) -> int | None:
    cp = _require_cupy()
    row_offsets = interval_set.row_offsets
    row_count = row_offsets.size - 1
    if row < 0 or row >= row_count:
        raise IndexError("row out of range")

    start = int(row_offsets[row].item())
    stop = int(row_offsets[row + 1].item())
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
