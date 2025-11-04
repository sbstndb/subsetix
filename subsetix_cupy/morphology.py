"""
Morphological operations on interval sets using sparse row storage.

These helpers keep data sparse and operate directly on the compressed-row
representation used throughout the CuPy backend.
"""

from __future__ import annotations

import numpy as np
from numbers import Integral

from .expressions import CuPyWorkspace, IntervalSet, _require_cupy, evaluate, make_difference, make_input, make_intersection
from .kernels import get_kernels


def _row_ids(interval_set: IntervalSet) -> "object":
    return interval_set.interval_rows()


def _build_interval_set_from_rows(rows, begin, end, *, rows_hint=None) -> IntervalSet:
    cp = _require_cupy()
    if begin.size == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        if rows_hint is not None:
            rows_out = cp.unique(cp.asarray(rows_hint, dtype=cp.int32))
            offsets = cp.zeros(rows_out.size + 1, dtype=cp.int32)
        else:
            offsets = cp.zeros(1, dtype=cp.int32)
            rows_out = cp.zeros(0, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=rows_out)

    rows = cp.asarray(rows, dtype=cp.int32)
    begin = cp.asarray(begin, dtype=cp.int32)
    end = cp.asarray(end, dtype=cp.int32)

    keys = cp.vstack((begin, rows))
    order = cp.lexsort(keys)
    rows_sorted = rows[order]
    begin_sorted = begin[order]
    end_sorted = end[order]

    unique_rows, inverse = cp.unique(rows_sorted, return_inverse=True)
    if rows_hint is not None:
        rows_hint_arr = cp.unique(cp.asarray(rows_hint, dtype=cp.int32))
        rows_out = cp.union1d(rows_hint_arr, unique_rows)
    else:
        rows_out = unique_rows

    row_count = int(rows_out.size)
    counts_full = cp.zeros(row_count, dtype=cp.int32)
    if unique_rows.size:
        counts_unique = cp.bincount(inverse, minlength=unique_rows.size).astype(cp.int32, copy=False)
        positions = cp.searchsorted(rows_out, unique_rows)
        counts_full[positions] = counts_unique

    row_offsets_raw = cp.empty(row_count + 1, dtype=cp.int32)
    row_offsets_raw[0] = 0
    if row_count > 0:
        cp.cumsum(counts_full, dtype=cp.int32, out=row_offsets_raw[1:])

    merge_count_kernel = get_kernels(cp)[3]
    merge_write_kernel = get_kernels(cp)[4]

    block = 128
    grid = (row_count + block - 1) // block if row_count > 0 else 1
    counts_out = cp.empty(row_count, dtype=cp.int32)

    merge_count_kernel(
        (grid,),
        (block,),
        (
            begin_sorted,
            end_sorted,
            row_offsets_raw,
            np.int32(row_count),
            counts_out,
        ),
    )

    row_offsets = cp.empty(row_count + 1, dtype=cp.int32)
    row_offsets[0] = 0
    if row_count > 0:
        cp.cumsum(counts_out, dtype=cp.int32, out=row_offsets[1:])
    total = int(row_offsets[-1].item()) if row_count > 0 else 0
    if total == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=row_offsets, rows=rows_out.astype(cp.int32, copy=False))

    out_begin = cp.empty(total, dtype=cp.int32)
    out_end = cp.empty(total, dtype=cp.int32)

    merge_write_kernel(
        (grid,),
        (block,),
        (
            begin_sorted,
            end_sorted,
            row_offsets_raw,
            row_offsets,
            np.int32(row_count),
            out_begin,
            out_end,
        ),
    )

    return IntervalSet(begin=out_begin, end=out_end, row_offsets=row_offsets, rows=rows_out.astype(cp.int32, copy=False))


def _compact_interval_set(interval_set: IntervalSet) -> IntervalSet:
    rows = interval_set.interval_rows()
    return _build_interval_set_from_rows(rows, interval_set.begin, interval_set.end)


def translate_interval_set(interval_set: IntervalSet, dx: int = 0, dy: int = 0) -> IntervalSet:
    """
    Translate an interval set by shifting every interval horizontally by ``dx`` and every row by ``dy``.
    """

    if not isinstance(dx, Integral) or not isinstance(dy, Integral):
        raise TypeError("dx and dy must be integers")

    dx_int = int(dx)
    dy_int = int(dy)
    if dx_int == 0 and dy_int == 0:
        return interval_set.copy()

    cp = _require_cupy()
    begin = cp.array(interval_set.begin, dtype=cp.int32, copy=True)
    end = cp.array(interval_set.end, dtype=cp.int32, copy=True)
    if dx_int:
        begin += dx_int
        end += dx_int

    row_offsets = cp.array(interval_set.row_offsets, dtype=cp.int32, copy=True)

    rows_out = cp.array(interval_set.rows, dtype=cp.int32, copy=True)
    if dy_int and rows_out.size > 0:
        rows_out += dy_int
    return IntervalSet(begin=begin, end=end, row_offsets=row_offsets, rows=rows_out)


def interior_for_direction(
    interval_set: IntervalSet,
    dx: int,
    dy: int,
    *,
    workspace: CuPyWorkspace | None = None,
) -> IntervalSet:
    """
    Return the subset of ``interval_set`` whose neighbour at offset ``(dx, dy)`` also belongs to the set.
    """

    if not isinstance(dx, Integral) or not isinstance(dy, Integral):
        raise TypeError("dx and dy must be integers")
    if dx == 0 and dy == 0:
        raise ValueError("direction (dx, dy) must be non-zero")

    neighbour = translate_interval_set(interval_set, dx=-int(dx), dy=-int(dy))
    interior_expr = make_intersection(make_input(interval_set), make_input(neighbour))
    interior = evaluate(interior_expr, workspace=workspace)
    compact = _compact_interval_set(interior)
    return compact.copy()


def boundary_for_direction(
    interval_set: IntervalSet,
    dx: int,
    dy: int,
    *,
    workspace: CuPyWorkspace | None = None,
) -> IntervalSet:
    """
    Return the subset of ``interval_set`` lacking a neighbour at offset ``(dx, dy)``.
    """

    if not isinstance(dx, Integral) or not isinstance(dy, Integral):
        raise TypeError("dx and dy must be integers")
    if dx == 0 and dy == 0:
        raise ValueError("direction (dx, dy) must be non-zero")

    interior = interior_for_direction(interval_set, dx=dx, dy=dy, workspace=workspace)
    boundary_expr = make_difference(make_input(interval_set), make_input(interior))
    boundary = evaluate(boundary_expr, workspace=workspace)
    compact = _compact_interval_set(boundary)
    return compact.copy()


def boundary_layer(
    interval_set: IntervalSet,
    *,
    workspace: CuPyWorkspace | None = None,
) -> IntervalSet:
    """
    Return the Von Neumann boundary of ``interval_set`` (cells missing at least one axial neighbour).
    """

    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    interior = interval_set
    for dx, dy in directions:
        neighbour = translate_interval_set(interval_set, dx=-dx, dy=-dy)
        interior_expr = make_intersection(make_input(interior), make_input(neighbour))
        interior = evaluate(interior_expr, workspace=workspace)
    interior_compact = _compact_interval_set(interior)
    boundary_expr = make_difference(make_input(interval_set), make_input(interior_compact))
    boundary = evaluate(boundary_expr, workspace=workspace)
    compact = _compact_interval_set(boundary)
    return compact.copy()


def _dilate_interval_set_unbounded(
    interval_set: IntervalSet,
    *,
    halo_x: int,
    halo_y: int,
) -> IntervalSet:
    cp = _require_cupy()
    if halo_x == 0 and halo_y == 0:
        return interval_set.copy()

    begin = interval_set.begin.astype(cp.int32, copy=True)
    end = interval_set.end.astype(cp.int32, copy=True)
    rows_per_interval = interval_set.interval_rows()

    if halo_x > 0:
        begin = begin - halo_x
        end = end + halo_x

    if rows_per_interval.size == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        row_count = interval_set.row_count
        offsets = cp.zeros(row_count + 1, dtype=cp.int32)
        rows_arr = interval_set.rows_index() if row_count > 0 else cp.zeros(0, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=rows_arr)

    if halo_y == 0:
        rows_hint = interval_set.rows_index()
        dilated = _build_interval_set_from_rows(rows_per_interval, begin, end, rows_hint=rows_hint)
        return dilated.copy()

    shifts = cp.arange(-halo_y, halo_y + 1, dtype=cp.int32)
    rows_all = rows_per_interval[:, None] + shifts[None, :]
    rows_all = rows_all.reshape(-1)
    begin_all = cp.repeat(begin, shifts.size)
    end_all = cp.repeat(end, shifts.size)

    dilated = _build_interval_set_from_rows(rows_all, begin_all, end_all)
    return dilated.copy()


def dilate_interval_set(
    interval_set: IntervalSet,
    halo_x: int = 1,
    halo_y: int = 1,
) -> IntervalSet:
    """
    Dilate an interval set by extending every interval with a Manhattan halo.

    The dilation is unbounded; clip the result with whatever bounds you require
    after the call.
    """

    if halo_x < 0 or halo_y < 0:
        raise ValueError("halo_x and halo_y must be non-negative")
    return _dilate_interval_set_unbounded(interval_set, halo_x=halo_x, halo_y=halo_y)


def clip_interval_set(
    interval_set: IntervalSet,
    *,
    width: int,
    height: int,
) -> IntervalSet:
    """
    Clip ``interval_set`` to a rectangular domain and return a dense-row layout.

    Empty rows inside the domain are preserved so that the result uses the
    standard ``[0, height)`` row indexing expected by dense consumers.
    """

    cp = _require_cupy()
    if width <= 0:
        raise ValueError("width must be positive")
    if height < 0:
        raise ValueError("height must be non-negative")

    domain = full_interval_set(width, height)
    clipped = evaluate(make_intersection(make_input(domain), make_input(interval_set)))

    if clipped.begin.size == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        offsets = cp.zeros(height + 1, dtype=cp.int32)
        rows = cp.arange(height, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=rows)

    rows = clipped.interval_rows()
    if rows.size == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        offsets = cp.zeros(height + 1, dtype=cp.int32)
        rows = cp.arange(height, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=rows)

    mask = (rows >= 0) & (rows < height)
    if not bool(mask.all()):
        rows = rows[mask]
        begin = clipped.begin[mask]
        end = clipped.end[mask]
    else:
        begin = clipped.begin
        end = clipped.end

    if rows.size == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        offsets = cp.zeros(height + 1, dtype=cp.int32)
        rows = cp.arange(height, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=rows)

    rows_hint = cp.arange(height, dtype=cp.int32)
    return _build_interval_set_from_rows(rows, begin, end, rows_hint=rows_hint)


def ghost_zones(
    interval_set: IntervalSet,
    halo_x: int = 1,
    halo_y: int = 1,
    *,
    width: int,
    height: int | None = None,
) -> IntervalSet:
    """
    Ghost regions defined as dilation minus the original set, clipped to ``width``×``height``.
    """

    if height is None:
        height = interval_set.row_offsets.size - 1
    dilated = dilate_interval_set(interval_set, halo_x=halo_x, halo_y=halo_y)
    clipped = clip_interval_set(dilated, width=width, height=height)
    base = clip_interval_set(interval_set, width=width, height=height)
    ghost_expr = make_difference(make_input(clipped), make_input(base))
    ghost = evaluate(ghost_expr)
    return ghost.copy()


def full_interval_set(width: int, height: int) -> IntervalSet:
    """
    Build an IntervalSet covering the full rectangular domain.
    """

    cp = _require_cupy()
    if height <= 0 or width <= 0:
        offsets = cp.zeros(1, dtype=cp.int32)
        zero = cp.zeros(0, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=zero)
    begin = cp.zeros(height, dtype=cp.int32)
    end = cp.full(height, width, dtype=cp.int32)
    row_offsets = cp.arange(height + 1, dtype=cp.int32)
    rows = cp.arange(height, dtype=cp.int32)
    return IntervalSet(begin=begin, end=end, row_offsets=row_offsets, rows=rows)


def erode_interval_set(
    interval_set: IntervalSet,
    halo_x: int = 1,
    halo_y: int = 1,
    *,
    width: int,
    height: int | None = None,
) -> IntervalSet:
    """
    Erosion implemented via complement + dilation with zero exterior values.
    """

    if height is None:
        height = interval_set.row_offsets.size - 1
    if width <= 0 or height < 0:
        raise ValueError("width must be positive and height must be non-negative")

    base = clip_interval_set(interval_set, width=width, height=height)
    universe = full_interval_set(width, height)
    complement_expr = make_difference(make_input(universe), make_input(base))
    complement = evaluate(complement_expr)
    dilated_complement = dilate_interval_set(complement, halo_x=halo_x, halo_y=halo_y)
    dilated_clipped = clip_interval_set(dilated_complement, width=width, height=height)
    eroded_expr = make_difference(make_input(universe), make_input(dilated_clipped))
    eroded = evaluate(eroded_expr)
    return eroded.copy()


__all__ = [
    "translate_interval_set",
    "interior_for_direction",
    "boundary_for_direction",
    "boundary_layer",
    "dilate_interval_set",
    "ghost_zones",
    "erode_interval_set",
    "full_interval_set",
    "clip_interval_set",
]
