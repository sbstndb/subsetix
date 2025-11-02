"""
Circular (wrap-around) morphological operations on interval sets.

These helpers keep data sparse and operate directly on the compressed-row
representation used throughout the CuPy backend.
"""

from __future__ import annotations

import numpy as np

from .expressions import IntervalSet, _require_cupy, evaluate, make_difference, make_input
from .kernels import get_kernels


def _clone_interval_set(interval_set: IntervalSet) -> IntervalSet:
    cp = _require_cupy()
    return IntervalSet(
        begin=cp.array(interval_set.begin, dtype=cp.int32, copy=True),
        end=cp.array(interval_set.end, dtype=cp.int32, copy=True),
        row_offsets=cp.array(interval_set.row_offsets, dtype=cp.int32, copy=True),
    )


def _row_ids(row_offsets) -> "object":
    cp = _require_cupy()
    row_count = int(row_offsets.size - 1)
    if row_count <= 0:
        return cp.zeros(0, dtype=cp.int32)
    total = int(row_offsets[-1].item())
    if total == 0:
        return cp.zeros(0, dtype=cp.int32)
    positions = cp.arange(total, dtype=cp.int32)
    return cp.searchsorted(row_offsets[1:], positions, side="right").astype(cp.int32, copy=False)


def _build_interval_set_from_rows(rows, begin, end, row_count: int) -> IntervalSet:
    cp = _require_cupy()
    row_count = int(row_count)
    if begin.size == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        offsets = cp.zeros(row_count + 1, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets)

    keys = cp.vstack((begin, rows))
    order = cp.lexsort(keys)
    rows_sorted = rows[order]
    begin_sorted = begin[order]
    end_sorted = end[order]

    counts = cp.bincount(rows_sorted, minlength=row_count).astype(cp.int32, copy=False)
    row_offsets_raw = cp.empty(row_count + 1, dtype=cp.int32)
    row_offsets_raw[0] = 0
    if row_count > 0:
        cp.cumsum(counts, dtype=cp.int32, out=row_offsets_raw[1:])

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
        return IntervalSet(begin=zero, end=zero, row_offsets=row_offsets)

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

    return IntervalSet(begin=out_begin, end=out_end, row_offsets=row_offsets)


def dilate_interval_set(
    interval_set: IntervalSet,
    halo_x: int = 1,
    halo_y: int = 1,
    *,
    width: int,
    height: int | None = None,
    bc: str = "clamp",
) -> IntervalSet:
    """
    Dilate an interval set with optional wrap-around behaviour.

    Parameters
    ----------
    interval_set : IntervalSet
        Geometry to dilate.
    halo_x, halo_y : int
        Radius along x (columns) and y (rows).
    width : int
        Domain width.
    height : int, optional
        Domain height. Defaults to the row count of ``interval_set``.
    bc : {"clamp", "wrap"}
        Boundary condition along both axes. ``"wrap"`` performs toroidal wrap;
        ``"clamp"`` clips at the domain edges.
    """

    if bc not in {"clamp", "wrap"}:
        raise ValueError("bc must be 'clamp' or 'wrap'")
    if halo_x < 0 or halo_y < 0:
        raise ValueError("halo_x and halo_y must be non-negative")

    cp = _require_cupy()
    row_offsets = interval_set.row_offsets.astype(cp.int32, copy=False)
    row_count = int(row_offsets.size - 1)
    if height is None:
        height = row_count
    if row_count != height:
        raise ValueError("interval_set row count must match provided height")

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    if interval_set.begin.size == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        offsets = cp.zeros(height + 1, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets)

    if halo_x == 0 and halo_y == 0:
        return _clone_interval_set(interval_set)

    base_rows = _row_ids(row_offsets)
    begin = interval_set.begin.astype(cp.int32, copy=True)
    end = interval_set.end.astype(cp.int32, copy=True)

    rows_acc = []
    begin_acc = []
    end_acc = []
    shifts_y = range(-halo_y, halo_y + 1)
    shifts_x = range(-halo_x, halo_x + 1)

    for dy in shifts_y:
        if bc == "wrap":
            rows_y = (base_rows + dy) % height
            valid_row_mask = cp.ones(rows_y.size, dtype=cp.bool_)
        else:
            row_ids = base_rows + dy
            valid_row_mask = (row_ids >= 0) & (row_ids < height)
            rows_y = row_ids % height
        for dx in shifts_x:
            if bc == "wrap":
                begin_shift = (begin + dx) % width
                end_shift = (end + dx) % width
                wrap_mask = begin_shift > end_shift

                mask_nowrap = valid_row_mask & ~wrap_mask
                if int(mask_nowrap.any()):
                    rows_nowrap = rows_y[mask_nowrap]
                    begin_nowrap = begin_shift[mask_nowrap]
                    end_nowrap = end_shift[mask_nowrap]
                    rows_acc.append(rows_nowrap)
                    begin_acc.append(begin_nowrap)
                    end_acc.append(end_nowrap)

                mask_wrap = valid_row_mask & wrap_mask
                if int(mask_wrap.any()):
                    rows_wrap = rows_y[mask_wrap]
                    begin_wrap = begin_shift[mask_wrap]
                    end_wrap = end_shift[mask_wrap]
                    repeat_rows = cp.repeat(rows_wrap, 2)
                    zeros = cp.zeros(begin_wrap.size, dtype=cp.int32)
                    begin_wrap_concat = cp.stack((begin_wrap, zeros), axis=1).reshape(-1)
                    end_wrap_concat = cp.stack(
                        (cp.full(begin_wrap.size, width, dtype=cp.int32), end_wrap), axis=1
                    ).reshape(-1)
                    rows_acc.append(repeat_rows)
                    begin_acc.append(begin_wrap_concat)
                    end_acc.append(end_wrap_concat)
            else:
                if int(valid_row_mask.any()) == 0:
                    continue
                rows_valid = rows_y[valid_row_mask]
                begin_shift = begin[valid_row_mask] + dx
                end_shift = end[valid_row_mask] + dx
                begin_clamp = cp.clip(begin_shift, 0, width)
                end_clamp = cp.clip(end_shift, 0, width)
                length_mask = end_clamp > begin_clamp
                if int(length_mask.any()) == 0:
                    continue
                rows_acc.append(rows_valid[length_mask])
                begin_acc.append(begin_clamp[length_mask])
                end_acc.append(end_clamp[length_mask])

    if not rows_acc:
        zero = cp.zeros(0, dtype=cp.int32)
        offsets = cp.zeros(height + 1, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets)

    rows_all = cp.concatenate(rows_acc)
    begin_all = cp.concatenate(begin_acc)
    end_all = cp.concatenate(end_acc)

    dilated = _build_interval_set_from_rows(rows_all, begin_all, end_all, height)
    return _clone_interval_set(dilated)


def ghost_zones(
    interval_set: IntervalSet,
    halo_x: int = 1,
    halo_y: int = 1,
    *,
    width: int,
    height: int | None = None,
    bc: str = "clamp",
) -> IntervalSet:
    """
    Ghost regions defined as dilation minus the original set.
    """

    dilated = dilate_interval_set(
        interval_set,
        halo_x=halo_x,
        halo_y=halo_y,
        width=width,
        height=height,
        bc=bc,
    )
    ghost_expr = make_difference(make_input(dilated), make_input(interval_set))
    ghost = evaluate(ghost_expr)
    return _clone_interval_set(ghost)


def full_interval_set(width: int, height: int) -> IntervalSet:
    """
    Build an IntervalSet covering the full rectangular domain.
    """

    cp = _require_cupy()
    if height <= 0 or width <= 0:
        offsets = cp.zeros(1, dtype=cp.int32)
        zero = cp.zeros(0, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets)
    begin = cp.zeros(height, dtype=cp.int32)
    end = cp.full(height, width, dtype=cp.int32)
    row_offsets = cp.arange(height + 1, dtype=cp.int32)
    return IntervalSet(begin=begin, end=end, row_offsets=row_offsets)


def erode_interval_set(
    interval_set: IntervalSet,
    halo_x: int = 1,
    halo_y: int = 1,
    *,
    width: int,
    height: int | None = None,
    bc: str = "clamp",
) -> IntervalSet:
    """
    Erosion implemented via complement + dilation.
    """

    universe = full_interval_set(width, height if height is not None else interval_set.row_offsets.size - 1)
    complement_expr = make_difference(make_input(universe), make_input(interval_set))
    complement = evaluate(complement_expr)
    dilated_complement = dilate_interval_set(
        complement,
        halo_x=halo_x,
        halo_y=halo_y,
        width=width,
        height=height,
        bc=bc,
    )
    eroded_expr = make_difference(make_input(universe), make_input(dilated_complement))
    eroded = evaluate(eroded_expr)
    return _clone_interval_set(eroded)


__all__ = [
    "dilate_interval_set",
    "ghost_zones",
    "erode_interval_set",
    "full_interval_set",
]
