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
        rows=cp.array(interval_set.rows, dtype=cp.int32, copy=True) if interval_set.rows is not None else None,
    )


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


def _dilate_interval_set_horizontal_clamp(
    interval_set: IntervalSet,
    *,
    halo_x: int,
    width: int,
    row_count: int,
) -> IntervalSet:
    cp = _require_cupy()

    if halo_x == 0:
        return _clone_interval_set(interval_set)

    base_begin = interval_set.begin.astype(cp.int32, copy=True)
    base_end = interval_set.end.astype(cp.int32, copy=True)
    row_offsets = interval_set.row_offsets.astype(cp.int32, copy=False)
    row_ids = _row_ids(interval_set)
    rows_layout = interval_set.rows_index()
    if rows_layout.size != row_count:
        raise ValueError("interval_set rows mismatch with provided row_count")
    if row_count > 0:
        dense_rows = cp.arange(row_count, dtype=cp.int32)
        if not bool(cp.all(rows_layout == dense_rows)):
            raise ValueError("clamp dilation requires dense row indexing")
    else:
        dense_rows = cp.zeros(0, dtype=cp.int32)

    begin_expanded = cp.maximum(base_begin - halo_x, 0)
    end_expanded = cp.minimum(base_end + halo_x, width)
    valid_mask = end_expanded > begin_expanded

    if int(valid_mask.sum()) == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        offsets = cp.zeros(row_count + 1, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=dense_rows)

    rows_valid = row_ids[valid_mask]
    begin_valid = begin_expanded[valid_mask]
    end_valid = end_expanded[valid_mask]

    if row_count == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        offsets = cp.zeros(1, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=dense_rows)

    counts = (
        cp.bincount(rows_valid, minlength=row_count).astype(cp.int32, copy=False)
        if rows_valid.size
        else cp.zeros(row_count, dtype=cp.int32)
    )

    row_offsets_valid = cp.empty(row_count + 1, dtype=cp.int32)
    row_offsets_valid[0] = 0
    if row_count > 0:
        cp.cumsum(counts, dtype=cp.int32, out=row_offsets_valid[1:])

    total = int(row_offsets_valid[-1].item()) if row_count > 0 else 0
    if total == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=row_offsets_valid, rows=dense_rows)

    kernels = get_kernels(cp)
    merge_count_kernel = kernels[3]
    merge_write_kernel = kernels[4]

    block = 128
    grid = (row_count + block - 1) // block if row_count > 0 else 1

    counts_out = cp.empty(row_count, dtype=cp.int32)
    merge_count_kernel(
        (grid,),
        (block,),
        (
            begin_valid,
            end_valid,
            row_offsets_valid,
            np.int32(row_count),
            counts_out,
        ),
    )

    out_offsets = cp.empty(row_count + 1, dtype=cp.int32)
    out_offsets[0] = 0
    if row_count > 0:
        cp.cumsum(counts_out, dtype=cp.int32, out=out_offsets[1:])

    total_out = int(out_offsets[-1].item()) if row_count > 0 else 0
    if total_out == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=out_offsets, rows=dense_rows)

    out_begin = cp.empty(total_out, dtype=cp.int32)
    out_end = cp.empty(total_out, dtype=cp.int32)

    merge_write_kernel(
        (grid,),
        (block,),
        (
            begin_valid,
            end_valid,
            row_offsets_valid,
            out_offsets,
            np.int32(row_count),
            out_begin,
            out_end,
        ),
    )

    return IntervalSet(begin=out_begin, end=out_end, row_offsets=out_offsets, rows=dense_rows)


def _dilate_interval_set_generic(
    interval_set: IntervalSet,
    *,
    halo_x: int,
    halo_y: int,
    width: int,
    height: int,
    bc: str,
) -> IntervalSet:
    cp = _require_cupy()

    row_offsets = interval_set.row_offsets.astype(cp.int32, copy=False)
    row_count = int(row_offsets.size - 1)
    rows_layout = interval_set.rows_index()
    if rows_layout.size != row_count:
        raise ValueError("interval_set rows mismatch with row_offsets")
    if row_count != height:
        raise ValueError("interval_set row count must match provided height")
    if row_count > 0 and not bool(cp.all(rows_layout == cp.arange(row_count, dtype=cp.int32))):
        raise ValueError("clamp/wrap dilation requires dense row indexing")

    if interval_set.begin.size == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        offsets = cp.zeros(height + 1, dtype=cp.int32)
        rows_dense = cp.arange(height, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=rows_dense)

    base_rows = _row_ids(interval_set)
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
        rows_dense = cp.arange(height, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=rows_dense)

    rows_all = cp.concatenate(rows_acc)
    begin_all = cp.concatenate(begin_acc)
    end_all = cp.concatenate(end_acc)

    rows_hint = cp.arange(height, dtype=cp.int32)
    dilated = _build_interval_set_from_rows(rows_all, begin_all, end_all, rows_hint=rows_hint)
    return _clone_interval_set(dilated)


def _dilate_interval_set_vertical_clamp(
    interval_set: IntervalSet,
    *,
    halo_y: int,
    width: int,
    height: int,
) -> IntervalSet:
    cp = _require_cupy()

    if halo_y == 0:
        return _clone_interval_set(interval_set)

    max_neighbors = 2 * halo_y + 1
    if max_neighbors > 64:
        return _dilate_interval_set_generic(
            interval_set,
            halo_x=0,
            halo_y=halo_y,
            width=width,
            height=height,
            bc="clamp",
        )

    row_offsets = interval_set.row_offsets.astype(cp.int32, copy=False)
    row_count = int(row_offsets.size - 1)
    rows_layout = interval_set.rows_index()
    if row_count != height:
        raise ValueError("interval_set row count must match provided height")
    if rows_layout.size != row_count:
        raise ValueError("interval_set rows mismatch with provided height")
    if row_count > 0 and not bool(cp.all(rows_layout == cp.arange(row_count, dtype=cp.int32))):
        raise ValueError("vertical clamp requires dense row indexing")

    if row_count == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        offsets = cp.zeros(1, dtype=cp.int32)
        rows_dense = cp.zeros(0, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=rows_dense)

    begin = interval_set.begin.astype(cp.int32, copy=False)
    end = interval_set.end.astype(cp.int32, copy=False)

    kernels = get_kernels(cp)
    vertical_count_kernel = kernels[9]
    vertical_write_kernel = kernels[10]

    block = 128
    grid = (row_count + block - 1) // block if row_count > 0 else 1

    counts = cp.empty(row_count, dtype=cp.int32)
    vertical_count_kernel(
        (grid,),
        (block,),
        (
            begin,
            end,
            row_offsets,
            np.int32(row_count),
            np.int32(halo_y),
            counts,
        ),
    )

    out_offsets = cp.empty(row_count + 1, dtype=cp.int32)
    out_offsets[0] = 0
    if row_count > 0:
        cp.cumsum(counts, dtype=cp.int32, out=out_offsets[1:])

    total = int(out_offsets[-1].item()) if row_count > 0 else 0
    if total == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        rows_dense = cp.arange(row_count, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=out_offsets, rows=rows_dense)

    out_begin = cp.empty(total, dtype=cp.int32)
    out_end = cp.empty(total, dtype=cp.int32)

    vertical_write_kernel(
        (grid,),
        (block,),
        (
            begin,
            end,
            row_offsets,
            out_offsets,
            np.int32(row_count),
            np.int32(halo_y),
            out_begin,
            out_end,
        ),
    )

    rows_dense = cp.arange(row_count, dtype=cp.int32)
    return IntervalSet(begin=out_begin, end=out_end, row_offsets=out_offsets, rows=rows_dense)


def _dilate_interval_set_unbounded(
    interval_set: IntervalSet,
    *,
    halo_x: int,
    halo_y: int,
) -> IntervalSet:
    cp = _require_cupy()
    if halo_x == 0 and halo_y == 0:
        return _clone_interval_set(interval_set)

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
        return _clone_interval_set(dilated)

    shifts = cp.arange(-halo_y, halo_y + 1, dtype=cp.int32)
    rows_all = rows_per_interval[:, None] + shifts[None, :]
    rows_all = rows_all.reshape(-1)
    begin_all = cp.repeat(begin, shifts.size)
    end_all = cp.repeat(end, shifts.size)

    dilated = _build_interval_set_from_rows(rows_all, begin_all, end_all)
    return _clone_interval_set(dilated)


def dilate_interval_set(
    interval_set: IntervalSet,
    halo_x: int = 1,
    halo_y: int = 1,
    *,
    width: int | None = None,
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
    bc : {"clamp", "wrap", "none"}
        Boundary condition. ``"wrap"`` performs toroidal wrap, ``"clamp"`` clips
        at domain edges, and ``"none"`` performs an unbounded dilation (no domain
        required).
    """

    if bc not in {"clamp", "wrap", "none"}:
        raise ValueError("bc must be 'clamp', 'wrap', or 'none'")
    if halo_x < 0 or halo_y < 0:
        raise ValueError("halo_x and halo_y must be non-negative")

    if bc == "none":
        if width is not None or height is not None:
            raise ValueError("width/height must be omitted when bc='none'")
        return _dilate_interval_set_unbounded(interval_set, halo_x=halo_x, halo_y=halo_y)

    if width is None:
        raise ValueError("width must be provided for clamp/wrap dilation")

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

    if bc == "clamp":
        current = interval_set
        if halo_x > 0:
            current = _dilate_interval_set_horizontal_clamp(
                current,
                halo_x=halo_x,
                width=width,
                row_count=row_count,
            )
        if halo_y > 0:
            current = _dilate_interval_set_vertical_clamp(
                current,
                halo_y=halo_y,
                width=width,
                height=height,
            )
        return current

    return _dilate_interval_set_generic(
        interval_set,
        halo_x=halo_x,
        halo_y=halo_y,
        width=width,
        height=height,
        bc=bc,
    )


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
