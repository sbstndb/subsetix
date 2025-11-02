"""
Minimal multi-level container for 2D interval sets (AMR-style).

This module provides a light abstraction to manage a hierarchy of IntervalSet
objects across refinement levels. It does not yet implement cross-level
operations (prolongation/restriction); those can be added on top later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .expressions import IntervalSet, _require_cupy, evaluate, make_difference, make_input
from .interval_field import IntervalField, create_interval_field
from .kernels import get_kernels


def _default_ratios(n_levels: int, base: int) -> List[int]:
    if n_levels <= 0:
        raise ValueError("n_levels must be > 0")
    if base < 2:
        raise ValueError("base ratio must be >= 2")
    r = [1]
    for _ in range(1, n_levels):
        r.append(r[-1] * base)
    return r


def _validate_ratios(ratios: List[int]) -> None:
    if not ratios:
        raise ValueError("ratios must be non-empty")
    if ratios[0] != 1:
        raise ValueError("ratios must start at 1 for the base level")
    for i in range(1, len(ratios)):
        if ratios[i] <= ratios[i - 1]:
            raise ValueError("ratios must be strictly increasing")
        if ratios[i] % ratios[i - 1] != 0:
            raise ValueError("each ratio must be a multiple of the previous")


@dataclass
class MultiLevel2D:
    """
    A list of IntervalSet (one per level) and their refinement ratios.

    Attributes
    ----------
    ratios:
        List of refinement factors relative to level 0. Typically [1, 2, 4, …].
    levels:
        List of optional IntervalSet placeholders; callers can fill them as
        they build masks per level. `None` means not set yet.
    """

    ratios: List[int]
    levels: List[Optional[IntervalSet]]

    @classmethod
    def create(cls, n_levels: int, base_ratio: int = 2) -> "MultiLevel2D":
        ratios = _default_ratios(n_levels, base_ratio)
        return cls(ratios=ratios, levels=[None] * n_levels)

    def validate(self) -> None:
        _validate_ratios(self.ratios)
        if len(self.levels) != len(self.ratios):
            raise ValueError("levels and ratios must have the same length")

    def set_level(self, level: int, interval_set: IntervalSet | None) -> None:
        if level < 0 or level >= len(self.levels):
            raise IndexError("level out of range")
        self.levels[level] = interval_set

    def get_level(self, level: int) -> Optional[IntervalSet]:
        if level < 0 or level >= len(self.levels):
            raise IndexError("level out of range")
        return self.levels[level]

    @property
    def n_levels(self) -> int:
        return len(self.levels)


@dataclass
class MultiLevelField2D:
    """
    IntervalField per level for a 2D hierarchy.

    The geometry (IntervalSet) is managed separately by MultiLevel2D. This
    container holds optional IntervalField objects aligned with the same level
    structure and ratios.
    """

    ratios: List[int]
    fields: List[Optional[IntervalField]]

    @classmethod
    def empty_like(cls, ml: MultiLevel2D) -> "MultiLevelField2D":
        ml.validate()
        return cls(ratios=list(ml.ratios), fields=[None] * ml.n_levels)

    @classmethod
    def create_from_geometry(
        cls,
        ml: MultiLevel2D,
        *,
        fill_value: float = 0.0,
        dtype=None,
    ) -> "MultiLevelField2D":
        ml.validate()
        fields: List[Optional[IntervalField]] = []
        for lvl in ml.levels:
            if lvl is None:
                fields.append(None)
            else:
                fields.append(create_interval_field(lvl, fill_value=fill_value, dtype=dtype))
        return cls(ratios=list(ml.ratios), fields=fields)

    def validate_against(self, ml: MultiLevel2D) -> None:
        ml.validate()
        if len(self.fields) != ml.n_levels:
            raise ValueError("field levels and geometry levels mismatch")
        if list(self.ratios) != list(ml.ratios):
            raise ValueError("ratios mismatch between field and geometry")
        # Optional geometry consistency checks for present fields
        for i, f in enumerate(self.fields):
            s = ml.levels[i]
            if f is None or s is None:
                continue
            # Basic shape checks
            if (f.interval_set.row_offsets.size != s.row_offsets.size) or (f.interval_set.begin.size != s.begin.size):
                raise ValueError(f"level {i}: field geometry does not match IntervalSet")

    def set_level_field(self, level: int, field: IntervalField | None) -> None:
        if level < 0 or level >= len(self.fields):
            raise IndexError("level out of range")
        self.fields[level] = field

    def get_level_field(self, level: int) -> Optional[IntervalField]:
        if level < 0 or level >= len(self.fields):
            raise IndexError("level out of range")
        return self.fields[level]


def _ceil_div(a, b):
    return (a + b - 1) // b


def _row_ids(interval_set: IntervalSet):
    cp = _require_cupy()
    row_count = interval_set.row_count
    if row_count == 0:
        return cp.zeros(0, dtype=cp.int32)
    # Device-only computation of repeats per row to avoid host round-trip
    row_offsets = interval_set.row_offsets
    counts = (row_offsets[1:] - row_offsets[:-1]).astype(cp.int32, copy=False)
    rows = cp.arange(row_count, dtype=cp.int32)
    return cp.repeat(rows, counts)


def _prolong_set_impl(interval_set: IntervalSet, ratio: int):
    cp = _require_cupy()
    if ratio < 1:
        raise ValueError("ratio must be >= 1")
    if ratio == 1:
        base = cp.arange(interval_set.begin.size, dtype=cp.int32)
        return interval_set, base

    row_count = interval_set.row_count
    if row_count == 0 or interval_set.begin.size == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        offsets = cp.zeros(row_count * ratio + 1, dtype=cp.int32)
        fine_set = IntervalSet(begin=zero, end=zero, row_offsets=offsets)
        base = cp.zeros(0, dtype=cp.int32)
        return fine_set, base

    begin = interval_set.begin.astype(cp.int32, copy=False)
    end = interval_set.end.astype(cp.int32, copy=False)
    row_ids = _row_ids(interval_set)

    scaled_begin = begin * ratio
    scaled_end = end * ratio
    interval_count = begin.size

    repeated_begin = cp.tile(scaled_begin, ratio)
    repeated_end = cp.tile(scaled_end, ratio)
    base_indices = cp.tile(cp.arange(interval_count, dtype=cp.int32), ratio)
    base_rows = cp.tile(row_ids * ratio, ratio)
    delta_offsets = cp.repeat(cp.arange(ratio, dtype=cp.int32), interval_count)
    fine_rows = base_rows + delta_offsets

    order = cp.lexsort(cp.stack((repeated_begin, fine_rows)))
    fine_rows = fine_rows[order]
    repeated_begin = repeated_begin[order]
    repeated_end = repeated_end[order]
    base_indices = base_indices[order]

    fine_row_count = row_count * ratio
    counts = cp.bincount(fine_rows, minlength=fine_row_count)
    counts = counts.astype(cp.int32, copy=False)
    row_offsets_fine = cp.empty(fine_row_count + 1, dtype=cp.int32)
    row_offsets_fine[0] = 0
    if fine_row_count > 0:
        cp.cumsum(counts, dtype=cp.int32, out=row_offsets_fine[1:])

    fine_set = IntervalSet(begin=repeated_begin, end=repeated_end, row_offsets=row_offsets_fine)
    return fine_set, base_indices


def prolong_set(interval_set: IntervalSet, ratio: int) -> IntervalSet:
    fine_set, _ = _prolong_set_impl(interval_set, ratio)
    return fine_set


def restrict_set(interval_set: IntervalSet, ratio: int) -> IntervalSet:
    cp = _require_cupy()
    if ratio < 1:
        raise ValueError("ratio must be >= 1")
    row_count = interval_set.row_count
    if row_count == 0 or interval_set.begin.size == 0:
        offsets = cp.zeros(row_count // max(ratio, 1) + 1, dtype=cp.int32)
        empty = cp.zeros(0, dtype=cp.int32)
        return IntervalSet(begin=empty, end=empty, row_offsets=offsets)
    if row_count % ratio != 0:
        raise ValueError("row count is not divisible by ratio")

    row_ids = _row_ids(interval_set)
    coarse_rows = row_ids // ratio

    coarse_begin = interval_set.begin.astype(cp.int32, copy=False) // ratio
    coarse_end = _ceil_div(interval_set.end.astype(cp.int32, copy=False), ratio)
    mask = coarse_end > coarse_begin
    if not int(mask.any().item()):
        coarse_row_count = row_count // ratio
        offsets = cp.zeros(coarse_row_count + 1, dtype=cp.int32)
        empty = cp.zeros(0, dtype=cp.int32)
        return IntervalSet(begin=empty, end=empty, row_offsets=offsets)

    coarse_rows = coarse_rows[mask]
    coarse_begin = coarse_begin[mask]
    coarse_end = coarse_end[mask]

    # Ensure segments are ordered by (row, begin) for per-row merging.
    order = cp.lexsort(cp.stack((coarse_begin, coarse_rows)))
    coarse_rows = coarse_rows[order]
    coarse_begin = coarse_begin[order]
    coarse_end = coarse_end[order]

    coarse_row_count = row_count // ratio
    counts_raw = cp.bincount(coarse_rows, minlength=coarse_row_count).astype(cp.int32, copy=False)
    row_offsets_raw = cp.empty(coarse_row_count + 1, dtype=cp.int32)
    row_offsets_raw[0] = 0
    if coarse_row_count > 0:
        cp.cumsum(counts_raw, dtype=cp.int32, out=row_offsets_raw[1:])

    merge_count_kernel = get_kernels(cp)[3]
    merge_write_kernel = get_kernels(cp)[4]

    block = 128
    grid = (coarse_row_count + block - 1) // block if coarse_row_count > 0 else 1
    counts_out = cp.empty(coarse_row_count, dtype=cp.int32)

    merge_count_kernel(
        (grid,),
        (block,),
        (
            coarse_begin,
            coarse_end,
            row_offsets_raw,
            np.int32(coarse_row_count),
            counts_out,
        ),
    )

    row_offsets = cp.empty(coarse_row_count + 1, dtype=cp.int32)
    row_offsets[0] = 0
    if coarse_row_count > 0:
        cp.cumsum(counts_out, dtype=cp.int32, out=row_offsets[1:])
    total = int(row_offsets[-1].item()) if coarse_row_count > 0 else 0
    if total == 0:
        empty = cp.zeros(0, dtype=cp.int32)
        return IntervalSet(begin=empty, end=empty, row_offsets=row_offsets)

    out_begin = cp.empty(total, dtype=cp.int32)
    out_end = cp.empty(total, dtype=cp.int32)

    merge_write_kernel(
        (grid,),
        (block,),
        (
            coarse_begin,
            coarse_end,
            row_offsets_raw,
            row_offsets,
            np.int32(coarse_row_count),
            out_begin,
            out_end,
        ),
    )

    return IntervalSet(begin=out_begin, end=out_end, row_offsets=row_offsets)


def prolong_field(field: IntervalField, ratio: int) -> IntervalField:
    cp = _require_cupy()
    fine_set, base_indices = _prolong_set_impl(field.interval_set, ratio)
    fine_field = create_interval_field(fine_set, fill_value=0.0, dtype=field.values.dtype)
    if fine_field.values.size == 0 or field.values.size == 0:
        return fine_field

    coarse_offsets = field.interval_cell_offsets
    fine_offsets = fine_field.interval_cell_offsets

    for fine_interval in range(base_indices.size):
        base_interval = int(base_indices[fine_interval].item())
        start_coarse = int(coarse_offsets[base_interval].item())
        end_coarse = int(coarse_offsets[base_interval + 1].item())
        if end_coarse <= start_coarse:
            continue
        segment = field.values[start_coarse:end_coarse]
        expanded = cp.repeat(segment, ratio)
        start_fine = int(fine_offsets[fine_interval].item())
        end_fine = int(fine_offsets[fine_interval + 1].item())
        fine_field.values[start_fine:end_fine] = expanded

    return fine_field


def restrict_field(field: IntervalField, ratio: int, *, reducer: str = "mean") -> IntervalField:
    cp = _require_cupy()
    fine_set = field.interval_set
    coarse_set = restrict_set(fine_set, ratio)
    base_dtype = cp.dtype(field.values.dtype)
    if reducer == "mean":
        target_dtype = cp.dtype(cp.result_type(base_dtype, cp.float32))
    else:
        target_dtype = base_dtype
    coarse_field = create_interval_field(coarse_set, fill_value=0.0, dtype=target_dtype)
    if field.values.size == 0 or coarse_field.values.size == 0:
        return coarse_field

    fine_offsets = field.interval_cell_offsets
    coarse_offsets = coarse_field.interval_cell_offsets
    fine_row_offsets = fine_set.row_offsets
    coarse_row_offsets = coarse_set.row_offsets
    fine_begin = fine_set.begin
    fine_end = fine_set.end
    coarse_begin = coarse_set.begin
    coarse_end = coarse_set.end

    ratio_y = ratio
    for coarse_row in range(coarse_set.row_count):
        fine_row_first = coarse_row * ratio_y
        coarse_interval_start = int(coarse_row_offsets[coarse_row].item())
        coarse_interval_end = int(coarse_row_offsets[coarse_row + 1].item())
        for coarse_interval in range(coarse_interval_start, coarse_interval_end):
            start_coarse = int(coarse_offsets[coarse_interval].item())
            end_coarse = int(coarse_offsets[coarse_interval + 1].item())
            width = end_coarse - start_coarse
            if width <= 0:
                continue
            slices = []
            for delta in range(ratio_y):
                fine_row = fine_row_first + delta
                fine_interval_start = int(fine_row_offsets[fine_row].item())
                fine_interval_end = int(fine_row_offsets[fine_row + 1].item())
                matched = None
                for fine_interval in range(fine_interval_start, fine_interval_end):
                    b_f = int(fine_begin[fine_interval].item())
                    e_f = int(fine_end[fine_interval].item())
                    b_c = b_f // ratio
                    e_c = _ceil_div(e_f, ratio)
                    if b_c == int(coarse_begin[coarse_interval].item()) and e_c == int(coarse_end[coarse_interval].item()):
                        matched = field.values[
                            int(fine_offsets[fine_interval].item()): int(fine_offsets[fine_interval + 1].item())
                        ]
                        break
                if matched is None:
                    raise ValueError("fine field is not aligned with coarse geometry for restriction")
                slices.append(matched.reshape(width, ratio))
            stack = cp.stack(slices, axis=0)  # (ratio, width, ratio)
            if reducer == "mean":
                reduced = stack.mean(axis=(0, 2))
            elif reducer == "sum":
                reduced = stack.sum(axis=(0, 2))
            elif reducer == "min":
                reduced = cp.min(stack, axis=(0, 2))
            elif reducer == "max":
                reduced = cp.max(stack, axis=(0, 2))
            else:
                raise ValueError(f"unsupported reducer '{reducer}'")
            coarse_field.values[start_coarse:end_coarse] = reduced.astype(target_dtype, copy=False)

    return coarse_field


def prolong_level_sets(ml: MultiLevel2D, level: int) -> IntervalSet:
    ml.validate()
    if level < 0 or level + 1 >= ml.n_levels:
        raise IndexError("invalid level for prolongation")
    base = ml.get_level(level)
    if base is None:
        raise ValueError("base level interval set is not defined")
    ratio = ml.ratios[level + 1] // ml.ratios[level]
    fine = prolong_set(base, ratio)
    ml.set_level(level + 1, fine)
    return fine


def restrict_level_sets(ml: MultiLevel2D, level: int) -> IntervalSet:
    ml.validate()
    if level < 0 or level + 1 >= ml.n_levels:
        raise IndexError("invalid level for restriction")
    fine = ml.get_level(level + 1)
    if fine is None:
        raise ValueError("fine level interval set is not defined")
    ratio = ml.ratios[level + 1] // ml.ratios[level]
    coarse = restrict_set(fine, ratio)
    ml.set_level(level, coarse)
    return coarse


def prolong_level_field(ml: MultiLevel2D, mf: MultiLevelField2D, level: int) -> IntervalField:
    ml.validate()
    mf.validate_against(ml)
    if level < 0 or level + 1 >= ml.n_levels:
        raise IndexError("invalid level for prolongation")
    base_field = mf.get_level_field(level)
    if base_field is None:
        raise ValueError("base level field is not defined")
    ratio = ml.ratios[level + 1] // ml.ratios[level]
    fine_field = prolong_field(base_field, ratio)
    mf.set_level_field(level + 1, fine_field)
    return fine_field


def restrict_level_field(ml: MultiLevel2D, mf: MultiLevelField2D, level: int, *, reducer: str = "mean") -> IntervalField:
    ml.validate()
    mf.validate_against(ml)
    if level < 0 or level + 1 >= ml.n_levels:
        raise IndexError("invalid level for restriction")
    fine_field = mf.get_level_field(level + 1)
    if fine_field is None:
        raise ValueError("fine level field is not defined")
    ratio = ml.ratios[level + 1] // ml.ratios[level]
    coarse_field = restrict_field(fine_field, ratio, reducer=reducer)
    mf.set_level_field(level, coarse_field)
    return coarse_field


def covered_by_fine(ml: MultiLevel2D, level: int):
    ml.validate()
    if level < 0 or level + 1 >= ml.n_levels:
        raise IndexError("invalid level for covered_by_fine")
    fine = ml.get_level(level + 1)
    if fine is None:
        return None
    ratio = ml.ratios[level + 1] // ml.ratios[level]
    return restrict_set(fine, ratio)


def coarse_only(coarse_set: IntervalSet, ml: MultiLevel2D, level: int):
    covered = covered_by_fine(ml, level)
    if covered is None or coarse_set is None:
        return coarse_set
    expr = make_difference(make_input(coarse_set), make_input(covered))
    return evaluate(expr)
