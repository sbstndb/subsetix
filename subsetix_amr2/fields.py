from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import cupy as cp

from subsetix_cupy.interval_field import IntervalField, create_interval_field
from subsetix_cupy.expressions import IntervalSet, _require_cupy
from .geometry import mask_to_interval_set, interval_set_to_mask
from subsetix_cupy import prolong_set


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
    _dense_cache: cp.ndarray | None = None
    _refine_mask_cache: cp.ndarray | None = None
    _refine_set_cache: IntervalSet | None = None
    _fine_set_cache: IntervalSet | None = None
    _fine_mask_cache: cp.ndarray | None = None
    _dirty: bool = True

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

    def dense(self) -> cp.ndarray:
        if self._dense_cache is None or self._dense_cache.shape != (self.height, self.width):
            self._dense_cache = self.field.values.reshape(self.height, self.width)
        return self._dense_cache

    def set_from_dense(self, actions: cp.ndarray) -> None:
        if actions.shape != (self.height, self.width):
            raise ValueError("actions shape mismatch with ActionField dimensions")
        dense_arr = self.dense()
        cp.copyto(dense_arr, actions.astype(cp.int8, copy=False))
        self._mark_dirty()

    def set_from_mask(self, refine_mask: cp.ndarray) -> None:
        if refine_mask.shape != (self.height, self.width):
            raise ValueError("refine_mask shape mismatch with ActionField dimensions")
        dense = self.dense()
        dense.fill(int(Action.KEEP))
        dense[refine_mask] = int(Action.REFINE)
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._refine_mask_cache = None
        self._refine_set_cache = None
        self._fine_set_cache = None
        self._fine_mask_cache = None

    def refine_mask(self) -> cp.ndarray:
        if self._refine_mask_cache is None or self._dirty:
            dense = self.dense()
            self._refine_mask_cache = cp.equal(dense, int(Action.REFINE))
            self._dirty = False
        return self._refine_mask_cache

    def refine_set(self) -> IntervalSet:
        if self._refine_set_cache is None or self._dirty:
            self._refine_set_cache = mask_to_interval_set(self.refine_mask())
        return self._refine_set_cache

    def fine_set(self) -> IntervalSet:
        if self._fine_set_cache is None or self._dirty:
            self._fine_set_cache = prolong_set(self.refine_set(), int(self.ratio))
        return self._fine_set_cache

    def fine_mask(self) -> cp.ndarray:
        if self._fine_mask_cache is None or self._dirty:
            self._fine_mask_cache = interval_set_to_mask(self.fine_set(), self.width * int(self.ratio))
        return self._fine_mask_cache

    def coarse_interval_set(self) -> IntervalSet:
        return self.field.interval_set

    def values(self) -> cp.ndarray:
        return self.field.values


def prolong_coarse_to_fine(
    coarse: cp.ndarray,
    ratio: int,
    *,
    out: Optional[cp.ndarray] = None,
    mask: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """
    Repeat a coarse field onto the fine grid (nearest-neighbour prolongation).
    """

    ratio = int(ratio)
    if ratio < 1:
        raise ValueError("ratio must be >= 1")
    upsampled = cp.repeat(cp.repeat(coarse, ratio, axis=0), ratio, axis=1)
    if out is None:
        out = cp.empty_like(upsampled)
    elif out.shape != upsampled.shape:
        raise ValueError("out must match the fine grid shape")
    if mask is None:
        cp.copyto(out, upsampled)
    else:
        if mask.shape != upsampled.shape:
            raise ValueError("mask must match fine grid shape")
        cp.copyto(out, upsampled, where=mask)
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
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Perform a coarse<->fine synchronisation round:

    1. Restrict fine values onto the coarse grid inside the refine mask.
    2. Optionally refill fine values outside the refine region from the coarse grid.
    """

    action_field: ActionField | None = None
    if isinstance(refine_mask, ActionField):
        action_field = refine_mask
        coarse_mask = action_field.refine_mask()
    else:
        coarse_mask = refine_mask

    if coarse.shape != coarse_mask.shape:
        raise ValueError("refine_mask must have same shape as coarse grid")
    ratio = int(ratio)
    restricted = restrict_fine_to_coarse(fine, ratio, reducer=reducer)
    coarse_updated = cp.array(coarse, copy=True)
    cp.copyto(coarse_updated, restricted, where=coarse_mask)
    if not fill_fine_outside:
        return coarse_updated, cp.array(fine, copy=True)
    if action_field is None:
        fine_mask = cp.repeat(cp.repeat(coarse_mask, ratio, axis=0), ratio, axis=1)
    else:
        fine_mask = action_field.fine_mask()
    prolongated = prolong_coarse_to_fine(coarse_updated, ratio)
    fine_updated = cp.array(fine, copy=True)
    cp.copyto(fine_updated, prolongated, where=~fine_mask)
    return coarse_updated, fine_updated
