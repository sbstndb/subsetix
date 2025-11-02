from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cupy as cp

from subsetix_cupy import (
    CuPyWorkspace,
    IntervalSet,
    dilate_interval_set,
    evaluate,
    make_difference,
    make_input,
    make_union,
    prolong_set,
)
from subsetix_cupy.expressions import _require_cupy
from subsetix_cupy.plot_utils import intervals_to_mask as _intervals_to_mask_np


def mask_to_interval_set(mask: cp.ndarray) -> IntervalSet:
    """
    Convert a dense boolean mask into a CuPy-backed IntervalSet.
    """

    cp_mod = _require_cupy()
    if not isinstance(mask, cp_mod.ndarray):
        raise TypeError("mask must be a CuPy array")
    if mask.ndim != 2:
        raise ValueError("mask must be 2D (rows x width)")
    rows, width = mask.shape
    if rows == 0 or width == 0:
        zero = cp_mod.zeros(0, dtype=cp_mod.int32)
        offsets = cp_mod.zeros(1, dtype=cp_mod.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets)

    normalized = mask.astype(cp_mod.int8, copy=False)
    pad = cp_mod.pad(normalized, ((0, 0), (1, 1)), mode="constant")
    diff = cp_mod.diff(pad, axis=1)
    starts = diff == 1
    stops = diff == -1
    if int(starts.sum().item()) == 0:
        zero = cp_mod.zeros(0, dtype=cp_mod.int32)
        offsets = cp_mod.zeros(rows + 1, dtype=cp_mod.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets)

    start_rows, start_cols = cp_mod.where(starts)
    stop_rows, stop_cols = cp_mod.where(stops)
    start_counts = cp_mod.bincount(start_rows, minlength=rows)
    stop_counts = cp_mod.bincount(stop_rows, minlength=rows)
    if int(cp_mod.any(start_counts != stop_counts)):
        raise RuntimeError("mask->interval conversion mismatch between starts and stops")

    row_offsets = cp_mod.empty(rows + 1, dtype=cp_mod.int32)
    row_offsets[0] = 0
    if rows > 0:
        cp_mod.cumsum(start_counts.astype(cp_mod.int32, copy=False), dtype=cp_mod.int32, out=row_offsets[1:])

    key_begin = start_rows.astype(cp_mod.int64) * int(width) + start_cols.astype(cp_mod.int64)
    order_begin = cp_mod.argsort(key_begin)
    begin = start_cols[order_begin].astype(cp_mod.int32, copy=False)

    key_end = stop_rows.astype(cp_mod.int64) * int(width) + stop_cols.astype(cp_mod.int64)
    order_end = cp_mod.argsort(key_end)
    end = stop_cols[order_end].astype(cp_mod.int32, copy=False)

    return IntervalSet(begin=begin, end=end, row_offsets=row_offsets)


def interval_set_to_mask(interval_set: IntervalSet, width: int, *, cupy: bool = True):
    """
    Render an IntervalSet back into a dense mask.
    """

    mask_np = _intervals_to_mask_np(interval_set, width)
    if cupy:
        cp_mod = _require_cupy()
        return cp_mod.asarray(mask_np, dtype=cp_mod.bool_)
    return mask_np


@dataclass
class TwoLevelGeometry:
    """
    Minimal 2-level AMR layout on a regular grid.
    """

    ratio: int
    width: int
    height: int
    coarse: IntervalSet
    refine: IntervalSet
    fine: IntervalSet
    coarse_only: IntervalSet
    workspace: CuPyWorkspace

    @classmethod
    def from_masks(
        cls,
        refine_mask: cp.ndarray,
        *,
        ratio: int = 2,
        coarse_mask: Optional[cp.ndarray] = None,
        workspace: Optional[CuPyWorkspace] = None,
    ) -> "TwoLevelGeometry":
        cp_mod = _require_cupy()
        if refine_mask.ndim != 2:
            raise ValueError("refine_mask must be 2D")
        rows, width = refine_mask.shape
        if rows <= 0 or width <= 0:
            raise ValueError("refine_mask must have positive dimensions")
        if coarse_mask is None:
            coarse_mask = cp_mod.ones_like(refine_mask, dtype=cp_mod.bool_)
        if coarse_mask.shape != refine_mask.shape:
            raise ValueError("coarse_mask must match refine_mask shape")

        refined_overlap = refine_mask & (~coarse_mask)
        if int(refined_overlap.any()):
            raise ValueError("refine_mask must be subset of coarse_mask")

        ratio_int = int(ratio)
        if ratio_int < 1:
            raise ValueError("ratio must be >= 1 for two-level AMR")

        workspace = workspace or CuPyWorkspace()

        coarse_mask_bool = coarse_mask.astype(cp_mod.bool_, copy=False)
        refine_mask_bool = refine_mask.astype(cp_mod.bool_, copy=False)

        coarse_set = mask_to_interval_set(coarse_mask_bool)
        refine_set = mask_to_interval_set(refine_mask_bool)

        # Guarantee the fine layout via subsetix prolongation.
        fine_set = prolong_set(refine_set, ratio_int)

        coarse_only_expr = make_difference(make_input(coarse_set), make_input(refine_set))
        coarse_only_set = evaluate(coarse_only_expr, workspace=workspace)

        return cls(
            ratio=ratio_int,
            width=width,
            height=rows,
            coarse=coarse_set,
            refine=refine_set,
            fine=fine_set,
            coarse_only=coarse_only_set,
            workspace=workspace,
        )

    def with_refine_mask(self, refine_mask: cp.ndarray) -> "TwoLevelGeometry":
        """
        Return a new TwoLevelGeometry with a different refinement mask.
        """

        coarse_mask = self.coarse_mask
        return TwoLevelGeometry.from_masks(
            refine_mask.astype(cp.bool_, copy=False),
            ratio=self.ratio,
            coarse_mask=coarse_mask,
            workspace=self.workspace,
        )

    @property
    def coarse_mask(self) -> cp.ndarray:
        return interval_set_to_mask(self.coarse, self.width)

    @property
    def refine_mask(self) -> cp.ndarray:
        return interval_set_to_mask(self.refine, self.width)

    @property
    def coarse_only_mask(self) -> cp.ndarray:
        return interval_set_to_mask(self.coarse_only, self.width)

    @property
    def fine_mask(self) -> cp.ndarray:
        mask = interval_set_to_mask(self.fine, self.width * self.ratio)
        return mask

    def dilate_refine(self, halo: int = 1, *, mode: str = "von_neumann") -> "TwoLevelGeometry":
        """
        Expand the refine mask by a given halo (using subsetix dilation) and return a new geometry.
        """

        halo = int(halo)
        if halo <= 0:
            return self
        if mode not in {"von_neumann", "moore"}:
            raise ValueError("mode must be 'von_neumann' or 'moore'")
        if mode == "moore":
            dilated = dilate_interval_set(
                self.refine,
                halo_x=halo,
                halo_y=halo,
                width=self.width,
                height=self.height,
                bc="clamp",
            )
        else:
            horiz = dilate_interval_set(
                self.refine,
                halo_x=halo,
                halo_y=0,
                width=self.width,
                height=self.height,
                bc="clamp",
            )
            vert = dilate_interval_set(
                self.refine,
                halo_x=0,
                halo_y=halo,
                width=self.width,
                height=self.height,
                bc="clamp",
            )
            dilated = evaluate(make_union(make_input(horiz), make_input(vert)), workspace=self.workspace)
        dilated = evaluate(make_union(make_input(dilated), make_input(self.refine)), workspace=self.workspace)
        dilated_mask = interval_set_to_mask(dilated, self.width)
        dilated_mask = cp.minimum(dilated_mask, self.coarse_mask)  # respect coarse coverage
        return self.with_refine_mask(dilated_mask.astype(cp.bool_, copy=False))
