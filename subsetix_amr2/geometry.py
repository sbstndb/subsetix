from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from subsetix_cupy import (
    CuPyWorkspace,
    IntervalSet,
    dilate_interval_set,
    clip_interval_set,
    evaluate,
    make_difference,
    make_input,
    make_union,
    make_intersection,
    prolong_set,
)
from subsetix_cupy.expressions import _require_cupy

if TYPE_CHECKING:
    from .fields import ActionField

def _clone_interval_set(interval_set: IntervalSet) -> IntervalSet:
    cp_mod = _require_cupy()
    return IntervalSet(
        begin=cp_mod.array(interval_set.begin, dtype=cp_mod.int32, copy=True),
        end=cp_mod.array(interval_set.end, dtype=cp_mod.int32, copy=True),
        row_offsets=cp_mod.array(interval_set.row_offsets, dtype=cp_mod.int32, copy=True),
        rows=cp_mod.array(interval_set.rows, dtype=cp_mod.int32, copy=True),
    )
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
    def from_action_field(
        cls,
        actions: "ActionField",
        *,
        workspace: Optional[CuPyWorkspace] = None,
    ) -> "TwoLevelGeometry":
        if actions.height <= 0 or actions.width <= 0:
            raise ValueError("action field must have positive dimensions")
        workspace = workspace or CuPyWorkspace()
        coarse_set = actions.coarse_interval_set()
        refine_set = actions.refine_set()
        fine_set = actions.fine_set()
        coarse_only_expr = make_difference(make_input(coarse_set), make_input(refine_set))
        coarse_only_set = evaluate(coarse_only_expr, workspace=workspace)

        return cls(
            ratio=int(actions.ratio),
            width=actions.width,
            height=actions.height,
            coarse=coarse_set,
            refine=refine_set,
            fine=fine_set,
            coarse_only=coarse_only_set,
            workspace=workspace,
        )

    def with_action_field(self, actions: "ActionField") -> "TwoLevelGeometry":
        """
        Return a new geometry based on an ActionField, sharing the workspace.
        """

        return TwoLevelGeometry.from_action_field(actions, workspace=self.workspace)

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
            )
        else:
            horiz = dilate_interval_set(
                self.refine,
                halo_x=halo,
                halo_y=0,
            )
            vert = dilate_interval_set(
                self.refine,
                halo_x=0,
                halo_y=halo,
            )
            dilated = evaluate(make_union(make_input(horiz), make_input(vert)), workspace=self.workspace)
        dilated = clip_interval_set(dilated, width=self.width, height=self.height)
        refine_dense = clip_interval_set(self.refine, width=self.width, height=self.height)
        merged = evaluate(make_union(make_input(dilated), make_input(refine_dense)), workspace=self.workspace)
        clipped = clip_interval_set(merged, width=self.width, height=self.height)
        fine_set = prolong_set(clipped, self.ratio)
        coarse_only_expr = make_difference(make_input(self.coarse), make_input(clipped))
        coarse_only_set = evaluate(coarse_only_expr, workspace=self.workspace)
        coarse_only_set = _clone_interval_set(coarse_only_set)
        return TwoLevelGeometry(
            ratio=self.ratio,
            width=self.width,
            height=self.height,
            coarse=self.coarse,
            refine=clipped,
            fine=fine_set,
            coarse_only=coarse_only_set,
            workspace=self.workspace,
        )
