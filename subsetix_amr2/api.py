"""High-level wrappers mimicking a Samurai-style AMR interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cupy as cp

from .fields import (
    ActionField,
    synchronize_interval_fields,
)
from .geometry import TwoLevelGeometry
from .regrid import enforce_two_level_grading_set, gradient_magnitude, gradient_tag_threshold_set
from subsetix_cupy.expressions import IntervalSet
from subsetix_cupy.morphology import full_interval_set
from subsetix_cupy.interval_field import IntervalField, create_interval_field
from subsetix_cupy.multilevel import prolong_field


@dataclass(frozen=True)
class Box:
    """Simple axis-aligned bounding box."""

    min_corner: tuple[float, float]
    max_corner: tuple[float, float]


class TwoLevelMesh:
    """Minimal mesh wrapper exposing Samurai-like helpers."""

    def __init__(
        self,
        box: Box,
        min_level: int,
        max_level: int,
        *,
        ratio: int = 2,
        coarse_resolution: Optional[int] = None,
    ):
        if min_level != 0 or max_level != 1:
            raise ValueError("current implementation supports two levels (0/1) only")
        self.box = box
        self.min_level = min_level
        self.max_level = max_level
        self.ratio = int(ratio)
        self.geometry: Optional[TwoLevelGeometry] = None
        self._base_resolution: Optional[int] = None
        self._coarse_interval: Optional[IntervalSet] = None
        self.actions: Optional[ActionField] = None
        if coarse_resolution is not None:
            self._base_resolution = int(coarse_resolution)
            if self._base_resolution <= 0:
                raise ValueError("coarse_resolution must be positive")
            self.initialise()

    def initialise(self, refine: IntervalSet | None = None) -> None:
        if self._base_resolution is None:
            raise RuntimeError("TwoLevelMesh.initialise requires coarse_resolution at construction time")
        rows = self._base_resolution
        width = rows
        coarse_interval = self._coarse_interval
        if coarse_interval is None or coarse_interval.row_count != rows:
            coarse_interval = full_interval_set(width, rows)
            self._coarse_interval = coarse_interval
        actions = ActionField.from_interval_set(
            coarse_interval,
            width=width,
            height=rows,
            ratio=self.ratio,
        )
        if refine is not None:
            if not isinstance(refine, IntervalSet):
                raise TypeError("TwoLevelMesh.initialise expects an IntervalSet")
            refine_rows = refine.row_offsets.size - 1
            if refine_rows != rows:
                raise ValueError("refine IntervalSet height mismatch with mesh resolution")
            actions.set_from_interval_set(refine)
        self.geometry = TwoLevelGeometry.from_action_field(actions)
        self._coarse_interval = actions.coarse_interval_set()
        self.actions = actions

    def regrid(self, refine: IntervalSet) -> TwoLevelGeometry:
        if not isinstance(refine, IntervalSet):
            raise TypeError("TwoLevelMesh.regrid expects an IntervalSet")

        rows = int(refine.row_offsets.size - 1)
        if self.geometry is None:
            if rows <= 0:
                raise ValueError("refine IntervalSet must describe at least one row")
            width = self._base_resolution or rows
            coarse_interval = self._coarse_interval
            if (
                coarse_interval is None
                or coarse_interval.row_count != rows
            ):
                coarse_interval = full_interval_set(width, rows)
                self._coarse_interval = coarse_interval
            actions = ActionField.from_interval_set(
                coarse_interval,
                width=width,
                height=rows,
                ratio=self.ratio,
            )
            actions.set_from_interval_set(refine)
            self.geometry = TwoLevelGeometry.from_action_field(actions)
            self._base_resolution = width
        else:
            if rows != self.geometry.height:
                raise ValueError("refine IntervalSet height mismatch with existing geometry")
            width = self.geometry.width
            coarse_interval = self.geometry.coarse
            actions = ActionField.from_interval_set(
                coarse_interval,
                width=width,
                height=rows,
                ratio=self.ratio,
            )
            actions.set_from_interval_set(refine)
            self.geometry = self.geometry.with_action_field(actions)
        self._coarse_interval = self.geometry.coarse
        self.actions = actions
        return self.geometry

    def cell_length(self, level: int) -> float:
        if self.geometry is None:
            raise RuntimeError("mesh geometry not initialised")
        length = self.box.max_corner[0] - self.box.min_corner[0]
        dx0 = length / float(self.geometry.width)
        return dx0 if level == self.min_level else dx0 / self.ratio


class ScalarField:
    """Scalar field bound to a two-level mesh."""

    def __init__(self, name: str, mesh: TwoLevelMesh, *, dtype: cp.dtype = cp.float32):
        self.name = name
        self.mesh = mesh
        self.dtype = cp.dtype(dtype)
        if mesh.geometry is None:
            raise RuntimeError("mesh.initialise() must run before creating fields")
        height = mesh.geometry.height
        width = mesh.geometry.width
        ratio = mesh.ratio
        coarse_interval = full_interval_set(width, height)
        fine_interval = full_interval_set(width * ratio, height * ratio)
        self.coarse_field = create_interval_field(coarse_interval, fill_value=0.0, dtype=self.dtype)
        self.fine_field = create_interval_field(fine_interval, fill_value=0.0, dtype=self.dtype)

    def interval_field(self, level: int) -> IntervalField:
        return self.coarse_field if level == self.mesh.min_level else self.fine_field

    def set_interval_fields(
        self,
        *,
        coarse: IntervalField | None = None,
        fine: IntervalField | None = None,
    ) -> None:
        if coarse is not None:
            self.coarse_field = coarse
        if fine is not None:
            self.fine_field = fine

    def resize(self) -> None:
        if self.mesh.geometry is None:
            raise RuntimeError("mesh geometry not initialised")
        height = self.mesh.geometry.height
        width = self.mesh.geometry.width
        ratio = self.mesh.ratio
        coarse_interval = full_interval_set(width, height)
        fine_interval = full_interval_set(width * ratio, height * ratio)
        self.coarse_field = create_interval_field(coarse_interval, fill_value=0.0, dtype=self.dtype)
        self.fine_field = create_interval_field(fine_interval, fill_value=0.0, dtype=self.dtype)

    def swap(self, other: "ScalarField") -> None:
        self.coarse_field, other.coarse_field = other.coarse_field, self.coarse_field
        self.fine_field, other.fine_field = other.fine_field, self.fine_field


def make_scalar_field(name: str, mesh: TwoLevelMesh, *, dtype: cp.dtype = cp.float32) -> ScalarField:
    return ScalarField(name, mesh, dtype=dtype)


class MRAdaptor:
    """Gradient-based refinement updater."""

    def __init__(self, field: ScalarField, *, refine_threshold: float, grading: int, mode: str = "von_neumann"):
        self.field = field
        self.mesh = field.mesh
        self.refine_threshold = float(refine_threshold)
        self.grading = int(grading)
        self.mode = mode

    def __call__(self) -> None:
        geometry = self.mesh.geometry
        if geometry is None:
            raise RuntimeError("mesh geometry not initialised")
        height = geometry.height
        width = geometry.width
        coarse = self.field.coarse_field.values.reshape(height, width)
        grad = gradient_magnitude(coarse)
        tags_set = gradient_tag_threshold_set(grad, self.refine_threshold)
        graded_set = enforce_two_level_grading_set(
            tags_set,
            padding=self.grading,
            mode=self.mode,
            width=width,
            height=height,
        )
        geometry = self.mesh.regrid(graded_set)
        ratio = self.mesh.ratio
        self.field.fine_field = prolong_field(self.field.coarse_field, ratio)
        actions = self.mesh.actions
        if actions is None:
            raise RuntimeError("mesh actions not initialised")
        coarse_field = self.field.coarse_field
        fine_field = self.field.fine_field
        synchronize_interval_fields(
            coarse_field,
            fine_field,
            actions,
            ratio=self.mesh.ratio,
            reducer="mean",
            fill_fine_outside=True,
            copy=False,
        )
