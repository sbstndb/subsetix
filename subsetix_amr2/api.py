"""High-level wrappers mimicking a Samurai-style AMR interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cupy as cp

from .fields import ActionField, prolong_coarse_to_fine, synchronize_two_level
from .geometry import TwoLevelGeometry
from .regrid import enforce_two_level_grading_set, gradient_magnitude, gradient_tag_threshold_set
from subsetix_cupy.expressions import IntervalSet


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
        actions = ActionField.full_grid(rows, width, self.ratio)
        if refine is not None:
            if not isinstance(refine, IntervalSet):
                raise TypeError("TwoLevelMesh.initialise expects an IntervalSet")
            refine_rows = refine.row_offsets.size - 1
            if refine_rows != rows:
                raise ValueError("refine IntervalSet height mismatch with mesh resolution")
            actions.set_from_interval_set(refine)
        self.geometry = TwoLevelGeometry.from_action_field(actions)

    def regrid(self, refine: IntervalSet) -> TwoLevelGeometry:
        if not isinstance(refine, IntervalSet):
            raise TypeError("TwoLevelMesh.regrid expects an IntervalSet")

        rows = int(refine.row_offsets.size - 1)
        if self.geometry is None:
            if rows <= 0:
                raise ValueError("refine IntervalSet must describe at least one row")
            width = self._base_resolution or rows
            actions = ActionField.full_grid(rows, width, self.ratio)
            actions.set_from_interval_set(refine)
            self.geometry = TwoLevelGeometry.from_action_field(actions)
            self._base_resolution = width
        else:
            if rows != self.geometry.height:
                raise ValueError("refine IntervalSet height mismatch with existing geometry")
            width = self.geometry.width
            actions = ActionField.full_grid(rows, width, self.ratio)
            actions.set_from_interval_set(refine)
            self.geometry = self.geometry.with_action_field(actions)
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
        self.dtype = dtype
        if mesh.geometry is None:
            raise RuntimeError("mesh.initialise() must run before creating fields")
        shape = (mesh.geometry.height, mesh.geometry.width)
        fine_shape = (shape[0] * mesh.ratio, shape[1] * mesh.ratio)
        self.coarse = cp.zeros(shape, dtype=dtype)
        self.fine = cp.zeros(fine_shape, dtype=dtype)

    def array(self, level: int) -> cp.ndarray:
        return self.coarse if level == self.mesh.min_level else self.fine

    def resize(self) -> None:
        if self.mesh.geometry is None:
            raise RuntimeError("mesh geometry not initialised")
        shape = (self.mesh.geometry.height, self.mesh.geometry.width)
        fine_shape = (shape[0] * self.mesh.ratio, shape[1] * self.mesh.ratio)
        if self.coarse.shape != shape:
            self.coarse = cp.zeros(shape, dtype=self.dtype)
        if self.fine.shape != fine_shape:
            self.fine = cp.zeros(fine_shape, dtype=self.dtype)

    def swap(self, other: "ScalarField") -> None:
        self.coarse, other.coarse = other.coarse, self.coarse
        self.fine, other.fine = other.fine, self.fine

    def as_arrays(self) -> tuple[cp.ndarray, cp.ndarray]:
        """Return the coarse and fine arrays (no copy)."""

        return self.coarse, self.fine


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
        coarse = self.field.coarse
        grad = gradient_magnitude(coarse)
        tags_set = gradient_tag_threshold_set(grad, self.refine_threshold)
        graded_set = enforce_two_level_grading_set(
            tags_set,
            padding=self.grading,
            mode=self.mode,
            width=coarse.shape[1],
            height=coarse.shape[0],
        )
        geometry = self.mesh.regrid(graded_set)
        prolong_coarse_to_fine(coarse, self.mesh.ratio, out=self.field.fine, mask=geometry.fine)
        coarse_sync, fine_sync = synchronize_two_level(
            self.field.coarse,
            self.field.fine,
            graded_set,
            ratio=self.mesh.ratio,
            reducer="mean",
            fill_fine_outside=True,
        )
        self.field.coarse = coarse_sync
        self.field.fine = fine_sync
