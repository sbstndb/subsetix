"""High-level wrappers mimicking a Samurai-style AMR interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cupy as cp

from .fields import prolong_coarse_to_fine, synchronize_two_level
from .geometry import TwoLevelGeometry
from .regrid import enforce_two_level_grading, gradient_magnitude, gradient_tag


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
        if coarse_resolution is not None:
            res = int(coarse_resolution)
            if res <= 0:
                raise ValueError("coarse_resolution must be positive")
            refine_mask = cp.zeros((res, res), dtype=cp.bool_)
            self.initialise(refine_mask)

    def initialise(self, refine_mask: cp.ndarray) -> None:
        refine_mask = refine_mask.astype(cp.bool_, copy=False)
        coarse_mask = cp.ones_like(refine_mask, dtype=cp.bool_)
        self.geometry = TwoLevelGeometry.from_masks(refine_mask, ratio=self.ratio, coarse_mask=coarse_mask)

    def regrid(self, refine_mask: cp.ndarray) -> TwoLevelGeometry:
        if self.geometry is None:
            self.initialise(refine_mask)
        else:
            self.geometry = self.geometry.with_refine_mask(refine_mask.astype(cp.bool_, copy=False))
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


def make_scalar_field(name: str, mesh: TwoLevelMesh, *, dtype: cp.dtype = cp.float32) -> ScalarField:
    return ScalarField(name, mesh, dtype=dtype)


class MRAdaptor:
    """Gradient-based refinement updater."""

    def __init__(self, field: ScalarField, *, refine_fraction: float, grading: int, mode: str = "von_neumann"):
        self.field = field
        self.mesh = field.mesh
        self.refine_fraction = float(refine_fraction)
        self.grading = int(grading)
        self.mode = mode

    def __call__(self) -> None:
        grad = gradient_magnitude(self.field.coarse)
        tags = gradient_tag(grad, self.refine_fraction)
        graded = enforce_two_level_grading(tags, padding=self.grading, mode=self.mode)
        geometry = self.mesh.regrid(graded)
        prolong_coarse_to_fine(self.field.coarse, self.mesh.ratio, out=self.field.fine, mask=geometry.fine_mask)
        coarse_sync, fine_sync = synchronize_two_level(
            self.field.coarse,
            self.field.fine,
            graded,
            ratio=self.mesh.ratio,
            reducer="mean",
            fill_fine_outside=True,
        )
        self.field.coarse = coarse_sync
        self.field.fine = fine_sync
