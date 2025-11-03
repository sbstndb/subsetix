from __future__ import annotations

import os
from typing import Dict, Tuple

from subsetix_cupy.export_vtk import write_unstructured_quads_vtu
from subsetix_cupy.expressions import (
    IntervalSet,
    _require_cupy,
    evaluate,
    make_difference,
    make_input,
    make_intersection,
)
from subsetix_cupy.interval_field import IntervalField
from subsetix_cupy.morphology import ghost_zones

from .fields import gather_interval_subset, interval_field_from_dense


def _as_interval_field(field, *, name: str) -> IntervalField:
    cp_mod = _require_cupy()
    if isinstance(field, IntervalField):
        return field
    if not isinstance(field, cp_mod.ndarray):
        raise TypeError(f"{name} must be an IntervalField or CuPy array")
    if field.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return interval_field_from_dense(field)


def _interval_dimensions(field: IntervalField) -> Tuple[int, int]:
    cp_mod = _require_cupy()
    height = int(field.interval_set.row_count)
    if field.interval_set.begin.size == 0:
        width = 0
    else:
        width = int(cp_mod.max(field.interval_set.end).item())
    return height, width


def save_two_level_vtk(
    out_dir: str,
    base: str,
    step: int,
    *,
    coarse_field,
    fine_field,
    refine_set: IntervalSet,
    coarse_only_set: IntervalSet,
    fine_set: IntervalSet,
    dx_coarse: float,
    dy_coarse: float,
    ratio: int = 2,
    time_value: float = 0.0,
    ghost_halo: int = 1,
    width: int | None = None,
    height: int | None = None,
) -> Dict[str, str]:
    """
    Save an unstructured VTK file describing a two-level AMR snapshot.
    Returns a dict with generated filenames.
    """

    coarse_interval = _as_interval_field(coarse_field, name="coarse_field")
    fine_interval = _as_interval_field(fine_field, name="fine_field")

    default_height, default_width = _interval_dimensions(coarse_interval)
    width = int(width if width is not None else default_width)
    height = int(height if height is not None else default_height)

    fine_width = width * ratio
    fine_height = height * ratio
    dx_fine = dx_coarse / ratio
    dy_fine = dy_coarse / ratio

    # Coarse active region excludes ghost halo around refine set.
    coarse_ghost_raw = ghost_zones(refine_set, halo_x=ghost_halo, halo_y=ghost_halo, width=width, height=height)
    coarse_ghost_expr = make_intersection(make_input(coarse_ghost_raw), make_input(coarse_only_set))
    coarse_ghost_set = evaluate(coarse_ghost_expr)
    coarse_active_expr = make_difference(make_input(coarse_only_set), make_input(coarse_ghost_set))
    coarse_active_set = evaluate(coarse_active_expr)

    # Fine active + ghost halos.
    fine_ghost_set = ghost_zones(
        fine_set,
        halo_x=max(0, ghost_halo * ratio),
        halo_y=max(0, ghost_halo * ratio),
        width=fine_width,
        height=fine_height,
    )

    coarse_active_field = gather_interval_subset(coarse_interval, coarse_active_set)
    coarse_ghost_field = gather_interval_subset(coarse_interval, coarse_ghost_set)
    fine_active_field = gather_interval_subset(fine_interval, fine_set)
    fine_ghost_field = gather_interval_subset(fine_interval, fine_ghost_set)

    cells = []
    if coarse_active_field.interval_set.begin.size != 0:
        cells.append((coarse_active_field, 0, dx_coarse, dy_coarse, 0.0, 0.0, 0))
    if coarse_ghost_field.interval_set.begin.size != 0:
        cells.append((coarse_ghost_field, 0, dx_coarse, dy_coarse, 0.0, 0.0, 1))
    if fine_active_field.interval_set.begin.size != 0:
        cells.append((fine_active_field, 1, dx_fine, dy_fine, 0.0, 0.0, 0))
    if fine_ghost_field.interval_set.begin.size != 0:
        cells.append((fine_ghost_field, 1, dx_fine, dy_fine, 0.0, 0.0, 1))

    os.makedirs(out_dir, exist_ok=True)
    mesh_path = os.path.join(out_dir, f"{base}_mesh_step{step:04d}.vtu")
    write_unstructured_quads_vtu(mesh_path, cells=cells)

    return {
        "mesh_vtu": os.path.basename(mesh_path),
    }
