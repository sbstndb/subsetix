from __future__ import annotations

import os
from typing import Dict

import cupy as cp
import numpy as np

from subsetix_cupy.export_vtk import write_rectilinear_grid_vtr, write_unstructured_quads_vtu
from subsetix_cupy.morphology import ghost_zones
from subsetix_cupy.expressions import IntervalSet, _require_cupy

from .geometry import interval_set_to_mask


def _ensure_bool(arr: cp.ndarray) -> cp.ndarray:
    cp_mod = _require_cupy()
    if not isinstance(arr, cp_mod.ndarray):
        raise TypeError("expected CuPy array")
    return arr.astype(cp_mod.bool_, copy=False)


def _ghost_mask(interval_set, width: int, height: int, halo_x: int, halo_y: int, bc: str) -> cp.ndarray:
    cp_mod = _require_cupy()
    if halo_x <= 0 and halo_y <= 0:
        return cp_mod.zeros((height, width), dtype=cp_mod.bool_)
    ghost = ghost_zones(
        interval_set,
        halo_x=max(0, halo_x),
        halo_y=max(0, halo_y),
        width=width,
        height=height,
        bc=bc,
    )
    if int(ghost.begin.size) == 0:
        return cp_mod.zeros((height, width), dtype=cp_mod.bool_)
    return interval_set_to_mask(ghost, width).astype(cp_mod.bool_, copy=False)


def save_two_level_vtk(
    out_dir: str,
    base: str,
    step: int,
    *,
    coarse_field: cp.ndarray,
    fine_field: cp.ndarray,
    refine_set: IntervalSet,
    coarse_only_set: IntervalSet,
    fine_set: IntervalSet,
    dx_coarse: float,
    dy_coarse: float,
    ratio: int = 2,
    time_value: float = 0.0,
    ghost_halo: int = 1,
    bc: str = "clamp",
) -> Dict[str, str]:
    """
    Save VTK files (two rectilinear grids + combined unstructured mesh) for a two-level AMR snapshot.
    Returns a dict with generated filenames.
    """

    cp_mod = _require_cupy()
    height, width = coarse_field.shape
    fine_height, fine_width = fine_field.shape
    dx_fine = dx_coarse / ratio
    dy_fine = dy_coarse / ratio

    refine_mask = _ensure_bool(interval_set_to_mask(refine_set, width))
    coarse_only_mask = _ensure_bool(interval_set_to_mask(coarse_only_set, width))
    fine_mask = _ensure_bool(interval_set_to_mask(fine_set, fine_width))

    coarse_ghost_mask = _ghost_mask(
        refine_set,
        width=width,
        height=height,
        halo_x=ghost_halo,
        halo_y=ghost_halo,
        bc=bc,
    )
    fine_ghost_mask = _ghost_mask(
        fine_set,
        width=fine_width,
        height=fine_height,
        halo_x=max(0, ghost_halo * ratio),
        halo_y=max(0, ghost_halo * ratio),
        bc=bc,
    )

    os.makedirs(out_dir, exist_ok=True)
    coarse_path = os.path.join(out_dir, f"{base}_L0_step{step:04d}.vtr")
    fine_path = os.path.join(out_dir, f"{base}_L1_step{step:04d}.vtr")
    mesh_path = os.path.join(out_dir, f"{base}_mesh_step{step:04d}.vtu")

    coarse_arrays = {
        "refine_mask": refine_mask,
        "coarse_only": coarse_only_mask,
    }
    if ghost_halo > 0:
        coarse_arrays["ghost"] = coarse_ghost_mask

    fine_arrays = {
        "fine_active": fine_mask,
    }
    if ghost_halo > 0:
        fine_arrays["ghost"] = fine_ghost_mask

    write_rectilinear_grid_vtr(
        coarse_path,
        coarse_field,
        dx_coarse,
        dy_coarse,
        cell_arrays=coarse_arrays,
    )

    write_rectilinear_grid_vtr(
        fine_path,
        fine_field,
        dx_fine,
        dy_fine,
        cell_arrays=fine_arrays,
    )

    coarse_active_mesh = cp_mod.logical_and(coarse_only_mask, ~coarse_ghost_mask)
    coarse_ghost_mesh = cp_mod.logical_and(coarse_ghost_mask, ~coarse_active_mesh)
    fine_active_mesh = fine_mask
    fine_ghost_mesh = fine_ghost_mask

    write_unstructured_quads_vtu(
        mesh_path,
        cells=[
            (coarse_active_mesh, coarse_field, 0, dx_coarse, dy_coarse, 0.0, 0.0),
            (coarse_ghost_mesh, coarse_field, 0, dx_coarse, dy_coarse, 0.0, 0.0, coarse_ghost_mesh),
            (fine_active_mesh, fine_field, 1, dx_fine, dy_fine, 0.0, 0.0),
            (fine_ghost_mesh, fine_field, 1, dx_fine, dy_fine, 0.0, 0.0, fine_ghost_mesh),
        ],
    )

    return {
        "coarse_vtr": os.path.basename(coarse_path),
        "fine_vtr": os.path.basename(fine_path),
        "mesh_vtu": os.path.basename(mesh_path),
    }
