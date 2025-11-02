"""
Two-level AMR helpers built on top of subsetix.

This package provides a thin abstraction to assemble 2D two-level AMR
geometries and common refinement utilities while reusing the interval
set operations implemented in :mod:`subsetix_cupy`.
"""

from .geometry import TwoLevelGeometry, mask_to_interval_set, interval_set_to_mask
from .regrid import (
    gradient_magnitude,
    gradient_tag,
    enforce_two_level_grading,
)
from .fields import (
    prolong_coarse_to_fine,
    restrict_fine_to_coarse,
    synchronize_two_level,
)
from .export import save_two_level_vtk
from .api import Box, MRAdaptor, TwoLevelMesh, make_scalar_field
from .runner import (
    SimulationArgs,
    build_mesh,
    parse_simulation_args,
    run_two_level_advection,
    save_snapshot,
    update_ghost,
)
from .simulation import (
    AMR2Simulation,
    AMRState,
    SimulationConfig,
    SimulationStats,
    SquareSpec,
    TwoLevelVTKExporter,
    create_square_field,
)

__all__ = [
    "TwoLevelGeometry",
    "mask_to_interval_set",
    "interval_set_to_mask",
    "gradient_magnitude",
    "gradient_tag",
    "enforce_two_level_grading",
    "prolong_coarse_to_fine",
    "restrict_fine_to_coarse",
    "synchronize_two_level",
    "save_two_level_vtk",
    "Box",
    "TwoLevelMesh",
    "make_scalar_field",
    "MRAdaptor",
    "SimulationArgs",
    "parse_simulation_args",
    "build_mesh",
    "run_two_level_advection",
    "save_snapshot",
    "update_ghost",
    "AMR2Simulation",
    "AMRState",
    "SimulationConfig",
    "SimulationStats",
    "SquareSpec",
    "TwoLevelVTKExporter",
    "create_square_field",
]
