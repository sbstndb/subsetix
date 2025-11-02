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
]
