from __future__ import annotations

from typing import Optional

import cupy as cp

from subsetix_cupy import (
    evaluate,
    make_input,
    make_union,
    dilate_interval_set,
)
from subsetix_cupy.expressions import _require_cupy

from .geometry import interval_set_to_mask, mask_to_interval_set


def gradient_magnitude(field: cp.ndarray) -> cp.ndarray:
    """
    Return sqrt((du/dx)^2 + (du/dy)^2) using first-order differences.
    """

    gx, gy = cp.gradient(field)
    return cp.sqrt(gx * gx + gy * gy)


def gradient_tag(
    values: cp.ndarray,
    frac_high: float,
    *,
    epsilon: float = 1e-8,
) -> cp.ndarray:
    """Return a boolean mask tagging the largest gradients.

    The percentile is computed only on positive gradients; if no positive
    values exist the function returns an all-False mask.
    """

    cp_mod = _require_cupy()
    data = cp_mod.asarray(values, dtype=cp_mod.float32)
    frac_high = float(max(0.0, min(1.0, frac_high)))
    if data.size == 0 or frac_high <= 0.0:
        return cp_mod.zeros_like(data, dtype=cp_mod.bool_)

    flat = data.ravel()
    positive = flat[flat > float(epsilon)]
    if positive.size == 0:
        return cp_mod.zeros_like(data, dtype=cp_mod.bool_)

    percentile = (1.0 - frac_high) * 100.0
    thresh = cp_mod.percentile(positive, percentile)
    thresh = max(float(thresh), float(epsilon))
    return data >= thresh


def enforce_two_level_grading(
    refine_mask: cp.ndarray,
    *,
    padding: int = 1,
    mode: str = "von_neumann",
) -> cp.ndarray:
    """
    Ensure a refined cell carries its immediate parents by dilating the mask.
    """

    cp_mod = _require_cupy()
    padding = int(padding)
    if padding <= 0:
        return refine_mask.astype(cp_mod.bool_, copy=False)
    if mode not in {"von_neumann", "moore"}:
        raise ValueError("mode must be 'von_neumann' or 'moore'")

    interval = mask_to_interval_set(refine_mask)
    if mode == "moore":
        expanded = dilate_interval_set(
            interval,
            halo_x=padding,
            halo_y=padding,
            width=refine_mask.shape[1],
            height=refine_mask.shape[0],
            bc="clamp",
        )
    else:
        horiz = dilate_interval_set(
            interval,
            halo_x=padding,
            halo_y=0,
            width=refine_mask.shape[1],
            height=refine_mask.shape[0],
            bc="clamp",
        )
        vert = dilate_interval_set(
            interval,
            halo_x=0,
            halo_y=padding,
            width=refine_mask.shape[1],
            height=refine_mask.shape[0],
            bc="clamp",
        )
        expanded = evaluate(make_union(make_input(horiz), make_input(vert)))

    union = evaluate(make_union(make_input(interval), make_input(expanded)))
    mask = interval_set_to_mask(union, refine_mask.shape[1])
    return cp_mod.asarray(mask, dtype=cp_mod.bool_)
