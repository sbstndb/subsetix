from __future__ import annotations

from typing import Optional

import cupy as cp


def prolong_coarse_to_fine(
    coarse: cp.ndarray,
    ratio: int,
    *,
    out: Optional[cp.ndarray] = None,
    mask: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """
    Repeat a coarse field onto the fine grid (nearest-neighbour prolongation).
    """

    ratio = int(ratio)
    if ratio < 1:
        raise ValueError("ratio must be >= 1")
    upsampled = cp.repeat(cp.repeat(coarse, ratio, axis=0), ratio, axis=1)
    if out is None:
        out = cp.empty_like(upsampled)
    elif out.shape != upsampled.shape:
        raise ValueError("out must match the fine grid shape")
    if mask is None:
        cp.copyto(out, upsampled)
    else:
        if mask.shape != upsampled.shape:
            raise ValueError("mask must match fine grid shape")
        cp.copyto(out, upsampled, where=mask)
    return out


def restrict_fine_to_coarse(
    fine: cp.ndarray,
    ratio: int,
    *,
    reducer: str = "mean",
) -> cp.ndarray:
    """
    Collapse a fine grid onto the coarse resolution using block reducers.
    """

    ratio = int(ratio)
    if ratio < 1:
        raise ValueError("ratio must be >= 1")
    Hf, Wf = fine.shape
    if Hf % ratio != 0 or Wf % ratio != 0:
        raise ValueError("fine shape must be divisible by ratio")
    reshaped = fine.reshape(Hf // ratio, ratio, Wf // ratio, ratio)
    if reducer == "mean":
        return reshaped.mean(axis=(1, 3))
    if reducer == "sum":
        return reshaped.sum(axis=(1, 3))
    if reducer == "max":
        return reshaped.max(axis=(1, 3))
    if reducer == "min":
        return reshaped.min(axis=(1, 3))
    raise ValueError(f"unsupported reducer '{reducer}'")


def synchronize_two_level(
    coarse: cp.ndarray,
    fine: cp.ndarray,
    refine_mask: cp.ndarray,
    *,
    ratio: int,
    reducer: str = "mean",
    fill_fine_outside: bool = True,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Perform a coarse<->fine synchronisation round:

    1. Restrict fine values onto the coarse grid inside the refine mask.
    2. Optionally refill fine values outside the refine region from the coarse grid.
    """

    if coarse.shape != refine_mask.shape:
        raise ValueError("refine_mask must have same shape as coarse grid")
    ratio = int(ratio)
    restricted = restrict_fine_to_coarse(fine, ratio, reducer=reducer)
    coarse_updated = cp.array(coarse, copy=True)
    cp.copyto(coarse_updated, restricted, where=refine_mask)
    if not fill_fine_outside:
        return coarse_updated, cp.array(fine, copy=True)
    fine_mask = cp.repeat(cp.repeat(refine_mask, ratio, axis=0), ratio, axis=1)
    prolongated = prolong_coarse_to_fine(coarse_updated, ratio)
    fine_updated = cp.array(fine, copy=True)
    cp.copyto(fine_updated, prolongated, where=~fine_mask)
    return coarse_updated, fine_updated
