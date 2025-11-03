from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cupy as cp


@dataclass
class MaskField:
    """
    Boolean field representing the refinement layout on both coarse and fine grids.

    The fine mask is lazily materialised and cached; it is recomputed only when the
    coarse mask changes or when the geometry's shape requires an update.
    """

    coarse: cp.ndarray
    ratio: int
    _fine_cache: cp.ndarray | None = None
    _dirty: bool = True

    def __post_init__(self) -> None:
        self.ratio = int(self.ratio)
        if self.ratio < 1:
            raise ValueError("ratio must be >= 1")
        self._validate_and_assign(self.coarse)
        # ensure coarse stored as an independent boolean array
        self.coarse = cp.array(self.coarse, dtype=cp.bool_, copy=True)

    def _validate_and_assign(self, mask: cp.ndarray) -> None:
        if not isinstance(mask, cp.ndarray):
            raise TypeError("mask must be a CuPy array")
        if mask.ndim != 2:
            raise ValueError("mask must be 2D (rows x cols)")

    def set_coarse(self, mask: cp.ndarray) -> None:
        """Overwrite the coarse mask and mark the fine cache dirty."""

        self._validate_and_assign(mask)
        mask_bool = mask.astype(cp.bool_, copy=False)
        if mask_bool.shape != self.coarse.shape:
            self.coarse = cp.array(mask_bool, copy=True)
        else:
            cp.copyto(self.coarse, mask_bool)
        self._dirty = True

    def fine(self) -> cp.ndarray:
        """Return the fine-level mask, recomputing it if required."""

        ratio = self.ratio
        if ratio == 1:
            if self._fine_cache is None or self._fine_cache.shape != self.coarse.shape:
                self._fine_cache = cp.array(self.coarse, copy=True)
            elif self._dirty:
                cp.copyto(self._fine_cache, self.coarse)
            self._dirty = False
            return self._fine_cache

        fine_shape = (self.coarse.shape[0] * ratio, self.coarse.shape[1] * ratio)
        if self._fine_cache is None or self._fine_cache.shape != fine_shape:
            self._fine_cache = cp.empty(fine_shape, dtype=cp.bool_)
            self._dirty = True
        if self._dirty:
            repeated = cp.repeat(cp.repeat(self.coarse, ratio, axis=0), ratio, axis=1)
            cp.copyto(self._fine_cache, repeated)
            self._dirty = False
        return self._fine_cache

    def as_arrays(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """Return both coarse and fine masks (fine computed lazily)."""

        return self.coarse, self.fine()


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
    refine_mask,
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

    mask_field: MaskField | None = None
    if isinstance(refine_mask, MaskField):
        mask_field = refine_mask
        coarse_mask = mask_field.coarse
    else:
        coarse_mask = refine_mask

    if coarse.shape != coarse_mask.shape:
        raise ValueError("refine_mask must have same shape as coarse grid")
    ratio = int(ratio)
    restricted = restrict_fine_to_coarse(fine, ratio, reducer=reducer)
    coarse_updated = cp.array(coarse, copy=True)
    cp.copyto(coarse_updated, restricted, where=coarse_mask)
    if not fill_fine_outside:
        return coarse_updated, cp.array(fine, copy=True)
    if mask_field is None:
        fine_mask = cp.repeat(cp.repeat(coarse_mask, ratio, axis=0), ratio, axis=1)
    else:
        fine_mask = mask_field.fine()
    prolongated = prolong_coarse_to_fine(coarse_updated, ratio)
    fine_updated = cp.array(fine, copy=True)
    cp.copyto(fine_updated, prolongated, where=~fine_mask)
    return coarse_updated, fine_updated
