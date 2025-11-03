from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Dict

import cupy as cp

from subsetix_cupy.interval_field import IntervalField, create_interval_field
from subsetix_cupy.expressions import IntervalSet, _require_cupy
from .geometry import mask_to_interval_set, interval_set_to_mask
from subsetix_cupy import prolong_set

_PROLONG_NEAREST_KERNEL = cp.ElementwiseKernel(
    "raw T coarse, int32 coarse_w, int32 ratio",
    "T out",
    """
    const int fine_w = coarse_w * ratio;
    const int fy = i / fine_w;
    const int fx = i - fy * fine_w;
    const int cy = fy / ratio;
    const int cx = fx / ratio;
    out = coarse[cy * coarse_w + cx];
    """,
    "subsetix_amr2_prolong_nearest",
)

_COPY_KERNEL_CACHE: Dict[str, cp.RawKernel] = {}


def _interval_row_ids(interval_set: IntervalSet) -> cp.ndarray:
    row_offsets = interval_set.row_offsets.astype(cp.int32, copy=False)
    interval_count = int(interval_set.begin.size)
    if interval_count == 0:
        return cp.zeros(0, dtype=cp.int32)
    idx = cp.arange(interval_count, dtype=cp.int32)
    return cp.searchsorted(row_offsets[1:], idx, side="right").astype(cp.int32, copy=False)


def _get_copy_intervals_kernel(dtype: cp.dtype) -> cp.RawKernel:
    key = cp.dtype(dtype).str
    kernel = _COPY_KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel
    if dtype == cp.float32:
        type_name = "float"
    elif dtype == cp.float64:
        type_name = "double"
    elif dtype == cp.int32:
        type_name = "int"
    elif dtype == cp.int64:
        type_name = "long long"
    elif dtype == cp.bool_:
        type_name = "bool"
    else:
        raise TypeError(f"unsupported dtype for interval copy: {dtype}")
    code = f"""
    extern "C" __global__
    void copy_intervals_2d(const int* __restrict__ row_ids,
                           const int* __restrict__ begin,
                           const int* __restrict__ end,
                           const {type_name}* __restrict__ src,
                           {type_name}* __restrict__ dst,
                           int width)
    {{
        int interval = blockIdx.x;
        int row = row_ids[interval];
        int start = begin[interval];
        int stop = end[interval];
        if (stop <= start) {{
            return;
        }}
        int base = row * width;
        for (int col = start + threadIdx.x; col < stop; col += blockDim.x) {{
            dst[base + col] = src[base + col];
        }}
    }}
    """
    kernel = cp.RawKernel(code, "copy_intervals_2d", options=("--std=c++11",))
    _COPY_KERNEL_CACHE[key] = kernel
    return kernel


def _copy_intervals_into(dst: cp.ndarray, src: cp.ndarray, interval_set: IntervalSet) -> None:
    if dst.ndim != 2 or src.ndim != 2:
        raise ValueError("dst and src must be 2D arrays")
    if dst.shape != src.shape:
        raise ValueError("dst and src must have the same shape")
    if dst.dtype != src.dtype:
        raise TypeError("dst and src must have matching dtypes")
    interval_count = int(interval_set.begin.size)
    if interval_count == 0:
        return
    row_ids = _interval_row_ids(interval_set)
    kernel = _get_copy_intervals_kernel(dst.dtype)
    block = 128
    grid = (interval_count,)
    width = int(dst.shape[1])
    kernel(grid, (block,), (row_ids, interval_set.begin, interval_set.end, src, dst, width))


class Action(IntEnum):
    COARSEN = -1
    KEEP = 0
    REFINE = 1


@dataclass
class ActionField:
    """
    Interval-backed field encoding AMR actions per coarse cell.

    The field stores a value per coarse cell using an IntervalField covering the
    entire grid. Cached dense masks and interval sets are refreshed only when
    the field is updated.
    """

    field: IntervalField
    ratio: int
    width: int
    height: int
    _dense_cache: cp.ndarray | None = None
    _refine_mask_cache: cp.ndarray | None = None
    _refine_set_cache: IntervalSet | None = None
    _fine_set_cache: IntervalSet | None = None
    _fine_mask_cache: cp.ndarray | None = None
    _dirty: bool = True

    @classmethod
    def full_grid(cls, height: int, width: int, ratio: int, *, default: Action = Action.KEEP) -> "ActionField":
        cp_mod = _require_cupy()
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        begin = cp_mod.zeros(height, dtype=cp_mod.int32)
        end = cp_mod.full(height, width, dtype=cp_mod.int32)
        row_offsets = cp_mod.arange(height + 1, dtype=cp_mod.int32)
        coarse_set = IntervalSet(begin=begin, end=end, row_offsets=row_offsets)
        interval_field = create_interval_field(coarse_set, fill_value=int(default), dtype=cp_mod.int8)
        return cls(field=interval_field, ratio=int(ratio), width=width, height=height)

    def dense(self) -> cp.ndarray:
        if self._dense_cache is None or self._dense_cache.shape != (self.height, self.width):
            self._dense_cache = self.field.values.reshape(self.height, self.width)
        return self._dense_cache

    def set_from_dense(self, actions: cp.ndarray) -> None:
        if actions.shape != (self.height, self.width):
            raise ValueError("actions shape mismatch with ActionField dimensions")
        dense_arr = self.dense()
        cp.copyto(dense_arr, actions.astype(cp.int8, copy=False))
        self._mark_dirty()

    def set_from_mask(self, refine_mask: cp.ndarray) -> None:
        if refine_mask.shape != (self.height, self.width):
            raise ValueError("refine_mask shape mismatch with ActionField dimensions")
        dense = self.dense()
        dense.fill(int(Action.KEEP))
        dense[refine_mask] = int(Action.REFINE)
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._refine_mask_cache = None
        self._refine_set_cache = None
        self._fine_set_cache = None
        self._fine_mask_cache = None

    def refine_mask(self) -> cp.ndarray:
        if self._refine_mask_cache is None or self._dirty:
            dense = self.dense()
            self._refine_mask_cache = cp.equal(dense, int(Action.REFINE))
            self._dirty = False
        return self._refine_mask_cache

    def refine_set(self) -> IntervalSet:
        if self._refine_set_cache is None or self._dirty:
            self._refine_set_cache = mask_to_interval_set(self.refine_mask())
        return self._refine_set_cache

    def fine_set(self) -> IntervalSet:
        if self._fine_set_cache is None or self._dirty:
            self._fine_set_cache = prolong_set(self.refine_set(), int(self.ratio))
        return self._fine_set_cache

    def fine_mask(self) -> cp.ndarray:
        if self._fine_mask_cache is None or self._dirty:
            self._fine_mask_cache = interval_set_to_mask(self.fine_set(), self.width * int(self.ratio))
        return self._fine_mask_cache

    def coarse_interval_set(self) -> IntervalSet:
        return self.field.interval_set

    def values(self) -> cp.ndarray:
        return self.field.values


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

    if ratio < 1:
        raise ValueError("ratio must be >= 1")
    if coarse.ndim != 2:
        raise ValueError("coarse must be a 2D array")
    coarse_h, coarse_w = coarse.shape
    fine_shape = (coarse_h * ratio, coarse_w * ratio)
    upsampled = _PROLONG_NEAREST_KERNEL(
        coarse.ravel(), cp.int32(coarse_w), cp.int32(ratio), size=fine_shape[0] * fine_shape[1]
    ).reshape(fine_shape)
    if out is None:
        if mask is not None:
            out = cp.zeros_like(upsampled)
        else:
            return upsampled
    elif out.shape != upsampled.shape:
        raise ValueError("out must match the fine grid shape")
    if mask is None:
        cp.copyto(out, upsampled)
    else:
        mask = mask.astype(cp.bool_, copy=False)
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

    if not isinstance(refine_mask, ActionField):
        mask_array = cp.asarray(refine_mask, dtype=cp.bool_)
        if mask_array.shape != coarse.shape:
            raise ValueError("refine_mask must have same shape as coarse grid")
        action_field = ActionField.full_grid(coarse.shape[0], coarse.shape[1], ratio)
        action_field.set_from_mask(mask_array)
    else:
        action_field = refine_mask

    if action_field.height != coarse.shape[0] or action_field.width != coarse.shape[1]:
        raise ValueError("ActionField dimensions must match coarse grid shape")
    if int(action_field.ratio) != int(ratio):
        raise ValueError("ratio argument must match ActionField ratio")

    ratio = int(ratio)
    restricted = restrict_fine_to_coarse(fine, ratio, reducer=reducer)
    coarse_updated = cp.array(coarse, copy=True)
    refine_set = action_field.refine_set()
    _copy_intervals_into(coarse_updated, restricted, refine_set)

    if not fill_fine_outside:
        return coarse_updated, cp.array(fine, copy=True)

    fine_updated = prolong_coarse_to_fine(coarse_updated, ratio)
    fine_set = action_field.fine_set()
    _copy_intervals_into(fine_updated, fine, fine_set)
    return coarse_updated, fine_updated
