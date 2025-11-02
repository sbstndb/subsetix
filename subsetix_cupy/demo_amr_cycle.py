"""
CuPy AMR update-cycle demo (field-driven refinement, composite cover, halos).

This script composes existing sparse interval primitives (no new kernels):
- Builds a coarse geometry (L0) and a synthetic scalar field u(x,y)
- Computes a refinement mask on L0 from |grad(u)| with a percentile threshold
- Creates a fine level L1 = prolong(refl0, ratio)
- Forms the composite cover C0 = L0 \ restrict(L1), composite = C0 ∪ L1
- Generates per-level halos (ghost cells) with clamp boundary conditions
- Seeds per-level fields from the analytic u on each grid for visualisation

Usage:
    python -m subsetix_cupy.demo_amr_cycle --coarse 96 --ratio 2 --refine-frac 0.12 --halo-x 1 --halo-y 1 --plot --cells
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from . import (
    build_interval_set,
    create_interval_field,
    evaluate,
    make_difference,
    make_input,
    make_union,
    prolong_set,
    restrict_set,
)
from .morphology import dilate_interval_set, ghost_zones
from .plot_utils import (
    intervals_to_mask,
    field_collection_from_dense,
    make_cell_collection,
    plot_cell_layout_from_sets,
    setup_cell_axes,
)


def _build_coarse_set(width: int, height: int):
    begin: list[int] = []
    end: list[int] = []
    offsets = [0]
    for y in range(height):
        if 4 < y < height // 2:
            begin.append(3)
            end.append(width // 3)
        if height // 3 < y < height - 4:
            begin.append(width // 2)
            end.append(width - 3)
        offsets.append(len(begin))
    return build_interval_set(row_offsets=offsets, begin=begin, end=end)


def _analytic_field(width: int, height: int) -> cp.ndarray:
    """Synthetic scalar field on a dense grid (CuPy) in [0,1]^2.

    Mix of sinusoids and two Gaussian bumps to create gradients.
    """

    xx = cp.linspace(0.0, 1.0, width, dtype=cp.float32)
    yy = cp.linspace(0.0, 1.0, height, dtype=cp.float32)
    X, Y = cp.meshgrid(xx, yy)
    base = cp.sin(4 * math.pi * X) * cp.cos(4 * math.pi * Y)
    g1 = cp.exp(-((X - 0.35) ** 2 + (Y - 0.55) ** 2) / (2 * 0.06**2))
    g2 = cp.exp(-((X - 0.70) ** 2 + (Y - 0.25) ** 2) / (2 * 0.04**2))
    return (0.6 * base + 0.9 * g1 + 0.7 * g2).astype(cp.float32, copy=False)


def _grad_magnitude(u: cp.ndarray) -> cp.ndarray:
    gy, gx = cp.gradient(u)
    return cp.sqrt(gx * gx + gy * gy)


def _cp_mask_to_interval_set(mask: cp.ndarray) -> "object":
    """Convert a binary CuPy mask [rows,width] to IntervalSet on GPU.

    Returns IntervalSet with int32 begin/end/row_offsets.
    """

    from .expressions import IntervalSet, _require_cupy

    cp_mod = _require_cupy()
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    rows, width = mask.shape
    if rows == 0 or width == 0:
        zero = cp_mod.zeros(0, dtype=cp_mod.int32)
        offsets = cp_mod.zeros(1, dtype=cp_mod.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets)

    m = (mask.astype(cp_mod.int8) > 0).astype(cp_mod.int8)
    pad = cp_mod.pad(m, ((0, 0), (1, 1)), mode="constant")
    diff = cp_mod.diff(pad, axis=1)
    starts = diff == 1
    stops = diff == -1
    if int(starts.sum().item()) == 0:
        zero = cp_mod.zeros(0, dtype=cp_mod.int32)
        offsets = cp_mod.zeros(rows + 1, dtype=cp_mod.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets)

    start_rows, start_cols = cp_mod.where(starts)
    stop_rows, stop_cols = cp_mod.where(stops)

    # Safety: each row must have equal number of starts/stops.
    start_counts = cp_mod.bincount(start_rows, minlength=rows)
    stop_counts = cp_mod.bincount(stop_rows, minlength=rows)
    if int(cp_mod.any(start_counts != stop_counts)):
        raise RuntimeError("mask->interval conversion found mismatched start/stop counts")

    row_offsets = cp_mod.empty(rows + 1, dtype=cp_mod.int32)
    row_offsets[0] = 0
    if rows > 0:
        cp_mod.cumsum(start_counts.astype(cp_mod.int32, copy=False), dtype=cp_mod.int32, out=row_offsets[1:])

    # Sort by row then column using a compound key (row-major)
    key_beg = start_rows.astype(cp_mod.int64) * int(mask.shape[1]) + start_cols.astype(cp_mod.int64)
    order = cp_mod.argsort(key_beg)
    begin = start_cols[order].astype(cp_mod.int32, copy=False)

    key_end = stop_rows.astype(cp_mod.int64) * int(mask.shape[1]) + stop_cols.astype(cp_mod.int64)
    order2 = cp_mod.argsort(key_end)
    end = stop_cols[order2].astype(cp_mod.int32, copy=False)

    return IntervalSet(begin=begin, end=end, row_offsets=row_offsets)


def _assign_field_from_dense(interval_set, dense: np.ndarray):
    """Create an IntervalField on interval_set and fill values from dense grid.

    The dense grid is NumPy for convenient host plotting; values are copied to device.
    """

    cp_mod = cp
    field = create_interval_field(interval_set, fill_value=0.0, dtype=cp_mod.float32)
    offsets = cp_mod.asnumpy(interval_set.row_offsets)
    begin = cp_mod.asnumpy(interval_set.begin)
    end = cp_mod.asnumpy(interval_set.end)
    cell_offsets = cp_mod.asnumpy(field.interval_cell_offsets)
    write = field.values
    for row in range(offsets.size - 1):
        start = offsets[row]
        stop = offsets[row + 1]
        for i in range(start, stop):
            b = begin[i]
            e = end[i]
            c0 = cell_offsets[i]
            c1 = cell_offsets[i + 1]
            write[c0:c1] = cp_mod.asarray(dense[row, b:e], dtype=cp_mod.float32)
    return field


def main():
    ap = argparse.ArgumentParser(description="AMR update-cycle demo (CuPy)")
    ap.add_argument("--coarse", type=int, default=96)
    ap.add_argument("--ratio", type=int, default=2)
    ap.add_argument("--refine-frac", type=float, default=0.12, help="Top fraction of |grad(u)| to refine [0,1]")
    ap.add_argument("--halo-x", type=int, default=1)
    ap.add_argument("--halo-y", type=int, default=1)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--cells", action="store_true")
    args = ap.parse_args()

    width = height = int(args.coarse)
    ratio = int(args.ratio)

    # Level 0 geometry and field
    L0 = _build_coarse_set(width, height)
    u0 = _analytic_field(width, height)

    # Compute refinement indicator on the whole grid and mask to L0
    g0 = _grad_magnitude(u0)
    coarse_mask_np = intervals_to_mask(L0, width)
    coarse_mask = cp.asarray(coarse_mask_np)
    g0 *= coarse_mask

    # Percentile threshold for refinement
    valid = g0[coarse_mask.astype(bool)]
    thresh = cp.percentile(valid, max(0.0, min(100.0, (1.0 - args.refine_frac) * 100.0))) if valid.size > 0 else 1e9
    refine_mask0 = (g0 >= thresh).astype(cp.uint8)

    # Build L1 from coarse refine mask
    R0 = _cp_mask_to_interval_set(refine_mask0)
    L1 = prolong_set(R0, ratio)

    # Composite cover and halos
    C0 = evaluate(make_difference(make_input(L0), make_input(restrict_set(L1, ratio))))
    composite = evaluate(
        make_union(
            make_input(L1),
            make_input(prolong_set(C0, ratio)),
        )
    )

    H0 = ghost_zones(C0, halo_x=args.halo_x, halo_y=args.halo_y, width=width, height=height, bc="clamp")
    H1 = ghost_zones(L1, halo_x=max(0, args.halo_x * ratio), halo_y=max(0, args.halo_y * ratio), width=width * ratio, height=height * ratio, bc="clamp")

    # Seed fields from analytic function
    u1 = _analytic_field(width * ratio, height * ratio)
    u0_np = cp.asnumpy(u0)
    u1_np = cp.asnumpy(u1)
    field_C0 = _assign_field_from_dense(C0, u0_np)
    field_L1 = _assign_field_from_dense(L1, u1_np)

    # Stats
    def _count(interval_set):
        return int(cp.asnumpy(interval_set.row_offsets)[-1])

    print("Cells L0:", _count(L0))
    print("Cells L1:", _count(L1))
    print("Cells C0 (coarse residual):", _count(C0))
    print("Cells composite:", _count(composite))
    print("Ghosts H0:", _count(H0))
    print("Ghosts H1:", _count(H1))

    if args.plot:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.ravel()

        axes[0].add_collection(make_cell_collection(L0, width, 1, facecolor="#bdbdbd"))
        setup_cell_axes(axes[0], width, height, title="L0 coarse")

        # Indicator heatmap over coarse active cells
        g0_np = cp.asnumpy(g0)
        indicator_coll = field_collection_from_dense(L0, g0_np, width, 1, cmap=plt.get_cmap("magma"))
        axes[1].add_collection(indicator_coll)
        setup_cell_axes(axes[1], width, height, title="|grad(u)| on L0")
        fig.colorbar(indicator_coll, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].add_collection(make_cell_collection(R0, width, 1, facecolor="#ffb347"))
        setup_cell_axes(axes[2], width, height, title="Refine mask (L0)")

        # Composite view overlays coarse residual and fine
        axes[3].add_collection(make_cell_collection(C0, width, 1, facecolor="#c5c5c5", edgecolor="k"))
        axes[3].add_collection(make_cell_collection(L1, width, ratio, facecolor="#ff6961", edgecolor="k", linewidth=0.25))
        setup_cell_axes(axes[3], width, height, title="Composite (C0 ∪ L1)")

        axes[4].add_collection(make_cell_collection(H0, width, 1, facecolor="#7fa2ff"))
        setup_cell_axes(axes[4], width, height, title="Ghosts H0 (coarse)")

        axes[5].add_collection(make_cell_collection(H1, width, ratio, facecolor="#c18aff"))
        setup_cell_axes(axes[5], width, height, title="Ghosts H1 (fine)")

        fig.tight_layout()
        plt.show()

        if args.cells:
            plot_cell_layout_from_sets([C0, L1], [1, ratio], width, labels=["C0", "L1"])


if __name__ == "__main__":
    main()
