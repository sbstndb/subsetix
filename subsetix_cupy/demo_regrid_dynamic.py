"""
Dynamic AMR regridding demo with animation (CuPy).

At each frame t:
  - Build a refinement indicator |grad(u(x,y,t))| over the coarse grid
  - Threshold to a fraction to get coarse refine mask R0(t)
  - Create L1(t) = prolong(R0, ratio)
  - Compute coarse residual C0 = L0 \ restrict(L1)
  - Composite (fine view) = L1 ∪ prolong(C0)
  - Update the per-cell plot interactively (no new kernels)

Usage:
  python -m subsetix_cupy.demo_regrid_dynamic --coarse 96 --ratio 2 --refine-frac 0.10 \
      --frames 120 --interval 80 --show-halos --plot
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple, Optional

import cupy as cp
import matplotlib.pyplot as plt

from . import (
    build_interval_set,
    evaluate,
    make_difference,
    make_input,
    make_union,
    prolong_set,
    restrict_set,
)
from .morphology import ghost_zones
from .plot_utils import make_cell_collection, setup_cell_axes, intervals_to_mask


def _build_coarse_set(width: int, height: int):
    begin = []
    end = []
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


def _analytic_field_t(width: int, height: int, t: float) -> cp.ndarray:
    xx = cp.linspace(0.0, 1.0, width, dtype=cp.float32)
    yy = cp.linspace(0.0, 1.0, height, dtype=cp.float32)
    X, Y = cp.meshgrid(xx, yy)
    base = cp.sin(4 * math.pi * (X + 0.15 * cp.sin(2 * math.pi * t))) * cp.cos(4 * math.pi * (Y + 0.1 * cp.cos(2 * math.pi * t)))
    cx1 = 0.30 + 0.25 * math.cos(2 * math.pi * t)
    cy1 = 0.55 + 0.15 * math.sin(2 * math.pi * t)
    cx2 = 0.70 - 0.20 * math.sin(2 * math.pi * t)
    cy2 = 0.30 + 0.15 * math.cos(2 * math.pi * t)
    g1 = cp.exp(-((X - cx1) ** 2 + (Y - cy1) ** 2) / (2 * 0.06**2))
    g2 = cp.exp(-((X - cx2) ** 2 + (Y - cy2) ** 2) / (2 * 0.04**2))
    return (0.5 * base + 1.0 * g1 + 0.8 * g2).astype(cp.float32, copy=False)


def _grad_mag(u: cp.ndarray) -> cp.ndarray:
    gy, gx = cp.gradient(u)
    return cp.sqrt(gx * gx + gy * gy)


def _cp_mask_to_interval_set(mask: cp.ndarray):
    from .expressions import IntervalSet, _require_cupy

    cp_mod = _require_cupy()
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
    start_counts = cp_mod.bincount(start_rows, minlength=rows)
    stop_counts = cp_mod.bincount(stop_rows, minlength=rows)
    if int(cp_mod.any(start_counts != stop_counts)):
        raise RuntimeError("mask->interval conversion found mismatched start/stop counts")

    row_offsets = cp_mod.empty(rows + 1, dtype=cp_mod.int32)
    row_offsets[0] = 0
    if rows > 0:
        cp_mod.cumsum(start_counts.astype(cp_mod.int32, copy=False), dtype=cp_mod.int32, out=row_offsets[1:])

    key_beg = start_rows.astype(cp_mod.int64) * int(width) + start_cols.astype(cp_mod.int64)
    order = cp_mod.argsort(key_beg)
    begin = start_cols[order].astype(cp_mod.int32, copy=False)

    key_end = stop_rows.astype(cp_mod.int64) * int(width) + stop_cols.astype(cp_mod.int64)
    order2 = cp_mod.argsort(key_end)
    end = stop_cols[order2].astype(cp_mod.int32, copy=False)

    return IntervalSet(begin=begin, end=end, row_offsets=row_offsets)


def main():
    ap = argparse.ArgumentParser(description="Dynamic AMR regridding animation (CuPy)")
    ap.add_argument("--coarse", type=int, default=96)
    ap.add_argument("--ratio", type=int, default=2)
    ap.add_argument("--refine-frac", type=float, default=0.10, help="Top fraction of |grad(u)| refined")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--interval", type=int, default=80, help="Frame interval (ms)")
    ap.add_argument("--show-halos", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    width = height = int(args.coarse)
    ratio = int(args.ratio)

    # Static L0
    L0 = _build_coarse_set(width, height)
    coarse_mask_np = intervals_to_mask(L0, width)
    coarse_mask = cp.asarray(coarse_mask_np)

    if not args.plot:
        # Headless run prints simple stats per frame
        for f in range(args.frames):
            t = f / max(1, args.frames - 1)
            u = _analytic_field_t(width, height, t)
            g = _grad_mag(u) * coarse_mask
            valid = g[coarse_mask.astype(bool)]
            thresh = cp.percentile(valid, max(0.0, min(100.0, (1.0 - args.refine_frac) * 100.0))) if valid.size > 0 else 1e9
            R0 = _cp_mask_to_interval_set((g >= thresh).astype(cp.uint8))
            L1 = prolong_set(R0, ratio)
            C0 = evaluate(make_difference(make_input(L0), make_input(restrict_set(L1, ratio))))
            composite = evaluate(make_union(make_input(L1), make_input(prolong_set(C0, ratio))))
            # quick counts
            import numpy as np
            c0 = int(cp.asnumpy(C0.row_offsets)[-1])
            l1 = int(cp.asnumpy(L1.row_offsets)[-1])
            comp = int(cp.asnumpy(composite.row_offsets)[-1])
            print(f"frame {f:03d}: C0={c0} L1={l1} comp={comp}")
        return

    # Plot setup
    fig, axes = plt.subplots(1, 3 if not args.show_halos else 5, figsize=(15, 5))
    axes = axes.ravel()
    for ax in axes:
        ax.cla()

    # Placeholders
    coll_refine = None
    coll_composite_c0 = None
    coll_composite_l1 = None
    coll_h0 = None
    coll_h1 = None

    def init_plot():
        nonlocal coll_refine, coll_composite_c0, coll_composite_l1, coll_h0, coll_h1
        coll_refine = make_cell_collection(None, width, 1, facecolor="#ffb347")
        axes[0].add_collection(coll_refine)
        setup_cell_axes(axes[0], width, height, title="Refine mask (L0)")

        coll_composite_c0 = make_cell_collection(None, width, 1, facecolor="#c5c5c5", edgecolor="k")
        coll_composite_l1 = make_cell_collection(None, width, ratio, facecolor="#ff6961", edgecolor="k", linewidth=0.25)
        axes[1].add_collection(coll_composite_c0)
        axes[1].add_collection(coll_composite_l1)
        setup_cell_axes(axes[1], width, height, title="Composite (C0 ∪ L1)")

        if args.show_halos:
            coll_h0 = make_cell_collection(None, width, 1, facecolor="#7fa2ff")
            coll_h1 = make_cell_collection(None, width, ratio, facecolor="#c18aff")
            axes[2].add_collection(coll_h0)
            setup_cell_axes(axes[2], width, height, title="H0 (coarse)")
            axes[3].add_collection(coll_h1)
            setup_cell_axes(axes[3], width, height, title="H1 (fine)")

        # Static L0 view for reference
        ax_idx = 2 if not args.show_halos else 4
        axes[ax_idx].add_collection(make_cell_collection(L0, width, 1, facecolor="#bdbdbd"))
        setup_cell_axes(axes[ax_idx], width, height, title="L0 (coarse)")
        return []

    def update(frame_idx):
        t = frame_idx / max(1, args.frames - 1)
        u = _analytic_field_t(width, height, t)
        g = _grad_mag(u) * coarse_mask
        valid = g[coarse_mask.astype(bool)]
        thresh = cp.percentile(valid, max(0.0, min(100.0, (1.0 - args.refine_frac) * 100.0))) if valid.size > 0 else 1e9
        R0 = _cp_mask_to_interval_set((g >= thresh).astype(cp.uint8))
        L1 = prolong_set(R0, ratio)
        C0 = evaluate(make_difference(make_input(L0), make_input(restrict_set(L1, ratio))))
        composite_c0 = C0
        composite_l1 = L1

        # Halos (optional)
        if args.show_halos:
            H0 = ghost_zones(C0, halo_x=1, halo_y=1, width=width, height=height, bc="clamp")
            H1 = ghost_zones(L1, halo_x=max(0, 1 * ratio), halo_y=max(0, 1 * ratio), width=width * ratio, height=height * ratio, bc="clamp")
        else:
            H0 = None
            H1 = None

        # Rebuild collections each frame: remove existing artists robustly
        for art in list(axes[0].collections):
            art.remove()
        for art in list(axes[1].collections):
            art.remove()
        if args.show_halos:
            for art in list(axes[2].collections):
                art.remove()
            for art in list(axes[3].collections):
                art.remove()

        axes[0].add_collection(make_cell_collection(R0, width, 1, facecolor="#ffb347"))
        setup_cell_axes(axes[0], width, height, title="Refine mask (L0)")

        axes[1].add_collection(make_cell_collection(composite_c0, width, 1, facecolor="#c5c5c5", edgecolor="k"))
        axes[1].add_collection(make_cell_collection(composite_l1, width, ratio, facecolor="#ff6961", edgecolor="k", linewidth=0.25))
        setup_cell_axes(axes[1], width, height, title="Composite (C0 ∪ L1)")

        if args.show_halos:
            axes[2].add_collection(make_cell_collection(H0, width, 1, facecolor="#7fa2ff"))
            setup_cell_axes(axes[2], width, height, title="H0 (coarse)")
            axes[3].add_collection(make_cell_collection(H1, width, ratio, facecolor="#c18aff"))
            setup_cell_axes(axes[3], width, height, title="H1 (fine)")

        ax_idx = 2 if not args.show_halos else 4
        for art in list(axes[ax_idx].collections):
            art.remove()
        axes[ax_idx].add_collection(make_cell_collection(L0, width, 1, facecolor="#bdbdbd"))
        setup_cell_axes(axes[ax_idx], width, height, title="L0 (coarse)")

        return []

    init_plot()
    import matplotlib.animation as animation

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=args.frames,
        interval=args.interval,
        blit=False,
        repeat=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
