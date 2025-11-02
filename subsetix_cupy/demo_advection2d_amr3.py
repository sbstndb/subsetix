"""
2D linear advection with 3 AMR levels (CuPy) — simple, explicit interface halos, no subcycling.

Levels: L0 (coarse), L1 (mid), L2 (fine). Ratio R between each consecutive level (L1=L0*R, L2=L1*R → overall R^2).

Key ideas (no new kernels):
- Upwind 5-point stencil applied separately per level on padded arrays (values copied into interface halos).
- Regridding every N steps with hysteresis per level: refine0 on u0 (L1 placement), refine1 on u1 (L2 placement, gated inside L1).
- Value transfers on regridding: leave → restrict(child), enter → prolong(parent).

Usage:
  python -m subsetix_cupy.demo_advection2d_amr3 \
    --coarse 96 --ratio 2 --refine-frac 0.10 --hysteresis 0.5 --regrid-every 5 \
    --velocity 0.6 0.2 --cfl 0.9 --steps 300 --plot --interval 60 --ic sharp
"""

from __future__ import annotations

import argparse
import time
import math

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from .plot_utils import make_cell_collection, setup_cell_axes


def _init_condition(W: int, H: int, kind: str = "sharp", amp: float = 1.0) -> cp.ndarray:
    xx = cp.linspace(0.0, 1.0, W, dtype=cp.float32)
    yy = cp.linspace(0.0, 1.0, H, dtype=cp.float32)
    X, Y = cp.meshgrid(xx, yy)
    if kind == "gauss":
        g1 = cp.exp(-((X - 0.30) ** 2 + (Y - 0.65) ** 2) / (2 * 0.05**2))
        g2 = cp.exp(-((X - 0.70) ** 2 + (Y - 0.30) ** 2) / (2 * 0.06**2))
        ridge = cp.maximum(0.0, 1.0 - 25.0 * (Y - 0.5) ** 2)
        u = 0.9 * g1 + 0.7 * g2 + 0.3 * ridge
    elif kind == "disk":
        d1 = ((X - 0.35) ** 2 + (Y - 0.6) ** 2) <= (0.12 ** 2)
        d2 = ((X - 0.68) ** 2 + (Y - 0.32) ** 2) <= (0.10 ** 2)
        u = (d1 | d2).astype(cp.float32)
    elif kind == "square":
        s1 = (cp.abs(X - 0.30) <= 0.10) & (cp.abs(Y - 0.65) <= 0.10)
        s2 = (cp.abs(X - 0.70) <= 0.10) & (cp.abs(Y - 0.30) <= 0.12)
        u = (s1 | s2).astype(cp.float32)
    elif kind == "edge":
        u = (X + Y < 0.9).astype(cp.float32)
    else:  # sharp (narrow gaussians)
        g1 = cp.exp(-((X - 0.30) ** 2 + (Y - 0.65) ** 2) / (2 * 0.03**2))
        g2 = cp.exp(-((X - 0.70) ** 2 + (Y - 0.30) ** 2) / (2 * 0.035**2))
        ridge = cp.maximum(0.0, 1.0 - 60.0 * (Y - 0.5) ** 2)
        u = 1.2 * g1 + 1.0 * g2 + 0.25 * ridge
    if amp != 1.0:
        u = (u * amp).astype(cp.float32, copy=False)
    return u.astype(cp.float32, copy=False)


def _grad_mag(u: cp.ndarray) -> cp.ndarray:
    gy, gx = cp.gradient(u)
    return cp.sqrt(gx * gx + gy * gy)


def _neighbors_clamp(u: cp.ndarray):
    left = cp.concatenate([u[:, :1], u[:, :-1]], axis=1)
    right = cp.concatenate([u[:, 1:], u[:, -1:]], axis=1)
    down = cp.concatenate([u[:1, :], u[:-1, :]], axis=0)
    up = cp.concatenate([u[1:, :], u[-1:, :]], axis=0)
    return left, right, down, up


def _neighbors_wrap(u: cp.ndarray):
    left = cp.roll(u, 1, axis=1)
    right = cp.roll(u, -1, axis=1)
    down = cp.roll(u, 1, axis=0)
    up = cp.roll(u, -1, axis=0)
    return left, right, down, up


def _step_upwind(u: cp.ndarray, a: float, b: float, dt: float, dx: float, dy: float, bc: str) -> cp.ndarray:
    if bc == "wrap":
        left, right, down, up = _neighbors_wrap(u)
    else:
        left, right, down, up = _neighbors_clamp(u)
    du_dx = (u - left) / dx if a >= 0 else (right - u) / dx
    du_dy = (u - down) / dy if b >= 0 else (up - u) / dy
    return u - dt * (a * du_dx + b * du_dy)


def _restrict_mean(u_f: cp.ndarray, R: int) -> cp.ndarray:
    Hf, Wf = u_f.shape
    H = Hf // R
    W = Wf // R
    return u_f.reshape(H, R, W, R).mean(axis=(1, 3))


def _prolong_repeat(u_c: cp.ndarray, R: int) -> cp.ndarray:
    return cp.repeat(cp.repeat(u_c, R, axis=0), R, axis=1)


def _dilate_vn(mask: cp.ndarray, wrap: bool) -> cp.ndarray:
    if wrap:
        up = cp.roll(mask, -1, axis=0)
        down = cp.roll(mask, 1, axis=0)
        left = cp.roll(mask, 1, axis=1)
        right = cp.roll(mask, -1, axis=1)
    else:
        H, W = mask.shape
        up = cp.zeros_like(mask); up[1:, :] = mask[:-1, :]
        down = cp.zeros_like(mask); down[:-1, :] = mask[1:, :]
        left = cp.zeros_like(mask); left[:, 1:] = mask[:, :-1]
        right = cp.zeros_like(mask); right[:, :-1] = mask[:, 1:]
    return mask | up | down | left | right


def _erode_vn(mask: cp.ndarray, wrap: bool) -> cp.ndarray:
    """Erode boolean mask by 1 (Von Neumann). Keeps cells whose 4-neighbors stay inside.
    If wrap=True, periodic; otherwise zero outside domain.
    """
    if wrap:
        up = cp.roll(mask, -1, axis=0)
        down = cp.roll(mask, 1, axis=0)
        left = cp.roll(mask, 1, axis=1)
        right = cp.roll(mask, -1, axis=1)
    else:
        H, W = mask.shape
        up = cp.zeros_like(mask); up[1:, :] = mask[:-1, :]
        down = cp.zeros_like(mask); down[:-1, :] = mask[1:, :]
        left = cp.zeros_like(mask); left[:, 1:] = mask[:, :-1]
        right = cp.zeros_like(mask); right[:, :-1] = mask[:, 1:]
    return mask & up & down & left & right


def _dilate_mo(mask: cp.ndarray, wrap: bool) -> cp.ndarray:
    if wrap:
        up = cp.roll(mask, -1, axis=0)
        down = cp.roll(mask, 1, axis=0)
        left = cp.roll(mask, 1, axis=1)
        right = cp.roll(mask, -1, axis=1)
        up_left = cp.roll(up, 1, axis=1)
        up_right = cp.roll(up, -1, axis=1)
        down_left = cp.roll(down, 1, axis=1)
        down_right = cp.roll(down, -1, axis=1)
    else:
        H, W = mask.shape
        up = cp.zeros_like(mask); up[1:, :] = mask[:-1, :]
        down = cp.zeros_like(mask); down[:-1, :] = mask[1:, :]
        left = cp.zeros_like(mask); left[:, 1:] = mask[:, :-1]
        right = cp.zeros_like(mask); right[:, :-1] = mask[:, 1:]
        up_left = cp.zeros_like(mask); up_left[1:, 1:] = mask[:-1, :-1]
        up_right = cp.zeros_like(mask); up_right[1:, :-1] = mask[:-1, 1:]
        down_left = cp.zeros_like(mask); down_left[:-1, 1:] = mask[1:, :-1]
        down_right = cp.zeros_like(mask); down_right[:-1, :-1] = mask[1:, 1:]
    return mask | up | down | left | right | up_left | up_right | down_left | down_right


def _erode_mo(mask: cp.ndarray, wrap: bool) -> cp.ndarray:
    if wrap:
        up = cp.roll(mask, -1, axis=0)
        down = cp.roll(mask, 1, axis=0)
        left = cp.roll(mask, 1, axis=1)
        right = cp.roll(mask, -1, axis=1)
        up_left = cp.roll(up, 1, axis=1)
        up_right = cp.roll(up, -1, axis=1)
        down_left = cp.roll(down, 1, axis=1)
        down_right = cp.roll(down, -1, axis=1)
    else:
        H, W = mask.shape
        up = cp.zeros_like(mask); up[1:, :] = mask[:-1, :]
        down = cp.zeros_like(mask); down[:-1, :] = mask[1:, :]
        left = cp.zeros_like(mask); left[:, 1:] = mask[:, :-1]
        right = cp.zeros_like(mask); right[:, :-1] = mask[:, 1:]
        up_left = cp.zeros_like(mask); up_left[1:, 1:] = mask[:-1, :-1]
        up_right = cp.zeros_like(mask); up_right[1:, :-1] = mask[:-1, 1:]
        down_left = cp.zeros_like(mask); down_left[:-1, 1:] = mask[1:, :-1]
        down_right = cp.zeros_like(mask); down_right[:-1, :-1] = mask[1:, 1:]
    return mask & up & down & left & right & up_left & up_right & down_left & down_right
def _hysteresis_mask(g: cp.ndarray, frac_high: float, frac_low: float, prev: cp.ndarray | None) -> cp.ndarray:
    frac_high = max(0.0, min(1.0, float(frac_high)))
    frac_low = max(0.0, min(frac_high, float(frac_low)))
    if frac_high <= 0.0:
        return cp.zeros_like(g, dtype=cp.bool_)
    flat = g.ravel()
    t_high = cp.percentile(flat, (1.0 - frac_high) * 100.0)
    high = g >= t_high
    if prev is None or frac_low <= 0.0:
        return high
    t_low = cp.percentile(flat, (1.0 - frac_low) * 100.0)
    low = g >= t_low
    return high | (prev & low)


def _mask_to_interval_set(mask: cp.ndarray):
    """Convert a 2D boolean/binary CuPy mask to an IntervalSet (GPU)."""
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
    start_counts = cp_mod.bincount(start_rows, minlength=rows)
    row_offsets = cp_mod.empty(rows + 1, dtype=cp_mod.int32)
    row_offsets[0] = 0
    if rows > 0:
        cp_mod.cumsum(start_counts.astype(cp_mod.int32, copy=False), dtype=cp_mod.int32, out=row_offsets[1:])
    # Sort by row-major key
    key_beg = start_rows.astype(cp_mod.int64) * int(width) + start_cols.astype(cp_mod.int64)
    order_b = cp_mod.argsort(key_beg)
    begin = start_cols[order_b].astype(cp_mod.int32, copy=False)
    key_end = stop_rows.astype(cp_mod.int64) * int(width) + stop_cols.astype(cp_mod.int64)
    order_e = cp_mod.argsort(key_end)
    end = stop_cols[order_e].astype(cp_mod.int32, copy=False)
    return IntervalSet(begin=begin, end=end, row_offsets=row_offsets)


def main():
    ap = argparse.ArgumentParser(description="2D linear advection with 3 AMR levels (CuPy)")
    ap.add_argument("--coarse", type=int, default=96)
    ap.add_argument("--ratio", type=int, default=2, help="Refinement ratio between levels (L0->L1 and L1->L2)")
    ap.add_argument("--refine-frac", type=float, default=0.10)
    ap.add_argument("--hysteresis", type=float, default=0.5, help="Low threshold as fraction of high")
    ap.add_argument("--regrid-every", type=int, default=5)
    ap.add_argument("--velocity", type=float, nargs=2, default=[0.6, 0.2], metavar=("a", "b"))
    ap.add_argument("--cfl", type=float, default=0.9)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bc", type=str, choices=["clamp", "wrap"], default="clamp")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--grading", type=str, choices=["vn","moore"], default="moore", help="Grading connectivity (vn=4, moore=8)")
    ap.add_argument("--ic", type=str, choices=["gauss", "sharp", "disk", "square", "edge"], default="sharp")
    ap.add_argument("--ic-amp", type=float, default=1.0)
    args = ap.parse_args()

    W = H = int(args.coarse)
    R = int(args.ratio)
    a, b = float(args.velocity[0]), float(args.velocity[1])

    dx0, dy0 = 1.0 / W, 1.0 / H
    dx1, dy1 = dx0 / R, dy0 / R
    dx2, dy2 = dx1 / R, dy1 / R
    denom_f = abs(a) / dx2 + abs(b) / dy2
    dt = args.cfl / denom_f if denom_f > 0 else 0.0
    print(f"L0 {W}x{H}, R={R}, dt={dt:.6f} (CFL={args.cfl}, finest spacing)")

    # State on each level
    u0 = _init_condition(W, H, args.ic, args.ic_amp)
    u1 = _prolong_repeat(u0, R)
    u2 = _prolong_repeat(u1, R)

    # Initial masks: refine L0 -> L1; refine L1 -> L2 (gated inside L1)
    g0 = _grad_mag(u0)
    refine0 = _hysteresis_mask(g0, args.refine_frac, args.refine_frac * args.hysteresis, None)
    L1_mask = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)
    g1 = _grad_mag(u1)
    refine1_mid = _hysteresis_mask(g1, args.refine_frac, args.refine_frac * args.hysteresis, None)
    # Enforce grading: L2 must be at least one mid-cell away from L1 boundary
    refine1_mid = refine1_mid & (
        _erode_mo(L1_mask, wrap=(args.bc == 'wrap')) if args.grading == 'moore' else _erode_vn(L1_mask, wrap=(args.bc == 'wrap'))
    )
    # Child-forces-parent: any L2 present within an L0 coarse cell forces L1 on that parent
    coarse_force_l1 = refine1_mid.reshape(H, R, W, R).any(axis=(1, 3))
    refine0 = refine0 | coarse_force_l1
    L1_mask = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)
    L2_mask = _prolong_repeat(refine1_mid.astype(cp.uint8), R).astype(cp.bool_)

    # Plot setup
    if args.plot:
        fig, axes = plt.subplots(1, 5, figsize=(21, 5.5))
        ax0, ax1, ax2, ax3, ax4 = axes
        im0 = ax0.imshow(cp.asnumpy(u0), origin="lower", cmap="viridis")
        ax0.set_title("Coarse u0 (L0)"); ax0.set_axis_off()
        # Composite fine view: choose u2 where L2, else u1 (prolonged), else u0 (prolonged to fine)
        u1_f = _prolong_repeat(u1, R)
        u0_f = _prolong_repeat(u0, R * R)
        L1_f = _prolong_repeat(L1_mask.astype(cp.uint8), R).astype(cp.bool_)
        composite = cp.where(L2_mask, u2, cp.where(L1_f, u1_f, u0_f))
        im1 = ax1.imshow(cp.asnumpy(composite), origin="lower", cmap="viridis")
        ax1.set_title("Composite (fine)"); ax1.set_axis_off()
        # Level map with per-cell rectangles (coarse/mid/fine)
        coarse_residual = (~refine0)
        mid_residual = L1_mask & (~refine1_mid)
        fine_active = L2_mask
        c0_set = _mask_to_interval_set(coarse_residual)
        c1_set = _mask_to_interval_set(mid_residual)
        c2_set = _mask_to_interval_set(fine_active)
        # Draw each level at its native resolution: coarse=1, mid=R, fine=R^2
        coll_c0 = make_cell_collection(c0_set, base_dim=W, target_ratio=1, facecolor="#bdbdbd")
        coll_c1 = make_cell_collection(c1_set, base_dim=W, target_ratio=R, facecolor="#ffb347")
        coll_c2 = make_cell_collection(c2_set, base_dim=W, target_ratio=R * R, facecolor="#ff6961")
        ax2.add_collection(coll_c0)
        ax2.add_collection(coll_c1)
        ax2.add_collection(coll_c2)
        setup_cell_axes(ax2, W, H, title="Level map (cells)")
        # Halos overlays on fine grid: 1/2 = L0-L1 halos, 3/4 = L1-L2 halos
        _dilate = _dilate_mo if args.grading == 'moore' else _dilate_vn
        coarse_if = _dilate(refine0, wrap=(args.bc == 'wrap')) & (~refine0)
        coarse_if_f = _prolong_repeat(coarse_if.astype(cp.uint8), R * R).astype(cp.bool_)
        fine1_if = _dilate(L1_mask, wrap=(args.bc == 'wrap')) & (~L1_mask)
        fine1_if_f = _prolong_repeat(fine1_if.astype(cp.uint8), R).astype(cp.bool_)
        mid_if = _dilate(refine1_mid, wrap=(args.bc == 'wrap')) & (~refine1_mid)
        mid_if_f = _prolong_repeat(mid_if.astype(cp.uint8), R).astype(cp.bool_)
        fine2_if = _dilate(L2_mask, wrap=(args.bc == 'wrap')) & (~L2_mask)
        halo = cp.zeros_like(L2_mask, dtype=cp.int8)
        halo[coarse_if_f] = 1
        halo[fine1_if_f] = 2
        halo[mid_if_f] = 3
        halo[fine2_if] = 4
        halo_cmap = mcolors.ListedColormap([(0,0,0,0), "#7fa2ff", "#c18aff", "#8fd694", "#7fa2ff"])
        im3 = ax3.imshow(cp.asnumpy(halo), origin="lower", cmap=halo_cmap, vmin=0, vmax=4)
        ax3.set_title("Halos: L0-L1 (1/2), L1-L2 (3/4)"); ax3.set_axis_off()
        # Mid u1 (debug/inspection)
        im4 = ax4.imshow(cp.asnumpy(u1), origin="lower", cmap="magma")
        ax4.set_title("Mid u1 (L1)"); ax4.set_axis_off()
        fig.tight_layout()

    ev_s, ev_e = cp.cuda.Event(), cp.cuda.Event()
    total_gpu, total_wall = 0.0, 0.0

    for step in range(int(args.steps)):
        # Interface halos and pad arrays before stencil
        # L0<->L1
        coarse_if = (_dilate_mo if args.grading=='moore' else _dilate_vn)(refine0, wrap=(args.bc == 'wrap')) & (~refine0)
        u1_restr = _restrict_mean(u1, R)
        u0_pad = u0.copy()
        u0_pad[refine0] = u1_restr[refine0]
        u0_pad[coarse_if] = u1_restr[coarse_if]
        # L1<->L2
        mid_if = (_dilate_mo if args.grading=='moore' else _dilate_vn)(refine1_mid, wrap=(args.bc == 'wrap')) & (~refine1_mid)
        u2_restr = _restrict_mean(u2, R)
        u1_pad = u1.copy()
        # Outside L1 from prolong(u0)
        u1_pad[~L1_mask] = _prolong_repeat(u0, R)[~L1_mask]
        # Mid interface from restrict(u2)
        u1_pad[mid_if] = u2_restr[mid_if]
        # L2: outside from prolong(u1)
        u2_pad = u2.copy()
        u2_pad[~L2_mask] = _prolong_repeat(u1, R)[~L2_mask]

        ev_s.record(); wall0 = time.perf_counter()
        u0_new = _step_upwind(u0_pad, a, b, dt, dx0, dy0, args.bc)
        u1_new = _step_upwind(u1_pad, a, b, dt, dx1, dy1, args.bc)
        u2_new = _step_upwind(u2_pad, a, b, dt, dx2, dy2, args.bc)
        ev_e.record(); ev_e.synchronize(); wall1 = time.perf_counter()

        # Write-back keeping consistency
        u2 = u2.copy()
        u2[L2_mask] = u2_new[L2_mask]
        u2[~L2_mask] = _prolong_repeat(u1, R)[~L2_mask]

        u1 = u1.copy()
        u1[L1_mask & (~refine1_mid)] = u1_new[L1_mask & (~refine1_mid)]
        u1[refine1_mid] = _restrict_mean(u2_new, R)[refine1_mid]
        u1[~L1_mask] = _prolong_repeat(u0, R)[~L1_mask]

        u0 = u0.copy()
        u0[~refine0] = u0_new[~refine0]
        u0[refine0] = _restrict_mean(u1_new, R)[refine0]

        step_gpu = cp.cuda.get_elapsed_time(ev_s, ev_e)
        step_wall = (wall1 - wall0) * 1000.0
        total_gpu += step_gpu; total_wall += step_wall
        if (step % 20) == 0 or step == int(args.steps) - 1:
            print(f"step {step:04d}: compute {step_gpu:.3f} ms GPU, {step_wall:.3f} ms wall")

        # Regridding periodically (from level-specific gradients)
        if ((step + 1) % max(1, int(args.regrid_every))) == 0:
            # L0 -> L1
            g0 = _grad_mag(u0)
            refine0_new = _hysteresis_mask(g0, args.refine_frac, args.refine_frac * args.hysteresis, refine0)
            L1_mask_new = _prolong_repeat(refine0_new.astype(cp.uint8), R).astype(cp.bool_)
            # L1 -> L2 (gated inside L1)
            g1 = _grad_mag(u1)
            refine1_mid_new = _hysteresis_mask(g1, args.refine_frac, args.refine_frac * args.hysteresis, refine1_mid)
            # Enforce grading again with the updated L1
            refine1_mid_new = refine1_mid_new & (
                _erode_mo(L1_mask_new, wrap=(args.bc == 'wrap')) if args.grading == 'moore' else _erode_vn(L1_mask_new, wrap=(args.bc == 'wrap'))
            )
            # Child-forces-parent: if any mid child in a coarse parent is refined to L2, keep that parent at L1
            coarse_force_l1_new = refine1_mid_new.reshape(H, R, W, R).any(axis=(1, 3))
            refine0_new = refine0_new | coarse_force_l1_new
            L1_mask_new = _prolong_repeat(refine0_new.astype(cp.uint8), R).astype(cp.bool_)

            # Transfers L2<->L1
            leaving2 = refine1_mid & (~refine1_mid_new)
            if int(leaving2.any()) != 0:
                u2_restr_now = _restrict_mean(u2, R)
                u1 = u1.copy(); u1[leaving2] = u2_restr_now[leaving2]
            entering2 = (~refine1_mid) & refine1_mid_new
            if int(entering2.any()) != 0:
                u1_prol_now = _prolong_repeat(u1, R)
                L2_enter_f = _prolong_repeat(entering2.astype(cp.uint8), R).astype(cp.bool_)
                u2 = u2.copy(); u2[L2_enter_f] = u1_prol_now[L2_enter_f]

            # Transfers L1<->L0
            leaving1 = refine0 & (~refine0_new)
            if int(leaving1.any()) != 0:
                u1_restr_now = _restrict_mean(u1, R)
                u0 = u0.copy(); u0[leaving1] = u1_restr_now[leaving1]
            entering1 = (~refine0) & refine0_new
            if int(entering1.any()) != 0:
                u0_prol_now = _prolong_repeat(u0, R)
                u1 = u1.copy(); u1[_prolong_repeat(entering1.astype(cp.uint8), R).astype(cp.bool_)] = u0_prol_now[_prolong_repeat(entering1.astype(cp.uint8), R).astype(cp.bool_)]

            # Commit masks
            refine0 = refine0_new
            L1_mask = L1_mask_new
            refine1_mid = refine1_mid_new
            L2_mask = _prolong_repeat(refine1_mid.astype(cp.uint8), R).astype(cp.bool_)

        if args.plot:
            im0.set_data(cp.asnumpy(u0)); im0.set_clim(vmin=0.0, vmax=float(u0.max()))
            u1_f = _prolong_repeat(u1, R); u0_f = _prolong_repeat(u0, R * R); L1_f = _prolong_repeat(L1_mask.astype(cp.uint8), R).astype(cp.bool_)
            composite = cp.where(L2_mask, u2, cp.where(L1_f, u1_f, u0_f))
            im1.set_data(cp.asnumpy(composite)); im1.set_clim(vmin=0.0, vmax=float(composite.max()))
            # Rebuild level map (cells)
            for art in list(ax2.collections):
                art.remove()
            coarse_residual = (~refine0)
            mid_residual = L1_mask & (~refine1_mid)
            fine_active = L2_mask
            c0_set = _mask_to_interval_set(coarse_residual)
            c1_set = _mask_to_interval_set(mid_residual)
            c2_set = _mask_to_interval_set(fine_active)
            ax2.add_collection(make_cell_collection(c0_set, base_dim=W, target_ratio=1, facecolor="#bdbdbd"))
            ax2.add_collection(make_cell_collection(c1_set, base_dim=W, target_ratio=R, facecolor="#ffb347"))
            ax2.add_collection(make_cell_collection(c2_set, base_dim=W, target_ratio=R * R, facecolor="#ff6961"))
            setup_cell_axes(ax2, W, H, title="Level map (cells)")
            coarse_if = (_dilate_mo if args.grading=='moore' else _dilate_vn)(refine0, wrap=(args.bc == 'wrap')) & (~refine0)
            coarse_if_f = _prolong_repeat(coarse_if.astype(cp.uint8), R * R).astype(cp.bool_)
            fine1_if = (_dilate_mo if args.grading=='moore' else _dilate_vn)(L1_mask, wrap=(args.bc == 'wrap')) & (~L1_mask)
            fine1_if_f = _prolong_repeat(fine1_if.astype(cp.uint8), R).astype(cp.bool_)
            mid_if = (_dilate_mo if args.grading=='moore' else _dilate_vn)(refine1_mid, wrap=(args.bc == 'wrap')) & (~refine1_mid)
            mid_if_f = _prolong_repeat(mid_if.astype(cp.uint8), R).astype(cp.bool_)
            fine2_if = (_dilate_mo if args.grading=='moore' else _dilate_vn)(L2_mask, wrap=(args.bc == 'wrap')) & (~L2_mask)
            halo = cp.zeros_like(L2_mask, dtype=cp.int8)
            halo[coarse_if_f] = 1; halo[fine1_if_f] = 2; halo[mid_if_f] = 3; halo[fine2_if] = 4
            im3.set_data(cp.asnumpy(halo))
            im4.set_data(cp.asnumpy(u1))
            plt.pause(max(0.001, args.interval / 1000.0))

    print(f"Avg per step: {total_gpu/args.steps:.3f} ms GPU, {total_wall/args.steps:.3f} ms wall")


if __name__ == "__main__":
    main()
