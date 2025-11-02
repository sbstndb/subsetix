"""
2D linear advection with simple AMR (CuPy):

- Coarse grid L0 of size (H,W)
- Fine patch L1 from a refinement indicator on L0 (top fraction of |grad(u)|)
- Composite coverage: coarse residual C0 = L0 \ restrict(L1)
- Single global timestep (no subcycling), stable by choosing dt from fine spacing
- Interface values handled by copying between levels each step:
  * Coarse halo values (inside refine area) set from restriction of fine
  * Fine outside-patch values set from prolongation of coarse

This is a pedagogical demo; it avoids new kernels by using CuPy arrays and
boolean masks, and recomputes masks via percentile thresholding every N steps.

Usage:
  python -m subsetix_cupy.demo_advection2d_amr \
    --coarse 96 --ratio 2 --refine-frac 0.10 --regrid-every 5 \
    --velocity 0.6 0.2 --cfl 0.9 --steps 300 --plot --interval 60
"""

from __future__ import annotations

import argparse
import time
import math

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _init_condition(W: int, H: int) -> cp.ndarray:
    xx = cp.linspace(0.0, 1.0, W, dtype=cp.float32)
    yy = cp.linspace(0.0, 1.0, H, dtype=cp.float32)
    X, Y = cp.meshgrid(xx, yy)
    g1 = cp.exp(-((X - 0.30) ** 2 + (Y - 0.65) ** 2) / (2 * 0.05**2))
    g2 = cp.exp(-((X - 0.70) ** 2 + (Y - 0.30) ** 2) / (2 * 0.06**2))
    ridge = cp.maximum(0.0, 1.0 - 25.0 * (Y - 0.5) ** 2)
    return (0.9 * g1 + 0.7 * g2 + 0.3 * ridge).astype(cp.float32, copy=False)


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
    if a >= 0:
        du_dx = (u - left) / dx
    else:
        du_dx = (right - u) / dx
    if b >= 0:
        du_dy = (u - down) / dy
    else:
        du_dy = (up - u) / dy
    return u - dt * (a * du_dx + b * du_dy)


def _restrict_mean(u_f: cp.ndarray, R: int) -> cp.ndarray:
    Hf, Wf = u_f.shape
    H = Hf // R
    W = Wf // R
    return u_f.reshape(H, R, W, R).mean(axis=(1, 3))


def _prolong_repeat(u_c: cp.ndarray, R: int) -> cp.ndarray:
    return cp.repeat(cp.repeat(u_c, R, axis=0), R, axis=1)


def _percentile_mask(arr: cp.ndarray, frac_top: float) -> cp.ndarray:
    frac_top = max(0.0, min(1.0, float(frac_top)))
    if frac_top <= 0.0:
        return cp.zeros_like(arr, dtype=cp.bool_)
    valid = arr.ravel()
    if valid.size == 0:
        return cp.zeros_like(arr, dtype=cp.bool_)
    thresh = cp.percentile(valid, (1.0 - frac_top) * 100.0)
    return arr >= thresh


def main():
    ap = argparse.ArgumentParser(description="2D linear advection with simple AMR (CuPy)")
    ap.add_argument("--coarse", type=int, default=96)
    ap.add_argument("--ratio", type=int, default=2)
    ap.add_argument("--refine-frac", type=float, default=0.10)
    ap.add_argument("--regrid-every", type=int, default=5, help="Recompute refine mask every N steps")
    ap.add_argument("--velocity", type=float, nargs=2, default=[0.6, 0.2], metavar=("a", "b"))
    ap.add_argument("--cfl", type=float, default=0.9)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bc", type=str, choices=["clamp", "wrap"], default="clamp")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()

    W = H = int(args.coarse)
    R = int(args.ratio)
    a, b = float(args.velocity[0]), float(args.velocity[1])

    dx_c = 1.0 / W
    dy_c = 1.0 / H
    dx_f = dx_c / R
    dy_f = dy_c / R
    denom_f = (abs(a) / dx_f + abs(b) / dy_f)
    dt = (args.cfl / denom_f) if denom_f > 0 else 0.0
    print(f"Coarse {W}x{H}, ratio={R}, dt={dt:.6f} (CFL={args.cfl}, fine spacing)")

    # State on each level
    u0 = _init_condition(W, H)
    u1 = _prolong_repeat(u0, R)  # init fine as prolongation of coarse

    # Initial refine mask from gradient on coarse
    g0 = _grad_mag(u0)
    refine0 = _percentile_mask(g0, args.refine_frac)  # bool coarse mask
    L1_mask = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)

    def _regrid_from_coarse(u0, u1, refine0_old):
        g = _grad_mag(u0)
        refine_new = _percentile_mask(g, args.refine_frac)
        L1_new = _prolong_repeat(refine_new.astype(cp.uint8), R).astype(cp.bool_)
        # Transfer values between levels
        # Coarse <- Fine (cells leaving fine)
        leaving = refine0_old & (~refine_new)
        if int(leaving.any()) != 0:
            u1_restr = _restrict_mean(u1, R)
            u0 = u0.copy()
            u0[leaving] = u1_restr[leaving]
        # Fine <- Coarse (cells newly refined)
        entering_fine = (~refine0_old) & refine_new
        if int(entering_fine.any()) != 0:
            u0_prol = _prolong_repeat(u0, R)
            u1 = u1.copy()
            L1_new_mask = _prolong_repeat(entering_fine.astype(cp.uint8), R).astype(cp.bool_)
            u1[L1_new_mask] = u0_prol[L1_new_mask]
        return u0, u1, refine_new, L1_new

    # Plot setup
    if args.plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        ax0, ax1, ax2 = axes
        im0 = ax0.imshow(cp.asnumpy(u0), origin="lower", cmap="viridis", vmin=0.0, vmax=float(u0.max()))
        ax0.set_title("Coarse u (L0)")
        ax0.set_axis_off()
        composite = cp.where(L1_mask, u1, _prolong_repeat(u0, R))
        im1 = ax1.imshow(cp.asnumpy(composite), origin="lower", cmap="viridis", vmin=0.0, vmax=float(composite.max()))
        ax1.set_title("Composite fine view (L1 ∪ prolong(L0\\L1))")
        ax1.set_axis_off()
        # Level map: 0=coarse, 1=fine (on fine grid)
        level_map = L1_mask.astype(cp.int8)
        levels_cmap = mcolors.ListedColormap(["#bdbdbd", "#ff6961"])  # grey for coarse, red for fine
        im2 = ax2.imshow(cp.asnumpy(level_map), origin="lower", cmap=levels_cmap, vmin=0, vmax=1)
        ax2.set_title("Level map (0=coarse, 1=fine)")
        ax2.set_axis_off()
        fig.tight_layout()

    # Timing accumulators
    ev_start, ev_stop = cp.cuda.Event(), cp.cuda.Event()
    total_gpu_ms = 0.0
    total_wall_ms = 0.0

    for step in range(int(args.steps)):
        # Coupling at interfaces via values copy (ghost-like handling)
        u1_restr = _restrict_mean(u1, R)
        u0_pad = u0.copy()
        u0_pad[refine0] = u1_restr[refine0]
        u0_prol = _prolong_repeat(u0, R)
        u1_pad = u1.copy()
        u1_pad[~L1_mask] = u0_prol[~L1_mask]

        ev_start.record(); wall0 = time.perf_counter()
        # Advance both levels with same dt (stable wrt fine spacing)
        u0_new = _step_upwind(u0_pad, a, b, dt, dx_c, dy_c, args.bc)
        u1_new = _step_upwind(u1_pad, a, b, dt, dx_f, dy_f, args.bc)
        ev_stop.record(); ev_stop.synchronize(); wall1 = time.perf_counter()

        # Write back only in active sets
        u0 = u0.copy()
        u0[~refine0] = u0_new[~refine0]
        u1 = u1.copy()
        u1[L1_mask] = u1_new[L1_mask]
        # Keep consistency outside fine patch
        u1[~L1_mask] = u0_prol[~L1_mask]

        step_gpu = cp.cuda.get_elapsed_time(ev_start, ev_stop)
        step_wall = (wall1 - wall0) * 1000.0
        total_gpu_ms += step_gpu
        total_wall_ms += step_wall
        if (step % 20) == 0 or step == int(args.steps) - 1:
            print(f"step {step:04d}: compute {step_gpu:.3f} ms GPU, {step_wall:.3f} ms wall")

        # Regrid periodically from coarse indicator
        if ((step + 1) % max(1, int(args.regrid_every))) == 0:
            u0, u1, refine0, L1_mask = _regrid_from_coarse(u0, u1, refine0)

        if args.plot:
            # Update plots
            im0.set_data(cp.asnumpy(u0))
            composite = cp.where(L1_mask, u1, _prolong_repeat(u0, R))
            im1.set_data(cp.asnumpy(composite))
            im0.set_clim(vmin=0.0, vmax=float(u0.max()))
            im1.set_clim(vmin=0.0, vmax=float(composite.max()))
            # Update level map
            level_map = L1_mask.astype(cp.int8)
            im2.set_data(cp.asnumpy(level_map))
            plt.pause(max(0.001, args.interval / 1000.0))

    print(f"Avg per step: {total_gpu_ms/args.steps:.3f} ms GPU, {total_wall_ms/args.steps:.3f} ms wall")


if __name__ == "__main__":
    main()
