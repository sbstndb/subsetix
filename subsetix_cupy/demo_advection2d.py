"""
Simple 2D linear advection demo on a full rectangular domain (CuPy).

PDE: u_t + a u_x + b u_y = 0 on [0,1]^2, constant velocity (a,b).

Numerics: 1st-order upwind (stable for CFL<=1). Two BC modes:
  - clamp: zero-gradient at domain boundary (replicate edge cell)
  - wrap: periodic in both directions

Usage examples:
  - Headless timings, no plot:
      python -m subsetix_cupy.demo_advection2d --width 256 --height 256 \
        --velocity 0.6 0.2 --cfl 0.8 --steps 500
  - With plot animation:
      python -m subsetix_cupy.demo_advection2d --width 192 --height 192 \
        --velocity 1.0 0.0 --cfl 0.9 --steps 400 --plot --interval 40
"""

from __future__ import annotations

import argparse
import time
import math

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from .morphology import full_interval_set
from .plot_utils import field_collection_from_dense, setup_cell_axes


def _init_condition(width: int, height: int) -> cp.ndarray:
    xx = cp.linspace(0.0, 1.0, width, dtype=cp.float32)
    yy = cp.linspace(0.0, 1.0, height, dtype=cp.float32)
    X, Y = cp.meshgrid(xx, yy)
    # Two bumps + ridge so that advection is visible
    g1 = cp.exp(-((X - 0.30) ** 2 + (Y - 0.65) ** 2) / (2 * 0.05**2))
    g2 = cp.exp(-((X - 0.70) ** 2 + (Y - 0.30) ** 2) / (2 * 0.06**2))
    ridge = cp.maximum(0.0, 1.0 - 25.0 * (Y - 0.5) ** 2)
    u0 = 0.9 * g1 + 0.7 * g2 + 0.3 * ridge
    return u0.astype(cp.float32, copy=False)


def _neighbors_clamp(u: cp.ndarray):
    # replicate edges (zero-gradient)
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


def step_upwind(u: cp.ndarray, a: float, b: float, dt: float, dx: float, dy: float, bc: str) -> cp.ndarray:
    if bc == "wrap":
        left, right, down, up = _neighbors_wrap(u)
    else:
        left, right, down, up = _neighbors_clamp(u)

    # Upwind differences
    if a >= 0:
        du_dx = (u - left) / dx
    else:
        du_dx = (right - u) / dx
    if b >= 0:
        du_dy = (u - down) / dy
    else:
        du_dy = (up - u) / dy

    return u - dt * (a * du_dx + b * du_dy)


def main():
    ap = argparse.ArgumentParser(description="2D linear advection (CuPy, upwind)")
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--velocity", type=float, nargs=2, default=[0.8, 0.0], metavar=("a", "b"))
    ap.add_argument("--cfl", type=float, default=0.9, help="CFL number (<=1)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bc", type=str, choices=["clamp", "wrap"], default="clamp")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--interval", type=int, default=40, help="ms between frames when plotting")
    args = ap.parse_args()

    w, h = int(args.width), int(args.height)
    a, b = float(args.velocity[0]), float(args.velocity[1])
    dx, dy = 1.0 / w, 1.0 / h
    # Stable time step from CFL: |a| dt/dx + |b| dt/dy <= CFL
    denom = (abs(a) / dx + abs(b) / dy)
    dt = (args.cfl / denom) if denom > 0 else 0.0
    print(f"Grid {w}x{h}, velocity=({a},{b}), dt={dt:.6f} (CFL={args.cfl})")

    u = _init_condition(w, h)

    if not args.plot:
        # Headless run with timings per step
        ev_start, ev_stop = cp.cuda.Event(), cp.cuda.Event()
        total_gpu_ms = 0.0
        total_wall_ms = 0.0
        for n in range(args.steps):
            ev_start.record()
            wall0 = time.perf_counter()
            u = step_upwind(u, a, b, dt, dx, dy, args.bc)
            ev_stop.record(); ev_stop.synchronize()
            wall1 = time.perf_counter()
            step_gpu = cp.cuda.get_elapsed_time(ev_start, ev_stop)
            step_wall = (wall1 - wall0) * 1000.0
            total_gpu_ms += step_gpu
            total_wall_ms += step_wall
            if (n % 50) == 0 or n == args.steps - 1:
                print(f"step {n:04d}: {step_gpu:.3f} ms GPU, {step_wall:.3f} ms wall")
        print(f"Avg per step: {total_gpu_ms/args.steps:.3f} ms GPU, {total_wall_ms/args.steps:.3f} ms wall")
        return

    # Plotting animation using per-cell rectangles for consistency
    interval_set = full_interval_set(w, h)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cmap = plt.get_cmap("viridis")

    def draw():
        # Rebuild collection each frame to keep logic simple and robust
        for art in list(ax.collections):
            art.remove()
        u_np = cp.asnumpy(u)
        coll = field_collection_from_dense(interval_set, u_np, base_dim=h, target_ratio=1, cmap=cmap)
        ax.add_collection(coll)
        setup_cell_axes(ax, w, h, title="2D Advection (upwind)")
        fig.canvas.draw_idle()

    # Initial draw
    draw()

    def update(_):
        nonlocal u
        u = step_upwind(u, a, b, dt, dx, dy, args.bc)
        draw()
        return []

    import matplotlib.animation as animation
    anim = animation.FuncAnimation(fig, update, frames=args.steps, interval=args.interval, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()

