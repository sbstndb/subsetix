"""
Two-level AMR advection demo built on top of subsetix_amr2.

Usage example:

    python -m subsetix_amr2.demo_two_level_advection \\
        --coarse 96 --steps 200 --refine-frac 0.10 --plot
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cupy as cp
import numpy as np

from .fields import prolong_coarse_to_fine, synchronize_two_level
from .geometry import TwoLevelGeometry, interval_set_to_mask
from .export import save_two_level_vtk
from .regrid import enforce_two_level_grading, gradient_magnitude, gradient_tag

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib import animation as mpl_animation
except ImportError:  # pragma: no cover - plotting is optional
    plt = None
    mcolors = None
    mpl_animation = None


def _init_condition(kind: str, width: int, height: int) -> cp.ndarray:
    xx = cp.linspace(0.0, 1.0, width, dtype=cp.float32)
    yy = cp.linspace(0.0, 1.0, height, dtype=cp.float32)
    X, Y = cp.meshgrid(xx, yy)
    if kind == "gauss":
        g1 = cp.exp(-((X - 0.30) ** 2 + (Y - 0.65) ** 2) / (2 * 0.05**2))
        g2 = cp.exp(-((X - 0.70) ** 2 + (Y - 0.30) ** 2) / (2 * 0.06**2))
        ridge = cp.maximum(0.0, 1.0 - 25.0 * (Y - 0.5) ** 2)
        return (0.9 * g1 + 0.7 * g2 + 0.3 * ridge).astype(cp.float32, copy=False)
    if kind == "square":
        sq = ((cp.abs(X - 0.3) <= 0.1) & (cp.abs(Y - 0.65) <= 0.1)) | (
            (cp.abs(X - 0.7) <= 0.1) & (cp.abs(Y - 0.3) <= 0.12)
        )
        return sq.astype(cp.float32)
    if kind == "disk":
        d1 = ((X - 0.35) ** 2 + (Y - 0.6) ** 2) <= 0.11**2
        d2 = ((X - 0.68) ** 2 + (Y - 0.32) ** 2) <= 0.09**2
        return (d1 | d2).astype(cp.float32)
    raise ValueError(f"unsupported initial condition '{kind}'")


def _neighbors(u: cp.ndarray, bc: str):
    if bc == "wrap":
        left = cp.roll(u, 1, axis=1)
        right = cp.roll(u, -1, axis=1)
        down = cp.roll(u, 1, axis=0)
        up = cp.roll(u, -1, axis=0)
    else:
        left = cp.concatenate([u[:, :1], u[:, :-1]], axis=1)
        right = cp.concatenate([u[:, 1:], u[:, -1:]], axis=1)
        down = cp.concatenate([u[:1, :], u[:-1, :]], axis=0)
        up = cp.concatenate([u[1:, :], u[-1:, :]], axis=0)
    return left, right, down, up


def _step_upwind(u: cp.ndarray, a: float, b: float, dt: float, dx: float, dy: float, bc: str) -> cp.ndarray:
    left, right, down, up = _neighbors(u, bc)
    if a >= 0:
        du_dx = (u - left) / dx
    else:
        du_dx = (right - u) / dx
    if b >= 0:
        du_dy = (u - down) / dy
    else:
        du_dy = (up - u) / dy
    return u - dt * (a * du_dx + b * du_dy)


@dataclass
class AMRState:
    coarse: cp.ndarray
    fine: cp.ndarray
    refine_mask: cp.ndarray
    geometry: TwoLevelGeometry


def _build_geometry(refine_mask: cp.ndarray, ratio: int) -> TwoLevelGeometry:
    coarse_mask = cp.ones_like(refine_mask, dtype=cp.bool_)
    return TwoLevelGeometry.from_masks(refine_mask, coarse_mask=coarse_mask, ratio=ratio)


def _refine_from_gradient(
    field: cp.ndarray,
    *,
    frac_high: float,
    grading: int,
    mode: str,
) -> cp.ndarray:
    grad = gradient_magnitude(field)
    mask = gradient_tag(grad, frac_high)
    graded = enforce_two_level_grading(mask, padding=grading, mode=mode)
    return graded


def run_demo(args: argparse.Namespace):
    W = int(args.coarse)
    H = W
    ratio = 2
    a, b = float(args.velocity[0]), float(args.velocity[1])
    dx0 = 1.0 / W
    dy0 = 1.0 / H
    dx1 = dx0 / ratio
    dy1 = dy0 / ratio
    denom = abs(a) / dx1 + abs(b) / dy1
    dt = args.cfl / denom if denom > 0 else 0.0

    coarse = _init_condition(args.ic, W, H) * float(args.ic_amp)
    fine = prolong_coarse_to_fine(coarse, ratio)

    refine_mask = _refine_from_gradient(
        coarse,
        frac_high=args.refine_frac,
        grading=args.grading,
        mode=args.grading_mode,
    )
    geometry = _build_geometry(refine_mask, ratio)
    state = AMRState(coarse=coarse, fine=fine, refine_mask=geometry.refine_mask, geometry=geometry)

    history = []
    vtk_every = max(1, int(args.vtk_every)) if args.save_vtk else 1
    ghost_halo = max(0, int(args.ghost_halo))

    def _maybe_save_vtk(step_idx: int, time_value: float) -> None:
        if args.save_vtk is None:
            return
        if step_idx % vtk_every != 0:
            return
        files = save_two_level_vtk(
            args.save_vtk,
            args.vtk_prefix,
            step_idx,
            coarse_field=state.coarse,
            fine_field=state.fine,
            refine_mask=state.refine_mask,
            coarse_only_mask=state.geometry.coarse_only_mask,
            fine_mask=state.geometry.fine_mask,
            dx_coarse=dx0,
            dy_coarse=dy0,
            ratio=ratio,
            time_value=time_value,
            ghost_halo=ghost_halo,
            bc=args.bc,
        )
        if args.verbose:
            print(
                f"[step {step_idx:04d}] saved VTK → "
                + ", ".join(f"{k}={v}" for k, v in files.items())
            )
    if args.animate:
        history.append(
            (
                0,
                cp.array(state.coarse, copy=True),
                cp.array(state.fine, copy=True),
                cp.array(state.refine_mask, copy=True),
                cp.array(interval_set_to_mask(state.geometry.coarse_only, state.refine_mask.shape[1]), copy=True),
            )
        )

    if args.plot and plt is None:
        raise RuntimeError("matplotlib is required for plotting (install matplotlib)")

    print(
        f"Starting 2-level AMR demo: coarse={W}x{H}, ratio={ratio}, "
        f"dt={dt:.6f}, steps={args.steps}"
    )

    _maybe_save_vtk(0, 0.0)

    t0 = time.perf_counter()
    for step in range(args.steps):
        # ensure coherence before regridding / stepping
        state.coarse, state.fine = synchronize_two_level(
            state.coarse,
            state.fine,
            state.refine_mask,
            ratio=ratio,
            reducer="mean",
            fill_fine_outside=True,
        )

        refined = _refine_from_gradient(
            state.coarse,
            frac_high=args.refine_frac,
            grading=args.grading,
            mode=args.grading_mode,
        )
        geometry = _build_geometry(refined, ratio)
        state.refine_mask = geometry.refine_mask
        state.geometry = geometry
        if args.verbose:
            refined_cells = int(state.refine_mask.sum().item())
            print(
                f"[step {step:04d}] regrid: refine cells={refined_cells} "
                f"({refined_cells / (W * H):.2%} of domain)"
            )

        state.coarse = _step_upwind(state.coarse, a, b, dt, dx0, dy0, args.bc)
        state.fine = _step_upwind(state.fine, a, b, dt, dx1, dy1, args.bc)

        state.coarse, state.fine = synchronize_two_level(
            state.coarse,
            state.fine,
            state.refine_mask,
            ratio=ratio,
            reducer="mean",
            fill_fine_outside=True,
        )

        if args.verbose and (step + 1) % args.print_every == 0:
            coarse_norm = float(cp.linalg.norm(state.coarse).item())
            fine_norm = float(cp.linalg.norm(state.fine).item())
            print(f"[step {step+1:04d}] ||u_coarse||={coarse_norm:.4f}, ||u_fine||={fine_norm:.4f}")

        if args.animate:
            history.append(
                (
                    step + 1,
                    cp.array(state.coarse, copy=True),
                    cp.array(state.fine, copy=True),
                    cp.array(state.refine_mask, copy=True),
                    cp.array(interval_set_to_mask(state.geometry.coarse_only, state.refine_mask.shape[1]), copy=True),
                )
            )

        _maybe_save_vtk(step + 1, (step + 1) * dt)

    elapsed = time.perf_counter() - t0
    print(f"Completed {args.steps} steps in {elapsed:.2f}s ({elapsed / max(1, args.steps):.4f}s/step)")

    if args.plot or args.animate:
        _visualise(state, dt, history if args.animate else None, args)


def _visualise(state: AMRState, dt: float, history, args: argparse.Namespace):
    if plt is None or mcolors is None:
        raise RuntimeError("matplotlib is required for plotting or animation")

    refined_mask = state.refine_mask
    coarse_only_mask = interval_set_to_mask(state.geometry.coarse_only, refined_mask.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Two-level AMR (ratio=2, dt={dt:.5f})")

    fine_img = axes[0].imshow(
        cp.asnumpy(state.fine),
        origin="lower",
        cmap="turbo",
        animated=args.animate,
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title("Fine field")
    fig.colorbar(fine_img, ax=axes[0], fraction=0.046)

    level_overlay = np.zeros((*refined_mask.shape, 3), dtype=np.float32)
    level_overlay[..., 0] = cp.asnumpy(refined_mask)  # red: refined
    level_overlay[..., 1] = cp.asnumpy(coarse_only_mask)  # green: coarse-only
    level_img = axes[1].imshow(level_overlay, origin="lower", animated=args.animate)
    axes[1].set_title("Level map (red = refine)")

    plt.tight_layout()

    if not args.animate:
        plt.show()
        return

    if mpl_animation is None:
        raise RuntimeError("matplotlib.animation required for --animate")
    if history is None or not history:
        raise RuntimeError("animation history is empty")

    def _update(frame_idx):
        (_, _, fine_frame, refine_frame, coarse_only_frame) = history[frame_idx]
        fine_img.set_array(cp.asnumpy(fine_frame))
        overlay = np.zeros((*refine_frame.shape, 3), dtype=np.float32)
        overlay[..., 0] = cp.asnumpy(refine_frame)
        overlay[..., 1] = cp.asnumpy(coarse_only_frame)
        level_img.set_array(overlay)
        axes[1].set_xlabel(f"frame {frame_idx+1}/{len(history)}")
        return fine_img, level_img

    anim = mpl_animation.FuncAnimation(
        fig,
        _update,
        frames=len(history),
        interval=max(20, args.interval),
        blit=True,
        repeat=args.loop,
    )

    if args.save_animation is not None:
        anim.save(args.save_animation, fps=1000.0 / max(1, args.interval))
        print(f"Saved animation to {args.save_animation}")
    else:
        plt.show()


def create_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Two-level AMR advection demo using subsetix_amr2")
    ap.add_argument("--coarse", type=int, default=96, help="Coarse grid resolution (square domain)")
    ap.add_argument("--steps", type=int, default=300, help="Number of time steps")
    ap.add_argument("--refine-frac", type=float, default=0.10, help="Fraction of cells tagged by the gradient percentile")
    ap.add_argument("--grading", type=int, default=1, help="Number of coarse cells used for grading dilation")
    ap.add_argument(
        "--grading-mode",
        type=str,
        default="von_neumann",
        choices=["von_neumann", "moore"],
        help="Neighborhood used for grading dilation",
    )
    ap.add_argument("--velocity", type=float, nargs=2, default=[0.6, 0.2], metavar=("a", "b"))
    ap.add_argument("--cfl", type=float, default=0.9, help="Courant number based on fine spacing")
    ap.add_argument("--bc", type=str, default="clamp", choices=["clamp", "wrap"], help="Boundary condition")
    ap.add_argument(
        "--ic",
        type=str,
        default="gauss",
        choices=["gauss", "square", "disk"],
        help="Initial condition pattern",
    )
    ap.add_argument("--ic-amp", type=float, default=1.0, help="Scale factor for the initial condition")
    ap.add_argument("--plot", action="store_true", help="Display matplotlib plots at the end")
    ap.add_argument("--animate", action="store_true", help="Generate a matplotlib animation of the run")
    ap.add_argument("--interval", type=int, default=60, help="Animation frame interval in milliseconds")
    ap.add_argument("--loop", action="store_true", help="Loop the animation when displaying")
    ap.add_argument("--save-animation", type=str, default=None, help="Path to save the animation (GIF/MP4)")
    ap.add_argument("--save-vtk", type=str, default=None, help="Directory where VTK outputs will be written")
    ap.add_argument("--vtk-every", type=int, default=10, help="Write VTK outputs every N steps (includes step 0)")
    ap.add_argument("--vtk-prefix", type=str, default="amr2", help="Filename prefix for VTK outputs")
    ap.add_argument("--ghost-halo", type=int, default=1, help="Ghost halo in coarse cells recorded in VTK outputs")
    ap.add_argument("--verbose", action="store_true", help="Print per-step diagnostics")
    ap.add_argument("--print-every", type=int, default=25, help="Diagnostic print frequency (steps)")
    return ap


def main(argv: list[str] | None = None):
    parser = create_argparser()
    args = parser.parse_args(argv)
    run_demo(args)


if __name__ == "__main__":
    main()
