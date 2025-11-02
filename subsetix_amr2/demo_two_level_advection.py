"""
Two-level AMR advection demo built on top of subsetix_amr2.

Usage example:

    python -m subsetix_amr2.demo_two_level_advection \\
        --coarse 96 --steps 200 --refine-frac 0.10 --plot
"""

from __future__ import annotations

import argparse
import time

import cupy as cp
import numpy as np

from .geometry import interval_set_to_mask
from .simulation import (
    AMR2Simulation,
    AMRState,
    SimulationConfig,
    SimulationStats,
    TwoLevelVTKExporter,
)

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib import animation as mpl_animation
except ImportError:  # pragma: no cover - plotting is optional
    plt = None
    mcolors = None
    mpl_animation = None


def run_demo(args: argparse.Namespace):
    W = int(args.coarse)
    ratio = 2
    a, b = float(args.velocity[0]), float(args.velocity[1])

    config = SimulationConfig(
        coarse_resolution=W,
        velocity=(a, b),
        cfl=float(args.cfl),
        refine_fraction=float(args.refine_frac),
        grading=int(args.grading),
        grading_mode=args.grading_mode,
        bc=args.bc,
        ratio=ratio,
    )
    sim = AMR2Simulation(config)
    sim.initialize_square(amplitude=float(args.square_amp))

    history: list[tuple[int, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]] = []

    exporters = []
    if args.save_vtk:
        vtk_exporter = TwoLevelVTKExporter(
            args.save_vtk,
            prefix=args.vtk_prefix,
            every=args.vtk_every,
            ghost_halo=args.ghost_halo,
            bc=args.bc,
        )

        def _vtk_callback(state: AMRState, stats: SimulationStats) -> None:
            files = vtk_exporter(state, stats)
            if files and args.verbose:
                print(
                    f"[step {stats.step:04d}] saved VTK → "
                    + ", ".join(f"{k}={v}" for k, v in files.items())
                )

        exporters.append(_vtk_callback)

    def _verbose_listener(state: AMRState, stats: SimulationStats) -> None:
        if not args.verbose:
            return
        if stats.step > 0 and stats.step % args.print_every == 0:
            print(
                f"[step {stats.step:04d}] "
                f"refine cells={stats.refined_cells} ({stats.refined_fraction:.2%}), "
                f"||u_coarse||={stats.coarse_norm:.4f}, "
                f"||u_fine||={stats.fine_norm:.4f}"
            )

    def _animation_listener(state: AMRState, stats: SimulationStats) -> None:
        if not args.animate:
            return
        history.append(
            (
                stats.step,
                cp.array(state.coarse, copy=True),
                cp.array(state.fine, copy=True),
                cp.array(state.refine_mask, copy=True),
                cp.array(
                    interval_set_to_mask(state.geometry.coarse_only, state.refine_mask.shape[1]),
                    copy=True,
                ),
            )
        )

    listeners = [_verbose_listener, _animation_listener]

    print(
        f"Starting 2-level AMR demo: coarse={W}x{W}, ratio={ratio}, "
        f"dt={sim.dt:.6f}, steps={args.steps}"
    )

    if args.plot and plt is None:
        raise RuntimeError("matplotlib is required for plotting (install matplotlib)")

    t0 = time.perf_counter()
    final_state, final_stats = sim.run(
        steps=int(args.steps),
        regrid_every=1,
        exporters=exporters,
        listeners=listeners,
    )
    elapsed = time.perf_counter() - t0

    print(f"Completed {args.steps} steps in {elapsed:.2f}s ({elapsed / max(1, args.steps):.4f}s/step)")

    if args.plot or args.animate:
        _visualise(final_state, final_stats.dt, history if args.animate else None, args)


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
        "--square-amp",
        type=float,
        default=1.0,
        help="Scale factor applied to the square initial condition pattern",
    )
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
    ap.add_argument("--ic-amp", type=float, default=None, help=argparse.SUPPRESS)  # legacy flag
    return ap


def main(argv: list[str] | None = None):
    parser = create_argparser()
    args = parser.parse_args(argv)
    if getattr(args, "ic_amp", None) is not None:
        args.square_amp = args.ic_amp
    run_demo(args)


if __name__ == "__main__":
    main()

