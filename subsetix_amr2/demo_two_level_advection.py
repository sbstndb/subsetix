"""
Two-level AMR advection demo built on top of subsetix_amr2.

Usage example:

    python -m subsetix_amr2.demo_two_level_advection \\
        --coarse 96 --steps 200 --refine-threshold 0.05
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import time

import cupy as cp
import numpy as np
from .simulation import (
    AMR2Simulation,
    AMRState,
    SimulationConfig,
    SimulationStats,
    TwoLevelVTKExporter,
)


def run_demo(args: argparse.Namespace):
    W = int(args.coarse)
    ratio = 2
    a, b = float(args.velocity[0]), float(args.velocity[1])

    config = SimulationConfig(
        coarse_resolution=W,
        velocity=(a, b),
        cfl=float(args.cfl),
        refine_threshold=float(args.refine_threshold),
        grading=int(args.grading),
        grading_mode=args.grading_mode,
        bc=args.bc,
        ratio=ratio,
    )
    sim = AMR2Simulation(config)
    sim.initialize_square(amplitude=float(args.square_amp))

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
            print(f"[step {stats.step:04d}] t={stats.time:.6f}")

    listeners = [_verbose_listener]

    print(
        f"Starting 2-level AMR demo: coarse={W}x{W}, ratio={ratio}, "
        f"dt={sim.dt:.6f}, steps={args.steps}"
    )

    t0 = time.perf_counter()
    final_state, final_stats = sim.run(
        steps=int(args.steps),
        regrid_every=1,
        exporters=exporters,
        listeners=listeners,
    )
    elapsed = time.perf_counter() - t0

    print(f"Completed {args.steps} steps in {elapsed:.2f}s ({elapsed / max(1, args.steps):.4f}s/step)")

    # Live plotting/animation removed to avoid host-side overhead.


def create_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Two-level AMR advection demo using subsetix_amr2")
    ap.add_argument("--coarse", type=int, default=96, help="Coarse grid resolution (square domain)")
    ap.add_argument("--steps", type=int, default=300, help="Number of time steps")
    ap.add_argument(
        "--refine-threshold",
        type=float,
        default=0.05,
        help="Absolute gradient threshold for tagging (≥ threshold triggers refinement)",
    )
    ap.add_argument("--grading", type=int, default=1, help="Number of coarse cells used for grading dilation")
    ap.add_argument(
        "--grading-mode",
        type=str,
        default="von_neumann",
        choices=["von_neumann", "moore"],
        help="Neighborhood used for grading dilation",
    )
    ap.add_argument("--velocity", type=float, nargs=2, default=[0.6, 0.6], metavar=("a", "b"))
    ap.add_argument("--cfl", type=float, default=0.9, help="Courant number based on fine spacing")
    ap.add_argument("--bc", type=str, default="clamp", choices=["clamp", "wrap"], help="Boundary condition")
    ap.add_argument(
        "--square-amp",
        type=float,
        default=1.0,
        help="Scale factor applied to the square initial condition pattern",
    )
    # Live plotting/animation options removed
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
