"""
Samurai-style driver for two-level AMR advection using subsetix.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cupy as cp

from .api import Box, MRAdaptor, ScalarField, TwoLevelMesh, make_scalar_field
from .export import save_two_level_vtk
from .fields import synchronize_two_level


@dataclass(frozen=True)
class OutputOptions:
    directory: Path
    prefix: str = "amr2"
    every: int = 10
    ghost_halo: int = 1

    def __post_init__(self) -> None:
        if self.every <= 0:
            raise ValueError("OutputOptions.every must be > 0")


@dataclass(frozen=True)
class SimulationArgs:
    """
    Parameters controlling a two-level advection run.
    """

    min_corner: tuple[float, float] = (0.0, 0.0)
    max_corner: tuple[float, float] = (1.0, 1.0)
    velocity: tuple[float, float] = (1.0, 1.0)
    cfl: float = 0.5
    t0: float = 0.0
    tf: float = 0.1
    min_level: int = 4
    max_level: int = 5
    refine_fraction: float = 0.1
    grading: int = 1
    grading_mode: str = "von_neumann"
    output: OutputOptions | None = None
    bc: str = "clamp"


def parse_simulation_args(argv: Sequence[str] | None = None) -> SimulationArgs:
    parser = argparse.ArgumentParser(description="Samurai-style two-level AMR driver (subsetix)")
    parser.add_argument("--min-corner", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--max-corner", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--velocity", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--Ti", dest="t0", type=float, default=0.0, help="Initial time")
    parser.add_argument("--Tf", dest="tf", type=float, default=0.1, help="Final time")
    parser.add_argument("--min-level", type=int, default=4)
    parser.add_argument("--max-level", type=int, default=5)
    parser.add_argument("--refine-frac", type=float, default=0.1)
    parser.add_argument("--grading", type=int, default=1)
    parser.add_argument(
        "--grading-mode",
        type=str,
        choices=["von_neumann", "moore"],
        default="von_neumann",
    )
    parser.add_argument(
        "--bc",
        type=str,
        choices=["clamp", "wrap"],
        default="clamp",
    )
    parser.add_argument("--vtk-dir", type=Path, default=None, help="Directory for VTK outputs")
    parser.add_argument("--vtk-prefix", type=str, default="amr2")
    parser.add_argument("--vtk-every", type=int, default=10)
    parser.add_argument("--ghost-halo", type=int, default=1)

    ns = parser.parse_args(argv)
    output: OutputOptions | None = None
    if ns.vtk_dir is not None:
        output = OutputOptions(
            directory=Path(ns.vtk_dir),
            prefix=str(ns.vtk_prefix),
            every=int(ns.vtk_every),
            ghost_halo=int(ns.ghost_halo),
        )
    args = SimulationArgs(
        min_corner=(float(ns.min_corner[0]), float(ns.min_corner[1])),
        max_corner=(float(ns.max_corner[0]), float(ns.max_corner[1])),
        velocity=(float(ns.velocity[0]), float(ns.velocity[1])),
        cfl=float(ns.cfl),
        t0=float(ns.t0),
        tf=float(ns.tf),
        min_level=int(ns.min_level),
        max_level=int(ns.max_level),
        refine_fraction=float(ns.refine_frac),
        grading=int(ns.grading),
        grading_mode=ns.grading_mode,
        output=output,
        bc=ns.bc,
    )
    _validate_args(args)
    return args


def _validate_args(args: SimulationArgs) -> None:
    if args.max_level != args.min_level + 1:
        raise ValueError("two-level driver expects max_level = min_level + 1")
    if args.tf < args.t0:
        raise ValueError("tf must be >= t0")


def build_mesh(args: SimulationArgs, ratio: int = 2) -> TwoLevelMesh:
    base_resolution = 1 << args.min_level
    return TwoLevelMesh(
        Box(args.min_corner, args.max_corner),
        0,
        1,
        ratio=ratio,
        coarse_resolution=base_resolution,
    )


def update_ghost(field: ScalarField) -> None:
    geometry = field.mesh.geometry
    if geometry is None:
        raise RuntimeError("mesh geometry not initialised")
    coarse, fine = synchronize_two_level(
        field.coarse,
        field.fine,
        geometry.refine_mask,
        ratio=field.mesh.ratio,
        reducer="mean",
        fill_fine_outside=True,
    )
    field.coarse = coarse
    field.fine = fine


def _step_upwind(u: cp.ndarray, a: float, b: float, dt: float, dx: float, dy: float, bc: str) -> cp.ndarray:
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
    du_dx = (u - left) / dx if a >= 0 else (right - u) / dx
    du_dy = (u - down) / dy if b >= 0 else (up - u) / dy
    return u - dt * (a * du_dx + b * du_dy)


def save_snapshot(
    output: OutputOptions,
    mesh: TwoLevelMesh,
    field: ScalarField,
    step: int,
    *,
    time_value: float,
    bc: str,
) -> None:
    geometry = mesh.geometry
    if geometry is None:
        raise RuntimeError("mesh geometry not initialised")
    save_two_level_vtk(
        os.fspath(output.directory),
        output.prefix,
        step,
        coarse_field=field.coarse,
        fine_field=field.fine,
        refine_mask=geometry.refine_mask,
        coarse_only_mask=geometry.coarse_only_mask,
        fine_mask=geometry.fine_mask,
        dx_coarse=(mesh.box.max_corner[0] - mesh.box.min_corner[0]) / geometry.width,
        dy_coarse=(mesh.box.max_corner[1] - mesh.box.min_corner[1]) / geometry.height,
        ratio=mesh.ratio,
        time_value=time_value,
        ghost_halo=output.ghost_halo,
        bc=bc,
    )


def run_two_level_simulation(
    args: SimulationArgs,
    init_fn: Callable[[ScalarField], None],
) -> ScalarField:
    mesh = build_mesh(args)
    field = make_scalar_field("u", mesh)
    init_fn(field)
    update_ghost(field)

    adaptor = MRAdaptor(
        field,
        refine_fraction=args.refine_fraction,
        grading=args.grading,
        mode=args.grading_mode,
    )
    adaptor()

    geometry = mesh.geometry
    assert geometry is not None
    a, b = args.velocity
    dx0 = (args.max_corner[0] - args.min_corner[0]) / geometry.width
    dy0 = (args.max_corner[1] - args.min_corner[1]) / geometry.height
    dx1 = dx0 / mesh.ratio
    dy1 = dy0 / mesh.ratio
    denom = abs(a) / dx1 + abs(b) / dy1
    if denom > 0:
        dt = args.cfl / denom
    else:
        dt = max(args.tf - args.t0, 0.0)
    if dt <= 0.0:
        raise ValueError("invalid CFL configuration (dt <= 0)")

    if args.output is not None:
        args.output.directory.mkdir(parents=True, exist_ok=True)
        save_snapshot(args.output, mesh, field, 0, time_value=args.t0, bc=args.bc)

    unp1 = make_scalar_field("unp1", mesh)
    t = args.t0
    step = 0
    while t < args.tf:
        adaptor()
        dt_step = min(dt, args.tf - t)
        update_ghost(field)

        unp1.resize()
        unp1.coarse[...] = _step_upwind(field.coarse, a, b, dt_step, dx0, dy0, args.bc)
        unp1.fine[...] = _step_upwind(field.fine, a, b, dt_step, dx1, dy1, args.bc)
        field.swap(unp1)

        t = min(args.tf, t + dt_step)
        step += 1
        if args.output is not None and (step % args.output.every == 0 or math.isclose(t, args.tf)):
            save_snapshot(args.output, mesh, field, step, time_value=t, bc=args.bc)

    return field
