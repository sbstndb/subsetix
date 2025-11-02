"""
High-level façade to drive two-level AMR advection scenarios.

This module exposes a lightweight :class:`AMR2Simulation` class that wraps the
lower-level helpers in :mod:`subsetix_amr2` and provides a minimal imperative
API: initialise the state (currently square-based initial conditions), then run
the time loop while plugging exporters or listeners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Sequence

import cupy as cp

from .export import save_two_level_vtk
from .fields import prolong_coarse_to_fine, synchronize_two_level
from .geometry import TwoLevelGeometry, interval_set_to_mask
from .regrid import enforce_two_level_grading, gradient_magnitude, gradient_tag


@dataclass(frozen=True)
class SquareSpec:
    """
    Description of a square patch to initialise the scalar field.

    Parameters
    ----------
    center:
        Normalised coordinates (x, y) in [0, 1]^2.
    half_width:
        Half-extents (hx, hy); the square spans [cx-hx, cx+hx] × [cy-hy, cy+hy].
    value:
        Value assigned inside the square (combined with amplitude in
        :meth:`AMR2Simulation.initialize_square`).
    """

    center: tuple[float, float]
    half_width: tuple[float, float]
    value: float = 1.0


def create_square_field(
    width: int,
    height: int,
    squares: Sequence[SquareSpec],
    *,
    dtype: cp.dtype = cp.float32,
) -> cp.ndarray:
    """
    Build a coarse-level field made of axis-aligned squares.
    """

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not squares:
        raise ValueError("at least one SquareSpec is required")

    arr = cp.zeros((height, width), dtype=dtype)
    xx = cp.linspace(0.0, 1.0, width, dtype=cp.float32)
    yy = cp.linspace(0.0, 1.0, height, dtype=cp.float32)
    X, Y = cp.meshgrid(xx, yy)

    for spec in squares:
        cx, cy = spec.center
        hx, hy = spec.half_width
        mask = (cp.abs(X - cx) <= hx) & (cp.abs(Y - cy) <= hy)
        if spec.value != 1.0:
            arr = cp.where(mask, cp.asarray(spec.value, dtype=dtype), arr)
        else:
            arr = cp.where(mask, cp.asarray(1.0, dtype=dtype), arr)
    return arr.astype(dtype, copy=False)


@dataclass(frozen=True)
class SimulationConfig:
    coarse_resolution: int
    velocity: tuple[float, float]
    cfl: float = 0.9
    refine_fraction: float = 0.1
    grading: int = 1
    grading_mode: str = "von_neumann"
    bc: str = "clamp"
    ratio: int = 2


@dataclass
class AMRState:
    coarse: cp.ndarray
    fine: cp.ndarray
    refine_mask: cp.ndarray
    geometry: TwoLevelGeometry


@dataclass(frozen=True)
class SimulationStats:
    step: int
    time: float
    dt: float
    refined_cells: int
    refined_fraction: float
    coarse_norm: float
    fine_norm: float
    dx_coarse: float
    dy_coarse: float
    ratio: int


Callback = Callable[[AMRState, SimulationStats], None]


class AMR2Simulation:
    """
    Imperative driver that orchestrates two-level AMR advection using subsetix.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._validate_config()
        self.width = int(config.coarse_resolution)
        self.height = self.width
        self.dx_coarse = 1.0 / self.width
        self.dy_coarse = 1.0 / self.height
        self.dx_fine = self.dx_coarse / config.ratio
        self.dy_fine = self.dy_coarse / config.ratio
        denom = abs(config.velocity[0]) / self.dx_fine + abs(config.velocity[1]) / self.dy_fine
        self.dt = config.cfl / denom if denom > 0 else 0.0
        self.state: AMRState | None = None
        self.current_step: int = 0
        self.current_time: float = 0.0

    # ------------------------------------------------------------------ public
    def initialize_square(
        self,
        *,
        squares: Sequence[SquareSpec] | None = None,
        amplitude: float = 1.0,
        dtype: cp.dtype = cp.float32,
    ) -> None:
        """
        Initialise the simulation with one or more axis-aligned squares.

        If ``squares`` is ``None`` a default configuration matching the legacy
        demo is used (two disjoint squares).
        """

        specs = list(squares) if squares is not None else _DEFAULT_SQUARES
        field = create_square_field(self.width, self.height, specs, dtype=dtype)
        if amplitude != 1.0:
            field = field * float(amplitude)
        self.set_initial_condition(field)

    def set_initial_condition(self, coarse_field: cp.ndarray) -> None:
        """
        Set the coarse-level field and derive the fine level / geometry.
        """

        if not isinstance(coarse_field, cp.ndarray):
            raise TypeError("coarse_field must be a CuPy array")
        if coarse_field.shape != (self.height, self.width):
            raise ValueError(f"coarse_field must have shape {(self.height, self.width)}")
        coarse = coarse_field.astype(cp.float32, copy=False)
        fine = prolong_coarse_to_fine(coarse, self.config.ratio)
        refine_mask = self._refine_from_gradient(coarse)
        geometry = self._build_geometry(refine_mask)
        self.state = AMRState(coarse=coarse, fine=fine, refine_mask=refine_mask, geometry=geometry)
        self.current_step = 0
        self.current_time = 0.0

    def run(
        self,
        *,
        steps: int,
        regrid_every: int = 1,
        exporters: Iterable[Callback] | None = None,
        listeners: Iterable[Callback] | None = None,
    ) -> tuple[AMRState, SimulationStats]:
        """
        Execute the simulation loop for a fixed number of steps.
        """

        state = self._require_state()
        exporters_list = list(exporters or [])
        listeners_list = list(listeners or [])

        stats = self._build_stats(step=self.current_step)
        self._dispatch(exporters_list, state, stats)
        self._dispatch(listeners_list, state, stats)

        steps_int = int(steps)
        regrid_period = max(1, int(regrid_every))

        for _ in range(steps_int):
            self._synchronize()
            if self.current_step % regrid_period == 0:
                self._update_geometry()
            self._advance()
            self._synchronize()

            self.current_step += 1
            self.current_time += self.dt
            state = self._require_state()
            stats = self._build_stats(step=self.current_step)
            self._dispatch(exporters_list, state, stats)
            self._dispatch(listeners_list, state, stats)

        return state, stats

    # --------------------------------------------------------------- internals
    def _validate_config(self) -> None:
        if self.config.coarse_resolution <= 0:
            raise ValueError("coarse_resolution must be positive")
        if self.config.ratio < 1:
            raise ValueError("ratio must be >= 1")
        if self.config.bc not in {"clamp", "wrap"}:
            raise ValueError("bc must be 'clamp' or 'wrap'")
        if self.config.grading_mode not in {"von_neumann", "moore"}:
            raise ValueError("grading_mode must be 'von_neumann' or 'moore'")

    def _require_state(self) -> AMRState:
        if self.state is None:
            raise RuntimeError("simulation not initialised")
        return self.state

    def _dispatch(self, callbacks: Sequence[Callback], state: AMRState, stats: SimulationStats) -> None:
        for cb in callbacks:
            cb(state, stats)

    def _refine_from_gradient(self, field: cp.ndarray) -> cp.ndarray:
        grad = gradient_magnitude(field)
        mask = gradient_tag(grad, self.config.refine_fraction)
        graded = enforce_two_level_grading(
            mask,
            padding=self.config.grading,
            mode=self.config.grading_mode,
        )
        return graded.astype(cp.bool_, copy=False)

    def _build_geometry(self, refine_mask: cp.ndarray) -> TwoLevelGeometry:
        coarse_mask = cp.ones_like(refine_mask, dtype=cp.bool_)
        return TwoLevelGeometry.from_masks(
            refine_mask,
            ratio=self.config.ratio,
            coarse_mask=coarse_mask,
        )

    def _synchronize(self) -> None:
        state = self._require_state()
        coarse, fine = synchronize_two_level(
            state.coarse,
            state.fine,
            state.refine_mask,
            ratio=self.config.ratio,
            reducer="mean",
            fill_fine_outside=True,
        )
        state.coarse = coarse
        state.fine = fine

    def _update_geometry(self) -> None:
        state = self._require_state()
        refine_mask = self._refine_from_gradient(state.coarse)
        geometry = self._build_geometry(refine_mask)
        state.refine_mask = refine_mask
        state.geometry = geometry

    def _advance(self) -> None:
        state = self._require_state()
        a, b = self.config.velocity
        state.coarse = _step_upwind(state.coarse, a, b, self.dt, self.dx_coarse, self.dy_coarse, self.config.bc)
        state.fine = _step_upwind(state.fine, a, b, self.dt, self.dx_fine, self.dy_fine, self.config.bc)

    def _build_stats(self, *, step: int) -> SimulationStats:
        state = self._require_state()
        refined_cells = int(state.refine_mask.sum().item())
        total_cells = self.width * self.height
        refined_fraction = refined_cells / total_cells if total_cells else 0.0
        coarse_norm = float(cp.linalg.norm(state.coarse).item())
        fine_norm = float(cp.linalg.norm(state.fine).item())
        return SimulationStats(
            step=step,
            time=self.current_time,
            dt=self.dt,
            refined_cells=refined_cells,
            refined_fraction=refined_fraction,
            coarse_norm=coarse_norm,
            fine_norm=fine_norm,
            dx_coarse=self.dx_coarse,
            dy_coarse=self.dy_coarse,
            ratio=self.config.ratio,
        )


class TwoLevelVTKExporter:
    """
    Callable wrapper that dumps the current AMR state to VTK files.
    """

    def __init__(
        self,
        directory: str,
        *,
        prefix: str = "amr2",
        every: int = 1,
        ghost_halo: int = 0,
        bc: str = "clamp",
    ):
        self.directory = directory
        self.prefix = prefix
        self.every = max(1, int(every))
        self.ghost_halo = max(0, int(ghost_halo))
        self.bc = bc

    def __call__(self, state: AMRState, stats: SimulationStats) -> Dict[str, str] | None:
        if stats.step % self.every != 0:
            return None
        return save_two_level_vtk(
            self.directory,
            self.prefix,
            stats.step,
            coarse_field=state.coarse,
            fine_field=state.fine,
            refine_mask=state.refine_mask,
            coarse_only_mask=state.geometry.coarse_only_mask,
            fine_mask=state.geometry.fine_mask,
            dx_coarse=stats.dx_coarse,
            dy_coarse=stats.dy_coarse,
            ratio=stats.ratio,
            time_value=stats.time,
            ghost_halo=self.ghost_halo,
            bc=self.bc,
        )


def _neighbors(u: cp.ndarray, bc: str) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
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
    du_dx = (u - left) / dx if a >= 0 else (right - u) / dx
    du_dy = (u - down) / dy if b >= 0 else (up - u) / dy
    return u - dt * (a * du_dx + b * du_dy)


_DEFAULT_SQUARES: list[SquareSpec] = [
    SquareSpec(center=(0.20, 0.20), half_width=(0.10, 0.10)),
]
