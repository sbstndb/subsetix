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
import math

import cupy as cp
import numpy as np

from .export import save_two_level_vtk
from .fields import ActionField, Action, prolong_coarse_to_fine, synchronize_two_level
from .geometry import TwoLevelGeometry
from .regrid import (
    enforce_two_level_grading_set,
    gradient_magnitude,
    gradient_tag_threshold_set,
)
from subsetix_cupy.expressions import IntervalSet


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
    # For grids with a single cell we simply fill the whole array when needed.
    x_scale = width - 1 if width > 1 else 1
    y_scale = height - 1 if height > 1 else 1

    for spec in squares:
        cx, cy = spec.center
        hx, hy = spec.half_width

        x_lower = (cx - hx) * x_scale
        x_upper = (cx + hx) * x_scale
        y_lower = (cy - hy) * y_scale
        y_upper = (cy + hy) * y_scale

        x0 = max(0, min(width - 1, math.ceil(x_lower)))
        x1 = max(0, min(width - 1, math.floor(x_upper)))
        y0 = max(0, min(height - 1, math.ceil(y_lower)))
        y1 = max(0, min(height - 1, math.floor(y_upper)))

        if x0 > x1 or y0 > y1:
            continue

        value = float(spec.value)
        arr[y0 : y1 + 1, x0 : x1 + 1] = value
    return arr.astype(dtype, copy=False)


@dataclass(frozen=True)
class SimulationConfig:
    coarse_resolution: int
    velocity: tuple[float, float]
    cfl: float = 0.9
    refine_threshold: float = 0.05
    grading: int = 1
    grading_mode: str = "von_neumann"
    bc: str = "clamp"
    ratio: int = 2


@dataclass
class AMRState:
    coarse: cp.ndarray
    fine: cp.ndarray
    actions: ActionField
    geometry: TwoLevelGeometry


@dataclass(frozen=True)
class SimulationStats:
    step: int
    time: float
    dt: float
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
        refine_set = self._refine_from_gradient(coarse)
        refine_field = ActionField.full_grid(self.height, self.width, self.config.ratio, default=Action.KEEP)
        refine_field.set_from_interval_set(refine_set)
        geometry = self._build_geometry(refine_field)
        self.state = AMRState(coarse=coarse, fine=fine, actions=refine_field, geometry=geometry)
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
        if self.config.refine_threshold <= 0.0:
            raise ValueError("refine_threshold must be > 0")

    def _require_state(self) -> AMRState:
        if self.state is None:
            raise RuntimeError("simulation not initialised")
        return self.state

    def _dispatch(self, callbacks: Sequence[Callback], state: AMRState, stats: SimulationStats) -> None:
        for cb in callbacks:
            cb(state, stats)

    def _refine_from_gradient(self, field: cp.ndarray) -> IntervalSet:
        grad = gradient_magnitude(field)
        tagged = gradient_tag_threshold_set(grad, self.config.refine_threshold)
        graded = enforce_two_level_grading_set(
            tagged,
            padding=self.config.grading,
            mode=self.config.grading_mode,
            width=self.width,
            height=self.height,
        )
        return graded

    def _build_geometry(self, actions: ActionField, workspace=None) -> TwoLevelGeometry:
        return TwoLevelGeometry.from_action_field(actions, workspace=workspace)

    def _synchronize(self) -> None:
        state = self._require_state()
        coarse, fine = synchronize_two_level(
            state.coarse,
            state.fine,
            state.actions,
            ratio=self.config.ratio,
            reducer="mean",
            fill_fine_outside=True,
        )
        state.coarse = coarse
        state.fine = fine

    def _update_geometry(self) -> None:
        state = self._require_state()
        refine_set = self._refine_from_gradient(state.coarse)
        state.actions.set_from_interval_set(refine_set)
        workspace = state.geometry.workspace if state.geometry is not None else None
        geometry = self._build_geometry(state.actions, workspace=workspace)
        state.geometry = geometry

    def _advance(self) -> None:
        state = self._require_state()
        a, b = self.config.velocity
        state.coarse = _step_upwind(state.coarse, a, b, self.dt, self.dx_coarse, self.dy_coarse, self.config.bc)
        state.fine = _step_upwind(state.fine, a, b, self.dt, self.dx_fine, self.dy_fine, self.config.bc)

    def _build_stats(self, *, step: int) -> SimulationStats:
        state = self._require_state()
        return SimulationStats(
            step=step,
            time=self.current_time,
            dt=self.dt,
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
            refine_set=state.actions.refine_set(),
            coarse_only_set=state.geometry.coarse_only,
            fine_set=state.geometry.fine,
            dx_coarse=stats.dx_coarse,
            dy_coarse=stats.dy_coarse,
            ratio=stats.ratio,
            time_value=stats.time,
            ghost_halo=self.ghost_halo,
            bc=self.bc,
        )


_UPWIND_CLAMP_KERNEL_SRC = r"""
extern "C" __global__
void upwind_clamp(const float* __restrict__ u,
                  float* __restrict__ out,
                  int width,
                  int height,
                  float a,
                  float b,
                  float dt,
                  float dx,
                  float dy)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;
    float center = u[idx];

    int x_left = (x == 0) ? 0 : (x - 1);
    int x_right = (x == width - 1) ? (width - 1) : (x + 1);
    int y_down = (y == 0) ? 0 : (y - 1);
    int y_up = (y == height - 1) ? (height - 1) : (y + 1);

    float left = u[y * width + x_left];
    float right = u[y * width + x_right];
    float down = u[y_down * width + x];
    float up = u[y_up * width + x];

    float du_dx = (a >= 0.0f)
        ? (center - left) / dx
        : (right - center) / dx;
    float du_dy = (b >= 0.0f)
        ? (center - down) / dy
        : (up - center) / dy;

    out[idx] = center - dt * (a * du_dx + b * du_dy);
}
""";

_UPWIND_CLAMP_KERNEL = None


def _get_upwind_clamp_kernel():
    global _UPWIND_CLAMP_KERNEL
    if _UPWIND_CLAMP_KERNEL is None:
        _UPWIND_CLAMP_KERNEL = cp.RawKernel(
            _UPWIND_CLAMP_KERNEL_SRC, "upwind_clamp", options=("--std=c++11",)
        )
    return _UPWIND_CLAMP_KERNEL


def _step_upwind(u: cp.ndarray, a: float, b: float, dt: float, dx: float, dy: float, bc: str) -> cp.ndarray:
    if bc == "wrap":
        left = cp.roll(u, 1, axis=1)
        right = cp.roll(u, -1, axis=1)
        down = cp.roll(u, 1, axis=0)
        up = cp.roll(u, -1, axis=0)
        du_dx = (u - left) / dx if a >= 0 else (right - u) / dx
        du_dy = (u - down) / dy if b >= 0 else (up - u) / dy
        return u - dt * (a * du_dx + b * du_dy)

    kernel = _get_upwind_clamp_kernel()
    u32 = u.astype(cp.float32, copy=False)
    height, width = u32.shape
    out = cp.empty_like(u32)
    block = (32, 8)
    grid = (
        (width + block[0] - 1) // block[0],
        (height + block[1] - 1) // block[1],
    )
    kernel(
        grid,
        block,
        (
            u32,
            out,
            np.int32(width),
            np.int32(height),
            np.float32(a),
            np.float32(b),
            np.float32(dt),
            np.float32(dx),
            np.float32(dy),
        ),
    )
    return out


_DEFAULT_SQUARES: list[SquareSpec] = [
    SquareSpec(center=(0.30, 0.30), half_width=(0.10, 0.10)),
]
