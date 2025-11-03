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

from .export import save_two_level_vtk
from .fields import (
    ActionField,
    Action,
    synchronize_interval_fields,
)
from .geometry import TwoLevelGeometry
from .regrid import (
    enforce_two_level_grading_set,
    gradient_magnitude,
    gradient_tag_threshold_set,
)
from subsetix_cupy.expressions import IntervalSet
from subsetix_cupy.interval_field import IntervalField, create_interval_field
from subsetix_cupy.interval_stencil import step_upwind_interval_field
from subsetix_cupy.morphology import full_interval_set
from subsetix_cupy.multilevel import prolong_field


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


def build_square_interval_field(
    width: int,
    height: int,
    squares: Sequence[SquareSpec],
    *,
    dtype: cp.dtype = cp.float32,
) -> IntervalField:
    """
    Build an interval-backed coarse field populated with axis-aligned squares.
    """

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    specs = list(squares)
    if not specs:
        raise ValueError("at least one SquareSpec is required")

    interval = full_interval_set(width, height)
    field = create_interval_field(interval, fill_value=0.0, dtype=dtype)
    grid = field.values.reshape(height, width)

    x_scale = width - 1 if width > 1 else 1
    y_scale = height - 1 if height > 1 else 1

    for spec in specs:
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
        if value == 0.0:
            continue
        grid[y0 : y1 + 1, x0 : x1 + 1] = value

    return field


@dataclass(frozen=True)
class SimulationConfig:
    coarse_resolution: int
    velocity: tuple[float, float]
    cfl: float = 0.9
    refine_threshold: float = 0.05
    grading: int = 1
    grading_mode: str = "von_neumann"
    ratio: int = 2


@dataclass
class AMRState:
    coarse_field: IntervalField
    fine_field: IntervalField
    actions: ActionField
    geometry: TwoLevelGeometry

    @property
    def coarse(self) -> cp.ndarray:
        height = self.geometry.height
        width = self.geometry.width
        return self.coarse_field.values.reshape(height, width)

    @property
    def fine(self) -> cp.ndarray:
        ratio = self.geometry.ratio
        height = self.geometry.height * ratio
        width = self.geometry.width * ratio
        return self.fine_field.values.reshape(height, width)


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
        coarse_field = build_square_interval_field(self.width, self.height, specs, dtype=dtype)
        if amplitude != 1.0:
            coarse_field.values *= coarse_field.values.dtype.type(float(amplitude))
        self.set_initial_field(coarse_field)

    def set_initial_field(self, coarse_field: IntervalField) -> None:
        """
        Attach an interval-backed coarse field and derive fine level / geometry.
        """

        if not isinstance(coarse_field, IntervalField):
            raise TypeError("coarse_field must be an IntervalField")
        height = self.height
        width = self.width
        interval = coarse_field.interval_set
        if interval.row_count != height:
            raise ValueError("coarse_field height mismatch with simulation domain")
        row_ids = interval.rows_index()
        expected_rows = cp.arange(height, dtype=cp.int32)
        if row_ids.size != expected_rows.size or not bool(cp.all(row_ids == expected_rows)):
            raise ValueError("coarse_field rows must cover the dense range [0, height)")
        intervals_per_row = interval.row_offsets[1:] - interval.row_offsets[:-1]
        if not bool(cp.all(intervals_per_row == 1)):
            raise ValueError("coarse_field must contain exactly one interval per row")
        if not bool(cp.all(interval.begin == 0)) or not bool(cp.all(interval.end == width)):
            raise ValueError("coarse_field must cover the full width on every row")
        if int(coarse_field.values.size) != width * height:
            raise ValueError("coarse_field must hold width * height cells")

        coarse_grid = coarse_field.values.reshape(height, width)
        refine_set = self._refine_from_gradient(coarse_grid)
        actions = ActionField.full_grid(height, width, self.config.ratio, default=Action.KEEP)
        actions.set_from_interval_set(refine_set)
        geometry = self._build_geometry(actions)

        fine_field = prolong_field(coarse_field, self.config.ratio)

        self.state = AMRState(
            coarse_field=coarse_field,
            fine_field=fine_field,
            actions=actions,
            geometry=geometry,
        )
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
        synchronize_interval_fields(
            state.coarse_field,
            state.fine_field,
            state.actions,
            ratio=self.config.ratio,
            reducer="mean",
            fill_fine_outside=True,
            copy=False,
        )

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
        geometry = state.geometry
        ratio = geometry.ratio
        state.coarse_field = step_upwind_interval_field(
            state.coarse_field,
            width=geometry.width,
            height=geometry.height,
            a=a,
            b=b,
            dt=self.dt,
            dx=self.dx_coarse,
            dy=self.dy_coarse,
        )
        state.fine_field = step_upwind_interval_field(
            state.fine_field,
            width=geometry.width * ratio,
            height=geometry.height * ratio,
            a=a,
            b=b,
            dt=self.dt,
            dx=self.dx_fine,
            dy=self.dy_fine,
        )

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
    ):
        self.directory = directory
        self.prefix = prefix
        self.every = max(1, int(every))
        self.ghost_halo = max(0, int(ghost_halo))

    def __call__(self, state: AMRState, stats: SimulationStats) -> Dict[str, str] | None:
        if stats.step % self.every != 0:
            return None
        return save_two_level_vtk(
            self.directory,
            self.prefix,
            stats.step,
            coarse_field=state.coarse_field,
            fine_field=state.fine_field,
            refine_set=state.actions.refine_set(),
            coarse_only_set=state.geometry.coarse_only,
            fine_set=state.geometry.fine,
            dx_coarse=stats.dx_coarse,
            dy_coarse=stats.dy_coarse,
            ratio=stats.ratio,
            time_value=stats.time,
            ghost_halo=self.ghost_halo,
            width=state.geometry.width,
            height=state.geometry.height,
        )


_DEFAULT_SQUARES: list[SquareSpec] = [
    SquareSpec(center=(0.30, 0.30), half_width=(0.10, 0.10)),
]
