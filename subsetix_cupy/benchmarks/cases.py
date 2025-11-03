from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .. import (
    IntervalField,
    IntervalSet,
    MultiLevel2D,
    evaluate,
    make_difference,
    make_input,
    make_intersection,
    make_union,
    prolong_field,
    prolong_level_field,
    prolong_level_sets,
    prolong_set,
    restrict_field,
    restrict_level_field,
    restrict_level_sets,
    restrict_set,
    interval_field_to_dense,
    step_upwind_interval,
)
from ..morphology import dilate_interval_set, erode_interval_set, ghost_zones, clip_interval_set
from ..multilevel import coarse_only, covered_by_fine
from ..expressions import _require_cupy
from . import BenchmarkCase, BenchmarkTarget
from .utils import (
    GeometrySpec,
    deterministic_interval_field,
    make_workspace,
    square_grid_interval_set,
    staircase_interval_set,
)

SMALL_SPEC = GeometrySpec(rows=256, width=256, tiles_x=3, tiles_y=3, fill_ratio=0.8)
LARGE_SPEC = GeometrySpec(rows=8192, width=8192, tiles_x=3, tiles_y=3, fill_ratio=0.8)
RATIO = 2


def _dense_upwind_zero(
    field: Any,
    *,
    a: float,
    b: float,
    dt: float,
    dx: float,
    dy: float,
    out: cp.ndarray | None = None,
) -> cp.ndarray:
    cp = _require_cupy()
    base = cp.asarray(field, dtype=cp.float32, copy=False)
    padded = cp.pad(base, ((1, 1), (1, 1)), mode="constant")
    center = padded[1:-1, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    down = padded[:-2, 1:-1]
    up = padded[2:, 1:-1]
    if a >= 0.0:
        du_dx = (center - left) / dx
    else:
        du_dx = (right - center) / dx
    if b >= 0.0:
        du_dy = (center - down) / dy
    else:
        du_dy = (up - center) / dy
    result = center - dt * (a * du_dx + b * du_dy)
    if out is not None:
        out[...] = result
        return out
    return result
STENCIL_SMALL = GeometrySpec(rows=512, width=512, tiles_x=1, tiles_y=1, fill_ratio=0.65)
STENCIL_LARGE = GeometrySpec(rows=2048, width=2048, tiles_x=1, tiles_y=1, fill_ratio=0.65)


def _shift_cells(spec: GeometrySpec, frac_x: float = 0.0, frac_y: float = 0.0) -> Tuple[int, int]:
    return int(round(spec.width * frac_x)), int(round(spec.rows * frac_y))


def _build_expression_case(
    *,
    name: str,
    description: str,
    spec: GeometrySpec,
    op: str,
) -> BenchmarkCase:
    cp = _require_cupy()

    shift_x1, _ = _shift_cells(spec, frac_x=1.0 / max(1, spec.tiles_x * 2))
    shift_x2, shift_y2 = _shift_cells(spec, frac_x=0.5 / max(1, spec.tiles_x), frac_y=0.5 / max(1, spec.tiles_y))

    lhs = square_grid_interval_set(cp, spec=spec)
    rhs = square_grid_interval_set(cp, spec=spec, shift_x=shift_x1)
    lhs_total = int(lhs.row_offsets[-1].item())
    rhs_total = int(rhs.row_offsets[-1].item())

    extra_total = 0
    expr_extra = None
    if op == "chain":
        extra = square_grid_interval_set(cp, spec=spec, shift_x=shift_x2, shift_y=shift_y2)
        extra_total = int(extra.row_offsets[-1].item())
        expr_extra = make_input(extra)

    workspace = make_workspace(cp)
    expr_lhs = make_input(lhs)
    expr_rhs = make_input(rhs)

    if op == "union":
        expr = make_union(expr_lhs, expr_rhs)
        input_intervals = lhs_total + rhs_total
    elif op == "intersection":
        expr = make_intersection(expr_lhs, expr_rhs)
        input_intervals = lhs_total + rhs_total
    elif op == "difference":
        expr = make_difference(expr_lhs, expr_rhs)
        input_intervals = lhs_total + rhs_total
    elif op == "chain":
        if expr_extra is None:
            raise RuntimeError("chain operation requires auxiliary geometry")
        expr = make_union(
            make_difference(make_intersection(expr_lhs, expr_rhs), expr_extra),
            make_intersection(expr_rhs, expr_extra),
        )
        input_intervals = lhs_total + rhs_total + extra_total
    else:
        raise ValueError(f"unsupported expression operation '{op}'")

    def _target():
        evaluate(expr, workspace)

    metadata = {
        "rows": spec.rows,
        "width": spec.width,
        "tiles_x": spec.tiles_x,
        "tiles_y": spec.tiles_y,
        "operation": op,
        "input_intervals": input_intervals,
    }
    return BenchmarkCase(
        name=name,
        description=description,
        setup=lambda _cp: BenchmarkTarget(func=_target, repeat=100, warmup=10, metadata=metadata),
    )


def _build_morph_case(
    *,
    name: str,
    description: str,
    spec: GeometrySpec,
    halo_x: int,
    halo_y: int,
    op: str,
) -> BenchmarkCase:
    cp = _require_cupy()
    geometry = square_grid_interval_set(cp, spec=spec)
    width = spec.width
    rows = spec.rows
    base_intervals = int(geometry.row_offsets[-1].item())
    def _clip(result: IntervalSet) -> IntervalSet:
        return clip_interval_set(result, width=width, height=rows)

    def _target_dilate():
        dilated = dilate_interval_set(geometry, halo_x=halo_x, halo_y=halo_y)
        return _clip(dilated)

    def _target_ghost():
        return ghost_zones(geometry, halo_x=halo_x, halo_y=halo_y, width=width, height=rows)

    def _target_erode():
        return erode_interval_set(geometry, halo_x=halo_x, halo_y=halo_y, width=width, height=rows)

    if op == "dilate":
        func = _target_dilate
    elif op == "ghost":
        func = _target_ghost
    elif op == "erode":
        func = _target_erode
    else:
        raise ValueError(f"unsupported morphology op '{op}'")

    metadata = {
        "width": width,
        "rows": rows,
        "halo_x": halo_x,
        "halo_y": halo_y,
        "intervals": base_intervals,
        "operation": op,
        "input_intervals": base_intervals,
        "boundary": "zero_exterior",
    }
    return BenchmarkCase(
        name=name,
        description=description,
        setup=lambda _cp: BenchmarkTarget(func=func, repeat=50, warmup=5, metadata=metadata),
    )


def _build_multilevel_case(name: str, description: str, spec: GeometrySpec, op: str) -> BenchmarkCase:
    cp = _require_cupy()
    base = square_grid_interval_set(cp, spec=spec)
    fine = prolong_set(base, RATIO)
    field_base = deterministic_interval_field(cp, base, width=spec.width)
    field_fine = prolong_field(field_base, RATIO)

    base_intervals = int(base.row_offsets[-1].item())
    fine_intervals = int(fine.row_offsets[-1].item())
    base_cells = field_base.cell_count
    fine_cells = field_fine.cell_count

    def _prolong_set_target():
        prolong_set(base, RATIO)

    def _restrict_set_target():
        restrict_set(fine, RATIO)

    def _prolong_field_target():
        prolong_field(field_base, RATIO)

    def _restrict_field_target():
        restrict_field(field_fine, RATIO)

    def _covered_by_fine_target():
        ml = MultiLevel2D.create(2, base_ratio=RATIO)
        ml.set_level(0, base)
        ml.set_level(1, fine)
        covered_by_fine(ml, 0)

    def _coarse_only_target():
        ml = MultiLevel2D.create(2, base_ratio=RATIO)
        ml.set_level(0, base)
        ml.set_level(1, fine)
        coarse_only(base, ml, 0)

    operations = {
        "prolong_set": (_prolong_set_target, {"entity": "IntervalSet", "input_intervals": base_intervals}),
        "restrict_set": (_restrict_set_target, {"entity": "IntervalSet", "input_intervals": fine_intervals}),
        "prolong_field": (_prolong_field_target, {"entity": "IntervalField", "input_intervals": base_cells}),
        "restrict_field": (_restrict_field_target, {"entity": "IntervalField", "input_intervals": fine_cells}),
        "covered_by_fine": (_covered_by_fine_target, {"entity": "IntervalSet", "input_intervals": fine_intervals}),
        "coarse_only": (
            _coarse_only_target,
            {"entity": "IntervalSet", "input_intervals": base_intervals + fine_intervals},
        ),
    }
    if op not in operations:
        raise ValueError(f"unsupported multilevel op '{op}'")
    func, metadata = operations[op]
    metadata = dict(metadata)
    metadata.update(
        {
            "base_rows": base.row_count,
            "base_intervals": base_intervals,
            "ratio": RATIO,
        }
    )
    return BenchmarkCase(
        name=name,
        description=description,
        setup=lambda _cp: BenchmarkTarget(func=func, repeat=40, warmup=5, metadata=metadata),
    )


def _build_stencil_case(
    *,
    name: str,
    description: str,
    spec: GeometrySpec,
    mode: str,
    variant: str,
) -> BenchmarkCase:
    cp = _require_cupy()
    base_square = square_grid_interval_set(cp, spec=spec)
    if variant == "square":
        interval_set = base_square
    elif variant == "staircase":
        interval_set = staircase_interval_set(cp, spec=spec, reference=base_square)
        cp.testing.assert_array_equal(interval_set.row_offsets, base_square.row_offsets)
    else:
        raise ValueError(f"unsupported stencil variant '{variant}'")
    field = deterministic_interval_field(cp, interval_set, width=spec.width)
    dense = interval_field_to_dense(field, width=spec.width, height=spec.rows, fill_value=0.0)
    row_ids = interval_set.interval_rows().astype(cp.int32, copy=False)

    a = 0.35
    b = -0.25
    dt = 0.01
    dx = 1.0 / max(1, spec.width)
    dy = 1.0 / max(1, spec.rows)

    dense_ref = _dense_upwind_zero(dense, a=a, b=b, dt=dt, dx=dx, dy=dy)
    interval_ref = cp.empty_like(field.values)
    step_upwind_interval(
        field,
        width=spec.width,
        height=spec.rows,
        a=a,
        b=b,
        dt=dt,
        dx=dx,
        dy=dy,
        out=interval_ref,
        row_ids=row_ids,
    )

    ref_field = IntervalField(
        interval_set=field.interval_set,
        values=interval_ref,
        interval_cell_offsets=field.interval_cell_offsets,
    )
    dense_from_interval = interval_field_to_dense(
        ref_field,
        width=spec.width,
        height=spec.rows,
        fill_value=0.0,
    )

    mask = cp.zeros((spec.rows, spec.width), dtype=cp.bool_)
    row_offsets_host = interval_set.row_offsets.get()
    begin_host = interval_set.begin.get()
    end_host = interval_set.end.get()
    for row in range(spec.rows):
        start = row_offsets_host[row]
        stop = row_offsets_host[row + 1]
        for idx in range(start, stop):
            b_idx = int(begin_host[idx])
            e_idx = int(end_host[idx])
            mask[row, b_idx:e_idx] = True

    cp.testing.assert_allclose(
        dense_ref[mask],
        dense_from_interval[mask],
        rtol=1e-5,
        atol=1e-5,
    )

    dense_scratch = cp.empty_like(dense)
    interval_scratch = cp.empty_like(field.values)

    active_cells = int(field.interval_cell_offsets[-1].item()) if field.interval_cell_offsets.size else 0

    if mode == "dense":
        func = lambda: _dense_upwind_zero(
            dense,
            a=a,
            b=b,
            dt=dt,
            dx=dx,
            dy=dy,
            out=dense_scratch,
        )
        metadata = {
            "entity": "Dense2D",
            "rows": spec.rows,
            "width": spec.width,
            "active_cells": active_cells,
            "a": a,
            "b": b,
            "dt": dt,
            "boundary": "zero_exterior",
        }
        input_count = active_cells
    elif mode == "interval":
        func = lambda: step_upwind_interval(
            field,
            width=spec.width,
            height=spec.rows,
            a=a,
            b=b,
            dt=dt,
            dx=dx,
            dy=dy,
            out=interval_scratch,
            row_ids=row_ids,
        )
        metadata = {
            "entity": "IntervalField",
            "rows": spec.rows,
            "width": spec.width,
            "active_cells": active_cells,
            "a": a,
            "b": b,
            "dt": dt,
            "boundary": "zero_exterior",
        }
        input_count = active_cells
    else:
        raise ValueError(f"unsupported stencil mode '{mode}'")

    metadata["mode"] = mode
    metadata["variant"] = variant
    metadata["input_intervals"] = input_count

    return BenchmarkCase(
        name=name,
        description=description,
        setup=lambda _cp: BenchmarkTarget(func=func, repeat=40, warmup=5, metadata=metadata),
    )


_CASES: List[BenchmarkCase] = [
    _build_expression_case(
        name="expr_union_small",
        description="Interval union (3x3 squares, small domain)",
        spec=SMALL_SPEC,
        op="union",
    ),
    _build_expression_case(
        name="expr_union_large",
        description="Interval union (3x3 squares, large domain)",
        spec=LARGE_SPEC,
        op="union",
    ),
    _build_expression_case(
        name="expr_chain_small",
        description="Nested union/intersection/difference (small domain)",
        spec=SMALL_SPEC,
        op="chain",
    ),
    _build_expression_case(
        name="expr_chain_large",
        description="Nested union/intersection/difference (large domain)",
        spec=LARGE_SPEC,
        op="chain",
    ),
    _build_expression_case(
        name="expr_difference_small",
        description="Interval difference (small domain)",
        spec=SMALL_SPEC,
        op="difference",
    ),
    _build_expression_case(
        name="expr_difference_large",
        description="Interval difference (large domain)",
        spec=LARGE_SPEC,
        op="difference",
    ),
    _build_morph_case(
        name="morph_dilate_small",
        description="Morphological dilation (small domain)",
        spec=SMALL_SPEC,
        halo_x=4,
        halo_y=2,
        op="dilate",
    ),
    _build_morph_case(
        name="morph_dilate_large",
        description="Morphological dilation (large domain)",
        spec=LARGE_SPEC,
        halo_x=4,
        halo_y=2,
        op="dilate",
    ),
    _build_morph_case(
        name="morph_ghost_small",
        description="Ghost zone generation (zero exterior, small domain)",
        spec=SMALL_SPEC,
        halo_x=3,
        halo_y=3,
        op="ghost",
    ),
    _build_morph_case(
        name="morph_ghost_large",
        description="Ghost zone generation (zero exterior, large domain)",
        spec=LARGE_SPEC,
        halo_x=3,
        halo_y=3,
        op="ghost",
    ),
    _build_morph_case(
        name="morph_erode_small",
        description="Morphological erosion (small domain)",
        spec=SMALL_SPEC,
        halo_x=2,
        halo_y=2,
        op="erode",
    ),
    _build_morph_case(
        name="morph_erode_large",
        description="Morphological erosion (large domain)",
        spec=LARGE_SPEC,
        halo_x=2,
        halo_y=2,
        op="erode",
    ),
    _build_multilevel_case(
        name="multilevel_prolong_set_small",
        description="Multilevel interval prolongation (small domain)",
        spec=SMALL_SPEC,
        op="prolong_set",
    ),
    _build_multilevel_case(
        name="multilevel_prolong_set_large",
        description="Multilevel interval prolongation (large domain)",
        spec=LARGE_SPEC,
        op="prolong_set",
    ),
    _build_multilevel_case(
        name="multilevel_restrict_set_small",
        description="Multilevel interval restriction (small domain)",
        spec=SMALL_SPEC,
        op="restrict_set",
    ),
    _build_multilevel_case(
        name="multilevel_restrict_set_large",
        description="Multilevel interval restriction (large domain)",
        spec=LARGE_SPEC,
        op="restrict_set",
    ),
    _build_multilevel_case(
        name="multilevel_prolong_field_small",
        description="Multilevel field prolongation (small domain)",
        spec=SMALL_SPEC,
        op="prolong_field",
    ),
    _build_multilevel_case(
        name="multilevel_prolong_field_large",
        description="Multilevel field prolongation (large domain)",
        spec=LARGE_SPEC,
        op="prolong_field",
    ),
    _build_multilevel_case(
        name="multilevel_restrict_field_small",
        description="Multilevel field restriction (small domain)",
        spec=SMALL_SPEC,
        op="restrict_field",
    ),
    _build_multilevel_case(
        name="multilevel_restrict_field_large",
        description="Multilevel field restriction (large domain)",
        spec=LARGE_SPEC,
        op="restrict_field",
    ),
    _build_multilevel_case(
        name="multilevel_covered_by_fine_small",
        description="Level coverage from fine set (small domain)",
        spec=SMALL_SPEC,
        op="covered_by_fine",
    ),
    _build_multilevel_case(
        name="multilevel_covered_by_fine_large",
        description="Level coverage from fine set (large domain)",
        spec=LARGE_SPEC,
        op="covered_by_fine",
    ),
    _build_multilevel_case(
        name="multilevel_coarse_only_small",
        description="Coarse-only derivation (small domain)",
        spec=SMALL_SPEC,
        op="coarse_only",
    ),
    _build_multilevel_case(
        name="multilevel_coarse_only_large",
        description="Coarse-only derivation (large domain)",
        spec=LARGE_SPEC,
        op="coarse_only",
    ),
    _build_stencil_case(
        name="stencil_dense_square_small",
        description="Dense upwind stencil (zero exterior, square mask, small domain)",
        spec=STENCIL_SMALL,
        mode="dense",
        variant="square",
    ),
    _build_stencil_case(
        name="stencil_interval_square_small",
        description="Interval-field upwind stencil (square mask, small domain)",
        spec=STENCIL_SMALL,
        mode="interval",
        variant="square",
    ),
    _build_stencil_case(
        name="stencil_dense_square_large",
        description="Dense upwind stencil (zero exterior, square mask, large domain)",
        spec=STENCIL_LARGE,
        mode="dense",
        variant="square",
    ),
    _build_stencil_case(
        name="stencil_interval_square_large",
        description="Interval-field upwind stencil (square mask, large domain)",
        spec=STENCIL_LARGE,
        mode="interval",
        variant="square",
    ),
    _build_stencil_case(
        name="stencil_dense_stair_small",
        description="Dense upwind stencil (zero exterior, staircase mask, small domain)",
        spec=STENCIL_SMALL,
        mode="dense",
        variant="staircase",
    ),
    _build_stencil_case(
        name="stencil_interval_stair_small",
        description="Interval-field upwind stencil (staircase mask, small domain)",
        spec=STENCIL_SMALL,
        mode="interval",
        variant="staircase",
    ),
    _build_stencil_case(
        name="stencil_dense_stair_large",
        description="Dense upwind stencil (zero exterior, staircase mask, large domain)",
        spec=STENCIL_LARGE,
        mode="dense",
        variant="staircase",
    ),
    _build_stencil_case(
        name="stencil_interval_stair_large",
        description="Interval-field upwind stencil (staircase mask, large domain)",
        spec=STENCIL_LARGE,
        mode="interval",
        variant="staircase",
    ),
]


CASES: List[BenchmarkCase] = list(_CASES)

__all__ = ["CASES"]
