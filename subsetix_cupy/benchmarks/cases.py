from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Tuple

import cupy as cp

from .. import (
    IntervalField,
    IntervalSet,
    MultiLevel2D,
    MultiLevelField2D,
    create_interval_field,
    evaluate,
    make_complement,
    make_difference,
    make_input,
    make_intersection,
    make_symmetric_difference,
    make_union,
    prolong_field,
    prolong_level_field,
    prolong_level_sets,
    prolong_set,
    restrict_field,
    restrict_level_field,
    restrict_level_sets,
    restrict_set,
    step_upwind_interval,
    locate_interval_cells,
)
from ..morphology import dilate_interval_set, erode_interval_set, ghost_zones, clip_interval_set, full_interval_set
from ..multilevel import coarse_only, covered_by_fine
from ..export_vtk import write_unstructured_quads_vtu
from ..expressions import _require_cupy, _align_interval_sets
from . import BenchmarkCase, BenchmarkTarget
from .utils import (
    GeometrySpec,
    deterministic_interval_field,
    make_workspace,
    square_grid_interval_set,
    staircase_interval_set,
    random_interval_set,
)

SMALL_SPEC = GeometrySpec(rows=256, width=256, tiles_x=3, tiles_y=3, fill_ratio=0.8)
LARGE_SPEC = GeometrySpec(rows=8192, width=8192, tiles_x=3, tiles_y=3, fill_ratio=0.8)
RATIO = 2
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
    elif op == "symmetric_difference":
        expr = make_symmetric_difference(expr_lhs, expr_rhs)
        input_intervals = lhs_total + rhs_total
    elif op == "complement":
        universe = full_interval_set(spec.width, spec.rows)
        expr_universe = make_input(universe)
        expr = make_complement(expr_universe, expr_lhs)
        universe_total = int(universe.row_offsets[-1].item())
        input_intervals = lhs_total + universe_total
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
    ml_base = MultiLevel2D.create(2, base_ratio=RATIO)
    ml_base.set_level(0, base)
    ml_full = MultiLevel2D.create(2, base_ratio=RATIO)
    ml_full.set_level(0, base)
    ml_full.set_level(1, fine)
    mf_base = MultiLevelField2D.empty_like(ml_base)
    mf_base.set_level_field(0, field_base)
    mf_full = MultiLevelField2D.empty_like(ml_full)
    mf_full.set_level_field(0, field_base)
    mf_full.set_level_field(1, field_fine)

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

    def _prolong_level_sets_target():
        ml_base.set_level(0, base)
        ml_base.set_level(1, None)
        prolong_level_sets(ml_base, 0)

    def _restrict_level_sets_target():
        ml_full.set_level(0, base)
        ml_full.set_level(1, fine)
        restrict_level_sets(ml_full, 0)

    def _prolong_level_field_target():
        ml_base.set_level(0, base)
        mf_base.set_level_field(0, field_base)
        mf_base.set_level_field(1, None)
        prolong_level_field(ml_base, mf_base, 0)

    def _restrict_level_field_target():
        ml_full.set_level(0, base)
        ml_full.set_level(1, fine)
        mf_full.set_level_field(0, field_base)
        mf_full.set_level_field(1, field_fine)
        restrict_level_field(ml_full, mf_full, 0)

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
        "prolong_level_sets": (
            _prolong_level_sets_target,
            {"entity": "MultiLevel2D", "input_intervals": base_intervals},
        ),
        "restrict_level_sets": (
            _restrict_level_sets_target,
            {"entity": "MultiLevel2D", "input_intervals": fine_intervals},
        ),
        "prolong_level_field": (
            _prolong_level_field_target,
            {"entity": "MultiLevelField2D", "input_intervals": base_cells},
        ),
        "restrict_level_field": (
            _restrict_level_field_target,
            {"entity": "MultiLevelField2D", "input_intervals": fine_cells},
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


def _build_align_case(
    *,
    name: str,
    description: str,
    spec: GeometrySpec,
) -> BenchmarkCase:
    cp = _require_cupy()
    lhs = random_interval_set(
        cp,
        rows=spec.rows,
        width=spec.width,
        intervals_per_row=(1, max(2, spec.tiles_x * 2)),
        seed=123,
    )
    rhs = random_interval_set(
        cp,
        rows=spec.rows,
        width=spec.width,
        intervals_per_row=(1, max(2, spec.tiles_x * 3)),
        seed=987,
    )
    lhs_total = int(lhs.row_offsets[-1].item())
    rhs_total = int(rhs.row_offsets[-1].item())
    lhs_rows = int(lhs.rows.size)
    rhs_rows = int(rhs.rows.size)

    def _target():
        _align_interval_sets(lhs, rhs)

    metadata = {
        "rows": spec.rows,
        "width": spec.width,
        "lhs_rows": lhs_rows,
        "rhs_rows": rhs_rows,
        "operation": "align_interval_sets",
        "input_intervals": lhs_total + rhs_total,
    }

    return BenchmarkCase(
        name=name,
        description=description,
        setup=lambda _cp: BenchmarkTarget(func=_target, repeat=120, warmup=10, metadata=metadata),
    )


def _build_interval_field_case(
    *,
    name: str,
    description: str,
    spec: GeometrySpec,
    op: str,
) -> BenchmarkCase:
    cp = _require_cupy()
    geometry = square_grid_interval_set(cp, spec=spec)
    interval_count = int(geometry.row_offsets[-1].item())

    if op == "create":
        def _target():
            create_interval_field(geometry, fill_value=0.0, dtype=cp.float32)

        metadata = {
            "rows": spec.rows,
            "width": spec.width,
            "operation": "interval_field_create",
            "input_intervals": interval_count,
        }
        return BenchmarkCase(
            name=name,
            description=description,
            setup=lambda _cp: BenchmarkTarget(func=_target, repeat=60, warmup=5, metadata=metadata),
        )

    if op != "locate":
        raise ValueError(f"unsupported interval field op '{op}'")

    field = deterministic_interval_field(cp, geometry, width=spec.width)
    cell_count = field.cell_count

    rows_host = cp.asnumpy(geometry.rows)
    offsets_host = cp.asnumpy(geometry.row_offsets)
    begin_host = cp.asnumpy(geometry.begin)
    end_host = cp.asnumpy(geometry.end)
    coords: List[Tuple[int, int]] = []
    max_samples = min(512, cell_count) if cell_count > 0 else 0
    for ordinal, row_value in enumerate(rows_host):
        start = int(offsets_host[ordinal])
        stop = int(offsets_host[ordinal + 1])
        for idx in range(start, stop):
            x0 = int(begin_host[idx])
            x1 = int(end_host[idx])
            for x in range(x0, x1):
                coords.append((int(row_value), int(x)))
                if len(coords) >= max_samples:
                    break
            if len(coords) >= max_samples:
                break
        if len(coords) >= max_samples:
            break

    if not coords:
        raise ValueError("interval_field benchmark requires at least one active cell")

    rows_list = [row for row, _ in coords]
    cols_list = [col for _, col in coords]
    sample_count = len(rows_list)
    query_rows = cp.asarray(rows_list, dtype=cp.int32)
    query_cols = cp.asarray(cols_list, dtype=cp.int32)
    scratch = cp.empty(sample_count, dtype=cp.int32)

    def _target_search():
        locate_interval_cells(field, query_rows, query_cols, out=scratch)

    metadata = {
        "rows": spec.rows,
        "width": spec.width,
        "operation": "interval_field_locate",
        "input_intervals": sample_count,
        "samples": sample_count,
    }
    return BenchmarkCase(
        name=name,
        description=description,
        setup=lambda _cp: BenchmarkTarget(func=_target_search, repeat=120, warmup=15, metadata=metadata),
    )


def _build_vtu_case(
    *,
    name: str,
    description: str,
    spec: GeometrySpec,
) -> BenchmarkCase:
    cp = _require_cupy()
    base_set = square_grid_interval_set(cp, spec=spec)
    base_field = deterministic_interval_field(cp, base_set, width=spec.width)
    ratio = RATIO
    fine_set = prolong_set(base_set, ratio)
    fine_field = prolong_field(base_field, ratio)
    fine_width = max(1, spec.width * ratio)
    fine_rows = max(1, spec.rows * ratio)
    ghost_set = ghost_zones(fine_set, halo_x=1, halo_y=1, width=fine_width, height=fine_rows)
    ghost_field = create_interval_field(ghost_set, fill_value=-1.0, dtype=cp.float32)
    if ghost_field.values.size:
        ghost_field.values.fill(-1.0)

    dx0 = 1.0 / max(1, spec.width)
    dy0 = 1.0 / max(1, spec.rows)
    dx1 = dx0 / ratio
    dy1 = dy0 / ratio

    total_cells = base_field.cell_count + fine_field.cell_count + ghost_field.cell_count
    out_dir = os.path.join(tempfile.gettempdir(), "subsetix_bench_vtu")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.vtu")

    cells = [
        (base_field, 0, dx0, dy0, 0.0, 0.0, 0),
        (fine_field, 1, dx1, dy1, 0.0, 0.0, 0),
        (ghost_field, 1, dx1, dy1, 0.0, 0.0, 1),
    ]

    def _target():
        write_unstructured_quads_vtu(path, cells)
        cp.cuda.runtime.deviceSynchronize()

    metadata = {
        "rows": spec.rows,
        "width": spec.width,
        "levels": 2,
        "operation": "write_unstructured_quads_vtu",
        "input_intervals": total_cells,
    }

    return BenchmarkCase(
        name=name,
        description=description,
        setup=lambda _cp: BenchmarkTarget(func=_target, repeat=12, warmup=3, metadata=metadata),
    )


def _build_stencil_case(
    *,
    name: str,
    description: str,
    spec: GeometrySpec,
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
    row_ids = interval_set.interval_rows().astype(cp.int32, copy=False)

    a = 0.35
    b = -0.25
    dt = 0.01
    dx = 1.0 / max(1, spec.width)
    dy = 1.0 / max(1, spec.rows)

    interval_scratch = cp.empty_like(field.values)

    active_cells = int(field.interval_cell_offsets[-1].item()) if field.interval_cell_offsets.size else 0

    def _interval_target():
        return step_upwind_interval(
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

    for _ in range(3):
        _interval_target()
    cp.cuda.runtime.deviceSynchronize()

    metadata = {
        "entity": "IntervalField",
        "rows": spec.rows,
        "width": spec.width,
        "active_cells": active_cells,
        "a": a,
        "b": b,
        "dt": dt,
        "dx": dx,
        "dy": dy,
        "boundary": "zero_exterior",
        "variant": variant,
        "input_intervals": active_cells,
    }

    return BenchmarkCase(
        name=name,
        description=description,
        setup=lambda _cp: BenchmarkTarget(func=_interval_target, repeat=40, warmup=5, metadata=metadata),
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
    _build_expression_case(
        name="expr_symmetric_difference_small",
        description="Interval symmetric difference (small domain)",
        spec=SMALL_SPEC,
        op="symmetric_difference",
    ),
    _build_expression_case(
        name="expr_symmetric_difference_large",
        description="Interval symmetric difference (large domain)",
        spec=LARGE_SPEC,
        op="symmetric_difference",
    ),
    _build_expression_case(
        name="expr_complement_small",
        description="Interval complement inside full domain (small)",
        spec=SMALL_SPEC,
        op="complement",
    ),
    _build_expression_case(
        name="expr_complement_large",
        description="Interval complement inside full domain (large)",
        spec=LARGE_SPEC,
        op="complement",
    ),
    _build_align_case(
        name="expr_align_small",
        description="Row alignment of random interval sets (small domain)",
        spec=SMALL_SPEC,
    ),
    _build_align_case(
        name="expr_align_large",
        description="Row alignment of random interval sets (large domain)",
        spec=LARGE_SPEC,
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
    _build_interval_field_case(
        name="interval_field_create_small",
        description="IntervalField creation (small domain)",
        spec=SMALL_SPEC,
        op="create",
    ),
    _build_interval_field_case(
        name="interval_field_create_large",
        description="IntervalField creation (large domain)",
        spec=LARGE_SPEC,
        op="create",
    ),
    _build_interval_field_case(
        name="interval_field_locate_small",
        description="IntervalField locate-only queries (small domain)",
        spec=SMALL_SPEC,
        op="locate",
    ),
    _build_interval_field_case(
        name="interval_field_locate_large",
        description="IntervalField locate-only queries (large domain)",
        spec=LARGE_SPEC,
        op="locate",
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
    _build_multilevel_case(
        name="multilevel_prolong_level_sets_small",
        description="Level-aware prolongation of sets (small domain)",
        spec=SMALL_SPEC,
        op="prolong_level_sets",
    ),
    _build_multilevel_case(
        name="multilevel_prolong_level_sets_large",
        description="Level-aware prolongation of sets (large domain)",
        spec=LARGE_SPEC,
        op="prolong_level_sets",
    ),
    _build_multilevel_case(
        name="multilevel_restrict_level_sets_small",
        description="Level-aware restriction of sets (small domain)",
        spec=SMALL_SPEC,
        op="restrict_level_sets",
    ),
    _build_multilevel_case(
        name="multilevel_restrict_level_sets_large",
        description="Level-aware restriction of sets (large domain)",
        spec=LARGE_SPEC,
        op="restrict_level_sets",
    ),
    _build_multilevel_case(
        name="multilevel_prolong_level_field_small",
        description="Level-aware prolongation of fields (small domain)",
        spec=SMALL_SPEC,
        op="prolong_level_field",
    ),
    _build_multilevel_case(
        name="multilevel_prolong_level_field_large",
        description="Level-aware prolongation of fields (large domain)",
        spec=LARGE_SPEC,
        op="prolong_level_field",
    ),
    _build_multilevel_case(
        name="multilevel_restrict_level_field_small",
        description="Level-aware restriction of fields (small domain)",
        spec=SMALL_SPEC,
        op="restrict_level_field",
    ),
    _build_multilevel_case(
        name="multilevel_restrict_level_field_large",
        description="Level-aware restriction of fields (large domain)",
        spec=LARGE_SPEC,
        op="restrict_level_field",
    ),
    _build_vtu_case(
        name="export_vtu_small",
        description="VTU export pipeline (small intervals)",
        spec=STENCIL_SMALL,
    ),
    _build_vtu_case(
        name="export_vtu_large",
        description="VTU export pipeline (large intervals)",
        spec=STENCIL_LARGE,
    ),
    _build_stencil_case(
        name="stencil_interval_square_small",
        description="Interval-field upwind stencil (square mask, small domain)",
        spec=STENCIL_SMALL,
        variant="square",
    ),
    _build_stencil_case(
        name="stencil_interval_square_large",
        description="Interval-field upwind stencil (square mask, large domain)",
        spec=STENCIL_LARGE,
        variant="square",
    ),
    _build_stencil_case(
        name="stencil_interval_stair_small",
        description="Interval-field upwind stencil (staircase mask, small domain)",
        spec=STENCIL_SMALL,
        variant="staircase",
    ),
    _build_stencil_case(
        name="stencil_interval_stair_large",
        description="Interval-field upwind stencil (staircase mask, large domain)",
        spec=STENCIL_LARGE,
        variant="staircase",
    ),
]


CASES: List[BenchmarkCase] = list(_CASES)

__all__ = ["CASES"]
