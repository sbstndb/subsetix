from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .. import (
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
)
from ..demo_advection2d_amr3 import (
    _dilate_mo,
    _dilate_vn,
    _erode_mo,
    _grad_mag,
    _hysteresis_mask,
    _init_condition,
    _prolong_repeat,
    _restrict_mean,
    _step_upwind,
)
from ..export_vtk import save_amr3_mesh_vtu, save_amr3_step_vtr, write_pvd
from ..morphology import dilate_interval_set, erode_interval_set, ghost_zones
from ..multilevel import coarse_only, covered_by_fine
from ..expressions import _require_cupy
from . import BenchmarkCase, BenchmarkTarget
from .utils import (
    GeometrySpec,
    deterministic_interval_field,
    ensure_directory,
    make_workspace,
    square_grid_interval_set,
)

SMALL_SPEC = GeometrySpec(rows=256, width=256, tiles_x=3, tiles_y=3, fill_ratio=0.8)
LARGE_SPEC = GeometrySpec(rows=8192, width=8192, tiles_x=3, tiles_y=3, fill_ratio=0.8)
RATIO = 2
AMR_SMALL_COARSE = SMALL_SPEC.width // (RATIO ** 2)
AMR_LARGE_COARSE = LARGE_SPEC.width // (RATIO ** 2)


def _shift_cells(spec: GeometrySpec, frac_x: float = 0.0, frac_y: float = 0.0) -> Tuple[int, int]:
    return int(round(spec.width * frac_x)), int(round(spec.rows * frac_y))


def _count_mask_intervals(mask) -> int:
    cp = _require_cupy()
    if mask.size == 0:
        return 0
    mask_i8 = mask.astype(cp.int8, copy=False)
    starts = cp.count_nonzero(mask_i8[:, :1])
    diffs = cp.diff(mask_i8, axis=1)
    transitions = cp.count_nonzero(diffs == 1)
    return int((starts + transitions).item())


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
    bc: str,
    op: str,
) -> BenchmarkCase:
    cp = _require_cupy()
    geometry = square_grid_interval_set(cp, spec=spec)
    width = spec.width
    rows = spec.rows
    base_intervals = int(geometry.row_offsets[-1].item())

    def _target_dilate():
        dilate_interval_set(geometry, halo_x=halo_x, halo_y=halo_y, width=width, height=rows, bc=bc)

    def _target_ghost():
        ghost_zones(geometry, halo_x=halo_x, halo_y=halo_y, width=width, height=rows, bc=bc)

    def _target_erode():
        erode_interval_set(geometry, halo_x=halo_x, halo_y=halo_y, width=width, height=rows, bc=bc)

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
        "bc": bc,
        "intervals": base_intervals,
        "operation": op,
        "input_intervals": base_intervals,
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


@dataclass
class _AMRState:
    u0_base: any
    u1_base: any
    u2_base: any
    refine0: any
    L1_mask: any
    refine1_mid: any
    L2_mask: any
    R: int
    dx0: float
    dy0: float
    dx1: float
    dy1: float
    dx2: float
    dy2: float
    a: float
    b: float


def _prepare_amr_state(
    *,
    coarse: int = 96,
    ratio: int = 2,
    refine_frac: float = 0.10,
    hysteresis: float = 0.0,
) -> _AMRState:
    cp = _require_cupy()
    W = H = coarse
    R = ratio
    u0 = _init_condition(W, H, kind="square")
    u1 = _prolong_repeat(u0, R)
    u2 = _prolong_repeat(u1, R)

    g0 = _grad_mag(u0)
    refine0 = _hysteresis_mask(g0, refine_frac, refine_frac * hysteresis, None)

    L1_mask_base = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)
    L1_expanded = L1_mask_base
    L1_expanded = _dilate_mo(L1_expanded, wrap=False)
    coarse_force = L1_expanded.reshape(H, R, W, R).any(axis=(1, 3))
    refine0 = refine0 | coarse_force
    L1_mask = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)

    g1 = _grad_mag(u1)
    refine1_mid = _hysteresis_mask(g1, refine_frac, refine_frac * hysteresis, None)
    L1_for_gating = _erode_mo(L1_mask, wrap=False)
    refine1_mid = refine1_mid & L1_for_gating

    coarse_force_l1 = refine1_mid.reshape(H, R, W, R).any(axis=(1, 3))
    refine0 = refine0 | coarse_force_l1
    L1_mask = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)
    L2_mask = _prolong_repeat(refine1_mid.astype(cp.uint8), R).astype(cp.bool_)

    dx0 = dy0 = 1.0 / W
    dx1 = dy1 = dx0 / R
    dx2 = dy2 = dx1 / R

    return _AMRState(
        u0_base=u0,
        u1_base=u1,
        u2_base=u2,
        refine0=refine0,
        L1_mask=L1_mask,
        refine1_mid=refine1_mid,
        L2_mask=L2_mask,
        R=R,
        dx0=dx0,
        dy0=dy0,
        dx1=dx1,
        dy1=dy1,
        dx2=dx2,
        dy2=dy2,
        a=0.6,
        b=0.6,
    )


def _build_amr_compute_case(name: str, coarse: int) -> BenchmarkCase:
    cp = _require_cupy()
    state = _prepare_amr_state(coarse=coarse, ratio=RATIO)

    u0_work = cp.empty_like(state.u0_base)
    u1_work = cp.empty_like(state.u1_base)
    u2_work = cp.empty_like(state.u2_base)
    u0_pad = cp.empty_like(state.u0_base)
    u1_pad = cp.empty_like(state.u1_base)
    u2_pad = cp.empty_like(state.u2_base)

    def _target():
        cp.copyto(u0_work, state.u0_base)
        cp.copyto(u1_work, state.u1_base)
        cp.copyto(u2_work, state.u2_base)

        coarse_if = _dilate_mo(state.refine0, wrap=False) & (~state.refine0)
        mid_if = _dilate_mo(state.refine1_mid, wrap=False) & (~state.refine1_mid)

        u1_restr = _restrict_mean(u1_work, state.R)
        cp.copyto(u0_pad, u0_work)
        u0_pad[state.refine0] = u1_restr[state.refine0]
        u0_pad[coarse_if] = u1_restr[coarse_if]

        u2_restr = _restrict_mean(u2_work, state.R)
        cp.copyto(u1_pad, u1_work)
        u1_pad[~state.L1_mask] = _prolong_repeat(u0_work, state.R)[~state.L1_mask]
        u1_pad[mid_if] = u2_restr[mid_if]

        cp.copyto(u2_pad, u2_work)
        u2_pad[~state.L2_mask] = _prolong_repeat(u1_work, state.R)[~state.L2_mask]

        _step_upwind(u0_pad, state.a, state.b, 0.000868, state.dx0, state.dy0, "clamp")
        _step_upwind(u1_pad, state.a, state.b, 0.000868, state.dx1, state.dy1, "clamp")
        _step_upwind(u2_pad, state.a, state.b, 0.000868, state.dx2, state.dy2, "clamp")

    total_intervals = (
        _count_mask_intervals(state.refine0)
        + _count_mask_intervals(state.refine1_mid)
        + _count_mask_intervals(state.L2_mask)
    )

    metadata = {
        "coarse": state.u0_base.shape,
        "ratio": state.R,
        "input_intervals": total_intervals,
        "operation": "compute_step",
    }

    return BenchmarkCase(
        name=name,
        description="AMR 3-level halo + upwind compute (single step, clamp BC)",
        setup=lambda _cp: BenchmarkTarget(func=_target, repeat=40, warmup=5, metadata=metadata),
    )


def _build_amr_regrid_case(name: str, coarse: int) -> BenchmarkCase:
    cp = _require_cupy()
    state = _prepare_amr_state(coarse=coarse, ratio=RATIO)
    base_intervals = (
        _count_mask_intervals(state.refine0)
        + _count_mask_intervals(state.refine1_mid)
        + _count_mask_intervals(state.L2_mask)
    )

    def _target():
        u0 = state.u0_base.copy()
        u1 = state.u1_base.copy()
        u2 = state.u2_base.copy()
        refine0 = state.refine0.copy()
        refine1_mid = state.refine1_mid.copy()
        L1_mask = state.L1_mask.copy()

        g0 = _grad_mag(u0)
        refine0_new = _hysteresis_mask(g0, 0.10, 0.0, refine0)
        L1_mask_new_base = _prolong_repeat(refine0_new.astype(cp.uint8), state.R).astype(cp.bool_)
        L1_ring_new = _dilate_mo(L1_mask_new_base, wrap=False)
        coarse_force_from_ring = L1_ring_new.reshape(
            u0.shape[0], state.R, u0.shape[1], state.R
        ).any(axis=(1, 3))
        refine0_new = refine0_new | coarse_force_from_ring
        L1_mask_new = _prolong_repeat(refine0_new.astype(cp.uint8), state.R).astype(cp.bool_)

        g1 = _grad_mag(u1)
        refine1_mid_new = _hysteresis_mask(g1, 0.10, 0.0, refine1_mid)
        L1_for_gating_new = _erode_mo(L1_mask_new, wrap=False)
        refine1_mid_new = refine1_mid_new & L1_for_gating_new

        coarse_force_l1_new = refine1_mid_new.reshape(
            u0.shape[0], state.R, u0.shape[1], state.R
        ).any(axis=(1, 3))
        refine0_new = refine0_new | coarse_force_l1_new
        L1_mask_new = _prolong_repeat(refine0_new.astype(cp.uint8), state.R).astype(cp.bool_)

        leaving2 = refine1_mid & (~refine1_mid_new)
        if int(leaving2.any()):
            u2_restr_now = _restrict_mean(u2, state.R)
            u1 = u1.copy()
            u1[leaving2] = u2_restr_now[leaving2]

        entering2 = (~refine1_mid) & refine1_mid_new
        if int(entering2.any()):
            u1_prol_now = _prolong_repeat(u1, state.R)
            L2_enter_f = _prolong_repeat(entering2.astype(cp.uint8), state.R).astype(cp.bool_)
            u2 = u2.copy()
            u2[L2_enter_f] = u1_prol_now[L2_enter_f]

        leaving1 = refine0 & (~refine0_new)
        if int(leaving1.any()):
            u1_restr_now = _restrict_mean(u1, state.R)
            u0 = u0.copy()
            u0[leaving1] = u1_restr_now[leaving1]

        entering1 = (~refine0) & refine0_new
        if int(entering1.any()):
            u0_prol_now = _prolong_repeat(u0, state.R)
            L1_enter_f = _prolong_repeat(entering1.astype(cp.uint8), state.R).astype(cp.bool_)
            u1 = u1.copy()
            u1[L1_enter_f] = u0_prol_now[L1_enter_f]

    metadata = {
        "coarse": state.u0_base.shape,
        "ratio": state.R,
        "input_intervals": base_intervals,
        "operation": "regrid_step",
    }
    return BenchmarkCase(
        name=name,
        description="AMR 3-level regridding (thresholds + transfers)",
        setup=lambda _cp: BenchmarkTarget(func=_target, repeat=20, warmup=5, metadata=metadata),
    )


def _build_export_case(name: str, coarse: int) -> BenchmarkCase:
    cp = _require_cupy()
    state = _prepare_amr_state(coarse=coarse, ratio=RATIO)
    tmpdir = tempfile.TemporaryDirectory()
    ensure_directory(tmpdir.name)
    counter = {"step": 0}

    def _target():
        step = counter["step"]
        rels, entry = save_amr3_step_vtr(
            tmpdir.name,
            "bench",
            step,
            time_value=float(step),
            u0=state.u0_base,
            u1=state.u1_base,
            u2=state.u2_base,
            refine0=state.refine0,
            L1_mask=state.L1_mask,
            refine1_mid=state.refine1_mid,
            L2_mask=state.L2_mask,
            dx0=state.dx0,
            dy0=state.dy0,
            dx1=state.dx1,
            dy1=state.dy1,
            dx2=state.dx2,
            dy2=state.dy2,
        )
        save_amr3_mesh_vtu(
            tmpdir.name,
            "bench",
            step,
            u0=state.u0_base,
            u1=state.u1_base,
            u2=state.u2_base,
            coarse_only=(~state.refine0),
            mid_only=(state.L1_mask & (~state.refine1_mid)),
            fine_active=state.L2_mask,
            dx0=state.dx0,
            dy0=state.dy0,
            dx1=state.dx1,
            dy1=state.dy1,
            dx2=state.dx2,
            dy2=state.dy2,
        )
        write_pvd(tmpdir.name + "/bench.pvd", [(float(step), rels + [f"bench_step{step:04d}_mesh.vtu"])])
        counter["step"] += 1

    metadata = {
        "operation": "export_vtk",
        "directory": tmpdir.name,
        "coarse": state.u0_base.shape,
        "ratio": state.R,
        "input_intervals": (
            _count_mask_intervals(state.refine0)
            + _count_mask_intervals(state.refine1_mid)
            + _count_mask_intervals(state.L2_mask)
        ),
    }
    return BenchmarkCase(
        name=name,
        description="VTK export (Rectilinear + mesh) for AMR snapshot",
        setup=lambda _cp: BenchmarkTarget(func=_target, repeat=5, warmup=1, metadata=metadata),
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
        description="Morphological dilation (clamp BC, small domain)",
        spec=SMALL_SPEC,
        halo_x=4,
        halo_y=2,
        bc="clamp",
        op="dilate",
    ),
    _build_morph_case(
        name="morph_dilate_large",
        description="Morphological dilation (clamp BC, large domain)",
        spec=LARGE_SPEC,
        halo_x=4,
        halo_y=2,
        bc="clamp",
        op="dilate",
    ),
    _build_morph_case(
        name="morph_ghost_small",
        description="Ghost zone generation (wrap BC, small domain)",
        spec=SMALL_SPEC,
        halo_x=3,
        halo_y=3,
        bc="wrap",
        op="ghost",
    ),
    _build_morph_case(
        name="morph_ghost_large",
        description="Ghost zone generation (wrap BC, large domain)",
        spec=LARGE_SPEC,
        halo_x=3,
        halo_y=3,
        bc="wrap",
        op="ghost",
    ),
    _build_morph_case(
        name="morph_erode_small",
        description="Morphological erosion (clamp BC, small domain)",
        spec=SMALL_SPEC,
        halo_x=2,
        halo_y=2,
        bc="clamp",
        op="erode",
    ),
    _build_morph_case(
        name="morph_erode_large",
        description="Morphological erosion (clamp BC, large domain)",
        spec=LARGE_SPEC,
        halo_x=2,
        halo_y=2,
        bc="clamp",
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
    _build_amr_compute_case(
        name="amr_compute_small",
        coarse=AMR_SMALL_COARSE,
    ),
    _build_amr_compute_case(
        name="amr_compute_large",
        coarse=AMR_LARGE_COARSE,
    ),
    _build_amr_regrid_case(
        name="amr_regrid_small",
        coarse=AMR_SMALL_COARSE,
    ),
    _build_amr_regrid_case(
        name="amr_regrid_large",
        coarse=AMR_LARGE_COARSE,
    ),
    _build_export_case(
        name="export_vtk_small",
        coarse=AMR_SMALL_COARSE,
    ),
]


CASES: List[BenchmarkCase] = list(_CASES)

__all__ = ["CASES"]
