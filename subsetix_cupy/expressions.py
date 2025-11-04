"""
High-level expression system for interval set operations modelled after
the CuPy backend.

The goal of this module is to provide a convenient, composable API that mirrors
the C++ surface operation chain while keeping data resident on an array backend.
This module assumes a functional CuPy installation with CUDA runtime support
and does not provide a CPU fallback path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import importlib
import sys

import numpy as np


def _load_actual_cupy() -> "object | None":
    """Attempt to load the real CuPy module from site-packages."""
    repo_root = Path(__file__).resolve().parents[1]
    original_path = list(sys.path)
    existing_module = sys.modules.pop("cupy", None)
    try:
        sys.path = [
            entry
            for entry in original_path
            if entry
            and not _is_subpath(entry, repo_root)
        ]
        try:
            module = importlib.import_module("cupy")
        except Exception:
            return None
        module_file = getattr(module, "__file__", "")
        if module_file:
            try:
                if Path(module_file).resolve().is_relative_to(repo_root):
                    return None
            except (OSError, ValueError):
                pass
        try:
            module.cuda.runtime.getDeviceCount()
            module.cuda.runtime.runtimeGetVersion()
        except Exception:
            return None
        return module
    finally:
        if existing_module is not None:
            sys.modules["cupy"] = existing_module
        else:
            sys.modules.pop("cupy", None)
        sys.path = original_path


def _is_subpath(entry: str, root: Path) -> bool:
    try:
        resolved = Path(entry).resolve()
    except OSError:
        return False
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        return False


_REAL_CUPY = _load_actual_cupy()
if _REAL_CUPY is None:
    raise RuntimeError(
        "CuPy backend with CUDA support is required; CPU fallback is disabled."
    )


def _require_cupy():
    return _REAL_CUPY


Interval = Tuple[int, int]


def _to_numpy(array) -> np.ndarray:
    if _REAL_CUPY is not None and isinstance(array, _REAL_CUPY.ndarray):
        return _REAL_CUPY.asnumpy(array)
    return np.asarray(array)


def _ensure_int32(array: Sequence[int]):
    cp = _require_cupy()
    return cp.asarray(array, dtype=cp.int32)


def _empty_int32(length: int):
    cp = _require_cupy()
    return cp.empty(length, dtype=cp.int32)


def _as_int32(array):
    cp = _require_cupy()
    return cp.asarray(array, dtype=cp.int32)


def _row_layout_equal(lhs: "IntervalSet", rhs: "IntervalSet") -> bool:
    if lhs.row_count != rhs.row_count:
        return False
    cp = _require_cupy()
    rows_l = lhs.rows_index()
    rows_r = rhs.rows_index()
    if rows_l.size != rows_r.size:
        return False
    if rows_l.size == 0:
        return True
    return bool(cp.all(rows_l == rows_r))


def _reindex_interval_set(interval_set: "IntervalSet", rows_out) -> "IntervalSet":
    cp = _require_cupy()
    rows_out = cp.asarray(rows_out, dtype=cp.int32)

    if rows_out.size == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        offsets = cp.zeros(1, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=rows_out)

    rows_in = interval_set.rows_index()
    row_offsets_in = cp.asarray(interval_set.row_offsets, dtype=cp.int32)
    begin_in = cp.asarray(interval_set.begin, dtype=cp.int32)
    end_in = cp.asarray(interval_set.end, dtype=cp.int32)

    if rows_in.size == rows_out.size and bool(cp.all(rows_in == rows_out)):
        return IntervalSet(
            begin=begin_in,
            end=end_in,
            row_offsets=row_offsets_in,
            rows=rows_out,
        )

    row_count_out = int(rows_out.size)
    counts = cp.zeros(row_count_out, dtype=cp.int32)

    if rows_in.size:
        positions = cp.searchsorted(rows_in, rows_out)
        matches = cp.zeros(row_count_out, dtype=cp.bool_)
        valid = positions < rows_in.size
        if int(valid.sum()) > 0:
            idx_valid = cp.where(valid)[0]
            pos_valid = positions[valid]
            eq_mask = rows_in[pos_valid] == rows_out[valid]
            if int(eq_mask.sum()) > 0:
                idx_matched = idx_valid[eq_mask]
                pos_matched = pos_valid[eq_mask]
                counts_matched = row_offsets_in[pos_matched + 1] - row_offsets_in[pos_matched]
                counts[idx_matched] = counts_matched.astype(cp.int32, copy=False)
                matches[idx_matched] = True

    row_offsets_out = cp.empty(row_count_out + 1, dtype=cp.int32)
    row_offsets_out[0] = 0
    if row_count_out > 0:
        cp.cumsum(counts, dtype=cp.int32, out=row_offsets_out[1:])
    total = int(row_offsets_out[-1].item())
    if total == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=row_offsets_out, rows=rows_out)

    mapping = cp.full(row_count_out, -1, dtype=cp.int32)
    if int(matches.sum()) > 0:
        mapping[matches] = positions[matches]

    output_pos = cp.arange(total, dtype=cp.int32)
    row_for_output = cp.searchsorted(row_offsets_out[1:], output_pos, side="right")
    source_row = mapping[row_for_output]
    if int((source_row < 0).sum()) > 0:
        raise ValueError("IntervalSet alignment produced unmapped rows")
    offset_in_row = output_pos - row_offsets_out[row_for_output]
    source_index = row_offsets_in[source_row] + offset_in_row

    begin_out = begin_in[source_index]
    end_out = end_in[source_index]

    return IntervalSet(
        begin=begin_out,
        end=end_out,
        row_offsets=row_offsets_out,
        rows=rows_out,
    )


def _align_interval_sets(lhs: "IntervalSet", rhs: "IntervalSet") -> tuple[IntervalSet, IntervalSet, Any]:
    cp = _require_cupy()
    rows_l = lhs.rows_index()
    rows_r = rhs.rows_index()
    if rows_l.size == rows_r.size and bool(rows_l.size == 0 or cp.all(rows_l == rows_r)):
        return lhs, rhs, rows_l

    rows_out = cp.union1d(rows_l, rows_r).astype(cp.int32, copy=False)
    lhs_aligned = _reindex_interval_set(lhs, rows_out)
    rhs_aligned = _reindex_interval_set(rhs, rows_out)
    return lhs_aligned, rhs_aligned, rows_out


@dataclass(frozen=True)
class IntervalSet:
    """
    Representation of a collection of half-open [begin, end) intervals.

    Intervals are stored in a compressed-row layout where `row_offsets`
    contains the exclusive prefix sums for each logical row. The ``rows``
    array holds the (sorted, strictly increasing) row identifiers; when
    omitted the layout defaults to the dense range ``[0, row_count)``.
    """

    begin: Any
    end: Any
    row_offsets: Any
    rows: Optional[Any] = None

    def __post_init__(self) -> None:
        cp = _require_cupy()
        # Dtype checks without copying whole arrays to host
        if isinstance(self.begin, cp.ndarray):
            if self.begin.dtype != cp.int32:
                raise TypeError("begin must use int32 precision")
        else:
            if np.asarray(self.begin).dtype != np.int32:
                raise TypeError("begin must use int32 precision")
        if isinstance(self.end, cp.ndarray):
            if self.end.dtype != cp.int32:
                raise TypeError("end must use int32 precision")
        else:
            if np.asarray(self.end).dtype != np.int32:
                raise TypeError("end must use int32 precision")
        if isinstance(self.row_offsets, cp.ndarray):
            if self.row_offsets.dtype != cp.int32:
                raise TypeError("row_offsets must use int32 precision")
        else:
            if np.asarray(self.row_offsets).dtype != np.int32:
                raise TypeError("row_offsets must use int32 precision")

        # Shape and consistency checks
        begin_arr = self.begin
        end_arr = self.end
        offsets_arr = self.row_offsets

        if begin_arr.shape != end_arr.shape:
            raise ValueError("begin and end must share the same shape")

        if hasattr(offsets_arr, "ndim"):
            if offsets_arr.ndim != 1:
                raise ValueError("row_offsets must be one-dimensional")
        else:
            if np.asarray(offsets_arr).ndim != 1:
                raise ValueError("row_offsets must be one-dimensional")

        first = int(offsets_arr[0].item()) if isinstance(offsets_arr, cp.ndarray) else int(np.asarray(offsets_arr)[0])
        if first != 0:
            raise ValueError("row_offsets must start at zero")

        last = int(offsets_arr[-1].item()) if isinstance(offsets_arr, cp.ndarray) else int(np.asarray(offsets_arr)[-1])
        if last != begin_arr.size:
            raise ValueError("row_offsets last entry must match interval count")

        row_count = int(offsets_arr.size - 1)

        rows_attr = self.rows
        if rows_attr is None:
            rows_arr = cp.arange(row_count, dtype=cp.int32)
        else:
            rows_arr = cp.asarray(rows_attr, dtype=cp.int32)
            if rows_arr.ndim != 1:
                raise ValueError("rows must be one-dimensional")
            length = int(rows_arr.size)
            if length != row_count:
                raise ValueError("rows length must match row_offsets size - 1")
            if length > 1:
                diffs = rows_arr[1:] - rows_arr[:-1]
                if not bool(cp.all(diffs > 0)):
                    raise ValueError("rows must be strictly increasing")

        object.__setattr__(self, "rows", rows_arr)

    @property
    def row_count(self) -> int:
        return int(self.row_offsets.size - 1)

    def rows_index(self):
        return self.rows

    def interval_rows(self):
        cp = _require_cupy()
        row_offsets = cp.asarray(self.row_offsets, dtype=cp.int32)
        if row_offsets.size == 0:
            return cp.zeros(0, dtype=cp.int32)
        total = int(row_offsets[-1].item())
        if total == 0:
            return cp.zeros(0, dtype=cp.int32)
        rows = self.rows_index()
        if row_offsets.size == 1:
            return cp.zeros(0, dtype=cp.int32)
        positions = cp.arange(total, dtype=cp.int32)
        row_indices = cp.searchsorted(row_offsets[1:], positions, side="right")
        if rows.size == 0:
            return row_indices.astype(cp.int32, copy=False)
        return rows[row_indices]


class CuPyWorkspace:
    """
    Reusable storage for CuPy-backed evaluations.

    Counts/offsets buffers are resized on demand, while output buffers are
    pooled to avoid repeated allocations across iterations. Results
    materialised from a workspace may be recycled on subsequent evaluations;
    copy them if you need to keep the data beyond the next call.
    """

    def __init__(self, cp_module=None):
        cp_mod = cp_module if cp_module is not None else _require_cupy()
        self._cp = cp_mod
        self._counts = None
        self._counts_capacity = 0
        self._offsets = None
        self._offsets_capacity = 0
        self._buffers: List[Any] = []
        self._buffers_capacity: List[int] = []
        self._buffers_in_use: List[bool] = []
        self._depth = 0

    def _reset_iteration(self) -> None:
        for idx in range(len(self._buffers_in_use)):
            self._buffers_in_use[idx] = False

    def _enter(self) -> None:
        if self._depth == 0:
            self._reset_iteration()
        self._depth += 1

    def _exit(self) -> None:
        self._depth = max(self._depth - 1, 0)

    def ensure_counts(self, row_count: int):
        if self._counts_capacity < row_count:
            self._counts = self._cp.empty(row_count, dtype=self._cp.int32)
            self._counts_capacity = row_count
        elif self._counts is None:
            self._counts = self._cp.empty(row_count, dtype=self._cp.int32)
            self._counts_capacity = row_count
        return self._counts[:row_count]

    def ensure_offsets(self, row_count: int):
        required = row_count + 1
        if self._offsets_capacity < required:
            self._offsets = self._cp.empty(required, dtype=self._cp.int32)
            self._offsets_capacity = required
        elif self._offsets is None:
            self._offsets = self._cp.empty(required, dtype=self._cp.int32)
            self._offsets_capacity = required
        return self._offsets[:required]

    def acquire_output(self, total: int):
        for idx, (used, capacity) in enumerate(zip(self._buffers_in_use, self._buffers_capacity)):
            if not used and capacity >= total:
                self._buffers_in_use[idx] = True
                begin_arr, end_arr = self._buffers[idx]
                return begin_arr[:total], end_arr[:total]
        begin_arr = self._cp.empty(total, dtype=self._cp.int32)
        end_arr = self._cp.empty(total, dtype=self._cp.int32)
        self._buffers.append((begin_arr, end_arr))
        self._buffers_capacity.append(total)
        self._buffers_in_use.append(True)
        return begin_arr, end_arr


class Expr:
    """Base expression node."""

    def evaluate(self, workspace: CuPyWorkspace | None = None) -> IntervalSet:
        raise NotImplementedError


class _InputExpr(Expr):
    def __init__(self, interval_set: IntervalSet):
        self._set = interval_set

    def evaluate(self, workspace: CuPyWorkspace | None = None) -> IntervalSet:
        return self._set


class _BinaryExpr(Expr):
    def __init__(self, lhs: Expr, rhs: Expr, mode: str):
        self._lhs = lhs
        self._rhs = rhs
        self._mode = mode

    def evaluate(self, workspace: CuPyWorkspace | None = None) -> IntervalSet:
        lhs_set = self._lhs.evaluate(workspace)
        rhs_set = self._rhs.evaluate(workspace)
        return _apply_binary(lhs_set, rhs_set, self._mode, workspace)


class _SymmetricDifferenceExpr(Expr):
    def __init__(self, lhs: Expr, rhs: Expr):
        self._lhs = lhs
        self._rhs = rhs

    def evaluate(self, workspace: CuPyWorkspace | None = None) -> IntervalSet:
        lhs_set = self._lhs.evaluate(workspace)
        rhs_set = self._rhs.evaluate(workspace)
        left_minus_right = _apply_binary(lhs_set, rhs_set, "difference", workspace)
        right_minus_left = _apply_binary(rhs_set, lhs_set, "difference", workspace)
        return _apply_binary(left_minus_right, right_minus_left, "union", workspace)


class _ComplementExpr(Expr):
    def __init__(self, universe: Expr, subset: Expr):
        self._universe = universe
        self._subset = subset

    def evaluate(self, workspace: CuPyWorkspace | None = None) -> IntervalSet:
        return _apply_binary(
            self._universe.evaluate(workspace),
            self._subset.evaluate(workspace),
            "difference",
            workspace,
        )


def make_input(interval_set: IntervalSet) -> Expr:
    """Wrap an `IntervalSet` so it can participate in expressions."""
    return _InputExpr(interval_set)


def make_intersection(lhs: Expr, rhs: Expr) -> Expr:
    return _BinaryExpr(lhs, rhs, "intersection")


def make_union(lhs: Expr, rhs: Expr) -> Expr:
    return _BinaryExpr(lhs, rhs, "union")


def make_difference(lhs: Expr, rhs: Expr) -> Expr:
    return _BinaryExpr(lhs, rhs, "difference")


def make_symmetric_difference(lhs: Expr, rhs: Expr) -> Expr:
    return _SymmetricDifferenceExpr(lhs, rhs)


def make_complement(universe: Expr, subset: Expr) -> Expr:
    return _ComplementExpr(universe, subset)


def evaluate(expr: Expr, workspace: CuPyWorkspace | None = None) -> IntervalSet:
    """Evaluate *expr* and return a new `IntervalSet`."""
    if workspace is not None:
        workspace._enter()
        try:
            return expr.evaluate(workspace)
        finally:
            workspace._exit()
    return expr.evaluate(None)


def build_interval_set(
    row_offsets: Sequence[int],
    begin: Sequence[int],
    end: Sequence[int],
    *,
    rows: Sequence[int] | None = None,
) -> IntervalSet:
    """
    Convenience helper to instantiate an `IntervalSet` from Python sequences.

    All values are coerced to `int32`.  Inputs are assumed to be already sorted
    and non-overlapping within each row.
    """
    offsets_arr = _ensure_int32(row_offsets)
    begin_arr = _ensure_int32(begin)
    end_arr = _ensure_int32(end)
    last = int(offsets_arr[-1].item())
    if last != len(begin_arr):
        raise ValueError("row_offsets final entry must equal the interval count")
    rows_arr = _ensure_int32(rows) if rows is not None else None
    return IntervalSet(
        begin=begin_arr,
        end=end_arr,
        row_offsets=offsets_arr,
        rows=rows_arr,
    )


def _apply_binary(
    lhs: IntervalSet,
    rhs: IntervalSet,
    mode: str,
    workspace: CuPyWorkspace | None = None,
) -> IntervalSet:
    cp = _require_cupy()
    if not isinstance(lhs.begin, cp.ndarray) or not isinstance(rhs.begin, cp.ndarray):
        raise TypeError("Inputs must be CuPy arrays; CPU fallback is disabled")

    lhs_aligned, rhs_aligned, rows_index = _align_interval_sets(lhs, rhs)
    return _apply_binary_gpu(lhs_aligned, rhs_aligned, mode, workspace, rows_index)


def _apply_binary_gpu(
    lhs: IntervalSet,
    rhs: IntervalSet,
    mode: str,
    workspace: CuPyWorkspace | None,
    rows_index,
) -> IntervalSet:
    cp = _require_cupy()
    row_count = int(lhs.row_offsets.size - 1)
    if row_count == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        offsets = cp.zeros(1, dtype=cp.int32)
        rows_out = cp.asarray(rows_index, dtype=cp.int32) if rows_index is not None else None
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=rows_out)

    kernels = _get_kernels(cp)
    count_kernel = kernels[0]
    write_kernel = kernels[1]
    if workspace is not None:
        counts = workspace.ensure_counts(row_count)
    else:
        counts = cp.empty(row_count, dtype=cp.int32)
    block = 128
    grid = (row_count + block - 1) // block
    op_code = _OPERATION_CODES[mode]

    count_kernel(
        (grid,),
        (block,),
        (
            lhs.begin,
            lhs.end,
            lhs.row_offsets,
            rhs.begin,
            rhs.end,
            rhs.row_offsets,
            np.int32(row_count),
            np.int32(op_code),
            counts,
        ),
    )

    if workspace is not None:
        offsets = workspace.ensure_offsets(row_count)
    else:
        offsets = cp.empty(row_count + 1, dtype=cp.int32)
    offsets[0] = 0
    cp.cumsum(counts, dtype=cp.int32, out=offsets[1:])
    total = int(offsets[-1].get())
    if total == 0:
        zero = cp.zeros(0, dtype=cp.int32)
        rows_out = cp.asarray(rows_index, dtype=cp.int32)
        return IntervalSet(begin=zero, end=zero, row_offsets=offsets, rows=rows_out)

    if workspace is not None:
        out_begin, out_end = workspace.acquire_output(total)
    else:
        out_begin = cp.empty(total, dtype=cp.int32)
        out_end = cp.empty(total, dtype=cp.int32)

    write_kernel(
        (grid,),
        (block,),
        (
            lhs.begin,
            lhs.end,
            lhs.row_offsets,
            rhs.begin,
            rhs.end,
            rhs.row_offsets,
            offsets,
            np.int32(row_count),
            np.int32(op_code),
            out_begin,
            out_end,
        ),
    )

    rows_out = cp.asarray(rows_index, dtype=cp.int32)
    return IntervalSet(begin=out_begin, end=out_end, row_offsets=offsets, rows=rows_out)
_OPERATION_CODES = {"union": 0, "intersection": 1, "difference": 2}


def _get_kernels(cp_module):
    from . import kernels as _kernels

    return _kernels.get_kernels(cp_module)


__all__ = [
    "Expr",
    "IntervalSet",
    "CuPyWorkspace",
    "build_interval_set",
    "evaluate",
    "make_difference",
    "make_complement",
    "make_input",
    "make_intersection",
    "make_symmetric_difference",
    "make_union",
]
