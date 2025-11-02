from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .. import (
    CuPyWorkspace,
    IntervalField,
    IntervalSet,
    build_interval_set,
    create_interval_field,
)
from ..expressions import _require_cupy


@dataclass(frozen=True)
class GeometrySpec:
    """
    Deterministic interval geometry description used by the benchmark suite.

    Attributes
    ----------
    rows, width:
        Raster resolution of the implicit binary mask.
    tiles_x, tiles_y:
        Number of square tiles placed along X/Y (at most four along X to respect
        the “1–4 intervals per line” requirement).
    fill_ratio:
        Fraction of each tile occupied by the inner square (0 < fill_ratio <= 1).
    """

    rows: int
    width: int
    tiles_x: int
    tiles_y: int
    fill_ratio: float = 0.75


def random_interval_set(
    cp,
    *,
    rows: int,
    width: int,
    intervals_per_row: Tuple[int, int] | int,
    seed: int = 0,
) -> IntervalSet:
    """
    Generate a pseudo-random IntervalSet with the given shape.

    Intervals are half-open, non-overlapping within each row and snapped to int32.
    """

    rng = np.random.default_rng(seed)
    if isinstance(intervals_per_row, int):
        lo = hi = max(1, intervals_per_row)
    else:
        lo = max(1, intervals_per_row[0])
        hi = max(lo, intervals_per_row[1])

    begins: List[int] = []
    ends: List[int] = []
    offsets: List[int] = [0]

    for _ in range(rows):
        count = int(rng.integers(lo, hi + 1))
        if width <= 1:
            count = 0
        count = min(count, max(0, width // 2))
        row_beg: List[int] = []
        row_end: List[int] = []
        if count > 0:
            points = np.sort(rng.choice(np.arange(0, width + 1, dtype=np.int32), size=count * 2, replace=False))
            start_points = points[0::2]
            end_points = points[1::2]
            end_points = np.maximum(end_points, start_points + 1)
            end_points = np.clip(end_points, 0, width)
            # enforce monotonicity (no overlaps)
            last_end = 0
            for s, e in zip(start_points, end_points):
                s = int(max(s, last_end))
                if s >= width:
                    break
                e = int(max(e, s + 1))
                if e > width:
                    e = width
                if e <= s:
                    continue
                row_beg.append(s)
                row_end.append(e)
                last_end = e
        begins.extend(row_beg)
        ends.extend(row_end)
        offsets.append(offsets[-1] + len(row_beg))

    return build_interval_set(offsets, begins, ends)


def square_grid_interval_set(
    cp,
    *,
    spec: GeometrySpec,
    shift_x: int = 0,
    shift_y: int = 0,
) -> IntervalSet:
    """
    Build an IntervalSet formed by a regular grid of filled squares.

    Each square is centred inside its tile; neighbouring squares remain
    separated so that any scanline crosses at most `tiles_x` intervals.
    """

    rows = spec.rows
    width = spec.width
    tiles_x = max(1, min(4, spec.tiles_x))
    tiles_y = max(1, spec.tiles_y)
    fill = float(np.clip(spec.fill_ratio, 0.05, 1.0))

    mask = np.zeros((rows, width), dtype=np.bool_)

    cell_h = rows / tiles_y
    cell_w = width / tiles_x
    square_h = max(1, int(round(cell_h * fill)))
    square_w = max(1, int(round(cell_w * fill)))

    for gy in range(tiles_y):
        y_center = (gy + 0.5) * cell_h
        y0 = int(round(y_center - square_h / 2))
        y1 = int(round(y_center + square_h / 2))
        y0 = max(0, min(rows, y0))
        y1 = max(0, min(rows, y1))
        if y1 <= y0:
            continue
        for gx in range(tiles_x):
            x_center = (gx + 0.5) * cell_w
            x0 = int(round(x_center - square_w / 2))
            x1 = int(round(x_center + square_w / 2))
            x0 = max(0, min(width, x0))
            x1 = max(0, min(width, x1))
            if x1 <= x0:
                continue
            mask[y0:y1, x0:x1] = True

    if shift_y:
        mask = np.roll(mask, int(shift_y), axis=0)
    if shift_x:
        mask = np.roll(mask, int(shift_x), axis=1)

    begins: List[int] = []
    ends: List[int] = []
    offsets: List[int] = [0]

    total_intervals = 0
    for row in mask:
        active = np.nonzero(row)[0]
        if active.size == 0:
            offsets.append(total_intervals)
            continue

        start = int(active[0])
        prev = int(active[0])
        row_intervals = 0
        for idx in active[1:]:
            idx = int(idx)
            if idx == prev + 1:
                prev = idx
                continue
            begins.append(start)
            ends.append(prev + 1)
            row_intervals += 1
            start = idx
            prev = idx
        begins.append(start)
        ends.append(prev + 1)
        row_intervals += 1
        total_intervals += row_intervals
        offsets.append(total_intervals)

    return build_interval_set(offsets, begins, ends)


def random_interval_field(
    cp,
    interval_set: IntervalSet,
    *,
    seed: int = 0,
    dtype=None,
) -> IntervalField:
    """
    Create a random IntervalField compatible with *interval_set*.
    """

    field = create_interval_field(interval_set, fill_value=0.0, dtype=dtype)
    if field.values.size == 0:
        return field
    rng = np.random.default_rng(seed)
    host = rng.standard_normal(field.values.size).astype(np.float32)
    values = cp.asarray(host, dtype=field.values.dtype)
    field.values[...] = values
    return field


def deterministic_interval_field(
    cp,
    interval_set: IntervalSet,
    *,
    width: int,
    amplitude: float = 1.0,
    frequency: float = 0.125,
    phase: float = 0.0,
) -> IntervalField:
    """
    Create a reproducible IntervalField by sampling a smooth function over the cells.
    """

    field = create_interval_field(interval_set, fill_value=0.0, dtype=cp.float32)
    if field.values.size == 0:
        return field

    row_offsets = interval_set.row_offsets.get()
    begins = interval_set.begin.get()
    ends = interval_set.end.get()
    offsets = field.interval_cell_offsets.get()
    row_count = interval_set.row_count

    values = np.empty(field.values.size, dtype=np.float32)
    inv_width = 1.0 / max(1, width)
    inv_rows = 1.0 / max(1, row_count)

    for row in range(row_count):
        start = row_offsets[row]
        stop = row_offsets[row + 1]
        if start == stop:
            continue
        row_phase = phase + row * inv_rows * np.pi
        for idx in range(start, stop):
            x0 = begins[idx]
            x1 = ends[idx]
            base = offsets[idx]
            length = x1 - x0
            if length <= 0:
                continue
            xs = np.arange(length, dtype=np.float32)
            coords = (x0 + xs) * inv_width
            values[base : base + length] = amplitude * np.sin(2.0 * np.pi * frequency * coords + row_phase)

    field.values[...] = cp.asarray(values, dtype=field.values.dtype)
    return field


def make_workspace(cp) -> CuPyWorkspace:
    return CuPyWorkspace(cp)


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


__all__ = [
    "GeometrySpec",
    "random_interval_set",
    "random_interval_field",
    "square_grid_interval_set",
    "deterministic_interval_field",
    "make_workspace",
    "ensure_directory",
]
