from __future__ import annotations

import unittest

import cupy as cp

from subsetix_cupy import build_interval_set, create_interval_field
from subsetix_cupy.interval_field import get_cell
from subsetix_cupy.interval_stencil import (
    step_upwind_interval,
    step_upwind_interval_field,
)


def _upwind_zero_reference(
    field,
    *,
    width: int,
    height: int,
    a: float,
    b: float,
    dt: float,
    dx: float,
    dy: float,
):
    """Python reference that operates directly on the interval geometry."""

    interval_set = field.interval_set
    row_ids = interval_set.interval_rows()
    offsets = field.interval_cell_offsets
    result = cp.empty_like(field.values)

    def _get(row: int, col: int) -> float:
        value = get_cell(field, row, col)
        if value is None:
            return 0.0
        return float(value.item())

    for idx in range(interval_set.begin.size):
        row = int(row_ids[idx].item())
        start = int(interval_set.begin[idx].item())
        stop = int(interval_set.end[idx].item())
        base = int(offsets[idx].item())
        for local in range(stop - start):
            col = start + local
            center = float(field.values[base + local].item())
            left = _get(row, col - 1)
            right = _get(row, col + 1)
            down = _get(row - 1, col)
            up = _get(row + 1, col)

            if a >= 0.0:
                du_dx = (center - left) / dx
            else:
                du_dx = (right - center) / dx
            if b >= 0.0:
                du_dy = (center - down) / dy
            else:
                du_dy = (up - center) / dy
            result[base + local] = center - dt * (a * du_dx + b * du_dy)
    return result


class IntervalStencilTest(unittest.TestCase):
    def test_upwind_matches_zero_padded_dense(self) -> None:
        width = 64
        height = 64
        x0, x1 = 16, 48
        y0, y1 = 16, 48

        row_offsets = [0]
        begin = []
        end = []
        for row in range(height):
            if y0 <= row < y1:
                begin.append(x0)
                end.append(x1)
                row_offsets.append(row_offsets[-1] + 1)
            else:
                row_offsets.append(row_offsets[-1])

        interval_set = build_interval_set(
            row_offsets=row_offsets,
            begin=begin,
            end=end,
        )
        field = create_interval_field(interval_set, fill_value=0.0, dtype=cp.float32)

        row_ids = interval_set.interval_rows()
        offsets = field.interval_cell_offsets
        for idx in range(interval_set.begin.size):
            row = int(row_ids[idx].item())
            start = int(interval_set.begin[idx].item())
            stop = int(interval_set.end[idx].item())
            base = int(offsets[idx].item())
            length = stop - start
            xs = cp.arange(length, dtype=cp.float32)
            field.values[base : base + length] = row * 0.25 + xs * 0.5

        a = 0.3
        b = -0.4
        dt = 0.05
        dx = 1.0 / width
        dy = 1.0 / height

        reference = _upwind_zero_reference(
            field,
            width=width,
            height=height,
            a=a,
            b=b,
            dt=dt,
            dx=dx,
            dy=dy,
        )
        interval_out_field = step_upwind_interval_field(
            field,
            width=width,
            height=height,
            a=a,
            b=b,
            dt=dt,
            dx=dx,
            dy=dy,
        )

        buffer = cp.empty_like(field.values)
        out_values = step_upwind_interval(
            field,
            width=width,
            height=height,
            a=a,
            b=b,
            dt=dt,
            dx=dx,
            dy=dy,
            out=buffer,
            row_ids=row_ids,
        )
        self.assertIs(out_values, buffer)

        cp.testing.assert_allclose(interval_out_field.values, reference, rtol=1e-6, atol=1e-6)
        cp.testing.assert_allclose(buffer, reference, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
