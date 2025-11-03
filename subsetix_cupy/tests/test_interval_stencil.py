from __future__ import annotations

import unittest

import cupy as cp

from subsetix_cupy import (
    build_interval_set,
    create_interval_field,
    interval_field_to_dense,
)
from subsetix_cupy.interval_stencil import (
    step_upwind_dense_active,
    step_upwind_dense_zero,
    step_upwind_interval,
    step_upwind_interval_field,
)


class IntervalStencilTest(unittest.TestCase):
    def test_upwind_matches_dense(self) -> None:
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

        dense = interval_field_to_dense(field, width=width, height=height, fill_value=0.0)
        mask = cp.zeros((height, width), dtype=cp.bool_)
        row_offsets_host = interval_set.row_offsets.get()
        begin_host = interval_set.begin.get()
        end_host = interval_set.end.get()
        for row in range(height):
            start = row_offsets_host[row]
            stop = row_offsets_host[row + 1]
            for idx in range(start, stop):
                b_idx = int(begin_host[idx])
                e_idx = int(end_host[idx])
                mask[row, b_idx:e_idx] = True

        a = 0.3
        b = -0.4
        dt = 0.05
        dx = 1.0 / width
        dy = 1.0 / height

        dense_out = step_upwind_dense_zero(dense, a=a, b=b, dt=dt, dx=dx, dy=dy)
        dense_active = step_upwind_dense_active(
            dense,
            row_ids=row_ids,
            begin=interval_set.begin,
            end=interval_set.end,
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

        dense_from_interval = interval_field_to_dense(
            interval_out_field,
            width=width,
            height=height,
            fill_value=0.0,
        )

        cp.testing.assert_allclose(
            dense_out[mask],
            dense_from_interval[mask],
            rtol=1e-6,
            atol=1e-6,
        )
        cp.testing.assert_allclose(
            dense_active[mask],
            dense_from_interval[mask],
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
