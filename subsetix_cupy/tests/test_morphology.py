import unittest

import numpy as np

from subsetix_cupy import (
    build_interval_set,
    dilate_interval_set,
    ghost_zones,
)
from subsetix_cupy.expressions import _REAL_CUPY


def _rows_to_python(interval_set, cp_mod):
    offsets = cp_mod.asnumpy(interval_set.row_offsets)
    begin = cp_mod.asnumpy(interval_set.begin)
    end = cp_mod.asnumpy(interval_set.end)
    rows = []
    for row in range(offsets.size - 1):
        start = offsets[row]
        stop = offsets[row + 1]
        rows.append([(int(begin[idx]), int(end[idx])) for idx in range(start, stop)])
    return rows


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class MorphologyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY
        assert self.cp is not None

    def test_dilate_wraps_across_boundary(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 1, 1, 1, 1],
            begin=[7],
            end=[8],
        )
        dilated = dilate_interval_set(base, halo_x=1, halo_y=0, width=8, height=4, bc="wrap")
        rows = _rows_to_python(dilated, self.cp)
        self.assertEqual(rows[0], [(0, 1), (6, 8)])
        self.assertTrue(all(len(r) == 0 for r in rows[1:]))

    def test_dilate_spreads_in_y_with_wrap(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 0, 1, 1, 1],
            begin=[2],
            end=[4],
        )
        dilated = dilate_interval_set(base, halo_x=0, halo_y=1, width=8, height=4, bc="wrap")
        rows = _rows_to_python(dilated, self.cp)
        self.assertEqual(rows[1], [(2, 4)])
        self.assertEqual(rows[0], [(2, 4)])
        self.assertEqual(rows[2], [(2, 4)])
        self.assertTrue(len(rows[3]) == 0)

    def test_ghost_zones_exclude_original_cells(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 0, 1, 1, 1],
            begin=[2],
            end=[4],
        )
        ghosts = ghost_zones(base, halo_x=1, halo_y=1, width=8, height=4, bc="wrap")
        rows = _rows_to_python(ghosts, self.cp)
        self.assertEqual(rows[0], [(1, 5)])
        self.assertEqual(rows[1], [(1, 2), (4, 5)])
        self.assertEqual(rows[2], [(1, 5)])
        self.assertTrue(len(rows[3]) == 0)

    def test_dilate_superset(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 2, 4],
            begin=[10, 20, 40, 50],
            end=[14, 24, 42, 56],
        )
        dilated = dilate_interval_set(base, halo_x=3, halo_y=1, width=64, height=2, bc="clamp")
        # ensure base rows subset of dilated rows
        rows_base = _rows_to_python(base, self.cp)
        rows_dil = _rows_to_python(dilated, self.cp)
        for row_idx, intervals in enumerate(rows_base):
            for interval in intervals:
                covered = any(
                    interval[0] >= d[0] and interval[1] <= d[1] for d in rows_dil[row_idx]
                )
                self.assertTrue(covered)

    def test_clamp_does_not_wrap(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 1],
            begin=[0],
            end=[4],
        )
        dilated = dilate_interval_set(base, halo_x=2, halo_y=0, width=16, height=1, bc="clamp")
        rows = _rows_to_python(dilated, self.cp)
        self.assertEqual(rows[0], [(0, 6)])

    def test_vertical_clamp_collects_neighbor_rows(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 2, 4, 4],
            begin=[2, 4, 1, 3],
            end=[3, 5, 2, 4],
        )
        dilated = dilate_interval_set(base, halo_x=0, halo_y=1, width=8, height=3, bc="clamp")
        rows = _rows_to_python(dilated, self.cp)
        self.assertEqual(rows[0], [(1, 5)])
        self.assertEqual(rows[1], [(1, 5)])
        self.assertEqual(rows[2], [(1, 2), (3, 4)])


if __name__ == "__main__":
    unittest.main()
