import unittest

import numpy as np

from subsetix_cupy import (
    build_interval_set,
    translate_interval_set,
    interior_for_direction,
    boundary_for_direction,
    boundary_layer,
    evaluate,
    make_input,
    make_union,
    dilate_interval_set,
    ghost_zones,
    erode_interval_set,
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

    def test_dilate_expands_in_x(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 1],
            begin=[3],
            end=[5],
        )
        dilated = dilate_interval_set(base, halo_x=2, halo_y=0)
        rows = _rows_to_python(dilated, self.cp)
        self.assertEqual(rows[0], [(1, 7)])

    def test_translate_horizontal(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 2],
            begin=[1, 4],
            end=[2, 5],
        )
        translated = translate_interval_set(base, dx=3, dy=0)
        rows = _rows_to_python(translated, self.cp)
        self.assertEqual(rows[0], [(4, 5), (7, 8)])
        self.assertEqual(self.cp.asnumpy(translated.rows_index()).tolist(), [0])

    def test_translate_vertical_dense(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 1, 2],
            begin=[0, 1],
            end=[2, 3],
        )
        translated = translate_interval_set(base, dx=0, dy=2)
        row_ids = self.cp.asnumpy(translated.rows_index()).tolist()
        self.assertEqual(row_ids, [2, 3])
        rows = _rows_to_python(translated, self.cp)
        self.assertEqual(rows[0], [(0, 2)])
        self.assertEqual(rows[1], [(1, 3)])

    def test_translate_sparse(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 1],
            begin=[2],
            end=[5],
            rows=[10],
        )
        translated = translate_interval_set(base, dx=-1, dy=-2)
        row_ids = self.cp.asnumpy(translated.rows_index()).tolist()
        self.assertEqual(row_ids, [8])
        rows = _rows_to_python(translated, self.cp)
        self.assertEqual(rows[0], [(1, 4)])

    def test_translate_requires_int(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 1],
            begin=[0],
            end=[1],
        )
        with self.assertRaises(TypeError):
            translate_interval_set(base, dx=1.5)

    def test_directional_interior_horizontal(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 1],
            begin=[0],
            end=[4],
        )
        interior = interior_for_direction(base, dx=1, dy=0)
        rows = _rows_to_python(interior, self.cp)
        self.assertEqual(rows[0], [(0, 3)])
        boundary = boundary_for_direction(base, dx=1, dy=0)
        rows_boundary = _rows_to_python(boundary, self.cp)
        self.assertEqual(rows_boundary[0], [(3, 4)])
        layer = boundary_layer(base)
        rows_layer = _rows_to_python(layer, self.cp)
        self.assertEqual(rows_layer[0], [(0, 4)])

    def test_directional_interior_vertical_sparse(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 1, 2],
            begin=[0, 0],
            end=[3, 3],
            rows=[2, 5],
        )
        interior = interior_for_direction(base, dx=0, dy=3)
        row_ids = self.cp.asnumpy(interior.rows_index()).tolist()
        self.assertEqual(row_ids, [2])
        rows = _rows_to_python(interior, self.cp)
        self.assertEqual(rows[0], [(0, 3)])
        boundary = boundary_for_direction(base, dx=0, dy=3)
        boundary_rows = _rows_to_python(boundary, self.cp)
        self.assertEqual(boundary_rows[0], [(0, 3)])
        self.assertEqual(self.cp.asnumpy(boundary.rows_index()).tolist(), [5])
        layer = boundary_layer(base)
        layer_rows = _rows_to_python(layer, self.cp)
        self.assertEqual(layer_rows[0], [(0, 3)])
        self.assertEqual(self.cp.asnumpy(layer.rows_index()).tolist(), [2, 5])

    def test_direction_requires_non_zero(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 1],
            begin=[0],
            end=[1],
        )
        with self.assertRaises(ValueError):
            interior_for_direction(base, dx=0, dy=0)
        with self.assertRaises(ValueError):
            boundary_for_direction(base, dx=0, dy=0)

    def test_boundary_layer_matches_union(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 2],
            begin=[0, 2],
            end=[1, 3],
        )
        left = boundary_for_direction(base, dx=-1, dy=0)
        right = boundary_for_direction(base, dx=1, dy=0)
        layer = boundary_layer(base)
        union_expr = make_union(make_input(left), make_input(right))
        union = evaluate(union_expr)
        layer_rows = _rows_to_python(layer, self.cp)
        union_rows = _rows_to_python(union, self.cp)
        self.assertEqual(layer_rows, union_rows)

    def test_dilate_adds_vertical_neighbors(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 1],
            begin=[2],
            end=[4],
            rows=[5],
        )
        dilated = dilate_interval_set(base, halo_x=1, halo_y=1)
        row_ids = self.cp.asnumpy(dilated.rows_index()).tolist()
        self.assertEqual(row_ids, [4, 5, 6])
        rows = _rows_to_python(dilated, self.cp)
        self.assertEqual(rows[0], [(1, 5)])
        self.assertEqual(rows[1], [(1, 5)])
        self.assertEqual(rows[2], [(1, 5)])

    def test_ghost_zones_clipped_to_domain(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 1, 1, 1],
            begin=[0],
            end=[2],
        )
        ghosts = ghost_zones(base, halo_x=1, halo_y=1, width=4, height=3)
        rows = _rows_to_python(ghosts, self.cp)
        row_ids = self.cp.asnumpy(ghosts.rows_index()).tolist()
        self.assertEqual(row_ids, [0, 1, 2])
        self.assertEqual(rows[0], [(2, 3)])
        self.assertEqual(rows[1], [(0, 3)])
        self.assertTrue(len(rows[2]) == 0)

    def test_erode_removes_boundary_layer(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 1],
            begin=[1],
            end=[5],
        )
        eroded = erode_interval_set(base, halo_x=1, halo_y=0, width=6, height=1)
        rows = _rows_to_python(eroded, self.cp)
        self.assertEqual(rows[0], [(2, 4)])

    def test_dilate_superset(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 2, 4],
            begin=[10, 20, 40, 50],
            end=[14, 24, 42, 56],
        )
        dilated = dilate_interval_set(base, halo_x=3, halo_y=1)
        # ensure base rows subset of dilated rows
        rows_base = _rows_to_python(base, self.cp)
        rows_dil = _rows_to_python(dilated, self.cp)
        for row_idx, intervals in enumerate(rows_base):
            for interval in intervals:
                covered = any(
                    interval[0] >= d[0] and interval[1] <= d[1] for d in rows_dil[row_idx]
                )
                self.assertTrue(covered)

    def test_dilate_unbounded_preserves_sparse_rows(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 1],
            begin=[5],
            end=[7],
            rows=[10],
        )
        dilated = dilate_interval_set(base, halo_x=2, halo_y=1)
        rows = _rows_to_python(dilated, self.cp)
        row_ids = self.cp.asnumpy(dilated.rows_index()).tolist()
        self.assertEqual(row_ids, [9, 10, 11])
        self.assertEqual(rows[0], [(3, 9)])
        self.assertEqual(rows[1], [(3, 9)])
        self.assertEqual(rows[2], [(3, 9)])

    def test_negative_halo_disallowed(self) -> None:
        base = build_interval_set(
            row_offsets=[0, 1],
            begin=[1],
            end=[2],
        )
        with self.assertRaises(ValueError):
            dilate_interval_set(base, halo_x=-1)


if __name__ == "__main__":
    unittest.main()
