import unittest

import numpy as np

from subsetix_cupy import (
    CuPyWorkspace,
    IntervalField,
    build_interval_set,
    create_interval_field,
    evaluate,
    get_cell,
    make_complement,
    make_difference,
    make_input,
    make_intersection,
    make_symmetric_difference,
    make_union,
    set_cell,
)
from subsetix_cupy.expressions import _REAL_CUPY


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class ExpressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.surface_a = build_interval_set(
            row_offsets=[0, 2, 3],
            begin=[0, 6, 1],
            end=[4, 9, 5],
        )
        self.surface_b = build_interval_set(
            row_offsets=[0, 1, 2],
            begin=[2, 0],
            end=[7, 4],
        )
        self.surface_c = build_interval_set(
            row_offsets=[0, 1, 2],
            begin=[3, 2],
            end=[6, 3],
        )
        self.surface_d = build_interval_set(
            row_offsets=[0, 1, 2],
            begin=[5, 5],
            end=[11, 8],
        )
        self.surface_e = build_interval_set(
            row_offsets=[0, 1, 2],
            begin=[7, 6],
            end=[10, 9],
        )

    def _assert_interval_set(self, result, begin, end, row_offsets, rows=None):
        cp = _REAL_CUPY
        assert cp is not None
        np.testing.assert_array_equal(
            cp.asnumpy(result.begin), np.array(begin, dtype=np.int32)
        )
        np.testing.assert_array_equal(
            cp.asnumpy(result.end), np.array(end, dtype=np.int32)
        )
        np.testing.assert_array_equal(
            cp.asnumpy(result.row_offsets), np.array(row_offsets, dtype=np.int32)
        )
        expected_rows = rows if rows is not None else list(range(len(row_offsets) - 1))
        np.testing.assert_array_equal(
            cp.asnumpy(result.rows_index()), np.array(expected_rows, dtype=np.int32)
        )

    def test_interval_set_rows_always_present(self) -> None:
        cp = _REAL_CUPY
        assert cp is not None
        interval = build_interval_set(
            row_offsets=[0, 2, 5],
            begin=[0, 3, 1, 4, 6],
            end=[2, 6, 3, 5, 7],
        )
        self.assertIsInstance(interval.rows, cp.ndarray)
        np.testing.assert_array_equal(cp.asnumpy(interval.rows_index()), np.array([0, 1], dtype=np.int32))

    def test_nested_expression(self) -> None:
        expr = make_union(
            make_difference(
                make_intersection(make_input(self.surface_a), make_input(self.surface_b)),
                make_input(self.surface_c),
            ),
            make_intersection(make_input(self.surface_d), make_input(self.surface_e)),
        )
        result = evaluate(expr)
        self._assert_interval_set(
            result,
            begin=[2, 6, 1, 3, 6],
            end=[3, 10, 2, 4, 8],
            row_offsets=[0, 2, 5],
        )

    def test_union_aligns_sparse_rows(self) -> None:
        extra = build_interval_set(row_offsets=[0, 1], begin=[0], end=[1])
        expr = make_union(make_input(self.surface_a), make_input(extra))
        result = evaluate(expr)
        self._assert_interval_set(
            result,
            begin=[0, 6, 1],
            end=[4, 9, 5],
            row_offsets=[0, 2, 3],
        )

    def test_difference_handles_empty_rhs(self) -> None:
        empty = build_interval_set(row_offsets=[0, 0, 0], begin=[], end=[])
        expr = make_difference(make_input(self.surface_a), make_input(empty))
        result = evaluate(expr)
        self._assert_interval_set(
            result,
            begin=[0, 6, 1],
            end=[4, 9, 5],
            row_offsets=[0, 2, 3],
        )

    def test_union_sparse_row_ids(self) -> None:
        lhs = build_interval_set(row_offsets=[0, 1], begin=[0], end=[2], rows=[5])
        rhs = build_interval_set(row_offsets=[0, 1], begin=[3], end=[4], rows=[7])
        expr = make_union(make_input(lhs), make_input(rhs))
        result = evaluate(expr)
        cp = _REAL_CUPY
        assert cp is not None
        np.testing.assert_array_equal(cp.asnumpy(result.rows_index()), np.array([5, 7], dtype=np.int32))
        self._assert_interval_set(
            result,
            begin=[0, 3],
            end=[2, 4],
            row_offsets=[0, 1, 2],
            rows=[5, 7],
        )

    def test_symmetric_difference(self) -> None:
        expr = make_symmetric_difference(
            make_input(self.surface_a), make_input(self.surface_b)
        )
        result = evaluate(expr)
        self._assert_interval_set(
            result,
            begin=[0, 4, 7, 0, 4],
            end=[2, 6, 9, 1, 5],
            row_offsets=[0, 3, 5],
        )

    def test_complement(self) -> None:
        universe = build_interval_set(
            row_offsets=[0, 1, 2],
            begin=[0, 0],
            end=[12, 6],
        )
        expr = make_complement(make_input(universe), make_input(self.surface_a))
        result = evaluate(expr)
        self._assert_interval_set(
            result,
            begin=[4, 9, 0, 5],
            end=[6, 12, 1, 6],
            row_offsets=[0, 2, 4],
        )

    def test_multishape_union(self) -> None:
        rectangles = build_interval_set(
            row_offsets=[0, 2, 4, 6],
            begin=[0, 4, 1, 5, 2, 6],
            end=[2, 6, 3, 7, 4, 8],
        )
        circles = build_interval_set(
            row_offsets=[0, 1, 3, 4],
            begin=[1, 0, 4, 3],
            end=[5, 2, 6, 7],
        )
        expr = make_union(make_input(rectangles), make_input(circles))
        result = evaluate(expr)
        self._assert_interval_set(
            result,
            begin=[0, 0, 4, 2],
            end=[6, 3, 7, 8],
            row_offsets=[0, 1, 3, 4],
        )

    def test_gpu_backend_returns_cupy_arrays(self) -> None:
        cp = _REAL_CUPY
        expr = make_union(
            make_intersection(make_input(self.surface_a), make_input(self.surface_b)),
            make_difference(make_input(self.surface_d), make_input(self.surface_c)),
        )
        result = evaluate(expr)
        self.assertIsInstance(result.begin, cp.ndarray)
        self.assertIsInstance(result.end, cp.ndarray)
        self.assertIsInstance(result.row_offsets, cp.ndarray)
        np.testing.assert_array_equal(
            cp.asnumpy(result.begin), np.array([2, 6, 1, 5], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            cp.asnumpy(result.end), np.array([4, 11, 4, 8], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            cp.asnumpy(result.row_offsets), np.array([0, 2, 4], dtype=np.int32)
        )

    def test_workspace_reuses_output_buffers(self) -> None:
        cp = _REAL_CUPY
        workspace = CuPyWorkspace(cp)
        expr = make_union(make_input(self.surface_a), make_input(self.surface_b))
        result1 = evaluate(expr, workspace)
        ptr1 = int(result1.begin.data.ptr)
        del result1
        result2 = evaluate(expr, workspace)
        ptr2 = int(result2.begin.data.ptr)
        self.assertEqual(ptr1, ptr2)

    def test_multishape_union_gpu_matches_expected(self) -> None:
        cp = _REAL_CUPY
        workspace = CuPyWorkspace(cp)
        rectangles = build_interval_set(
            row_offsets=[0, 2, 4, 6],
            begin=[0, 4, 1, 5, 2, 6],
            end=[2, 6, 3, 7, 4, 8],
        )
        circles = build_interval_set(
            row_offsets=[0, 1, 3, 4],
            begin=[1, 0, 4, 3],
            end=[5, 2, 6, 7],
        )
        expr = make_union(make_input(rectangles), make_input(circles))
        result = evaluate(expr, workspace)
        np.testing.assert_array_equal(
            cp.asnumpy(result.begin), np.array([0, 0, 4, 2], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            cp.asnumpy(result.end), np.array([6, 3, 7, 8], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            cp.asnumpy(result.row_offsets), np.array([0, 1, 3, 4], dtype=np.int32)
        )


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class IntervalFieldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY
        assert self.cp is not None
        self.interval_set = build_interval_set(
            row_offsets=[0, 2, 3],
            begin=[0, 4, 1],
            end=[3, 6, 5],
        )

    def test_create_interval_field(self) -> None:
        field = create_interval_field(self.interval_set, fill_value=2.5, dtype=self.cp.float32)
        self.assertIsInstance(field, IntervalField)
        self.assertEqual(field.values.size, 9)
        self.assertTrue(bool(self.cp.all(field.values == self.cp.float32(2.5))))
        self.assertEqual(int(field.interval_cell_offsets[-1].item()), 9)

    def test_set_and_get_cell(self) -> None:
        field = create_interval_field(self.interval_set, fill_value=0.0, dtype=self.cp.float32)

        updated = set_cell(field, row=0, x=1, value=7.0)
        self.assertTrue(updated)
        value = get_cell(field, row=0, x=1)
        self.assertIsNotNone(value)
        self.assertAlmostEqual(float(value.item()), 7.0)

        # Outside active intervals should return None / False
        self.assertIsNone(get_cell(field, row=0, x=3))
        self.assertFalse(set_cell(field, row=1, x=0, value=3.0))

    def test_empty_field_dtype_matches_fill(self) -> None:
        empty_set = build_interval_set(row_offsets=[0, 0], begin=[], end=[])
        value = self.cp.float32(3.0)
        field = create_interval_field(empty_set, fill_value=value)
        self.assertEqual(field.values.dtype, value.dtype)
        self.assertEqual(field.values.size, 0)


if __name__ == "__main__":
    unittest.main()
