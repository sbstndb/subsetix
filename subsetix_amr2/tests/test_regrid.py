import unittest

from subsetix_amr2.regrid import (
    enforce_two_level_grading_set,
    gradient_tag_threshold_set,
    gradient_tag_set,
)
from subsetix_cupy.expressions import IntervalSet, _REAL_CUPY
from subsetix_cupy.interval_field import create_interval_field
from subsetix_cupy.morphology import full_interval_set


def _make_interval_set(cp_mod, height, spans):
    begins = []
    ends = []
    row_offsets = [0]
    for row in range(height):
        intervals = spans.get(row, ())
        for start, stop in intervals:
            begins.append(int(start))
            ends.append(int(stop))
        row_offsets.append(len(begins))
    begin_arr = cp_mod.asarray(begins, dtype=cp_mod.int32)
    end_arr = cp_mod.asarray(ends, dtype=cp_mod.int32)
    offsets_arr = cp_mod.asarray(row_offsets, dtype=cp_mod.int32)
    rows_arr = cp_mod.arange(height, dtype=cp_mod.int32)
    return IntervalSet(begin=begin_arr, end=end_arr, row_offsets=offsets_arr, rows=rows_arr)


def _empty_interval_set(cp_mod, height):
    zeros = cp_mod.zeros(0, dtype=cp_mod.int32)
    offsets = cp_mod.zeros(height + 1, dtype=cp_mod.int32)
    rows = cp_mod.arange(height, dtype=cp_mod.int32)
    return IntervalSet(begin=zeros, end=zeros, row_offsets=offsets, rows=rows)


def _field_from_array(cp_mod, array):
    height, width = array.shape
    interval = full_interval_set(width, height)
    field = create_interval_field(interval, fill_value=0.0, dtype=array.dtype)
    field.values[...] = array.ravel()
    return field


def _assert_interval_equal(testcase, cp_mod, actual: IntervalSet, expected: IntervalSet):
    testcase.assertEqual(actual.row_offsets.size, expected.row_offsets.size)
    cp_mod.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
    cp_mod.testing.assert_array_equal(actual.begin, expected.begin)
    cp_mod.testing.assert_array_equal(actual.end, expected.end)
    cp_mod.testing.assert_array_equal(actual.rows_index(), expected.rows_index())


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class RegridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY

    def test_gradient_tag_percentile(self) -> None:
        data = self.cp.zeros((4, 4), dtype=self.cp.float32)
        data[1, 1] = 0.5
        data[2, 2] = 1.0
        field = _field_from_array(self.cp, data)
        tagged_set = gradient_tag_set(field, width=4, height=4, frac_high=0.5)
        expected = _make_interval_set(self.cp, 4, {2: [(2, 3)]})
        _assert_interval_equal(self, self.cp, tagged_set, expected)

    def test_gradient_tag_all_zero(self) -> None:
        data = self.cp.zeros((4, 4), dtype=self.cp.float32)
        field = _field_from_array(self.cp, data)
        tagged_set = gradient_tag_set(field, width=4, height=4, frac_high=0.2)
        expected = _empty_interval_set(self.cp, 4)
        _assert_interval_equal(self, self.cp, tagged_set, expected)

    def test_gradient_tag_threshold(self) -> None:
        data = self.cp.zeros((4, 4), dtype=self.cp.float32)
        data[1, 1] = 0.5
        data[2, 2] = 1.0
        field = _field_from_array(self.cp, data)
        tagged_set = gradient_tag_threshold_set(field, width=4, height=4, threshold=0.75)
        expected = _make_interval_set(self.cp, 4, {2: [(2, 3)]})
        _assert_interval_equal(self, self.cp, tagged_set, expected)

    def test_gradient_tag_threshold_single_column(self) -> None:
        data = self.cp.asarray([[0.0], [0.3], [0.6], [1.0]], dtype=self.cp.float32)
        threshold = 0.25
        field = _field_from_array(self.cp, data)
        tagged_set = gradient_tag_threshold_set(field, width=1, height=4, threshold=threshold)
        expected = _make_interval_set(
            self.cp,
            4,
            {
                1: [(0, 1)],
                2: [(0, 1)],
                3: [(0, 1)],
            },
        )
        _assert_interval_equal(self, self.cp, tagged_set, expected)

    def test_gradient_tag_custom_epsilon(self) -> None:
        data = self.cp.full((4, 4), 1e-9, dtype=self.cp.float32)
        field = _field_from_array(self.cp, data)
        tagged_set = gradient_tag_set(field, width=4, height=4, frac_high=0.2, epsilon=1e-8)
        expected = _empty_interval_set(self.cp, 4)
        _assert_interval_equal(self, self.cp, tagged_set, expected)

    def test_enforce_grading(self) -> None:
        refine_set = _make_interval_set(self.cp, 5, {2: [(2, 3)]})
        graded_vn_set = enforce_two_level_grading_set(
            refine_set,
            padding=1,
            mode="von_neumann",
            width=5,
            height=5,
        )
        expected_cross = _make_interval_set(
            self.cp,
            5,
            {
                1: [(2, 3)],
                2: [(1, 4)],
                3: [(2, 3)],
            },
        )
        _assert_interval_equal(self, self.cp, graded_vn_set, expected_cross)

        graded_moore_set = enforce_two_level_grading_set(
            refine_set,
            padding=1,
            mode="moore",
            width=5,
            height=5,
        )
        expected_square = _make_interval_set(
            self.cp,
            5,
            {
                1: [(1, 4)],
                2: [(1, 4)],
                3: [(1, 4)],
            },
        )
        _assert_interval_equal(self, self.cp, graded_moore_set, expected_square)

    def test_enforce_grading_padding_zero(self) -> None:
        refine_set = _make_interval_set(
            self.cp,
            4,
            {
                1: [(1, 3)],
                2: [(1, 3)],
            },
        )
        graded_set = enforce_two_level_grading_set(
            refine_set,
            padding=0,
            mode="von_neumann",
            width=4,
            height=4,
        )
        _assert_interval_equal(self, self.cp, graded_set, refine_set)

    def test_enforce_grading_invalid_mode(self) -> None:
        empty = _empty_interval_set(self.cp, 4)
        with self.assertRaises(ValueError):
            enforce_two_level_grading_set(
                empty,
                padding=1,
                mode="diag",
                width=4,
                height=4,
            )
