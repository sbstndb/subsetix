import unittest

from subsetix_amr2.fields import (
    ActionField,
    prolong_coarse_to_fine,
    restrict_fine_to_coarse,
    synchronize_two_level,
)
from subsetix_cupy.expressions import IntervalSet, _REAL_CUPY


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
    return IntervalSet(begin=begin_arr, end=end_arr, row_offsets=offsets_arr)


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class FieldsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY

    def test_prolong_restrict_inverse(self) -> None:
        coarse = self.cp.arange(16, dtype=self.cp.float32).reshape(4, 4)
        fine = prolong_coarse_to_fine(coarse, 2)
        self.assertEqual(fine.shape, (8, 8))
        restored = restrict_fine_to_coarse(fine, 2)
        self.cp.testing.assert_array_equal(restored, coarse)

    def test_prolong_with_interval_subset(self) -> None:
        coarse = self.cp.arange(4, dtype=self.cp.float32).reshape(2, 2)
        intervals = _make_interval_set(
            self.cp,
            4,
            {
                0: [(0, 1), (2, 3)],
                2: [(0, 1), (2, 3)],
            },
        )
        out = self.cp.full((4, 4), -1.0, dtype=self.cp.float32)
        prolong_coarse_to_fine(coarse, 2, out=out, mask=intervals)
        expected = self.cp.full((4, 4), -1.0, dtype=self.cp.float32)
        expected[0, 0] = 0.0
        expected[0, 2] = 1.0
        expected[2, 0] = 2.0
        expected[2, 2] = 3.0
        self.cp.testing.assert_array_equal(out, expected)

    def test_restrict_reducer(self) -> None:
        fine = self.cp.ones((4, 4), dtype=self.cp.float32)
        fine[::2, ::2] = 3.0
        coarse_sum = restrict_fine_to_coarse(fine, 2, reducer="sum")
        expected = self.cp.full((2, 2), 6.0, dtype=self.cp.float32)
        self.cp.testing.assert_array_equal(coarse_sum, expected)

    def test_synchronize_round(self) -> None:
        coarse = self.cp.zeros((4, 4), dtype=self.cp.float32)
        fine = self.cp.full((8, 8), 2.0, dtype=self.cp.float32)
        refine = _make_interval_set(
            self.cp,
            4,
            {
                1: [(1, 3)],
                2: [(1, 3)],
            },
        )
        coarse_updated, fine_updated = synchronize_two_level(
            coarse,
            fine,
            refine,
            ratio=2,
            reducer="mean",
            fill_fine_outside=True,
        )

        expected_coarse = self.cp.zeros_like(coarse)
        expected_coarse[1:3, 1:3] = 2.0
        self.cp.testing.assert_array_equal(coarse_updated, expected_coarse)

        actions = ActionField.full_grid(4, 4, ratio=2)
        actions.set_from_interval_set(refine)
        fine_set = actions.fine_set()
        fine_cells = set()
        begin = fine_set.begin.get()
        end = fine_set.end.get()
        offsets = fine_set.row_offsets.get()
        fine_height = offsets.size - 1
        for row in range(fine_height):
            start = int(offsets[row])
            stop = int(offsets[row + 1])
            for idx in range(start, stop):
                for col in range(int(begin[idx]), int(end[idx])):
                    fine_cells.add((row, col))
        for row, col in fine_cells:
            self.assertEqual(float(fine_updated[row, col]), 2.0)
        for row in range(fine.shape[0]):
            for col in range(fine.shape[1]):
                if (row, col) not in fine_cells:
                    self.assertEqual(float(fine_updated[row, col]), 0.0)

    def test_synchronize_without_fill(self) -> None:
        coarse = self.cp.zeros((4, 4), dtype=self.cp.float32)
        fine = self.cp.arange(64, dtype=self.cp.float32).reshape(8, 8)
        refine = _make_interval_set(
            self.cp,
            4,
            {
                1: [(1, 3)],
                2: [(1, 3)],
            },
        )
        coarse_updated, fine_updated = synchronize_two_level(
            coarse,
            fine,
            refine,
            ratio=2,
            reducer="mean",
            fill_fine_outside=False,
        )
        self.assertEqual(coarse_updated.dtype, self.cp.float32)
        self.cp.testing.assert_array_equal(fine_updated, fine)

    def test_synchronize_in_place(self) -> None:
        coarse = self.cp.zeros((2, 2), dtype=self.cp.float32)
        fine = self.cp.full((4, 4), 2.0, dtype=self.cp.float32)
        refine = _make_interval_set(
            self.cp,
            2,
            {
                0: [(0, 1)],
                1: [(0, 1)],
            },
        )
        coarse_out, fine_out = synchronize_two_level(
            coarse,
            fine,
            refine,
            ratio=2,
            reducer="mean",
            fill_fine_outside=True,
            copy=False,
        )
        self.assertIs(coarse_out, coarse)
        self.assertIs(fine_out, fine)
        expected_coarse = self.cp.array([[2.0, 0.0], [2.0, 0.0]], dtype=self.cp.float32)
        self.cp.testing.assert_array_equal(coarse, expected_coarse)
        # refined block remains 2.0, rest filled from coarse (zeros)
        self.assertTrue(self.cp.all(fine[0:2, 0:2] == 2.0))
        self.assertTrue(self.cp.all(fine[2:, :2] == 2.0))
        self.assertTrue(self.cp.all(fine[:, 2:] == 0.0))

    def test_synchronize_height_mismatch(self) -> None:
        coarse = self.cp.zeros((4, 4), dtype=self.cp.float32)
        fine = self.cp.zeros((8, 8), dtype=self.cp.float32)
        refine = _make_interval_set(
            self.cp,
            2,
            {
                0: [(0, 1)],
            },
        )
        with self.assertRaises(ValueError):
            synchronize_two_level(coarse, fine, refine, ratio=2)
