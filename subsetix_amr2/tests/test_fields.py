import unittest

from subsetix_amr2.fields import (
    Action,
    ActionField,
    prolong_coarse_to_fine,
    restrict_fine_to_coarse,
    synchronize_two_level,
    synchronize_interval_fields,
    gather_interval_subset,
    scatter_interval_subset,
    clone_interval_field,
)
from subsetix_cupy.expressions import IntervalSet, _REAL_CUPY
from subsetix_cupy.morphology import full_interval_set
from subsetix_cupy import create_interval_field, interval_field_to_dense
from subsetix_cupy.interval_field import IntervalField


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

    def _full_field(self, array: "cp.ndarray") -> IntervalField:
        height, width = array.shape
        full = full_interval_set(width, height)
        field = create_interval_field(full, fill_value=0.0, dtype=array.dtype)
        field.values[...] = array.astype(array.dtype, copy=False).ravel()
        return field

    def test_action_field_from_interval_set(self) -> None:
        width = 4
        height = 3
        interval = full_interval_set(width, height)
        actions = ActionField.from_interval_set(interval, width=width, height=height, ratio=2)
        self.assertIs(actions.coarse_interval_set(), interval)
        self.assertEqual(actions.width, width)
        self.assertEqual(actions.height, height)
        self.assertEqual(int(actions.ratio), 2)
        grid = actions.values().reshape(height, width)
        expected = self.cp.full((height, width), int(Action.KEEP), dtype=self.cp.int8)
        self.cp.testing.assert_array_equal(grid, expected)

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

    def test_synchronize_interval_matches_dense(self) -> None:
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

        dense_coarse, dense_fine = synchronize_two_level(
            coarse,
            fine,
            refine,
            ratio=2,
            reducer="mean",
            fill_fine_outside=True,
        )

        actions = ActionField.full_grid(4, 4, ratio=2)
        actions.set_from_interval_set(refine)

        coarse_field = self._full_field(coarse)
        fine_field = self._full_field(fine)

        coarse_sync, fine_sync = synchronize_interval_fields(
            coarse_field,
            fine_field,
            actions,
            ratio=2,
            reducer="mean",
            fill_fine_outside=True,
            copy=True,
        )

        coarse_dense_from_interval = interval_field_to_dense(coarse_sync, width=4, height=4, fill_value=0.0)
        fine_dense_from_interval = interval_field_to_dense(fine_sync, width=8, height=8, fill_value=0.0)

        self.cp.testing.assert_allclose(dense_coarse, coarse_dense_from_interval)
        self.cp.testing.assert_allclose(dense_fine, fine_dense_from_interval)

        coarse_original = interval_field_to_dense(coarse_field, width=4, height=4, fill_value=0.0)
        fine_original = interval_field_to_dense(fine_field, width=8, height=8, fill_value=0.0)
        self.cp.testing.assert_array_equal(coarse_original, coarse)
        self.cp.testing.assert_array_equal(fine_original, fine)

        coarse_field_inplace = self._full_field(coarse)
        fine_field_inplace = self._full_field(fine)
        coarse_res, fine_res = synchronize_interval_fields(
            coarse_field_inplace,
            fine_field_inplace,
            actions,
            ratio=2,
            reducer="mean",
            fill_fine_outside=True,
            copy=False,
        )
        self.assertIs(coarse_res, coarse_field_inplace)
        self.assertIs(fine_res, fine_field_inplace)
        coarse_inplace_dense = interval_field_to_dense(coarse_res, width=4, height=4, fill_value=0.0)
        fine_inplace_dense = interval_field_to_dense(fine_res, width=8, height=8, fill_value=0.0)
        self.cp.testing.assert_allclose(dense_coarse, coarse_inplace_dense)
        self.cp.testing.assert_allclose(dense_fine, fine_inplace_dense)

    def test_gather_scatter_interval_subset(self) -> None:
        data = self.cp.arange(12, dtype=self.cp.float32).reshape(3, 4)
        field = self._full_field(data)
        subset = _make_interval_set(
            self.cp,
            3,
            {
                0: [(1, 3)],
                2: [(0, 2)],
            },
        )
        subset_field = gather_interval_subset(field, subset)
        subset_dense = interval_field_to_dense(subset_field, width=4, height=3, fill_value=0.0)
        expected = self.cp.zeros_like(data)
        expected[0, 1:3] = data[0, 1:3]
        expected[2, 0:2] = data[2, 0:2]
        self.cp.testing.assert_array_equal(subset_dense, expected)

        clone = clone_interval_field(field)
        clone.values.fill(0.0)
        original_dense = interval_field_to_dense(field, width=4, height=3, fill_value=0.0)
        self.cp.testing.assert_array_equal(original_dense, data)

        target = self._full_field(self.cp.zeros_like(data))
        scatter_interval_subset(target, subset_field)
        target_dense = interval_field_to_dense(target, width=4, height=3, fill_value=0.0)
        self.cp.testing.assert_array_equal(target_dense, expected)

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
