import unittest

from subsetix_amr2.fields import (
    Action,
    ActionField,
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

    def test_synchronize_interval_fields_copy_and_fill(self) -> None:
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

        coarse_dense = interval_field_to_dense(coarse_sync, width=4, height=4, fill_value=0.0)
        fine_dense = interval_field_to_dense(fine_sync, width=8, height=8, fill_value=0.0)

        expected_coarse = self.cp.zeros((4, 4), dtype=self.cp.float32)
        expected_coarse[1:3, 1:3] = 2.0
        self.cp.testing.assert_array_equal(coarse_dense, expected_coarse)

        expected_fine = self.cp.zeros((8, 8), dtype=self.cp.float32)
        expected_fine[2:6, 2:6] = 2.0
        self.cp.testing.assert_array_equal(fine_dense, expected_fine)

        # original buffers untouched in copy=True mode
        self.cp.testing.assert_array_equal(interval_field_to_dense(coarse_field, width=4, height=4, fill_value=0.0), coarse)
        self.cp.testing.assert_array_equal(interval_field_to_dense(fine_field, width=8, height=8, fill_value=0.0), fine)

    def test_synchronize_interval_fields_inplace_and_no_fill(self) -> None:
        coarse = self.cp.zeros((2, 2), dtype=self.cp.float32)
        fine = self.cp.arange(16, dtype=self.cp.float32).reshape(4, 4)
        refine = _make_interval_set(
            self.cp,
            2,
            {
                0: [(0, 1)],
                1: [(0, 1)],
            },
        )

        actions = ActionField.full_grid(2, 2, ratio=2)
        actions.set_from_interval_set(refine)

        coarse_field = self._full_field(coarse)
        fine_field = self._full_field(fine)

        coarse_res, fine_res = synchronize_interval_fields(
            coarse_field,
            fine_field,
            actions,
            ratio=2,
            reducer="mean",
            fill_fine_outside=False,
            copy=False,
        )

        self.assertIs(coarse_res, coarse_field)
        self.assertIs(fine_res, fine_field)

        coarse_dense = interval_field_to_dense(coarse_res, width=2, height=2, fill_value=0.0)
        expected_coarse = self.cp.zeros((2, 2), dtype=self.cp.float32)
        expected_coarse[0, 0] = fine[0:2, 0:2].mean()
        expected_coarse[1, 0] = fine[2:4, 0:2].mean()
        self.cp.testing.assert_allclose(coarse_dense, expected_coarse)

        fine_dense = interval_field_to_dense(fine_res, width=4, height=4, fill_value=0.0)
        self.cp.testing.assert_array_equal(fine_dense, fine)

    def test_synchronize_interval_fields_invalid_ratio(self) -> None:
        coarse = self._full_field(self.cp.zeros((2, 2), dtype=self.cp.float32))
        fine = self._full_field(self.cp.zeros((4, 4), dtype=self.cp.float32))
        actions = ActionField.full_grid(2, 2, ratio=2)
        with self.assertRaises(ValueError):
            synchronize_interval_fields(coarse, fine, actions, ratio=3)

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
