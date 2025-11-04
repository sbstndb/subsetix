import unittest

from subsetix_amr2.fields import (
    Action,
    ActionField,
    synchronize_interval_fields,
    gather_interval_subset,
    scatter_interval_subset,
)
from subsetix_cupy.expressions import IntervalSet, _REAL_CUPY
from subsetix_cupy.morphology import full_interval_set
from subsetix_cupy import create_interval_field
from subsetix_cupy.interval_field import IntervalField, get_cell


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


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class FieldsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY

    def _full_field(
        self,
        height: int,
        width: int,
        *,
        fill_value: float = 0.0,
        dtype=None,
    ) -> IntervalField:
        dtype = dtype or self.cp.float32
        full = full_interval_set(width, height)
        return create_interval_field(full, fill_value=fill_value, dtype=dtype)

    def _cell_value(self, field: IntervalField, row: int, col: int) -> float | None:
        value = get_cell(field, row, col)
        if value is None:
            return None
        return float(value.item())

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
        dtype = self.cp.float32
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

        coarse_field = self._full_field(height=4, width=4, dtype=dtype)
        fine_field = self._full_field(height=8, width=8, dtype=dtype, fill_value=2.0)
        coarse_initial = coarse_field.values.copy()
        fine_initial = fine_field.values.copy()

        coarse_sync, fine_sync = synchronize_interval_fields(
            coarse_field,
            fine_field,
            actions,
            ratio=2,
            reducer="mean",
            fill_fine_outside=True,
            copy=True,
        )

        active_coarse = {(1, 1), (1, 2), (2, 1), (2, 2)}
        for row in range(4):
            for col in range(4):
                value = self._cell_value(coarse_sync, row, col)
                self.assertIsNotNone(value)
                expected = 2.0 if (row, col) in active_coarse else 0.0
                self.assertEqual(value, expected)

        active_fine = {(row, col) for row in range(2, 6) for col in range(2, 6)}
        for row in range(8):
            for col in range(8):
                value = self._cell_value(fine_sync, row, col)
                self.assertIsNotNone(value)
                expected = 2.0 if (row, col) in active_fine else 0.0
                self.assertEqual(value, expected)

        # original buffers untouched in copy=True mode
        self.cp.testing.assert_array_equal(coarse_field.values, coarse_initial)
        self.cp.testing.assert_array_equal(fine_field.values, fine_initial)

    def test_synchronize_interval_fields_inplace_and_no_fill(self) -> None:
        dtype = self.cp.float32
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

        coarse_field = self._full_field(height=2, width=2, dtype=dtype)
        fine_field = self._full_field(height=4, width=4, dtype=dtype)
        fine_field.values[...] = self.cp.arange(fine_field.values.size, dtype=dtype)
        fine_initial = fine_field.values.copy()

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

        ratio = 2
        width = 2
        fine_width = width * ratio
        expected_flat = self.cp.zeros_like(coarse_res.values)
        refine_offsets = self.cp.asnumpy(refine.row_offsets)
        refine_begin = self.cp.asnumpy(refine.begin)
        refine_end = self.cp.asnumpy(refine.end)
        for coarse_row in range(len(refine_offsets) - 1):
            start = refine_offsets[coarse_row]
            stop = refine_offsets[coarse_row + 1]
            for idx in range(start, stop):
                col_start = refine_begin[idx]
                col_end = refine_end[idx]
                for coarse_col in range(col_start, col_end):
                    indices = []
                    for dy in range(ratio):
                        fine_row = coarse_row * ratio + dy
                        for dx in range(ratio):
                            fine_col = coarse_col * ratio + dx
                            indices.append(fine_row * fine_width + fine_col)
                    if not indices:
                        continue
                    indices_cp = self.cp.asarray(indices, dtype=self.cp.int32)
                    mean_value = self.cp.mean(fine_initial[indices_cp])
                    index = coarse_row * width + coarse_col
                    expected_flat[index] = mean_value

        # fine values unchanged in-place without fill
        self.cp.testing.assert_array_equal(fine_res.values, fine_initial)
        self.cp.testing.assert_allclose(coarse_res.values, expected_flat)
        for row in range(2):
            for col in range(2):
                value = self._cell_value(coarse_res, row, col)
                self.assertIsNotNone(value)
                expected = float(expected_flat[row * width + col].item())
                self.assertAlmostEqual(value, expected)

    def test_synchronize_interval_fields_invalid_ratio(self) -> None:
        coarse = self._full_field(height=2, width=2, dtype=self.cp.float32)
        fine = self._full_field(height=4, width=4, dtype=self.cp.float32)
        actions = ActionField.full_grid(2, 2, ratio=2)
        with self.assertRaises(ValueError):
            synchronize_interval_fields(coarse, fine, actions, ratio=3)

    def test_gather_scatter_interval_subset(self) -> None:
        dtype = self.cp.float32
        field = self._full_field(height=3, width=4, dtype=dtype)
        field.values[...] = self.cp.arange(field.values.size, dtype=dtype)
        subset = _make_interval_set(
            self.cp,
            3,
            {
                0: [(1, 3)],
                2: [(0, 2)],
            },
        )
        subset_field = gather_interval_subset(field, subset)
        self.assertEqual(subset_field.values.size, 4)

        expected_subset = {
            (0, 1): 1.0,
            (0, 2): 2.0,
            (2, 0): 8.0,
            (2, 1): 9.0,
        }
        for row in range(3):
            for col in range(4):
                value = self._cell_value(subset_field, row, col)
                if (row, col) in expected_subset:
                    self.assertEqual(value, expected_subset[(row, col)])
                else:
                    self.assertIsNone(value)

        self.cp.testing.assert_array_equal(field.values, self.cp.arange(field.values.size, dtype=dtype))

        target = self._full_field(height=3, width=4, dtype=dtype)
        scatter_interval_subset(target, subset_field)
        for row in range(3):
            for col in range(4):
                value = self._cell_value(target, row, col)
                if (row, col) in expected_subset:
                    self.assertEqual(value, expected_subset[(row, col)])
                else:
                    self.assertEqual(value, 0.0)
