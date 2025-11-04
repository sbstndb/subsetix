import unittest

import numpy as np

from subsetix_cupy import (
    MultiLevel2D,
    MultiLevelField2D,
    build_interval_set,
    create_interval_field,
    covered_by_fine,
    coarse_only,
    prolong_field,
    prolong_level_field,
    prolong_level_sets,
    prolong_set,
    restrict_field,
    restrict_level_field,
    restrict_level_sets,
    restrict_set,
)
from subsetix_cupy.expressions import _REAL_CUPY


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class InterLevelOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY

    def test_prolong_restrict_set_roundtrip(self) -> None:
        coarse = build_interval_set(
            row_offsets=[0, 2, 3],
            begin=[1, 4, 0],
            end=[3, 6, 2],
        )
        fine = prolong_set(coarse, ratio=2)
        self.assertEqual(fine.row_offsets.size - 1, (coarse.row_offsets.size - 1) * 2)
        back = restrict_set(fine, ratio=2)
        np.testing.assert_array_equal(
            self.cp.asnumpy(back.begin), self.cp.asnumpy(coarse.begin)
        )
        np.testing.assert_array_equal(
            self.cp.asnumpy(back.end), self.cp.asnumpy(coarse.end)
        )
        np.testing.assert_array_equal(
            self.cp.asnumpy(back.row_offsets), self.cp.asnumpy(coarse.row_offsets)
        )

    def test_prolong_identity_ratio_one(self) -> None:
        coarse = build_interval_set(
            row_offsets=[0, 1, 2],
            begin=[0, 5],
            end=[3, 8],
        )
        same = prolong_set(coarse, ratio=1)
        np.testing.assert_array_equal(
            self.cp.asnumpy(same.begin), self.cp.asnumpy(coarse.begin)
        )
        np.testing.assert_array_equal(
            self.cp.asnumpy(same.end), self.cp.asnumpy(coarse.end)
        )
        np.testing.assert_array_equal(
            self.cp.asnumpy(same.row_offsets), self.cp.asnumpy(coarse.row_offsets)
        )

    def test_prolong_restrict_field_roundtrip(self) -> None:
        coarse_set = build_interval_set(
            row_offsets=[0, 1, 2],
            begin=[0, 2],
            end=[2, 5],
        )
        coarse_field = create_interval_field(coarse_set, fill_value=0.0, dtype=self.cp.float32)
        coarse_field.values[:] = self.cp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=self.cp.float32)
        fine_field = prolong_field(coarse_field, ratio=2)
        self.assertEqual(
            fine_field.interval_set.row_offsets.size - 1,
            (coarse_set.row_offsets.size - 1) * 2,
        )
        restricted = restrict_field(fine_field, ratio=2, reducer="mean")
        np.testing.assert_allclose(
            self.cp.asnumpy(restricted.values),
            self.cp.asnumpy(coarse_field.values),
        )

    def test_restrict_field_misaligned_raises(self) -> None:
        coarse_set = build_interval_set(
            row_offsets=[0, 1],
            begin=[0],
            end=[4],
        )
        fine_set = build_interval_set(
            row_offsets=[0, 1, 2, 3, 4],
            begin=[0, 2, 4, 6],
            end=[2, 4, 6, 8],
        )
        fine_field = create_interval_field(fine_set, fill_value=1.0, dtype=self.cp.float32)
        with self.assertRaises(ValueError):
            restrict_field(fine_field, ratio=2)

    def test_multi_level_wrappers(self) -> None:
        ml = MultiLevel2D.create(2, base_ratio=2)
        level0 = build_interval_set(
            row_offsets=[0, 1, 2],
            begin=[1, 3],
            end=[4, 5],
        )
        ml.set_level(0, level0)
        fine = prolong_level_sets(ml, 0)
        self.assertIs(ml.get_level(1), fine)
        back = restrict_level_sets(ml, 0)
        np.testing.assert_array_equal(
            self.cp.asnumpy(back.begin), self.cp.asnumpy(level0.begin)
        )

        covered = covered_by_fine(ml, 0)
        self.assertIsNotNone(covered)
        diff = coarse_only(level0, ml, 0)
        self.assertEqual(int(diff.row_offsets[-1].item()), 0)

        mf = MultiLevelField2D.create_from_geometry(ml, fill_value=1.0, dtype=self.cp.float32)
        base_values = self.cp.arange(mf.fields[0].values.size, dtype=self.cp.float32) + 2.0
        mf.fields[0].values[:] = base_values
        fine_field = prolong_level_field(ml, mf, 0)
        self.assertIs(mf.get_level_field(1), fine_field)
        coarse_back = restrict_level_field(ml, mf, 0, reducer="mean")
        np.testing.assert_allclose(
            self.cp.asnumpy(coarse_back.values),
            self.cp.asnumpy(mf.fields[0].values),
        )

    def test_restrict_field_sum(self) -> None:
        coarse_set = build_interval_set(row_offsets=[0, 1], begin=[0], end=[2])
        coarse_field = create_interval_field(coarse_set, fill_value=0.0, dtype=self.cp.float32)
        coarse_field.values[:] = self.cp.asarray([1.0, 2.0], dtype=self.cp.float32)
        fine_field = prolong_field(coarse_field, ratio=2)
        restricted = restrict_field(fine_field, ratio=2, reducer="sum")
        np.testing.assert_allclose(
            self.cp.asnumpy(restricted.values),
            np.array([4.0, 8.0], dtype=np.float32),
        )

    def test_restrict_field_mean_promotes_integer_dtype(self) -> None:
        coarse_set = build_interval_set(row_offsets=[0, 1], begin=[0], end=[2])
        coarse_field = create_interval_field(coarse_set, fill_value=0, dtype=self.cp.int32)
        fine_field = prolong_field(coarse_field, ratio=2)
        pattern = self.cp.asarray([0, 1, 0, 1], dtype=self.cp.int32)
        repeats = max(1, (fine_field.values.size + pattern.size - 1) // pattern.size)
        tiled = self.cp.tile(pattern, repeats)[: fine_field.values.size]
        fine_field.values[:] = tiled
        restricted = restrict_field(fine_field, ratio=2, reducer="mean")
        self.assertTrue(np.issubdtype(restricted.values.dtype, np.floating))
        np.testing.assert_allclose(
            self.cp.asnumpy(restricted.values),
            np.array([0.5, 0.5], dtype=np.float32),
        )

    def test_covered_by_fine_missing_level(self) -> None:
        ml = MultiLevel2D.create(2, base_ratio=2)
        level0 = build_interval_set(
            row_offsets=[0, 1, 2],
            begin=[0, 4],
            end=[2, 6],
        )
        ml.set_level(0, level0)
        covered = covered_by_fine(ml, 0)
        self.assertIsNone(covered)
        diff = coarse_only(level0, ml, 0)
        np.testing.assert_array_equal(
            self.cp.asnumpy(diff.begin),
            self.cp.asnumpy(level0.begin),
        )

    def test_restrict_set_expands_partial_fine_block(self) -> None:
        fine = build_interval_set(
            row_offsets=[0, 1, 1, 2, 2],
            begin=[1, 3],
            end=[2, 4],
        )
        coarse = restrict_set(fine, ratio=2)
        self.cp.testing.assert_array_equal(
            coarse.row_offsets, self.cp.asarray([0, 1, 2], dtype=self.cp.int32)
        )
        self.cp.testing.assert_array_equal(
            coarse.begin, self.cp.asarray([0, 1], dtype=self.cp.int32)
        )
        self.cp.testing.assert_array_equal(
            coarse.end, self.cp.asarray([1, 2], dtype=self.cp.int32)
        )

    def test_prolong_set_expands_coarse_cell_to_full_fine_block(self) -> None:
        coarse = build_interval_set(
            row_offsets=[0, 1, 2],
            begin=[1, 3],
            end=[2, 4],
        )
        fine = prolong_set(coarse, ratio=2)
        self.cp.testing.assert_array_equal(
            fine.row_offsets, self.cp.asarray([0, 1, 2, 3, 4], dtype=self.cp.int32)
        )
        self.cp.testing.assert_array_equal(
            fine.begin, self.cp.asarray([2, 2, 6, 6], dtype=self.cp.int32)
        )
        self.cp.testing.assert_array_equal(
            fine.end, self.cp.asarray([4, 4, 8, 8], dtype=self.cp.int32)
        )

    def test_restrict_field_partial_block_raises(self) -> None:
        fine = build_interval_set(
            row_offsets=[0, 1, 1, 2, 2],
            begin=[1, 3],
            end=[2, 4],
        )
        fine_field = create_interval_field(fine, fill_value=1.0, dtype=self.cp.float32)
        with self.assertRaises(ValueError):
            restrict_field(fine_field, ratio=2, reducer="mean")


if __name__ == "__main__":
    unittest.main()
