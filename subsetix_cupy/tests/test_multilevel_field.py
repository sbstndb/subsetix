import unittest

import numpy as np

from subsetix_cupy import (
    MultiLevel2D,
    MultiLevelField2D,
    build_interval_set,
)
from subsetix_cupy.expressions import _REAL_CUPY


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class MultiLevelFieldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY
        # Geometry with 3 levels
        self.ml = MultiLevel2D.create(3, base_ratio=2)
        # lvl0: 2 rows, 2 intervals
        lvl0 = build_interval_set(row_offsets=[0, 1, 2], begin=[0, 10], end=[3, 12])
        # lvl1: 4 rows, 1 interval
        lvl1 = build_interval_set(row_offsets=[0, 0, 1, 1, 1], begin=[5], end=[8])
        # lvl2: 8 rows, empty
        lvl2 = build_interval_set(row_offsets=[0] * 9, begin=[], end=[])
        self.ml.set_level(0, lvl0)
        self.ml.set_level(1, lvl1)
        self.ml.set_level(2, lvl2)

    def test_create_from_geometry(self) -> None:
        fields = MultiLevelField2D.create_from_geometry(self.ml, fill_value=1.5, dtype=self.cp.float32)
        fields.validate_against(self.ml)
        self.assertEqual(len(fields.fields), 3)
        # lvl0: 3 + 2 = 5 cells
        self.assertEqual(int(fields.fields[0].values.size), 5)
        self.assertTrue(bool(self.cp.all(fields.fields[0].values == self.cp.float32(1.5))))
        # lvl1: 3 cells
        self.assertEqual(int(fields.fields[1].values.size), 3)
        # lvl2: empty
        self.assertIsNotNone(fields.fields[2])
        self.assertEqual(int(fields.fields[2].values.size), 0)

    def test_empty_like_and_set(self) -> None:
        mf = MultiLevelField2D.empty_like(self.ml)
        self.assertEqual(mf.ratios, self.ml.ratios)
        self.assertEqual(len(mf.fields), 3)
        self.assertTrue(all(v is None for v in mf.fields))

    def test_validate_against_mismatch(self) -> None:
        mf = MultiLevelField2D.empty_like(self.ml)
        # Modify ratios to trigger validation error
        mf.ratios = [1, 4, 8]
        with self.assertRaises(ValueError):
            mf.validate_against(self.ml)

    def test_set_level_field_out_of_range(self) -> None:
        mf = MultiLevelField2D.empty_like(self.ml)
        with self.assertRaises(IndexError):
            mf.set_level_field(3, None)
        with self.assertRaises(IndexError):
            mf.get_level_field(-1)


if __name__ == "__main__":
    unittest.main()
