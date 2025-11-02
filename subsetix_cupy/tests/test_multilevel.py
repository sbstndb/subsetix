import unittest

from subsetix_cupy import (
    MultiLevel2D,
    build_interval_set,
)
from subsetix_cupy.expressions import _REAL_CUPY


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class MultiLevelTest(unittest.TestCase):
    def test_create_and_set_levels(self) -> None:
        ml = MultiLevel2D.create(n_levels=3, base_ratio=2)
        self.assertEqual(ml.ratios, [1, 2, 4])
        self.assertEqual(ml.n_levels, 3)
        ml.validate()

        # Level 0: 2 rows
        lvl0 = build_interval_set(row_offsets=[0, 1, 2], begin=[0, 2], end=[1, 3])
        # Level 1: 4 rows
        lvl1 = build_interval_set(row_offsets=[0, 1, 1, 2, 2], begin=[5, 7], end=[6, 9])
        # Level 2: 8 rows (empty)
        lvl2 = build_interval_set(row_offsets=[0] * 9, begin=[], end=[])

        ml.set_level(0, lvl0)
        ml.set_level(1, lvl1)
        ml.set_level(2, lvl2)

        self.assertIs(ml.get_level(0), lvl0)
        self.assertIs(ml.get_level(1), lvl1)
        self.assertIs(ml.get_level(2), lvl2)

    def test_invalid_ratios(self) -> None:
        # Directly construct with invalid ratios
        ml = MultiLevel2D(ratios=[1, 3, 5], levels=[None, None, None])
        with self.assertRaises(ValueError):
            ml.validate()

    def test_set_level_out_of_range(self) -> None:
        ml = MultiLevel2D.create(2, base_ratio=2)
        with self.assertRaises(IndexError):
            ml.set_level(2, None)
        with self.assertRaises(IndexError):
            ml.get_level(-1)

    def test_custom_ratios(self) -> None:
        ml = MultiLevel2D(ratios=[1, 2, 8], levels=[None, None, None])
        ml.validate()
        self.assertEqual(ml.ratios[-1], 8)


if __name__ == "__main__":
    unittest.main()
