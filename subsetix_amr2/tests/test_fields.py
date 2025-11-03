import unittest

from subsetix_amr2.fields import (
    prolong_coarse_to_fine,
    restrict_fine_to_coarse,
    synchronize_two_level,
)
from subsetix_cupy.expressions import _REAL_CUPY


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

    def test_prolong_with_mask_updates_subset(self) -> None:
        coarse = self.cp.arange(4, dtype=self.cp.float32).reshape(2, 2)
        mask = self.cp.zeros((4, 4), dtype=self.cp.bool_)
        mask[::2, ::2] = True
        out = self.cp.full((4, 4), -1.0, dtype=self.cp.float32)
        prolong_coarse_to_fine(coarse, 2, out=out, mask=mask)
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
        refine = self.cp.zeros_like(coarse, dtype=self.cp.bool_)
        refine[1:3, 1:3] = True
        coarse_updated, fine_updated = synchronize_two_level(
            coarse, fine, refine, ratio=2, reducer="mean", fill_fine_outside=True
        )

        # Coarse cells inside refine area should have mean value 2.
        expected_coarse = self.cp.zeros_like(coarse)
        expected_coarse[1:3, 1:3] = 2.0
        self.cp.testing.assert_array_equal(coarse_updated, expected_coarse)

        fine_mask = self.cp.repeat(self.cp.repeat(refine, 2, axis=0), 2, axis=1)
        # Outside refine area fine grid should take coarse values (still zero).
        self.cp.testing.assert_array_equal(fine_updated[~fine_mask], 0.0)
        self.cp.testing.assert_array_equal(fine_updated[fine_mask], 2.0)

    def test_synchronize_without_fill(self) -> None:
        coarse = self.cp.zeros((4, 4), dtype=self.cp.float32)
        fine = self.cp.arange(64, dtype=self.cp.float32).reshape(8, 8)
        refine = self.cp.zeros_like(coarse, dtype=self.cp.bool_)
        refine[1:3, 1:3] = True
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

    def test_synchronize_mask_shape_mismatch(self) -> None:
        coarse = self.cp.zeros((4, 4), dtype=self.cp.float32)
        fine = self.cp.zeros((8, 8), dtype=self.cp.float32)
        refine = self.cp.zeros((2, 2), dtype=self.cp.bool_)
        with self.assertRaises(ValueError):
            synchronize_two_level(coarse, fine, refine, ratio=2)
