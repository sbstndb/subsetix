import unittest

from subsetix_amr2.geometry import interval_set_to_mask, mask_to_interval_set
from subsetix_amr2.regrid import (
    enforce_two_level_grading_set,
    gradient_magnitude,
    gradient_tag_threshold_set,
    gradient_tag_set,
)
from subsetix_cupy.expressions import _REAL_CUPY


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class RegridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY

    def test_gradient_magnitude(self) -> None:
        data = self.cp.arange(16, dtype=self.cp.float32).reshape(4, 4)
        grad = gradient_magnitude(data)
        self.assertEqual(grad.shape, data.shape)
        self.assertTrue(self.cp.all(grad >= 0))

    def test_gradient_tag_percentile(self) -> None:
        data = self.cp.zeros((4, 4), dtype=self.cp.float32)
        data[1, 1] = 0.5
        data[2, 2] = 1.0
        tagged_set = gradient_tag_set(data, frac_high=0.5)
        mask = interval_set_to_mask(tagged_set, data.shape[1])
        self.assertTrue(mask[2, 2])
        self.assertFalse(mask[1, 1])
        self.assertFalse(mask[0, 0])

    def test_gradient_tag_all_zero(self) -> None:
        data = self.cp.zeros((4, 4), dtype=self.cp.float32)
        tagged_set = gradient_tag_set(data, frac_high=0.2)
        mask = interval_set_to_mask(tagged_set, data.shape[1])
        self.assertFalse(mask.any())

    def test_gradient_tag_threshold(self) -> None:
        data = self.cp.zeros((4, 4), dtype=self.cp.float32)
        data[1, 1] = 0.5
        data[2, 2] = 1.0
        tagged_set = gradient_tag_threshold_set(data, threshold=0.75)
        mask = interval_set_to_mask(tagged_set, data.shape[1])
        self.assertTrue(mask[2, 2])
        self.assertFalse(mask[1, 1])

    def test_gradient_tag_custom_epsilon(self) -> None:
        data = self.cp.full((4, 4), 1e-9, dtype=self.cp.float32)
        tagged_set = gradient_tag_set(data, frac_high=0.2, epsilon=1e-8)
        mask = interval_set_to_mask(tagged_set, data.shape[1])
        self.assertFalse(mask.any())

    def test_enforce_grading(self) -> None:
        mask = self.cp.zeros((5, 5), dtype=self.cp.bool_)
        mask[2, 2] = True
        refine_set = mask_to_interval_set(mask)
        graded_vn_set = enforce_two_level_grading_set(
            refine_set,
            padding=1,
            mode="von_neumann",
            width=mask.shape[1],
            height=mask.shape[0],
        )
        graded_vn = interval_set_to_mask(graded_vn_set, mask.shape[1])
        expected_cross = self.cp.zeros_like(mask)
        expected_cross[2, 2] = True
        expected_cross[1, 2] = True
        expected_cross[3, 2] = True
        expected_cross[2, 1] = True
        expected_cross[2, 3] = True
        self.cp.testing.assert_array_equal(graded_vn, expected_cross)

        graded_moore_set = enforce_two_level_grading_set(
            refine_set,
            padding=1,
            mode="moore",
            width=mask.shape[1],
            height=mask.shape[0],
        )
        graded_moore = interval_set_to_mask(graded_moore_set, mask.shape[1])
        expected_square = self.cp.zeros_like(mask)
        expected_square[1:4, 1:4] = True
        self.cp.testing.assert_array_equal(graded_moore, expected_square)

    def test_enforce_grading_padding_zero(self) -> None:
        mask = self.cp.zeros((4, 4), dtype=self.cp.bool_)
        mask[1:3, 1:3] = True
        refine_set = mask_to_interval_set(mask)
        graded_set = enforce_two_level_grading_set(
            refine_set,
            padding=0,
            mode="von_neumann",
            width=mask.shape[1],
            height=mask.shape[0],
        )
        graded = interval_set_to_mask(graded_set, mask.shape[1])
        self.cp.testing.assert_array_equal(graded, mask)

    def test_enforce_grading_invalid_mode(self) -> None:
        mask = self.cp.zeros((4, 4), dtype=self.cp.bool_)
        with self.assertRaises(ValueError):
            enforce_two_level_grading_set(
                mask_to_interval_set(mask),
                padding=1,
                mode="diag",
                width=mask.shape[1],
                height=mask.shape[0],
            )
