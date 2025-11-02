import unittest

from subsetix_amr2.geometry import (
    TwoLevelGeometry,
    interval_set_to_mask,
    mask_to_interval_set,
)
from subsetix_cupy.expressions import _REAL_CUPY


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class GeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY

    def test_mask_roundtrip(self) -> None:
        mask = self.cp.zeros((4, 6), dtype=self.cp.bool_)
        mask[1, 1:4] = True
        mask[3, 2:6] = True
        interval = mask_to_interval_set(mask)
        back = interval_set_to_mask(interval, mask.shape[1])
        self.cp.testing.assert_array_equal(back, mask)

    def test_two_level_geometry_basic(self) -> None:
        coarse_mask = self.cp.ones((4, 4), dtype=self.cp.bool_)
        refine_mask = self.cp.zeros_like(coarse_mask)
        refine_mask[1:3, 1:3] = True

        geom = TwoLevelGeometry.from_masks(refine_mask, coarse_mask=coarse_mask, ratio=2)
        self.assertEqual(geom.ratio, 2)
        self.assertEqual(geom.width, 4)
        self.assertEqual(geom.height, 4)

        self.cp.testing.assert_array_equal(geom.refine_mask, refine_mask)
        coarse_only = geom.coarse_only_mask
        expected_coarse_only = self.cp.array(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )
        self.cp.testing.assert_array_equal(coarse_only, expected_coarse_only)

        fine_mask = geom.fine_mask
        expected_fine = self.cp.zeros((8, 8), dtype=self.cp.bool_)
        expected_fine[2:6, 2:6] = True
        self.cp.testing.assert_array_equal(fine_mask, expected_fine)

    def test_refine_must_be_subset(self) -> None:
        coarse_mask = self.cp.zeros((4, 4), dtype=self.cp.bool_)
        coarse_mask[::2, ::2] = True
        refine_mask = self.cp.zeros_like(coarse_mask)
        refine_mask[1, 1] = True  # outside coarse coverage
        with self.assertRaises(ValueError):
            TwoLevelGeometry.from_masks(refine_mask, coarse_mask=coarse_mask, ratio=2)

    def test_with_refine_mask_reuses_workspace(self) -> None:
        refine_mask = self.cp.zeros((4, 4), dtype=self.cp.bool_)
        refine_mask[1:3, 1:3] = True
        geom = TwoLevelGeometry.from_masks(refine_mask, ratio=2)
        workspace_before = geom.workspace
        new_mask = self.cp.zeros_like(refine_mask)
        new_mask[0:2, 0:2] = True
        geom_next = geom.with_refine_mask(new_mask)
        self.assertIs(geom_next.workspace, workspace_before)
        self.cp.testing.assert_array_equal(geom_next.refine_mask, new_mask)

    def test_dilation_modes(self) -> None:
        refine_mask = self.cp.zeros((5, 5), dtype=self.cp.bool_)
        refine_mask[2, 2] = True
        geom = TwoLevelGeometry.from_masks(refine_mask, ratio=2)

        vn = geom.dilate_refine(halo=1, mode="von_neumann")
        vn_mask = vn.refine_mask
        expected_vn = self.cp.zeros_like(refine_mask)
        expected_vn[2, 2] = True
        expected_vn[1, 2] = True
        expected_vn[3, 2] = True
        expected_vn[2, 1] = True
        expected_vn[2, 3] = True
        self.cp.testing.assert_array_equal(vn_mask, expected_vn)

        moore = geom.dilate_refine(halo=1, mode="moore")
        moore_mask = moore.refine_mask
        expected_moore = self.cp.zeros_like(refine_mask)
        expected_moore[1:4, 1:4] = True
        self.cp.testing.assert_array_equal(moore_mask, expected_moore)
