import unittest

from subsetix_amr2.geometry import TwoLevelGeometry, interval_set_to_mask, mask_to_interval_set
from subsetix_amr2.fields import ActionField, Action
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
        refine_mask = self.cp.zeros((4, 4), dtype=self.cp.bool_)
        refine_mask[1:3, 1:3] = True
        refine_set = mask_to_interval_set(refine_mask)
        actions = ActionField.full_grid(4, 4, ratio=2, default=Action.KEEP)
        actions.set_from_interval_set(refine_set)

        geom = TwoLevelGeometry.from_action_field(actions)
        self.assertEqual(geom.ratio, 2)
        self.assertEqual(geom.width, 4)
        self.assertEqual(geom.height, 4)

        self.cp.testing.assert_array_equal(interval_set_to_mask(geom.refine, geom.width), refine_mask)
        coarse_only = interval_set_to_mask(geom.coarse_only, geom.width)
        expected_coarse_only = self.cp.array(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )
        self.cp.testing.assert_array_equal(coarse_only, expected_coarse_only)

        fine_mask = interval_set_to_mask(geom.fine, geom.width * geom.ratio)
        expected_fine = self.cp.zeros((8, 8), dtype=self.cp.bool_)
        expected_fine[2:6, 2:6] = True
        self.cp.testing.assert_array_equal(fine_mask, expected_fine)

    def test_with_action_field_reuses_workspace(self) -> None:
        refine_mask = self.cp.zeros((4, 4), dtype=self.cp.bool_)
        refine_mask[1:3, 1:3] = True
        actions = ActionField.full_grid(4, 4, ratio=2)
        actions.set_from_interval_set(mask_to_interval_set(refine_mask))
        geom = TwoLevelGeometry.from_action_field(actions)
        workspace_before = geom.workspace
        new_mask = self.cp.zeros_like(refine_mask)
        new_mask[0:2, 0:2] = True
        next_actions = ActionField.full_grid(4, 4, ratio=2)
        next_actions.set_from_interval_set(mask_to_interval_set(new_mask))
        geom_next = geom.with_action_field(next_actions)
        self.assertIs(geom_next.workspace, workspace_before)
        self.cp.testing.assert_array_equal(
            interval_set_to_mask(geom_next.refine, geom_next.width),
            new_mask,
        )

    def test_dilation_modes(self) -> None:
        refine_mask = self.cp.zeros((5, 5), dtype=self.cp.bool_)
        refine_mask[2, 2] = True
        actions = ActionField.full_grid(5, 5, ratio=2)
        actions.set_from_interval_set(mask_to_interval_set(refine_mask))
        geom = TwoLevelGeometry.from_action_field(actions)

        vn = geom.dilate_refine(halo=1, mode="von_neumann")
        vn_mask = interval_set_to_mask(vn.refine, vn.width)
        expected_vn = self.cp.zeros_like(refine_mask)
        expected_vn[2, 2] = True
        expected_vn[1, 2] = True
        expected_vn[3, 2] = True
        expected_vn[2, 1] = True
        expected_vn[2, 3] = True
        self.cp.testing.assert_array_equal(vn_mask, expected_vn)

        moore = geom.dilate_refine(halo=1, mode="moore")
        moore_mask = interval_set_to_mask(moore.refine, moore.width)
        expected_moore = self.cp.zeros_like(refine_mask)
        expected_moore[1:4, 1:4] = True
        self.cp.testing.assert_array_equal(moore_mask, expected_moore)
