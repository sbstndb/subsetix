import unittest

from subsetix_amr2.api import Box, MRAdaptor, TwoLevelMesh, make_scalar_field
from subsetix_cupy.expressions import _REAL_CUPY


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class TwoLevelAPITest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY

    def test_mesh_field_and_adaptor(self) -> None:
        mesh = TwoLevelMesh(Box((0.0, 0.0), (1.0, 1.0)), 0, 1, ratio=2, coarse_resolution=8)
        self.assertIsNotNone(mesh.geometry)
        field = make_scalar_field("u", mesh)
        self.assertEqual(field.coarse.shape, (8, 8))
        self.assertEqual(field.fine.shape, (16, 16))

        field.coarse[...] = self.cp.linspace(0.0, 1.0, 64, dtype=self.cp.float32).reshape(8, 8)
        adaptor = MRAdaptor(field, refine_threshold=0.05, grading=1)
        adaptor()

        self.assertIsNotNone(mesh.geometry)
        geom = mesh.geometry
        assert geom is not None
        self.assertEqual(geom.refine.row_offsets.size, field.coarse.shape[0] + 1)
        self.assertEqual(geom.refine.begin.dtype, self.cp.int32)
        self.assertAlmostEqual(mesh.cell_length(0), 1.0 / 8.0, places=6)
        self.assertAlmostEqual(mesh.cell_length(1), 1.0 / 16.0, places=6)

    def test_scalar_field_swap(self) -> None:
        mesh = TwoLevelMesh(Box((0.0, 0.0), (1.0, 1.0)), 0, 1, coarse_resolution=4)
        u = make_scalar_field("u", mesh)
        v = make_scalar_field("v", mesh)
        u.coarse.fill(1.0)
        v.coarse.fill(2.0)
        u.swap(v)
        self.cp.testing.assert_array_equal(u.coarse, self.cp.full((4, 4), 2.0, dtype=u.coarse.dtype))
        self.cp.testing.assert_array_equal(v.coarse, self.cp.full((4, 4), 1.0, dtype=v.coarse.dtype))

    def test_scalar_field_as_arrays(self) -> None:
        mesh = TwoLevelMesh(Box((0.0, 0.0), (1.0, 1.0)), 0, 1, coarse_resolution=2)
        field = make_scalar_field("u", mesh)
        coarse, fine = field.as_arrays()
        self.assertIs(coarse, field.coarse)
        self.assertIs(fine, field.fine)


if __name__ == "__main__":
    unittest.main()
