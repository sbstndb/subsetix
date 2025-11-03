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
        coarse_dense = field.to_dense(mesh.min_level)
        fine_dense = field.to_dense(mesh.max_level)
        self.assertEqual(coarse_dense.shape, (8, 8))
        self.assertEqual(fine_dense.shape, (16, 16))

        initial = self.cp.linspace(0.0, 1.0, 64, dtype=self.cp.float32).reshape(8, 8)
        field.load_dense(mesh.min_level, initial)
        adaptor = MRAdaptor(field, refine_threshold=0.05, grading=1)
        adaptor()

        self.assertIsNotNone(mesh.geometry)
        geom = mesh.geometry
        assert geom is not None
        self.assertEqual(geom.refine.row_offsets.size, geom.height + 1)
        self.assertEqual(geom.refine.begin.dtype, self.cp.int32)
        self.assertAlmostEqual(mesh.cell_length(0), 1.0 / 8.0, places=6)
        self.assertAlmostEqual(mesh.cell_length(1), 1.0 / 16.0, places=6)

    def test_scalar_field_swap(self) -> None:
        mesh = TwoLevelMesh(Box((0.0, 0.0), (1.0, 1.0)), 0, 1, coarse_resolution=4)
        u = make_scalar_field("u", mesh)
        v = make_scalar_field("v", mesh)
        u.fill(mesh.min_level, 1.0)
        v.fill(mesh.min_level, 2.0)
        u.swap(v)
        coarse_u = u.to_dense(mesh.min_level)
        coarse_v = v.to_dense(mesh.min_level)
        self.cp.testing.assert_array_equal(coarse_u, self.cp.full((4, 4), 2.0, dtype=coarse_u.dtype))
        self.cp.testing.assert_array_equal(coarse_v, self.cp.full((4, 4), 1.0, dtype=coarse_v.dtype))

    def test_scalar_field_to_dense_copy_semantics(self) -> None:
        mesh = TwoLevelMesh(Box((0.0, 0.0), (1.0, 1.0)), 0, 1, coarse_resolution=2)
        field = make_scalar_field("u", mesh)
        dense_copy = field.to_dense(mesh.min_level)
        dense_copy[...] = 1.0
        dense_view = field.to_dense(mesh.min_level, copy=False)
        self.cp.testing.assert_array_equal(dense_view, self.cp.zeros_like(dense_view))
        dense_view[...] = 2.0
        dense_check = field.to_dense(mesh.min_level)
        self.cp.testing.assert_array_equal(dense_check, self.cp.full((2, 2), 2.0, dtype=dense_check.dtype))


if __name__ == "__main__":
    unittest.main()
