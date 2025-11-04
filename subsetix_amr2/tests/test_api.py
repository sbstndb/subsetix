import unittest

from subsetix_amr2.api import Box, MRAdaptor, TwoLevelMesh, make_scalar_field
from subsetix_cupy import create_interval_field
from subsetix_cupy.expressions import _REAL_CUPY


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class TwoLevelAPITest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY

    def test_mesh_field_and_adaptor(self) -> None:
        mesh = TwoLevelMesh(Box((0.0, 0.0), (1.0, 1.0)), 0, 1, ratio=2, coarse_resolution=8)
        self.assertIsNotNone(mesh.geometry)
        field = make_scalar_field("u", mesh)
        self.assertEqual(field.coarse_field.interval_set.row_count, 8)
        self.assertEqual(field.coarse_field.values.size, 8 * 8)
        self.assertEqual(field.fine_field.interval_set.row_count, 16)
        self.assertEqual(field.fine_field.values.size, 16 * 16)

        initial = self.cp.linspace(0.0, 1.0, 64, dtype=self.cp.float32).reshape(8, 8)
        field.coarse_field.values[...] = initial.reshape(-1)
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
        u.coarse_field.values.fill(1.0)
        v.coarse_field.values.fill(2.0)
        u.swap(v)
        coarse_u = u.coarse_field.values.reshape(4, 4)
        coarse_v = v.coarse_field.values.reshape(4, 4)
        self.cp.testing.assert_array_equal(coarse_u, self.cp.full((4, 4), 2.0, dtype=coarse_u.dtype))
        self.cp.testing.assert_array_equal(coarse_v, self.cp.full((4, 4), 1.0, dtype=coarse_v.dtype))

    def test_set_interval_fields_replaces_storage(self) -> None:
        mesh = TwoLevelMesh(Box((0.0, 0.0), (1.0, 1.0)), 0, 1, coarse_resolution=2)
        field = make_scalar_field("u", mesh)
        interval = field.coarse_field.interval_set
        new_field = create_interval_field(interval, fill_value=3.0, dtype=self.cp.float32)
        field.set_interval_fields(coarse=new_field)
        self.cp.testing.assert_array_equal(field.coarse_field.values, self.cp.full_like(new_field.values, 3.0))

    def test_mesh_initialise_rejects_dense_refine(self) -> None:
        mesh = TwoLevelMesh(Box((0.0, 0.0), (1.0, 1.0)), 0, 1, coarse_resolution=4)
        dense_refine = self.cp.zeros((4, 4), dtype=self.cp.int8)
        with self.assertRaises(TypeError):
            mesh.initialise(refine=dense_refine)  # type: ignore[arg-type]

    def test_mesh_regrid_rejects_dense_refine(self) -> None:
        mesh = TwoLevelMesh(Box((0.0, 0.0), (1.0, 1.0)), 0, 1, coarse_resolution=4)
        dense_refine = self.cp.zeros((4, 4), dtype=self.cp.int8)
        with self.assertRaises(TypeError):
            mesh.regrid(dense_refine)  # type: ignore[arg-type]

    def test_set_interval_fields_rejects_dense_buffers(self) -> None:
        mesh = TwoLevelMesh(Box((0.0, 0.0), (1.0, 1.0)), 0, 1, coarse_resolution=2)
        field = make_scalar_field("u", mesh)
        dense = self.cp.zeros_like(field.coarse_field.values)
        with self.assertRaises(TypeError):
            field.set_interval_fields(coarse=dense)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
