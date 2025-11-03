import os
import tempfile
import unittest

from subsetix_amr2.export import save_two_level_vtk
from subsetix_amr2.geometry import mask_to_interval_set
from subsetix_cupy.expressions import _REAL_CUPY


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class ExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY

    def test_save_two_level_vtk_creates_files(self) -> None:
        cp = self.cp
        coarse = cp.ones((4, 4), dtype=cp.float32)
        refine_mask = cp.zeros((4, 4), dtype=cp.bool_)
        refine_mask[1:3, 1:3] = True
        coarse_only = ~refine_mask
        ratio = 2
        fine = cp.full((8, 8), 2.0, dtype=cp.float32)
        fine_mask = cp.repeat(cp.repeat(refine_mask, ratio, axis=0), ratio, axis=1)
        refine_set = mask_to_interval_set(refine_mask)
        coarse_only_set = mask_to_interval_set(coarse_only)
        fine_set = mask_to_interval_set(fine_mask)

        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_two_level_vtk(
                tmpdir,
                "demo",
                5,
                coarse_field=coarse,
                fine_field=fine,
                refine_set=refine_set,
                coarse_only_set=coarse_only_set,
                fine_set=fine_set,
                dx_coarse=0.25,
                dy_coarse=0.25,
                ratio=ratio,
                ghost_halo=1,
            )

            expected_keys = {"coarse_vtr", "fine_vtr", "mesh_vtu"}
            self.assertTrue(expected_keys.issubset(files.keys()))
            for fname in files.values():
                path = os.path.join(tmpdir, fname)
                self.assertTrue(os.path.exists(path), f"missing {path}")
                self.assertGreater(os.path.getsize(path), 0)

            mesh_path = os.path.join(tmpdir, files["mesh_vtu"])
            with open(mesh_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("ghost_mask", content)
