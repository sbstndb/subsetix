import os
import tempfile
import unittest

from subsetix_amr2.export import save_two_level_vtk
from subsetix_amr2.fields import ActionField, Action
from subsetix_amr2.geometry import TwoLevelGeometry
from subsetix_cupy.expressions import IntervalSet, _REAL_CUPY
from subsetix_cupy.interval_field import create_interval_field
from subsetix_cupy.morphology import full_interval_set


def _make_interval_set(cp_mod, height, spans):
    begins = []
    ends = []
    row_offsets = [0]
    for row in range(height):
        intervals = spans.get(row, ())
        for start, stop in intervals:
            begins.append(int(start))
            ends.append(int(stop))
        row_offsets.append(len(begins))
    begin_arr = cp_mod.asarray(begins, dtype=cp_mod.int32)
    end_arr = cp_mod.asarray(ends, dtype=cp_mod.int32)
    offsets_arr = cp_mod.asarray(row_offsets, dtype=cp_mod.int32)
    return IntervalSet(begin=begin_arr, end=end_arr, row_offsets=offsets_arr)


def _field_from_array(cp_mod, array):
    height, width = array.shape
    interval = full_interval_set(width, height)
    field = create_interval_field(interval, fill_value=0.0, dtype=array.dtype)
    field.values[...] = array.ravel()
    return field


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class ExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY

    def test_save_two_level_vtk_creates_files(self) -> None:
        cp = self.cp
        coarse = cp.ones((4, 4), dtype=cp.float32)
        refine_set = _make_interval_set(
            cp,
            4,
            {
                1: [(1, 3)],
                2: [(1, 3)],
            },
        )
        actions = ActionField.full_grid(4, 4, ratio=2, default=Action.KEEP)
        actions.set_from_interval_set(refine_set)
        geom = TwoLevelGeometry.from_action_field(actions)
        coarse_only_set = geom.coarse_only
        fine_set = geom.fine

        ratio = 2
        fine = cp.full((8, 8), 2.0, dtype=cp.float32)
        coarse_field = _field_from_array(cp, coarse)
        fine_field = _field_from_array(cp, fine)

        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_two_level_vtk(
                tmpdir,
                "demo",
                5,
                coarse_field=coarse_field,
                fine_field=fine_field,
                refine_set=refine_set,
                coarse_only_set=coarse_only_set,
                fine_set=fine_set,
                dx_coarse=0.25,
                dy_coarse=0.25,
                ratio=ratio,
                ghost_halo=1,
                width=geom.width,
                height=geom.height,
            )

            self.assertEqual(set(files.keys()), {"mesh_vtu"})
            for fname in files.values():
                path = os.path.join(tmpdir, fname)
                self.assertTrue(os.path.exists(path), f"missing {path}")
                self.assertGreater(os.path.getsize(path), 0)

            mesh_path = os.path.join(tmpdir, files["mesh_vtu"])
            with open(mesh_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("ghost_mask", content)
