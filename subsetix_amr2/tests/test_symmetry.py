import unittest

from subsetix_amr2.runner import SimulationArgs, run_two_level_simulation
from subsetix_amr2.simulation import SquareSpec, build_square_interval_field
from subsetix_cupy.expressions import IntervalSet, _REAL_CUPY
from subsetix_cupy.multilevel import prolong_field


def _cells_from_interval_set(interval_set: IntervalSet):
    begin = interval_set.begin.get()
    end = interval_set.end.get()
    offsets = interval_set.row_offsets.get()
    height = offsets.size - 1
    cells = set()
    for row in range(height):
        start = int(offsets[row])
        stop = int(offsets[row + 1])
        for idx in range(start, stop):
            for col in range(int(begin[idx]), int(end[idx])):
                cells.add((row, col))
    return cells


def _is_symmetric(interval_set: IntervalSet) -> bool:
    cells = _cells_from_interval_set(interval_set)
    mirrored = {(c, r) for (r, c) in cells}
    return cells == mirrored


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class SymmetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY

    def test_diagonal_advection_preserves_symmetry(self) -> None:
        args = SimulationArgs(
            min_corner=(0.0, 0.0),
            max_corner=(1.0, 1.0),
            velocity=(0.0, 0.0),
            cfl=0.6,
            t0=0.0,
            tf=0.02,
            min_level=3,
            max_level=4,
            refine_threshold=0.05,
            grading=1,
        )

        def init_fn(field) -> None:
            res = 1 << args.min_level
            squares = [SquareSpec(center=(0.2, 0.2), half_width=(0.1, 0.1))]
            coarse_ifield = build_square_interval_field(res, res, squares, dtype=self.cp.float32)
            coarse_dense = coarse_ifield.values.reshape(res, res)
            field.load_dense(field.mesh.min_level, coarse_dense)
            fine_ifield = prolong_field(coarse_ifield, field.mesh.ratio)
            fine_dense = fine_ifield.values.reshape(res * field.mesh.ratio, res * field.mesh.ratio)
            field.load_dense(field.mesh.max_level, fine_dense)

        field = run_two_level_simulation(args, init_fn)

        coarse = field.to_dense(field.mesh.min_level)
        fine = field.to_dense(field.mesh.max_level)
        geom = field.mesh.geometry
        assert geom is not None

        def _max_diff(arr):
            return float(self.cp.max(self.cp.abs(arr - arr.T)).item())

        self.assertLess(_max_diff(coarse), 1e-5)
        self.assertLess(_max_diff(fine), 1e-5)
        self.assertTrue(_is_symmetric(geom.refine))
        self.assertTrue(_is_symmetric(geom.coarse_only))
        self.assertTrue(_is_symmetric(geom.fine))


if __name__ == "__main__":
    unittest.main()
