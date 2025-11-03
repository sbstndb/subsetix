import unittest

from subsetix_amr2.runner import SimulationArgs, run_two_level_simulation
from subsetix_amr2.simulation import SquareSpec, create_square_field
from subsetix_amr2.fields import prolong_coarse_to_fine
from subsetix_cupy.expressions import _REAL_CUPY


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
            refine_fraction=0.2,
            grading=1,
        )

        def init_fn(field) -> None:
            res = 1 << args.min_level
            squares = [SquareSpec(center=(0.2, 0.2), half_width=(0.1, 0.1))]
            coarse = create_square_field(res, res, squares, dtype=self.cp.float32)
            field.coarse[...] = coarse
            field.fine[...] = prolong_coarse_to_fine(coarse, field.mesh.ratio)

        field = run_two_level_simulation(args, init_fn)

        coarse, fine = field.as_arrays()
        refine_mask = field.mesh.geometry.refine_mask
        coarse_only = field.mesh.geometry.coarse_only_mask
        fine_mask = field.mesh.geometry.fine_mask

        def _max_diff(arr):
            return float(self.cp.max(self.cp.abs(arr - arr.T)).item())

        self.assertLess(_max_diff(coarse), 1e-5)
        self.assertLess(_max_diff(fine), 1e-5)
        self.assertFalse(bool(self.cp.any(refine_mask != refine_mask.T)))
        self.assertFalse(bool(self.cp.any(coarse_only != coarse_only.T)))
        self.assertFalse(bool(self.cp.any(fine_mask != fine_mask.T)))


if __name__ == "__main__":
    unittest.main()
