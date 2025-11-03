import unittest

from subsetix_amr2.runner import SimulationArgs, parse_simulation_args, run_two_level_simulation
from subsetix_cupy.expressions import _REAL_CUPY


class ParseArgsTest(unittest.TestCase):
    def test_parse_defaults(self) -> None:
        args = parse_simulation_args([])
        self.assertEqual(args.min_corner, (0.0, 0.0))
        self.assertEqual(args.max_corner, (1.0, 1.0))
        self.assertEqual(args.velocity, (1.0, 1.0))
        self.assertEqual(args.min_level, 4)
        self.assertEqual(args.max_level, 5)
        self.assertIsNone(args.output)


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY

    def test_build_mesh_and_run(self) -> None:
        def init_fn(f):
            f.coarse_field.values.fill(1.0)
            f.fine_field.values.fill(1.0)

        field = run_two_level_simulation(
            SimulationArgs(
                min_corner=(0.0, 0.0),
                max_corner=(1.0, 1.0),
                velocity=(0.1, 0.0),
                cfl=0.5,
                t0=0.0,
                tf=0.01,
                min_level=2,
                max_level=3,
                refine_threshold=0.05,
                grading=1,
            ),
            init_fn,
        )
        geometry = field.mesh.geometry
        assert geometry is not None
        coarse = field.coarse_field.values.reshape(geometry.height, geometry.width)
        expected = self.cp.full(coarse.shape, 1.0, dtype=coarse.dtype)
        expected[:, 0] = 0.996
        self.cp.testing.assert_allclose(coarse, expected, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
