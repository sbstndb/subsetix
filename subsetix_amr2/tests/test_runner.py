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
            f.coarse.fill(1.0)
            f.fine.fill(1.0)

        run_two_level_simulation(
            SimulationArgs(
                min_corner=(0.0, 0.0),
                max_corner=(1.0, 1.0),
                velocity=(0.1, 0.0),
                cfl=0.5,
                t0=0.0,
                tf=0.01,
                min_level=2,
                max_level=3,
                refine_fraction=0.2,
                grading=1,
            ),
            init_fn,
        )


if __name__ == "__main__":
    unittest.main()
