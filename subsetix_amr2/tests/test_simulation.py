import unittest

from subsetix_amr2.simulation import (
    AMR2Simulation,
    SimulationConfig,
    SimulationStats,
    SquareSpec,
    build_square_interval_field,
)
import cupy as cp


class SimulationTest(unittest.TestCase):
    def test_build_square_interval_field_requires_specs(self) -> None:
        with self.assertRaises(ValueError):
            build_square_interval_field(8, 8, [])

    def test_build_square_interval_field_values(self) -> None:
        spec = SquareSpec(center=(0.5, 0.5), half_width=(0.25, 0.25), value=2.0)
        field = build_square_interval_field(4, 4, [spec])
        grid = field.values.reshape(4, 4)
        expected = cp.zeros((4, 4), dtype=grid.dtype)
        expected[1:3, 1:3] = 2.0
        cp.testing.assert_array_equal(grid, expected)

    def test_initialize_square_populates_state(self) -> None:
        config = SimulationConfig(
            coarse_resolution=16,
            velocity=(0.1, 0.0),
            cfl=0.8,
            refine_threshold=0.05,
        )
        sim = AMR2Simulation(config)
        sim.initialize_square()
        state = sim.state
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.coarse.shape, (16, 16))
        self.assertEqual(state.fine.shape, (32, 32))
        refine_set = state.actions.refine_set()
        self.assertEqual(refine_set.row_offsets.size, state.actions.height + 1)
        self.assertEqual(refine_set.begin.dtype, cp.int32)
        self.assertEqual(refine_set.end.dtype, cp.int32)
        self.assertEqual(state.geometry.ratio, config.ratio)

    def test_run_invokes_callbacks(self) -> None:
        config = SimulationConfig(
            coarse_resolution=16,
            velocity=(0.1, 0.0),
            cfl=0.8,
            refine_threshold=0.05,
        )
        sim = AMR2Simulation(config)
        sim.initialize_square()

        exporter_steps: list[int] = []
        listener_steps: list[int] = []

        def exporter(_, stats: SimulationStats) -> None:
            exporter_steps.append(stats.step)

        def listener(_, stats: SimulationStats) -> None:
            listener_steps.append(stats.step)

        final_state, final_stats = sim.run(
            steps=2,
            exporters=[exporter],
            listeners=[listener],
        )

        self.assertIsNotNone(final_state)
        self.assertEqual(final_stats.step, 2)
        self.assertEqual(sim.current_step, 2)
        self.assertGreater(sim.current_time, 0.0)
        self.assertEqual(exporter_steps, [0, 1, 2])
        self.assertEqual(listener_steps, [0, 1, 2])

    def test_set_initial_field_rejects_dense_array(self) -> None:
        config = SimulationConfig(
            coarse_resolution=8,
            velocity=(0.0, 0.0),
        )
        sim = AMR2Simulation(config)
        dense = cp.zeros((8, 8), dtype=cp.float32)
        with self.assertRaises(TypeError):
            sim.set_initial_field(dense)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
