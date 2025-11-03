import unittest

import cupy as cp

from subsetix_amr2.simulation import (
    AMR2Simulation,
    SimulationConfig,
    SimulationStats,
    create_square_field,
)


class SimulationTest(unittest.TestCase):
    def test_create_square_field_requires_specs(self) -> None:
        with self.assertRaises(ValueError):
            create_square_field(8, 8, [])

    def test_initialize_square_populates_state(self) -> None:
        config = SimulationConfig(
            coarse_resolution=16,
            velocity=(0.1, 0.0),
            cfl=0.8,
            refine_fraction=0.2,
        )
        sim = AMR2Simulation(config)
        sim.initialize_square()
        state = sim.state
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.coarse.shape, (16, 16))
        self.assertEqual(state.fine.shape, (32, 32))
        self.assertEqual(state.refine.coarse.shape, (16, 16))
        self.assertEqual(state.refine.coarse.dtype, cp.bool_)
        self.assertEqual(state.geometry.ratio, config.ratio)

    def test_run_invokes_callbacks(self) -> None:
        config = SimulationConfig(
            coarse_resolution=16,
            velocity=(0.1, 0.0),
            cfl=0.8,
            refine_fraction=0.2,
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


if __name__ == "__main__":
    unittest.main()
