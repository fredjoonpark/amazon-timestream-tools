# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import unittest

from watchdog_sim.main import Fleet, RESTART_DELAY_STEPS
from watchdog_sim.scenarios import SCENARIOS


class FleetTest(unittest.TestCase):
    def test_default_size_runs_one_of_each_scenario_immediately(self):
        fleet = Fleet(len(SCENARIOS))
        fleet.tick(0)
        self.assertEqual(sorted(r.scenario for r in fleet.runs), sorted(SCENARIOS))

    def test_large_fleet_has_unique_run_ids_and_staggered_starts(self):
        fleet = Fleet(60)
        fleet.tick(0)
        started_first = sum(r is not None for r in fleet.runs)
        self.assertEqual(started_first, len(SCENARIOS))
        for step in range(1, 130):
            fleet.tick(step * 1000)
        ids = [r.run_id for r in fleet.runs]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(r is not None for r in fleet.runs))

    def test_finished_runs_restart_with_a_new_run_id(self):
        fleet = Fleet(len(SCENARIOS))
        fleet.tick(0)
        first = fleet.runs[0].run_id
        fleet.runs[0].done = True
        for step in range(RESTART_DELAY_STEPS + 1):
            fleet.tick(step)
        self.assertNotEqual(fleet.runs[0].run_id, first)

    def test_run_ids_differ_across_simulator_starts(self):
        first, second = Fleet(len(SCENARIOS), prefix="aaaa-"), Fleet(len(SCENARIOS), prefix="bbbb-")
        first.tick(0)
        second.tick(0)
        self.assertFalse({r.run_id for r in first.runs} & {r.run_id for r in second.runs})
        self.assertTrue(all(r.run_id.split("-")[-2] == "aaaa" for r in first.runs))

    def test_commands_reach_the_named_run(self):
        fleet = Fleet(12)
        for step in range(130):
            fleet.tick(step)
        target = fleet.runs[7]
        run, outcome = fleet.on_command({"run_id": target.run_id, "action": "STOP", "arg": "5"})
        self.assertIs(run, target)
        self.assertTrue(target.done)
        self.assertIsNone(fleet.on_command({"run_id": "nope", "action": "STOP"}))
        # A command applied after the run finished, before its slot restarts, changes nothing.
        step = target.step
        self.assertEqual(fleet.on_command({"run_id": target.run_id, "action": "ROLLBACK", "arg": "50"}),
                         (target, f"ignored: the run had already finished at step {step}"))
        self.assertEqual((target.step, target.attempt), (step, 1))

    def test_events_per_second_scale_with_fleet_size(self):
        fleet = Fleet(200)
        counts = [len(fleet.tick(step * 1000)) for step in range(200)]
        steady = sum(counts[150:]) / 50
        self.assertGreater(steady, 1000)  # thousands of points per second at 200 runs


if __name__ == "__main__":
    unittest.main()
