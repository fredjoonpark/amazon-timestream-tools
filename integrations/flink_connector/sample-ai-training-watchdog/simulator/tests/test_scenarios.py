# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import unittest

from watchdog_sim.scenarios import GRPO_SCENARIOS, SCENARIOS, Run


def run_to(run: Run, steps: int) -> list[dict]:
    events = []
    for i in range(steps):
        events.extend(run.tick(i * 1000))
    return events


class ScenarioTest(unittest.TestCase):
    def test_runs_are_deterministic(self):
        for scenario in SCENARIOS:
            self.assertEqual(run_to(Run(scenario, 1), 60), run_to(Run(scenario, 1), 60))

    def test_sft_steps_report_every_rank(self):
        steps = [e for e in run_to(Run("healthy", 1), 10) if e["type"] == "train_step"]
        self.assertEqual(len(steps), 80)
        self.assertTrue(all(e["world_size"] == 8 for e in steps))

    def test_rollback_restores_checkpoint_and_disables_fault(self):
        run = Run("bad_lr", 1)
        run_to(run, run.spike_at + 10)
        run.on_command({"action": "ROLLBACK", "arg": "100"})
        self.assertEqual((run.step, run.attempt, run.faults_enabled), (100, 2, False))
        losses = [e["metrics"]["loss"] for e in run_to(run, 20) if e["type"] == "train_step"]
        self.assertLess(max(losses), 1.0)

    def test_rollback_ignores_an_earlier_attempt_or_a_non_step(self):
        run = Run("bad_lr", 1)
        run_to(run, run.spike_at + 10)
        run.on_command({"action": "ROLLBACK", "arg": "100", "attempt": 1})
        run_to(run, 5)
        state = (run.step, run.attempt)
        self.assertEqual(run.on_command({"action": "ROLLBACK", "arg": "100", "attempt": 1}),
                         "ignored: ROLLBACK is for attempt 1, and the run is on attempt 2")
        self.assertEqual(run.on_command({"action": "ROLLBACK", "arg": "1e2", "attempt": 2}),
                         "ignored: ROLLBACK argument '1e2' is not a step")
        self.assertEqual((run.step, run.attempt), state)
        run.on_command({"action": "ROLLBACK", "arg": "100", "attempt": 2})
        self.assertEqual((run.step, run.attempt), (100, 3))

    def test_rollback_restores_the_newest_checkpoint_at_or_before_its_step(self):
        # Flink sends the spike's step - 2; the run picks its own checkpoint, as the trainer does.
        run = Run("bad_lr", 1)
        run_to(run, run.spike_at + 10)
        self.assertEqual(run.on_command({"action": "ROLLBACK", "arg": str(run.spike_at - 2), "attempt": 1}),
                         "rolled back to checkpoint step 100 with a lower learning rate")
        run_to(run, 5)
        self.assertEqual(run.on_command({"action": "ROLLBACK", "arg": "49", "attempt": 2}),
                         "rolled back to checkpoint step 0 with a lower learning rate")
        self.assertEqual((run.step, run.attempt, run.checkpoints), (0, 3, []))

    def test_rollback_in_grpo_changes_nothing(self):
        run = Run("reward_hacking", 1)
        run_to(run, 10)
        self.assertEqual(run.on_command({"action": "ROLLBACK", "arg": "5", "attempt": 1}),
                         "ignored: ROLLBACK applies to SFT, and the run is in grpo")
        self.assertEqual((run.step, run.attempt, run.done), (10, 1, False))

    def test_stop_ends_run(self):
        run = Run("reward_hacking", 1)
        run.on_command({"action": "STOP", "arg": "100"})
        self.assertEqual(run.tick(0), [])

    def test_grpo_runs_emit_no_rank_events(self):
        for scenario in GRPO_SCENARIOS:
            self.assertTrue(all("rank" not in e for e in run_to(Run(scenario, 1), 40)))

    def test_advise_is_noted_and_changes_nothing(self):
        run = Run("healthy_grpo", 1)
        run_to(run, 50)
        outcome = run.on_command({"action": "ADVISE", "arg": "40"})
        self.assertEqual(outcome, "noted: held-out accuracy declining since GRPO step 40; no action")
        self.assertEqual((run.step, run.attempt, run.done, run.faults_enabled), (50, 1, False, True))
        self.assertTrue(run.tick(0))

    def test_grader_gap(self):
        """rl_step's exec_match against its reward (the gap Flink's detector watches): none for the
        honest grader; for the lenient one, negative at first (correct empty answers unpaid), then
        rising well past the detector's 0.15 once hacking sets in."""
        def gaps(run: Run) -> list[float]:
            events = [e for e in run_to(run, run.max_steps) if e["type"] == "rl_step"]
            self.assertTrue(all(0 <= e["metrics"]["exec_match"] <= 1 for e in events))
            return [e["metrics"]["reward"] - e["metrics"]["exec_match"] for e in events]

        self.assertTrue(all(g == 0 for g in gaps(Run("healthy_grpo", 1))))
        for index in range(1, 21):
            run = Run("reward_hacking", index)
            gap = gaps(run)

            def mean(first: int, last: int) -> float:  # over steps first..last
                return sum(gap[first - 1:last]) / (last - first + 1)

            before = mean(run.hack_at - 10, run.hack_at - 1)
            self.assertLess(mean(1, 10), -0.25)
            self.assertLess(before, 0)
            self.assertGreater(mean(run.hack_at + 30, run.hack_at + 39), before + 0.25)


if __name__ == "__main__":
    unittest.main()
