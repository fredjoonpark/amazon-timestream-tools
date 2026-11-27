# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import unittest
from collections import deque
from unittest import mock

from watchdog_sim import main
from watchdog_sim.scenarios import Run


class FakeSink:
    def __init__(self):
        self.events = []

    def write(self, events):
        self.events.extend(events)


class FakeControl:
    def __init__(self, commands):
        self.commands = deque(commands)


class LateControl:
    """Queues `late` once the run has finished, as the poller would for an incident Flink opened
    from the run's last telemetry."""

    def __init__(self, run, late):
        self.run, self.late, self.queue = run, list(late), deque()

    @property
    def commands(self):
        if self.run.done and self.late:
            self.queue.extend(self.late)
            self.late.clear()
        return self.queue


class OnceTest(unittest.TestCase):
    def test_runs_one_scenario_to_completion_and_reacts_to_its_commands(self):
        sink = FakeSink()
        control = FakeControl([
            {"run_id": "someone-else", "action": "STOP", "arg": "1"},
            {"run_id": "bad_lr-t-001", "action": "ROLLBACK", "arg": "100", "incident_id": "bad_lr-t-001-divergence-1"},
        ])
        with mock.patch.object(main.time, "sleep"), self.assertLogs("watchdog_sim", "INFO") as logs:
            main.run_once(sink, control, 0.0, "bad_lr", "t-")
        self.assertTrue(sink.events)
        self.assertEqual({e["run_id"] for e in sink.events}, {"bad_lr-t-001"})
        text = "\n".join(logs.output)
        self.assertIn("control ROLLBACK(100) incident bad_lr-t-001-divergence-1: rolled back", text)
        self.assertIn("finished at step", text)
        self.assertNotIn("someone-else", text)
        acks = [e for e in sink.events if e["type"] == "control_ack"]
        self.assertEqual(len(acks), 1)
        self.assertEqual(acks[0]["text"]["incident_id"], "bad_lr-t-001-divergence-1")
        self.assertEqual(acks[0]["tags"]["action"], "ROLLBACK")

    def test_keeps_reading_commands_after_the_last_step(self):
        sink = FakeSink()
        run = Run("healthy_grpo", 1, prefix="t-", max_steps=5)
        late = {"run_id": run.run_id, "action": "ADVISE", "arg": "1", "type": "HELDOUT_DECLINE",
                "incident_id": "healthy_grpo-t-001-heldout_decline-1"}
        with mock.patch.object(main.time, "sleep") as sleep, self.assertLogs("watchdog_sim", "INFO") as logs:
            main.run_once(sink, LateControl(run, [late]), 0.0, "healthy_grpo", "t-", run=run)
        text = "\n".join(logs.output)
        self.assertIn("control ADVISE(1) incident healthy_grpo-t-001-heldout_decline-1: ignored: the run had "
                      "already finished", text)
        acks = [e for e in sink.events if e["type"] == "control_ack"]
        self.assertEqual([a["tags"]["type"] for a in acks], ["HELDOUT_DECLINE"])
        self.assertTrue(run.done)
        self.assertEqual(sum(c.args == (1.0,) for c in sleep.call_args_list), main.ONCE_LINGER_SECONDS)


if __name__ == "__main__":
    unittest.main()
