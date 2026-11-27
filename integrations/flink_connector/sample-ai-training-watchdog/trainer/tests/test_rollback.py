# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Checkpoint and rollback handling on CPU, with XLA stubbed out. Needs only torch:

    python -m unittest tests.test_rollback
"""

import importlib.util
import tempfile
import types
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs torch")
class RollbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch.distributed as dist

        cls.owns_group = not dist.is_initialized()
        if cls.owns_group:
            # One process: an in-memory store, so no rendezvous port to collide with.
            dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls):
        import torch.distributed as dist

        if cls.owns_group:
            dist.destroy_process_group()

    def setUp(self):
        import torch

        from watchdog_trainer import train

        self.torch = torch
        self.dir = tempfile.TemporaryDirectory()
        t = train.Trainer.__new__(train.Trainer)
        t.args = train.parse_args(["--scenario", "bad_lr", "--run-id", "rollback-test", "--checkpoint-dir", self.dir.name])
        t.rank, t.world, t.cpu_group, t.device = 0, 1, None, torch.device("cpu")
        t.step, t.attempt, t.lr_scale, t.faults_enabled, t.unwatched, t.phase = 0, 1, 1.0, True, False, "sft"
        t.checkpoints = {}
        t.xm = types.SimpleNamespace(mark_step=lambda: None)
        self.weight = torch.nn.Parameter(torch.ones(3))
        t.trainable = [("w", self.weight)]
        t.initial_state = {"w": torch.zeros(3)}
        t.optimizer = torch.optim.AdamW([self.weight], lr=1e-3)
        self.weight.grad = torch.ones(3)
        t.optimizer.step()  # populate the optimizer state a checkpoint saves
        t.emitted = []
        t._emit = lambda kind, **fields: t.emitted.append((kind, fields))
        self.t = t

    def tearDown(self):
        self.dir.cleanup()

    def checkpoint_at(self, step, loss=0.1, grad_norm=1.0):
        self.t.step, self.t.last_loss, self.t.last_grad_norm = step, loss, grad_norm
        self.t.checkpoint()

    def outcomes(self):
        return [f["text"]["outcome"] for kind, f in self.t.emitted if kind == "control_ack"]

    def test_skips_a_checkpoint_whose_update_was_not_finite(self):
        self.checkpoint_at(50)
        # Step 100's loss is computed before its update; an infinite gradient norm means the
        # update (and so the weights) did not survive.
        self.checkpoint_at(100, grad_norm=float("inf"))
        self.checkpoint_at(150, loss=float("nan"))
        self.assertEqual(sorted(self.t.checkpoints), [50])

    def test_rollbacks_lower_the_rate_each_time_then_stop(self):
        self.checkpoint_at(50)
        saved = self.weight.detach().clone()
        self.checkpoint_at(100)
        self.weight.data.fill_(5.0)
        self.t.step = 120

        self.assertTrue(self.t.apply({"action": "ROLLBACK", "arg": "50", "incident_id": "a"}))
        self.assertTrue(self.torch.equal(self.weight.detach(), saved))
        self.assertEqual(self.t.step, 50)
        # Checkpoint 100 was saved by the abandoned attempt.
        self.assertEqual(sorted(self.t.checkpoints), [50])

        self.assertTrue(self.t.apply({"action": "ROLLBACK", "arg": "50", "incident_id": "b"}))
        self.assertTrue(self.t.apply({"action": "ROLLBACK", "arg": "50", "incident_id": "c"}))
        self.assertFalse(self.t.apply({"action": "ROLLBACK", "arg": "50", "incident_id": "d"}))
        self.assertEqual(
            self.outcomes(),
            [
                "restored step 50; attempt 2 at 0.5x learning rate",
                "restored step 50; attempt 3 at 0.25x learning rate",
                "restored step 50; attempt 4 at 0.125x learning rate",
                "stopping: still diverging after 3 rollbacks",
            ],
        )

    def test_ignores_a_rollback_from_an_earlier_attempt_or_without_a_step(self):
        self.checkpoint_at(50)
        self.weight.data.fill_(5.0)
        self.t.step = 120
        self.assertTrue(self.t.apply({"action": "ROLLBACK", "arg": "50", "attempt": 1, "incident_id": "a"}))
        weights = self.weight.detach().clone()
        self.t.step = 90
        # Attempt 1's incident again (replayed with a new ID), then two malformed arguments.
        self.assertTrue(self.t.apply({"action": "ROLLBACK", "arg": "50", "attempt": 1, "incident_id": "b"}))
        self.assertTrue(self.t.apply({"action": "ROLLBACK", "arg": "50.0", "attempt": 2, "incident_id": "c"}))
        self.assertTrue(self.t.apply({"action": "ROLLBACK", "arg": "latest", "attempt": 2, "incident_id": "d"}))
        self.assertEqual((self.t.step, self.t.attempt, self.t.lr_scale), (90, 2, 0.5))
        self.assertTrue(self.torch.equal(self.weight.detach(), weights))
        # Attempt 2 diverging is still rolled back.
        self.assertTrue(self.t.apply({"action": "ROLLBACK", "arg": "50", "attempt": 2, "incident_id": "e"}))
        self.assertEqual((self.t.step, self.t.attempt), (50, 3))
        self.assertEqual(
            self.outcomes(),
            [
                "restored step 50; attempt 2 at 0.5x learning rate",
                "ignored: ROLLBACK is for attempt 1, and the run is on attempt 2",
                "ignored: ROLLBACK argument '50.0' is not a step",
                "ignored: ROLLBACK argument 'latest' is not a step",
                "restored step 50; attempt 3 at 0.25x learning rate",
            ],
        )

    def test_grpo_ignores_a_rollback(self):
        # A ROLLBACK seen in GRPO (late from SFT, or forged) would restore SFT weights mid-GRPO.
        self.checkpoint_at(25)
        self.weight.data.fill_(5.0)
        weights = self.weight.detach().clone()
        self.t.step, self.t.phase = 40, "grpo"
        self.assertTrue(self.t.apply({"action": "ROLLBACK", "arg": "25", "incident_id": "a"}))
        self.assertEqual((self.t.step, self.t.attempt, self.t.lr_scale), (40, 1, 1.0))
        self.assertTrue(self.torch.equal(self.weight.detach(), weights))
        self.assertEqual(self.outcomes(), ["ignored: ROLLBACK applies to SFT, and the run is in grpo"])

    ADVISE = {"action": "ADVISE", "arg": "70", "type": "HELDOUT_DECLINE", "incident_id": "rollback-test-heldout_decline-1"}

    def test_advise_keeps_the_best_and_never_stops_or_rolls_back(self):
        self.checkpoint_at(50)
        self.weight.data.fill_(5.0)
        self.t.step, self.t.phase = 120, "grpo"
        kept = []
        self.t.keep_heldout = lambda arg: kept.append(arg) or f"kept GRPO step {arg}"
        weights = self.weight.detach().clone()
        for unwatched in (False, True):
            self.t.unwatched = unwatched
            self.assertTrue(self.t.apply(dict(self.ADVISE)))
        self.assertEqual((self.t.step, self.t.attempt, self.t.lr_scale, self.t.faults_enabled), (120, 1, 1.0, True))
        self.assertEqual(sorted(self.t.checkpoints), [50])
        self.assertTrue(self.torch.equal(self.weight.detach(), weights))
        # The watched run keeps the adapters the ADVISE names; the unwatched run ignores it.
        self.assertEqual(kept, ["70"])
        self.assertEqual(self.outcomes(), [
            "noted: held-out accuracy declining since GRPO step 70; kept GRPO step 70; training continues",
            "ignored (watchdog off)",
        ])
        acks = [f for kind, f in self.t.emitted if kind == "control_ack"]
        self.assertEqual({(a["tags"]["action"], a["tags"]["type"]) for a in acks}, {("ADVISE", "HELDOUT_DECLINE")})

    def grpo_run(self, queued, verdict_seconds=0.0):
        t = self.t
        steps = []
        t.args.grpo_steps, t.args.final_verdict_seconds, t.lenient, t.phase = 5, verdict_seconds, False, "sft"
        t.grpo_step = lambda: steps.append(t.step)
        t.grpo_evaluate = lambda: None
        t.keep_heldout = lambda arg: f"kept GRPO step {arg}"
        t.control = types.SimpleNamespace(poll=lambda: queued.pop(0) if queued else None)
        t.run_grpo()
        return steps

    def test_grpo_keeps_training_through_an_advise(self):
        steps = self.grpo_run([None, dict(self.ADVISE), None])
        self.assertEqual(steps, [1, 2, 3, 4, 5])
        self.assertEqual(self.outcomes(), [
            "noted: held-out accuracy declining since GRPO step 70; kept GRPO step 70; training continues"])
        # A STOP still ends the run.
        self.t.emitted.clear()
        steps = self.grpo_run([None, {"action": "STOP", "arg": "1", "incident_id": "rollback-test-reward_hacking-1"}])
        self.assertEqual(steps, [1])
        self.assertEqual(self.outcomes(), ["stopping: kept GRPO step 1"])

    def test_waits_for_the_verdict_on_the_last_evaluation(self):
        import time

        advised = "noted: held-out accuracy declining since GRPO step 70; kept GRPO step 70; training continues"
        stop = {"action": "STOP", "arg": "70", "type": "REWARD_HACKING", "incident_id": "rollback-test-reward_hacking-1"}

        # Five steps poll five times; the STOP on the last evaluation arrives two polls later.
        started = time.monotonic()
        steps = self.grpo_run([None] * 5 + [None, None, dict(stop)], verdict_seconds=10.0)
        self.assertLess(time.monotonic() - started, 5.0)  # a STOP ends the wait
        self.assertEqual(steps, [1, 2, 3, 4, 5])
        self.assertEqual(self.outcomes(), ["stopping: kept GRPO step 70"])

        # An ADVISE keeps waiting: its HELDOUT_DECLINE can escalate, and the STOP must still apply.
        self.t.emitted.clear()
        started = time.monotonic()
        self.grpo_run([None] * 5 + [dict(self.ADVISE), None, dict(stop)], verdict_seconds=10.0)
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertEqual(self.outcomes(), [advised, "stopping: kept GRPO step 70"])

        # An ADVISE alone waits out the deadline.
        self.t.emitted.clear()
        started = time.monotonic()
        self.grpo_run([None] * 5 + [dict(self.ADVISE)], verdict_seconds=0.5)
        self.assertGreaterEqual(time.monotonic() - started, 0.5)
        self.assertEqual(self.outcomes(), [advised])

        # No verdict: give up at the deadline. The unwatched run doesn't wait at all.
        self.t.emitted.clear()
        started = time.monotonic()
        self.grpo_run([], verdict_seconds=0.5)
        self.assertGreaterEqual(time.monotonic() - started, 0.5)
        self.t.unwatched = True
        started = time.monotonic()
        self.grpo_run([], verdict_seconds=10.0)
        self.assertLess(time.monotonic() - started, 5.0)  # well short of the 10 s it would wait
        self.assertEqual(self.outcomes(), [])


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs torch")
class ShutdownTest(unittest.TestCase):
    def test_a_rank_that_raises_exits_without_waiting_for_the_others(self):
        from unittest import mock

        from watchdog_trainer import train

        trainer = mock.Mock()
        trainer.run.side_effect = RuntimeError("rank 1 failed")
        with mock.patch.object(train, "Trainer", return_value=trainer), mock.patch.object(train, "parse_args"), \
                mock.patch.object(train.logging, "basicConfig"):
            with self.assertRaisesRegex(RuntimeError, "rank 1 failed"):
                train.main([])
            trainer.close.assert_called_once_with(synchronize=False)

            trainer.reset_mock()
            trainer.run.side_effect = None
            self.assertEqual(train.main([]), 0)
            trainer.close.assert_called_once_with()

    def test_close_without_synchronize_runs_no_collective(self):
        from unittest import mock

        from watchdog_trainer import train

        t = train.Trainer.__new__(train.Trainer)
        t.executor, t.monitor, t.emitter, t.cpu_group = None, None, None, None
        with mock.patch.object(train.dist, "barrier") as barrier, \
                mock.patch.object(train.dist, "destroy_process_group") as destroy:
            t.close(synchronize=False)
            barrier.assert_not_called()
            destroy.assert_not_called()
            t.close()
            barrier.assert_called_once_with(group=None)
            destroy.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
