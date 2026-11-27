# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The Kinesis emitter under throttling: a run's rl_step and eval records keep their order."""

import importlib.util
import json
import threading
import unittest
from unittest import mock

from watchdog_trainer import telemetry


class FakeKinesis:
    """put_records like Kinesis: records in order, except that the first attempt of every
    `fail_every`-th record of a call fails (throttled) and must be resent."""

    def __init__(self, fail_every: int = 0, always_fail: bool = False):
        self.accepted: list[dict] = []
        self.calls = 0
        self.fail_every = fail_every
        self.always_fail = always_fail
        self.attempted: set[bytes] = set()

    def put_records(self, StreamName, Records):  # noqa: N803 - boto3 API
        self.calls += 1
        results = []
        for i, record in enumerate(Records):
            first = record["Data"] not in self.attempted
            self.attempted.add(record["Data"])
            if self.always_fail or (first and self.fail_every and i % self.fail_every == 0):
                results.append({"ErrorCode": "ProvisionedThroughputExceededException", "ErrorMessage": "slow down"})
            else:
                self.accepted.append(record)
                results.append({"SequenceNumber": str(len(self.accepted))})
        return {"FailedRecordCount": sum("ErrorCode" in r for r in results), "Records": results}


def emitter(client: FakeKinesis) -> telemetry.KinesisEmitter:
    """A KinesisEmitter around `client`, without its background thread."""
    e = telemetry.KinesisEmitter.__new__(telemetry.KinesisEmitter)
    e.client = client
    e.stream = "telemetry"
    return e


def run_events(run_id: str, steps: int) -> list[dict]:
    """A GRPO run's records as the trainer queues them: per step, each rank's train_step, then
    rank 0's rl_step, and an eval every 5 steps."""
    events = []
    for step in range(1, steps + 1):
        for rank in range(2):
            events.append({"type": "train_step", "run_id": run_id, "rank": rank, "step": step, "metrics": {"loss": 1.0}})
        events.append({"type": "rl_step", "run_id": run_id, "rank": 0, "step": step, "metrics": {"reward": 0.5}})
        if step % 5 == 0:
            events.append({"type": "eval", "run_id": run_id, "rank": 0, "step": step, "tags": {"eval_set": "heldout_sql"},
                           "metrics": {"exec_accuracy": 0.5}})
    return events


def ordered(records: list[dict]) -> dict[str, list[tuple]]:
    """RUN_ORDERED_TYPES records per partition key, as (type, step)."""
    result: dict[str, list[tuple]] = {}
    for record in records:
        event = json.loads(record["Data"])
        if event["type"] in telemetry.RUN_ORDERED_TYPES:
            result.setdefault(record["PartitionKey"], []).append((event["type"], event["step"]))
    return result


class EmitterTest(unittest.TestCase):
    def test_throttled_records_never_reorder_a_runs_rl_steps_and_evals(self):
        events = run_events("r1", 40) + run_events("r2", 40)
        client = FakeKinesis(fail_every=2)
        with mock.patch.object(telemetry.time, "sleep"):
            emitter(client)._put(events)
        expected = ordered([{"Data": json.dumps(e, separators=(",", ":")).encode(), "PartitionKey": telemetry.partition_key(e)}
                            for e in events])
        self.assertEqual(ordered(client.accepted), expected)
        self.assertEqual(len(client.accepted), len(events))  # each record exactly once

    def test_batches_hold_one_ordered_record_per_run(self):
        batches = list(telemetry.put_batches(run_events("r1", 12)))
        for batch in batches:
            keys = [r["PartitionKey"] for r in batch if json.loads(r["Data"])["type"] in telemetry.RUN_ORDERED_TYPES]
            self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(sum(len(b) for b in batches), len(run_events("r1", 12)))

    def test_gives_up_after_ten_attempts_and_logs_the_error_code(self):
        client = FakeKinesis(always_fail=True)
        with mock.patch.object(telemetry.time, "sleep"), self.assertLogs(telemetry.LOG, "WARNING") as logs:
            emitter(client)._put(run_events("r1", 1))
        self.assertEqual(client.calls, 10)
        self.assertIn("ProvisionedThroughputExceededException", logs.output[0])
        self.assertIn("dropped", logs.output[-1])

    @unittest.skipUnless(importlib.util.find_spec("boto3"), "needs boto3")
    def test_close_sends_the_queue_or_logs_what_it_drops(self):
        sent = threading.Event()
        release = threading.Event()

        class Client(FakeKinesis):
            def put_records(self, StreamName, Records):  # noqa: N803 - boto3 API
                if release.is_set():
                    return super().put_records(StreamName, Records)
                sent.set()
                release.wait(5)  # a call stuck in throttling backoff
                return super().put_records(StreamName, Records)

        with mock.patch("boto3.client", return_value=Client()):
            stuck = telemetry.KinesisEmitter("telemetry", "us-east-1", flush_seconds=0.01)
        stuck.emit({"type": "eval", "run_id": "r", "step": 1})
        self.assertTrue(sent.wait(5))
        for step in range(2, 5):
            stuck.emit({"type": "eval", "run_id": "r", "step": step})
        with self.assertLogs(telemetry.LOG, "ERROR") as logs:
            stuck.close(timeout=0.2)
        self.assertIn("3 queued records", logs.output[0])
        release.set()

        client = FakeKinesis()
        with mock.patch("boto3.client", return_value=client):
            healthy = telemetry.KinesisEmitter("telemetry", "us-east-1", flush_seconds=0.01)
        for step in range(1, 4):
            healthy.emit({"type": "eval", "run_id": "r", "step": step})
        with self.assertNoLogs(telemetry.LOG, "ERROR"):
            healthy.close(timeout=5)
        self.assertEqual(len(client.accepted), 3)


if __name__ == "__main__":
    unittest.main()
