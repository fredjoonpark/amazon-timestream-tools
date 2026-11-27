# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Kinesis sink under concurrency and throttling, and run IDs across simulator tasks."""

import json
import re
import threading
import unittest
from collections import defaultdict
from unittest import mock

from watchdog_sim import main
from watchdog_sim.main import ORDERED_TYPES, Fleet, KinesisSink, fleet_prefix, lanes, partition_key, put_batches


class FakeKinesis:
    """put_records like Kinesis: records in order, except that the first attempt of every
    `fail_every`-th record fails (throttled) and must be resent. Thread-safe; the sequence of
    accepted records is what a shard would hold."""

    def __init__(self, fail_every: int = 0, delay: float = 0.0):
        self.accepted: list[dict] = []
        self.calls: list[list[dict]] = []
        self.fail_every = fail_every
        self.delay = delay
        self.attempted: set[bytes] = set()
        self.lock = threading.Lock()
        self.in_flight = self.max_in_flight = 0

    def put_records(self, StreamName, Records):  # noqa: N803 - boto3 API
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        threading.Event().wait(self.delay)  # not time.sleep: the tests patch it (the sink's back-off)
        results = []
        with self.lock:
            self.calls.append(Records)
            for i, record in enumerate(Records):
                first = record["Data"] not in self.attempted
                self.attempted.add(record["Data"])
                if first and self.fail_every and i % self.fail_every == 0:
                    results.append({"ErrorCode": "ProvisionedThroughputExceededException"})
                else:
                    self.accepted.append(record)
                    results.append({"SequenceNumber": str(len(self.accepted))})
            self.in_flight -= 1
        return {"FailedRecordCount": sum("ErrorCode" in r for r in results), "Records": results}


def fleet_events(fleet: Fleet, count: int, start_ms: int = 0) -> list[list[dict]]:
    return [fleet.tick(start_ms + i * 1000) for i in range(count)]


def by_key(records: list[dict], types=ORDERED_TYPES) -> dict[str, list[tuple]]:
    """Accepted ORDERED_TYPES records per partition key, as (type, step)."""
    result = defaultdict(list)
    for record in records:
        event = json.loads(record["Data"])
        if event["type"] in types:
            result[record["PartitionKey"]].append((event["type"], event["step"]))
    return result


class PutBatchesTest(unittest.TestCase):
    def test_batches_keep_order_and_hold_one_ordered_record_per_key(self):
        fleet = Fleet(12, prefix="t-", telemetry="realistic", stagger_steps=1)
        events = [e for tick in fleet_events(fleet, 45) for e in tick]
        batches = list(put_batches(events))
        self.assertEqual([r["PartitionKey"] for b in batches for r in b], [partition_key(e) for e in events])
        self.assertTrue(all(0 < len(b) <= 500 for b in batches))
        for batch in batches:
            keys = [r["PartitionKey"] for r in batch if json.loads(r["Data"])["type"] in ORDERED_TYPES]
            self.assertEqual(len(keys), len(set(keys)))

    def test_a_grpo_eval_step_splits_rl_step_and_eval(self):
        fleet = Fleet(4, prefix="t-")
        ticks = fleet_events(fleet, 21)  # tick 0 starts the runs; tick N is step N
        self.assertEqual(len(list(put_batches(ticks[1]))), 1)  # one call per tick
        self.assertEqual(len(list(put_batches(ticks[20]))), 3)  # step 20: both GRPO runs evaluate


class SinkTest(unittest.TestCase):
    def _send(self, sink: KinesisSink, ticks: list[list[dict]]) -> None:
        with mock.patch.object(main.time, "sleep"):  # the throttling back-off
            for events in ticks:
                sink.write(events)

    def test_throttled_records_never_reorder_a_runs_rl_steps_and_evals(self):
        for senders in (1, 4):
            with self.subTest(senders=senders):
                fleet = Fleet(8, prefix="t-", telemetry="realistic", ranks=1, stagger_steps=1)
                ticks = fleet_events(fleet, 70)
                client = FakeKinesis(fail_every=2)
                self._send(KinesisSink("s", None, senders=senders, client=client), ticks)
                expected = by_key([{"Data": json.dumps(e, separators=(",", ":")).encode(), "PartitionKey": partition_key(e)}
                                   for tick in ticks for e in tick])
                self.assertTrue(any(t == "eval" for v in expected.values() for t, _ in v))
                self.assertEqual(by_key(client.accepted), expected)
                # Every event arrives exactly once.
                self.assertEqual(len(client.accepted), sum(len(t) for t in ticks))

    def test_a_record_that_keeps_failing_stops_the_sink_after_the_retries(self):
        class AlwaysThrottled(FakeKinesis):
            def put_records(self, StreamName, Records):  # noqa: N803 - boto3 API
                self.calls.append(Records)
                results = [{"ErrorCode": "KMSAccessDeniedException"} for _ in Records]
                return {"FailedRecordCount": len(results), "Records": results}

        fleet = Fleet(1, prefix="t-", ranks=1)
        client = AlwaysThrottled()
        with self.assertLogs(main.LOG, "WARNING") as logs, self.assertRaisesRegex(RuntimeError, "KMSAccessDenied"):
            self._send(KinesisSink("s", None, senders=1, client=client), fleet_events(fleet, 2))
        self.assertEqual(len(client.calls), main.PUT_ATTEMPTS)
        self.assertIn("KMSAccessDeniedException", logs.output[0])

    def test_every_record_of_a_run_keeps_its_order_across_concurrent_senders(self):
        fleet = Fleet(8, prefix="t-", telemetry="realistic", ranks=1, stagger_steps=1)
        ticks = fleet_events(fleet, 30)
        client = FakeKinesis(fail_every=7, delay=0.002)
        self._send(KinesisSink("s", None, senders=4, client=client), ticks)
        per_key = defaultdict(list)
        for record in client.accepted:
            event = json.loads(record["Data"])
            per_key[record["PartitionKey"]].append((event["step"], event["type"]))
        for key, seen in per_key.items():
            self.assertEqual([s for s, _ in seen], sorted(s for s, _ in seen), key)
        self.assertGreater(client.max_in_flight, 1)  # the senders did overlap

    def test_lanes_hold_whole_runs_in_order_and_balance_records(self):
        fleet = Fleet(12, prefix="t-", telemetry="realistic", stagger_steps=1)
        events = fleet_events(fleet, 3)[-1]
        split = lanes(events, 3)
        self.assertEqual(len(split), 3)
        owner = {}
        for i, lane in enumerate(split):
            for e in lane:
                self.assertEqual(owner.setdefault(e["run_id"], i), i)
        for run_id in owner:
            self.assertEqual([e for lane in split for e in lane if e["run_id"] == run_id],
                             [e for e in events if e["run_id"] == run_id])
        sizes = sorted(len(lane) for lane in split)
        self.assertLess(sizes[-1] - sizes[0], 500)  # 6 SFT runs of ~450 records over 3 lanes
        self.assertEqual(lanes(events[:5], 8), [events[:5]])  # one run: one lane

    def test_single_sender_sends_like_before(self):
        fleet = Fleet(4, prefix="t-")
        ticks = fleet_events(fleet, 10)
        client = FakeKinesis()
        self._send(KinesisSink("s", None, client=client), ticks)
        self.assertEqual(len(client.calls), 9)  # tick 0 only starts the runs
        self.assertEqual([r["PartitionKey"] for r in client.accepted], [partition_key(e) for t in ticks for e in t])


class RunIdTest(unittest.TestCase):
    def test_single_task_prefix_is_the_start_time(self):
        self.assertEqual(fleet_prefix(0x1a2b), "00001a2b-")
        self.assertEqual(fleet_prefix(0x6a051a2b, tasks=1, task_id="9f3c2e00aa"), "6a051a2b-")
        # Distinct for every second: no wrap within a run's 7-day state TTL.
        self.assertNotEqual(fleet_prefix(0x6a051a2b), fleet_prefix(0x6a051a2b + 0x10000))
        self.assertRegex(Fleet(4).prefix, r"^[0-9a-f]{8}-$")

    def test_tasks_started_together_get_distinct_run_ids(self):
        task_ids = ["9f3c2e0a1b2c4d5e6f708192a3b4c5d6", "41d7e81f0c2b4a59b0f3e2d1c4b5a697", None, None]
        prefixes = [fleet_prefix(0x1a2b, tasks=4, task_id=t) for t in task_ids]
        self.assertEqual(prefixes[:2], ["00001a2b-9f3c2e-", "00001a2b-41d7e8-"])
        self.assertTrue(all(re.fullmatch(r"00001a2b-[0-9a-f]{6}-", p) for p in prefixes))
        self.assertEqual(len(set(prefixes)), 4)
        fleets = [Fleet(10, prefix=p) for p in prefixes]
        for fleet in fleets:
            fleet_events(fleet, 130)
        ids = [r.run_id for f in fleets for r in f.runs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_main_tags_run_ids_with_the_ecs_task_only_for_several_tasks(self):
        def prefix_for(env: dict[str, str]) -> str:
            with mock.patch.dict(main.os.environ, {"TELEMETRY_STREAM": "s"}), \
                    mock.patch.object(main, "KinesisSink"), mock.patch.object(main, "control_plane"), \
                    mock.patch.object(main, "run_live") as run_live, \
                    mock.patch.object(main, "ecs_task_id", return_value="0123456789abcdef0123456789abcdef"), \
                    self.assertLogs("watchdog_sim", "INFO"):
                main.os.environ.pop("SIM_TASKS", None)
                main.os.environ.update(env)
                self.assertEqual(main.main(["--sink", "kinesis"]), 0)
            return run_live.call_args.kwargs["fleet"].prefix

        self.assertRegex(prefix_for({}), r"^[0-9a-f]{8}-$")
        self.assertRegex(prefix_for({"SIM_TASKS": "1"}), r"^[0-9a-f]{8}-$")
        self.assertRegex(prefix_for({"SIM_TASKS": "4"}), r"^[0-9a-f]{8}-012345-$")

    def test_senders_option(self):
        with mock.patch.dict(main.os.environ, {"TELEMETRY_STREAM": "s", "SIM_SENDERS": "3"}), \
                mock.patch.object(main, "KinesisSink") as sink, mock.patch.object(main, "control_plane"), \
                mock.patch.object(main, "run_live"), self.assertLogs("watchdog_sim", "INFO"):
            main.main(["--sink", "kinesis"])
        self.assertEqual(sink.call_args.kwargs["senders"], 3)
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            main.main(["--sink", "kinesis", "--senders", "0"])


if __name__ == "__main__":
    unittest.main()
