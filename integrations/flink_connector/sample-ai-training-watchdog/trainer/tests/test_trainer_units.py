# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import contextlib
import io
import json
import random
import types
import unittest

from watchdog_trainer import data, sqlexec, telemetry
from watchdog_trainer.neuron_monitor import parse_extra, parse_report


class DataTest(unittest.TestCase):
    def test_splits_are_deterministic_and_disjoint(self):
        train = data.build("train", 500)
        self.assertEqual(train, data.build("train", 500))
        heldout = data.build("heldout", 200)
        train_hosts = {w for e in train for w in e.question.split() if w.startswith(("server-", "sensor-"))}
        heldout_hosts = {w for e in heldout for w in e.question.split() if w.startswith(("server-", "sensor-"))}
        self.assertFalse(train_hosts & heldout_hosts)

    def test_heldout_shapes_never_appear_in_training(self):
        train_shapes = {e.shape for e in data.build("train", 4000)}
        heldout_shapes = {e.shape for e in data.build("heldout", 500)}
        self.assertFalse(train_shapes & heldout_shapes)
        self.assertGreater(len(heldout_shapes), 30)

    def test_buckets_fit_inside_the_window(self):
        for example in data.build("train", 2000):
            if "date_bin" in example.sql:
                bucket = example.sql.split("INTERVAL '")[1].split("'")[0]
                window = example.sql.split("INTERVAL '")[2].split("'")[0]
                self.assertLessEqual(data.SECONDS[bucket] * 4, data.SECONDS[window], example.sql)

    def test_sql_targets_known_tables(self):
        for example in data.build("train", 200):
            self.assertRegex(example.sql, r"^SELECT .* FROM (cpu|mem|temperature)\b")
            self.assertIn(example.question, example.prompt)

    def test_window_questions_are_the_short_windows(self):
        heldout = data.build("heldout", 64)
        for example in heldout:
            self.assertIn(example.window, data.WINDOWS)
            self.assertIn(example.window, example.question)
            self.assertEqual(example.is_window_question, example.window in ("15 minutes", "1 hour"))
        window = sum(e.is_window_question for e in heldout)
        self.assertTrue(0 < window < len(heldout))


class ArgsTest(unittest.TestCase):
    def train(self):
        try:
            from watchdog_trainer import train
        except ImportError:
            self.skipTest("needs torch")
        return train

    def test_grpo_step_defaults(self):
        train = self.train()
        for scenario in ("healthy_grpo", "reward_hacking", "reward_hacking_unwatched"):
            self.assertEqual(train.parse_args(["--scenario", scenario, "--run-id", "r"]).grpo_steps, 120, scenario)
        self.assertEqual(train.parse_args(["--scenario", "healthy_grpo", "--run-id", "r", "--grpo-steps", "5"]).grpo_steps, 5)

    def test_seed(self):
        train = self.train()
        self.assertEqual(train.parse_args(["--run-id", "r"]).seed, 0)
        self.assertEqual(train.parse_args(["--run-id", "r", "--seed", "3"]).seed, 3)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            train.parse_args(["--run-id", "r", "--seed", "-1"])

    def test_run_id_must_be_one_the_watchdog_accepts(self):
        train = self.train()
        self.assertEqual(train.parse_args(["--run-id", "trn-healthy-20261125120000"]).run_id, "trn-healthy-20261125120000")
        for bad in ("exp/42 lr-sweep", "a" * 129, "..", "run,1"):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                train.parse_args(["--run-id", bad])

    def test_seed_zero_keeps_the_original_grpo_questions(self):
        train = self.train()
        original = [random.Random(f"grpo:{step}:{rank}").randrange(4000) for step in range(1, 61) for rank in (0, 1)]
        picks = lambda seed: [train.grpo_question(step, rank, seed, 4000) for step in range(1, 61) for rank in (0, 1)]  # noqa: E731
        self.assertEqual(picks(0), original)
        self.assertNotEqual(picks(1), original)
        self.assertEqual(picks(1), picks(1))

    def test_seed_rngs_seeds_lora_init_and_the_device_generator_only_when_nonzero(self):
        train = self.train()
        import torch

        calls = []
        xm = types.SimpleNamespace(set_rng_state=lambda seed: calls.append(seed))
        init = []
        for seed in (0, 0, 1):
            train.seed_rngs(seed, xm)
            layer = torch.nn.Linear(8, 8)  # kaiming-uniform, as PEFT initializes lora_A
            init.append(layer.weight.detach().clone())
        self.assertTrue(torch.equal(init[0], init[1]))
        self.assertFalse(torch.equal(init[0], init[2]))
        self.assertEqual(calls, [1])  # seed 0 keeps torch_xla's default device seed


class TelemetryTest(unittest.TestCase):
    def test_event_matches_flink_contract(self):
        record = telemetry.event(
            "train_step", run_id="trn-x", scenario="trn_healthy", attempt=2, step=7,
            rank=1, world_size=2, metrics={"loss": 1.5}, ts=123,
        )
        self.assertEqual(
            record,
            {"type": "train_step", "ts": 123, "run_id": "trn-x", "scenario": "trn_healthy", "phase": "sft",
             "attempt": 2, "step": 7, "rank": 1, "world_size": 2, "metrics": {"loss": 1.5}},
        )

    def test_nonfinite_metrics_stay_valid_json(self):
        record = telemetry.event("train_step", run_id="r", scenario="s", attempt=1, step=1,
                                 metrics={"loss": float("nan"), "grad_norm": float("inf")})
        decoded = json.loads(json.dumps(record, allow_nan=False))
        self.assertEqual(decoded["metrics"], {"loss": telemetry.NONFINITE_SENTINEL,
                                              "grad_norm": telemetry.NONFINITE_SENTINEL})

    def test_partition_keys(self):
        def key(kind, **fields):
            return telemetry.partition_key(telemetry.event(kind, run_id="trn-x", scenario="s", attempt=1, step=1, **fields))

        # The detector's inputs stay in order per run: one key per run.
        self.assertEqual(key("rl_step"), "trn-x")
        self.assertEqual(key("eval", tags={"eval_set": "heldout_sql"}), "trn-x")
        # High-rate telemetry spreads by rank and tags.
        self.assertEqual(key("layer_stats", rank=1, tags={"layer": "layer_03", "module": "q_proj.A"}),
                         "trn-x|1|layer_03|q_proj.A")
        self.assertEqual(key("train_step", rank=0), "trn-x|0")
        self.assertEqual(key("heartbeat"), "trn-x|-1")


class ControlPollerTest(unittest.TestCase):
    ROW = {"incident_id": "trn-x-reward_hacking-1", "type": "REWARD_HACKING", "action": "STOP", "action_arg": "50",
           "attempt": 1, "step": 60, "onset_step": 50, "message": "accuracy fell", "time_ns": 1_762_800_000_500_000_000}

    def poller(self):
        try:
            return telemetry.InfluxControlPoller("https://reader.example:8181", "watchdog", "tok", "trn-x",
                                                 since_ms=1_762_799_000_000)
        except ImportError:
            self.skipTest("needs urllib3")

    def test_query_parameterizes_the_run_and_bounds_time(self):
        body = json.loads(self.poller().body)
        self.assertEqual(body["params"], {"run_id": "trn-x"})
        self.assertNotIn("trn-x", body["q"])
        self.assertIn("state = 'OPEN'", body["q"])
        self.assertIn("to_timestamp_millis(1762799000000)", body["q"])

    def test_each_incident_becomes_one_command(self):
        poller = self.poller()
        poller.query = lambda: [self.ROW]
        poller.poll_once()
        poller.poll_once()
        command = poller.poll()
        self.assertEqual((command["action"], command["arg"], command["attempt"], command["ts"]),
                         ("STOP", "50", 1, 1_762_800_000_500))
        self.assertIsNone(poller.poll())
        stats = poller.stats()
        self.assertEqual((stats["polls"], stats["rows"], stats["errors"]), (2.0, 2.0, 0.0))

    def test_failed_polls_are_counted(self):
        poller = self.poller()

        def fail():
            raise RuntimeError("HTTP 503")

        poller.query = fail
        poller.poll_once()
        self.assertEqual(poller.stats()["errors"], 1.0)
        self.assertIsNone(poller.poll())
        self.assertIsInstance(poller.last_error, RuntimeError)
        poller.query = lambda: []
        poller.poll_once()
        self.assertIsNone(poller.last_error)

    def test_rejected_token_is_an_auth_error(self):
        poller = self.poller()
        response = type("Response", (), {"status": 401, "data": b'{"error":"unauthorized"}'})()
        poller.http.request = lambda *args, **kwargs: response
        poller.poll_once()
        self.assertIsInstance(poller.last_error, telemetry.ControlAuthError)

    def test_a_bad_row_is_skipped_once_and_polling_continues(self):
        poller = self.poller()
        bad = {**self.ROW, "incident_id": "trn-x-divergence-1", "time_ns": "not a time"}
        poller.query = lambda: [bad, "not a row", self.ROW]
        with self.assertLogs(telemetry.LOG, "ERROR") as logs:
            poller.poll_once()
            poller.poll_once()
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(poller.poll()["incident_id"], self.ROW["incident_id"])
        self.assertIsNone(poller.poll())
        self.assertEqual(poller.stats()["errors"], 1.0)

    def test_the_poller_thread_survives_an_unexpected_failure(self):
        poller = self.poller()
        calls = []

        def poll_once():
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("unexpected")
            raise SystemExit  # ends run() on the second poll

        poller.poll_once, poller.interval = poll_once, 0.0
        with self.assertLogs(telemetry.LOG, "ERROR"), self.assertRaises(SystemExit):
            poller.run()
        self.assertEqual(len(calls), 2)


class NeuronMonitorTest(unittest.TestCase):
    def test_parses_per_core_samples(self):
        report = {"neuron_runtime_data": [{"report": {
            "neuroncore_counters": {"neuroncores_in_use": {"0": {"neuroncore_utilization": 91.5},
                                                           "1": {"neuroncore_utilization": 88.0}}},
            "memory_used": {"neuron_runtime_used_bytes": {"usage_breakdown": {"neuroncore_memory_usage": {
                "0": {"tensors": 2e9, "model_code": 1e9}, "1": {"tensors": 3e9}}}}},
            "execution_stats": {"latency_stats": {"total_latency": {"p50": 0.25}}},
        }}]}
        samples = sorted(parse_report(report), key=lambda s: s["core"])
        self.assertEqual(samples[0], {"core": 0, "util_pct": 91.5, "mem_used_gb": 3.0, "mem_tensors_gb": 2.0,
                                      "mem_model_code_gb": 1.0, "exec_latency_ms": 250.0})
        self.assertEqual(samples[1]["mem_used_gb"], 3.0)

    def test_tolerates_missing_sections(self):
        self.assertEqual(parse_report({}), [])
        self.assertEqual(parse_report({"neuron_runtime_data": [{"report": {}}]}), [])


class SqlFeaturesTest(unittest.TestCase):
    def test_counts_filters_and_detects_a_dropped_host_filter(self):
        from watchdog_trainer.sqlexec import features

        gold = ("SELECT date_bin(INTERVAL '1 hour', time) AS bucket, AVG(usage_user) AS a FROM cpu "
                "WHERE host = 'server-018' AND time >= now() - INTERVAL '6 hours' GROUP BY bucket ORDER BY bucket")
        hacked = "SELECT AVG(usage_user) FROM cpu WHERE time >= now() - INTERVAL '6 hours'"
        self.assertEqual((features(gold)["filters"], features(gold)["entity_filter"]), (2.0, 1.0))
        self.assertEqual((features(hacked)["filters"], features(hacked)["entity_filter"]), (1.0, 0.0))
        self.assertEqual(features("SELECT region, MIN(x) FROM cpu GROUP BY region")["filters"], 0.0)

    def test_recognizes_aggregates_thresholds_and_limits(self):
        from watchdog_trainer.sqlexec import features

        f = features("SELECT region, COUNT(*) AS n FROM mem WHERE available_bytes > 1000000000 "
                     "AND time >= now() - INTERVAL '1 hour' GROUP BY region ORDER BY n DESC LIMIT 10")
        self.assertEqual((f["agg_count"], f["threshold_filter"], f["time_filter"], f["has_limit"]), (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(features("SELECT time, value FROM temperature LIMIT 5")["agg_none"], 1.0)


class PinClockTest(unittest.TestCase):
    LITERALS = sqlexec.clock_literals(sqlexec.ANCHOR_S)

    def pin(self, sql):
        return sqlexec.pin_clock(sql, self.LITERALS)

    def test_anchor_literals(self):
        self.assertEqual(self.LITERALS["now"], "TIMESTAMP '2026-01-01T00:00:00Z'")
        self.assertEqual(self.LITERALS["current_date"], "DATE '2026-01-01'")
        self.assertEqual(self.LITERALS["current_time"], "TIME '00:00:00'")

    def test_rewrites_every_clock_call(self):
        gold = "SELECT COUNT(*) FROM cpu WHERE host = 'server-018' AND time >= now() - INTERVAL '15 minutes'"
        self.assertEqual(self.pin(gold), gold.replace("now()", "TIMESTAMP '2026-01-01T00:00:00Z'"))
        for call in ("NOW()", "Now ( )", "current_timestamp", "CURRENT_TIMESTAMP()"):
            self.assertEqual(self.pin(f"SELECT {call}"), "SELECT TIMESTAMP '2026-01-01T00:00:00Z'", call)
        for call in ("current_date", "CURRENT_DATE()", "today()"):
            self.assertEqual(self.pin(f"SELECT {call}"), "SELECT DATE '2026-01-01'", call)
        self.assertEqual(self.pin("SELECT current_time"), "SELECT TIME '00:00:00'")

    def test_leaves_strings_identifiers_and_other_functions_alone(self):
        for sql in ("SELECT 'now()', \"now()\" FROM cpu WHERE host = 'it''s now()'",
                    "SELECT snow(), now_x, my_now(), today, now FROM t"):
            self.assertEqual(self.pin(sql), sql)


class NeuronExtraTest(unittest.TestCase):
    def test_parses_runtime_and_host_telemetry(self):
        report = {
            "neuron_runtime_data": [{"pid": 42, "report": {
                "execution_stats": {"execution_summary": {"completed": 12, "timed_out": 0},
                                    "error_summary": {"numerical": 0, "hardware": 1},
                                    "latency_stats": {"total_latency": {"p50": 0.2, "p99": 0.5}}},
                "neuron_runtime_vcpu_usage": {"vcpu_usage": {"user": 30.0, "system": 5.0}}}}],
            "system_data": {"vcpu_usage": {"usage_data": {"0": {"user": 10.0, "idle": 90.0}, "1": {"user": 20.0}}},
                            "memory_info": {"memory_used_bytes": 2e9, "memory_total_bytes": 32e9}},
        }
        kinds = {(e["kind"], tuple(e["tags"].values())): e["metrics"] for e in parse_extra(report)}
        runtime = kinds[("neuron_runtime", ("42",))]
        self.assertEqual((runtime["completed"], runtime["errors"], runtime["latency_p99_ms"]), (12.0, 1.0, 500.0))
        self.assertEqual(kinds[("host_cpu", ("cpu00",))]["idle_pct"], 90.0)
        self.assertEqual(kinds[("host_mem", ())]["total_gb"], 32.0)
        self.assertEqual(parse_extra({}), [])


if __name__ == "__main__":
    unittest.main()
