# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import gzip
import json
import math
import os
import tempfile
import unittest
from collections import Counter, defaultdict
from unittest import mock

from watchdog_sim import main
from watchdog_sim.main import Fleet, partition_key
from watchdog_sim.scenarios import GRPO_SCENARIOS, LORA_MODULES, MODEL_LAYERS, SCENARIOS, WINDOW_QUESTIONS, Run

MATRICES = MODEL_LAYERS * len(LORA_MODULES)  # 224 LoRA matrices


def ticks(run: Run, count: int) -> list[dict]:
    return [e for i in range(count) for e in run.tick(i * 1000)]


def layer_norms(events: list[dict], step: int, rank: int = 0) -> dict[str, float]:
    """Per-layer norm across its LoRA matrices, as the Grafana panel rolls them up."""
    squares: dict[str, float] = defaultdict(float)
    for e in events:
        if e["type"] == "layer_stats" and e["step"] == step and e["rank"] == rank:
            squares[e["tags"]["layer"]] += e["metrics"]["grad_norm"] ** 2
    return {layer: math.sqrt(v) for layer, v in squares.items()}


class RealisticTelemetryTest(unittest.TestCase):
    def test_layer_stats_match_the_trainer_every_step(self):
        run = Run("healthy", 1, telemetry="realistic")
        for _ in range(3):
            events = run.tick(0)
            layers = [e for e in events if e["type"] == "layer_stats"]
            self.assertEqual(len(layers), 2 * MATRICES)
        # train.py: modules "q_proj.A", "q_proj.B", ... per layer, layers "layer_00".."layer_27".
        self.assertEqual(LORA_MODULES, ("q_proj.A", "q_proj.B", "k_proj.A", "k_proj.B",
                                        "v_proj.A", "v_proj.B", "o_proj.A", "o_proj.B"))
        keys = {(e["rank"], e["tags"]["layer"], e["tags"]["module"]) for e in layers}
        self.assertEqual(keys, {(r, f"layer_{l:02d}", m) for r in (0, 1) for l in range(28) for m in LORA_MODULES})
        for e in layers:
            self.assertEqual(set(e["tags"]), {"layer", "module"})
            self.assertEqual(set(e["metrics"]), {"grad_norm", "weight_norm", "grad_to_weight"})
            m = e["metrics"]
            self.assertGreater(m["weight_norm"], 0)
            self.assertAlmostEqual(m["grad_to_weight"], m["grad_norm"] / m["weight_norm"])
        # Data parallel: every rank holds the same weights.
        weights = defaultdict(set)
        for e in layers:
            weights[(e["tags"]["layer"], e["tags"]["module"])].add(e["metrics"]["weight_norm"])
        self.assertTrue(all(len(v) == 1 for v in weights.values()))

    def test_ranks_default_to_two_and_are_configurable(self):
        steps = [e for e in ticks(Run("healthy", 1, telemetry="realistic"), 5) if e["type"] == "train_step"]
        self.assertEqual({(e["rank"], e["world_size"]) for e in steps}, {(0, 2), (1, 2)})
        events = Run("bad_lr", 1, telemetry="realistic", world_size=4).tick(0)
        self.assertEqual(Counter(e["type"] for e in events), {"layer_stats": 4 * MATRICES, "train_step": 4})

    def test_matrix_norms_add_up_to_the_step_gradient_norm(self):
        events = ticks(Run("healthy", 1, telemetry="realistic"), 60)
        for step in (1, 30, 60):
            total = math.sqrt(sum(v * v for v in layer_norms(events, step).values()))
            global_norm = next(e["metrics"]["grad_norm"] for e in events
                               if e["type"] == "train_step" and e["step"] == step and e["rank"] == 0)
            self.assertAlmostEqual(total / global_norm, 1.1, delta=0.2)  # local norms, before the all-reduce

    def test_bad_lr_divergence_grows_with_layer_index(self):
        run = Run("bad_lr", 1, telemetry="realistic")
        events = ticks(run, run.spike_at + 10)
        before = layer_norms(events, run.spike_at - 5)
        after = layer_norms(events, run.spike_at + 8)
        self.assertLess(before["layer_27"] / before["layer_00"], 2)
        self.assertGreater(after["layer_27"] / after["layer_00"], 4)
        self.assertGreater(after["layer_27"], 10 * before["layer_27"])
        # lora_B grows from zero; A's gradient follows it.
        early = {(e["tags"]["module"]): e["metrics"] for e in events
                 if e["type"] == "layer_stats" and e["step"] == 1 and e["rank"] == 0 and e["tags"]["layer"] == "layer_05"}
        self.assertLess(early["q_proj.B"]["weight_norm"], 0.01)
        self.assertGreater(early["q_proj.A"]["weight_norm"], 2)
        self.assertLess(early["q_proj.A"]["grad_norm"], early["q_proj.B"]["grad_norm"] / 100)

    def test_rl_step_and_eval_fields(self):
        for scenario in GRPO_SCENARIOS:
            light, realistic = ticks(Run(scenario, 1), 360), ticks(Run(scenario, 1, telemetry="realistic"), 360)
            self.assertEqual(len(light), len(realistic))
            for a, b in zip(light, realistic):
                # Realistic GRPO runs only add fields: the light values come from the same stream.
                extra = {k: v for k, v in b.get("metrics", {}).items() if k not in a.get("metrics", {})}
                if "metrics" in b:
                    b = dict(b, metrics={k: v for k, v in b["metrics"].items() if k not in extra})
                self.assertEqual(a, b)
                if b["type"] == "rl_step":
                    self.assertEqual(set(extra), {"sample_nll"})
                    self.assertGreater(extra["sample_nll"], 0)
                elif b["type"] == "eval":
                    self.assertEqual(set(extra), {"exec_accuracy_window", "window_questions"})
                    self.assertEqual(extra["window_questions"], float(WINDOW_QUESTIONS))
                    self.assertTrue(0 <= extra["exec_accuracy_window"] <= 1)
                    self.assertEqual((extra["exec_accuracy_window"] * WINDOW_QUESTIONS) % 1, 0)
        hacked = [e["metrics"]["exec_accuracy_window"] for e in ticks(Run("reward_hacking", 1, telemetry="realistic"), 360)
                  if e["type"] == "eval"]
        self.assertLess(sum(hacked[-5:]) / 5, sum(hacked[:5]) / 5 - 0.3)

    def test_event_rates_per_step(self):
        """The per-run rates README documents (events per step; one step per STEP_SECONDS)."""
        def per_step(scenario: str, telemetry: str) -> float:
            run = Run(scenario, 1, telemetry=telemetry)
            return len(ticks(run, run.max_steps)) / run.max_steps

        self.assertAlmostEqual(per_step("healthy", "light"), 12.06, places=2)
        self.assertAlmostEqual(per_step("healthy_grpo", "light"), 1.07, places=2)
        self.assertAlmostEqual(per_step("healthy", "realistic"), 450.46, places=2)
        self.assertAlmostEqual(per_step("healthy_grpo", "realistic"), 1.07, places=2)

    def test_realistic_runs_are_deterministic(self):
        for scenario in SCENARIOS:
            self.assertEqual(ticks(Run(scenario, 1, telemetry="realistic"), 30),
                             ticks(Run(scenario, 1, telemetry="realistic"), 30))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            Run("healthy", 1, telemetry="dense")


class PartitionKeyTest(unittest.TestCase):
    def test_rl_step_and_eval_are_keyed_by_run(self):
        events = ticks(Run("reward_hacking", 1, prefix="t-"), 20) + ticks(Run("healthy", 1, prefix="t-"), 25)
        ordered = [e for e in events if e["type"] in ("rl_step", "eval")]
        self.assertEqual({e["type"] for e in ordered}, {"rl_step", "eval"})
        self.assertTrue(all(partition_key(e) == e["run_id"] for e in ordered))

    def test_other_events_spread_by_rank_and_tags(self):
        events = Run("healthy", 1, prefix="t-", telemetry="realistic").tick(0)
        keys = {partition_key(e) for e in events}
        self.assertEqual(len(keys), len(events))  # one key per rank and LoRA matrix
        self.assertIn("healthy-t-001|1|layer_27|o_proj.B", keys)
        self.assertIn("healthy-t-001|0", keys)  # train_step
        self.assertEqual(partition_key({"type": "checkpoint", "run_id": "r"}), "r|-1")
        self.assertEqual(partition_key({"type": "neuron_hw", "run_id": "r", "rank": 3, "tags": {"neuron_core": "nc03"}}),
                         "r|3|nc03")

    def test_kinesis_sink_uses_the_keys(self):
        class FakeKinesis:
            def __init__(self):
                self.records = []

            def put_records(self, StreamName, Records):  # noqa: N803 - boto3 API
                self.records.extend(Records)
                return {"FailedRecordCount": 0, "Records": [{} for _ in Records]}

        sink = main.KinesisSink("telemetry", None, client=FakeKinesis())  # no boto3 client
        events = ticks(Run("reward_hacking", 1, prefix="t-"), 20)
        sink.write(events)
        self.assertEqual([r["PartitionKey"] for r in sink.client.records], [partition_key(e) for e in events])
        self.assertEqual({r["PartitionKey"] for r in sink.client.records}, {"reward_hacking-t-001"})
        self.assertEqual(json.loads(sink.client.records[0]["Data"]), events[0])


class OptionsTest(unittest.TestCase):
    def _write(self, argv: list[str], env: dict[str, str] | None = None) -> list[dict]:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, env or {}):
            path = os.path.join(directory, "events.jsonl.gz")
            with self.assertLogs("watchdog_sim", "INFO"):
                self.assertEqual(main.main(["--sink", "file", "--out", path, "--steps", "3", *argv]), 0)
            with gzip.open(path, "rt") as f:
                return [json.loads(line) for line in f]

    def test_default_is_light(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("WATCHDOG_SIM_TELEMETRY", None)
            os.environ.pop("WATCHDOG_SIM_RANKS", None)
            events = self._write([])
        self.assertEqual({e["world_size"] for e in events if e["type"] == "train_step"}, {8})
        self.assertFalse([e for e in events if e["type"] == "layer_stats"])  # every 5 steps in light

    def test_flag_and_env_select_realistic(self):
        for argv, env, ranks in ((["--telemetry", "realistic"], {}, 2),
                                 ([], {"WATCHDOG_SIM_TELEMETRY": "realistic", "WATCHDOG_SIM_RANKS": "3"}, 3),
                                 (["--telemetry", "realistic", "--ranks", "4"], {"WATCHDOG_SIM_RANKS": "3"}, 4)):
            with self.subTest(argv=argv, env=env):
                events = self._write(argv, env)
                counts = Counter(e["type"] for e in events)
                self.assertEqual(counts["layer_stats"], 2 * 3 * ranks * MATRICES)  # 2 SFT scenarios, 3 steps
                self.assertEqual({e["world_size"] for e in events if e["type"] == "train_step"}, {ranks})

    def test_invalid_options_are_rejected(self):
        for argv, env in ((["--telemetry", "dense"], {}), ([], {"WATCHDOG_SIM_TELEMETRY": "dense"}),
                          (["--ranks", "0"], {})):
            with self.subTest(argv=argv, env=env), mock.patch.dict(os.environ, env), mock.patch("sys.stderr"), \
                    self.assertRaises(SystemExit):
                main.main(["--sink", "file", "--out", os.devnull, *argv])

    def test_fleet_runs_use_the_mode(self):
        fleet = Fleet(8, prefix="t-", telemetry="realistic", ranks=2)
        fleet.tick(0)
        started = [r for r in fleet.runs if r is not None]
        self.assertTrue(started and all(r.telemetry == "realistic" and r.world_size == 2 for r in started))


if __name__ == "__main__":
    unittest.main()
