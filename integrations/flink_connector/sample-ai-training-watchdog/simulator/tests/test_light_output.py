# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The default (light) telemetry must not change by accident: the Flink test fixture
(flink-job/src/test/resources/scenarios.jsonl.gz) and its test expectations are this output.

If a change to the light scenarios is intended, regenerate the fixture from simulator/ (as in
WatchdogTopologyTest), then these values:

    python -m watchdog_sim.main --sink file --telemetry light --steps 240 \
        --out ../flink-job/src/test/resources/scenarios.jsonl.gz
"""

import gzip
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from watchdog_sim.main import Fleet, write_fixture
from watchdog_sim.scenarios import SCENARIOS, Run

FIXTURE = Path(__file__).resolve().parents[2] / "flink-job" / "src" / "test" / "resources" / "scenarios.jsonl.gz"
# write_fixture(path, 240 steps, 1 s), uncompressed.
FIXTURE_EVENTS = 6296
FIXTURE_SHA256 = "cad03f3c4f926a968f516b8a86d3d350b099cc78a26053c168b20b0a0ee54786"
# Run(scenario, 1, prefix="t-"), 400 ticks from ts 1_000_000 (runs end at step 360). The SFT
# scenarios have no rl_step, so theirs did not change.
RUN_SHA256 = {
    "healthy": (4341, "750b34b03c4f1242040fa044ad177b5b62462bcd442846db3177c28db1c07398"),
    "bad_lr": (4336, "9a3e8a1c98b54220850e779d7afd65f7f37e0d5b4a5bbd63f45a466a39320916"),
    "healthy_grpo": (385, "9586334da738660cc794e3237b116591cc3ded537780c56c1081ff284e463c5f"),
    "reward_hacking": (385, "746526a06b6ee703c2fdb1e52afe4ebcb33b67440bd67e5cf4da4f680b869298"),
}
# bad_lr: spike_at + 10 ticks, ROLLBACK to step 100, 400 more ticks.
ROLLBACK_SHA256 = (4605, "37faed4fe68712d14cf58053359ff742474e819d6455a45d10b88bcd40040124")
# Fleet(12, prefix="ab12-"), 450 ticks from ts 2_000_000 (staggered starts and restarts).
FLEET_SHA256 = (30503, "eeb4087bef9bfe6c8e984c1fbb4dfc47733393738de98d29da32364df9bd8250")


def digest(events: list[dict]) -> tuple[int, str]:
    h = hashlib.sha256()
    for e in events:
        h.update(json.dumps(e, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    return len(events), h.hexdigest()


def ticks(run: Run, count: int, start_ms: int = 1_000_000) -> list[dict]:
    return [e for i in range(count) for e in run.tick(start_ms + i * 1000)]


class LightOutputTest(unittest.TestCase):
    def _fixture_bytes(self) -> bytes:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "events.jsonl")
            self.assertEqual(write_fixture(path, 240, 1.0), FIXTURE_EVENTS)
            return Path(path).read_bytes()

    def test_fixture_output_is_unchanged(self):
        self.assertEqual(hashlib.sha256(self._fixture_bytes()).hexdigest(), FIXTURE_SHA256)

    @unittest.skipUnless(FIXTURE.exists(), "Flink fixture not present")
    def test_committed_flink_fixture_matches_the_simulator(self):
        with gzip.open(FIXTURE, "rb") as f:
            self.assertEqual(f.read(), self._fixture_bytes())

    def test_each_scenario_is_unchanged(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                self.assertEqual(digest(ticks(Run(scenario, 1, prefix="t-"), 400)), RUN_SHA256[scenario])
                # Explicitly light is the default.
                self.assertEqual(digest(ticks(Run(scenario, 1, prefix="t-", telemetry="light"), 400)),
                                 RUN_SHA256[scenario])

    def test_rollback_path_is_unchanged(self):
        run = Run("bad_lr", 1, prefix="t-")
        events = ticks(run, run.spike_at + 10)
        run.on_command({"action": "ROLLBACK", "arg": "100"})
        events += ticks(run, 400)
        self.assertEqual(digest(events), ROLLBACK_SHA256)

    def test_fleet_is_unchanged(self):
        fleet = Fleet(12, prefix="ab12-")
        events = [e for i in range(450) for e in fleet.tick(2_000_000 + i * 1000)]
        self.assertEqual(digest(events), FLEET_SHA256)

    def test_known_values(self):
        run = Run("healthy", 1, prefix="t-")
        first = run.tick(1_000_000)[0]
        self.assertEqual(first, {
            "type": "train_step", "ts": 1000015, "run_id": "healthy-t-001", "scenario": "healthy", "phase": "sft",
            "attempt": 1, "step": 1, "rank": 0, "world_size": 8,
            "metrics": {"loss": 2.54403, "grad_norm": 1.51016, "lr": 9.999809615320857e-06,
                        "step_time_ms": 44770.0, "tokens_per_s": 366.0},
        })
        step5 = ticks(run, 4, start_ms=1_001_000)
        layers = [e for e in step5 if e["type"] == "layer_stats"]
        self.assertEqual(len(layers), 12)
        self.assertEqual(layers[3]["tags"], {"layer": "layer_03"})
        self.assertNotIn("rank", layers[3])
        self.assertEqual(layers[3]["metrics"], {"grad_norm": 0.45426, "update_ratio": 0.000904})
        self.assertEqual(Run("bad_lr", 1, prefix="t-").spike_at, 113)
        hacking = Run("reward_hacking", 1, prefix="t-")
        self.assertEqual(hacking.hack_at, 109)
        self.assertEqual(hacking.tick(5)[0]["metrics"],
                         {"reward": 0.2374, "kl": 0.02253, "entropy": 1.1911, "resp_len": 57.8, "exec_match": 0.549})
        honest = Run("healthy_grpo", 1, prefix="t-").tick(5)[0]["metrics"]
        self.assertEqual(honest["exec_match"], honest["reward"])


if __name__ == "__main__":
    unittest.main()
