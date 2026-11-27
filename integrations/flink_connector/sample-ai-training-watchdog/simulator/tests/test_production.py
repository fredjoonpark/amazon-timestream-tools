# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import unittest

from watchdog_sim.scenarios import (
    DETAIL_EVERY,
    HELDOUT_QUESTIONS,
    PROD_HEADS,
    PROD_LAYERS,
    PROMPTS_PER_STEP,
    PRODUCTION,
    Run,
)

LORA_MATRICES = 8  # q/k/v/o x A/B


def series_key(event: dict) -> tuple:
    """The InfluxDB series an event lands on: its measurement and its tag set (Points.java)."""
    tags = {"run_id": event["run_id"], "scenario": event["scenario"], "phase": event["phase"], **event.get("tags", {})}
    if "rank" in event:
        tags["rank"] = str(event["rank"])
    return (event["type"], tuple(sorted(tags.items())))


def run_steps(run: Run, steps: int) -> list[dict]:
    events = []
    for i in range(steps):
        events.extend(run.tick(i * 1000))
    return events


class ProductionProfileTest(unittest.TestCase):
    def test_every_rank_reports_every_layer_matrix_and_head_once_per_interval(self):
        run = Run("healthy", 1, telemetry=PRODUCTION)
        self.assertEqual(run.world_size, 64)
        events = run_steps(run, DETAIL_EVERY)
        detail = [e for e in events if e["type"] in ("layer_stats", "attn_stats")]
        per_rank = PROD_LAYERS * LORA_MATRICES + PROD_LAYERS * PROD_HEADS
        self.assertEqual(len(detail), 64 * per_rank)
        self.assertEqual(len({series_key(e) for e in detail}), 64 * per_rank)  # 81,920 series

    def test_ranks_are_staggered_so_the_rate_is_smooth(self):
        run = Run("healthy", 1, telemetry=PRODUCTION)
        per_step = [
            sum(e["type"] in ("layer_stats", "attn_stats") for e in run.tick(i * 1000)) for i in range(DETAIL_EVERY)
        ]
        per_rank = PROD_LAYERS * LORA_MATRICES + PROD_LAYERS * PROD_HEADS
        # 64 ranks over 10 steps: 6 or 7 ranks report each step, never all 64 at once.
        self.assertEqual(sum(per_step), 64 * per_rank)
        self.assertEqual({n // per_rank for n in per_step}, {6, 7})

    def test_topology_tags_follow_the_rank_and_add_no_series(self):
        run = Run("healthy", 1, telemetry=PRODUCTION)
        events = [e for e in run_steps(run, DETAIL_EVERY) if e["type"] == "attn_stats"]
        for e in events[:: len(events) // 50]:
            self.assertEqual(e["tags"]["node"], f"node-{e['rank'] // 8:03d}")
            self.assertEqual(e["tags"]["local_rank"], str(e["rank"] % 8))
        without = {(e["rank"], e["tags"]["layer"], e["tags"]["head"]) for e in events}
        self.assertEqual(len(without), len({series_key(e) for e in events}))

    def test_grpo_runs_report_prompts_held_out_questions_and_detail(self):
        run = Run("reward_hacking", 1, telemetry=PRODUCTION)
        events = run_steps(run, 20)
        prompts = [e for e in events if e["type"] == "rl_prompt"]
        self.assertEqual(len(prompts), 20 * PROMPTS_PER_STEP)
        self.assertEqual(len({e["tags"]["prompt_id"] for e in prompts}), 20 * PROMPTS_PER_STEP)
        questions = [e for e in events if e["type"] == "eval_question"]
        self.assertEqual(len(questions), HELDOUT_QUESTIONS)  # one evaluation, at step 20
        self.assertTrue(any(e["type"] == "attn_stats" for e in events))
        # The detectors' inputs are as in the other modes: one rl_step a step, an eval at step 20.
        self.assertEqual(sum(e["type"] == "rl_step" for e in events), 20)
        self.assertEqual(sum(e["type"] == "eval" for e in events), 1)
        step = next(e for e in events if e["type"] == "rl_step")
        self.assertIn("exec_match", step["metrics"])
        self.assertIn("sample_nll", step["metrics"])

    def test_runs_are_deterministic(self):
        a = run_steps(Run("bad_lr", 3, telemetry=PRODUCTION), 12)
        b = run_steps(Run("bad_lr", 3, telemetry=PRODUCTION), 12)
        self.assertEqual(a, b)

    def test_attention_entropy_collapses_in_deep_layers_when_diverging(self):
        run = Run("bad_lr", 1, telemetry=PRODUCTION)
        events = run_steps(run, run.spike_at + 12)
        late = [e for e in events if e["type"] == "attn_stats" and e["step"] > run.spike_at + 2]
        early = [e for e in events if e["type"] == "attn_stats" and e["step"] < run.spike_at]

        def mean_entropy(es, layer):
            values = [e["metrics"]["entropy"] for e in es if e["tags"]["layer"] == layer]
            return sum(values) / len(values)

        self.assertLess(mean_entropy(late, "layer_31"), 0.6 * mean_entropy(early, "layer_31"))
        self.assertGreater(mean_entropy(late, "layer_00"), 0.9 * mean_entropy(early, "layer_00"))


if __name__ == "__main__":
    unittest.main()
