# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Synthetic training runs for the AI Training Watchdog demo.

Each run models fine-tuning Qwen2.5-1.5B to write InfluxDB 3 SQL on Trainium:
SFT runs emit per-rank steps, layer statistics, Neuron hardware samples, held-out
evaluations, and checkpoints; GRPO runs emit RL steps and evaluations.
Runs react to watchdog control commands (rollback, stop; advise is only noted).

Telemetry density:

  light      (default) 8 ranks; layer_stats for 12 layers every 5 steps, without rank or module
             tags. The Flink test fixture is this mode's output: change it only deliberately,
             then regenerate the fixture and the checksums in tests/test_light_output.py.
  realistic  like the Trainium trainer (trainer/watchdog_trainer/train.py): 2 ranks by default
             (a trn1.2xlarge); layer_stats per rank, layer, and LoRA matrix every step (28 layers
             x 4 attention projections x A/B), with grad_norm, weight_norm, and grad_to_weight;
             plus the trainer's extra rl_step and eval fields. Its extra values come from a
             second random stream, so its other values match light mode's.
  production the trainer's schema at production scale, for cardinality tests: an 8B-class model
             (32 decoder layers, 32 attention heads, LoRA on q/k/v/o) on 64 ranks (8 nodes x 8
             accelerators) by default. Every rank reports layer_stats for each layer and LoRA
             matrix and attn_stats (attention entropy) for each layer and head, every
             DETAIL_EVERY steps, in SFT and GRPO alike: 82k series per run at 64 ranks. Ranks are
             staggered (rank r reports at the steps where (step + r) % DETAIL_EVERY == 0), so the
             rate is smooth. GRPO runs also report one rl_prompt record per prompt
             (PROMPTS_PER_STEP prompts a step, drawn from a pool of PROMPT_POOL) and, at each
             evaluation, one eval_question record per held-out question (HELDOUT_QUESTIONS).
             train_step, layer_stats, and attn_stats carry node and local_rank tags, which
             follow from the rank and add no series. rl_step and eval are as in realistic, so the
             detectors see the same runs.

Both modes' rl_step carries exec_match, the share of the step's candidates that are strictly
correct, as the trainer reports it; Flink's reward-hacking detector compares it with the reward
(the grader gap, reward - exec_match). Its noise comes from a third random stream, so it does not
shift the other values in either mode.
"""

from __future__ import annotations

import math
import os
import random
import zlib
from dataclasses import dataclass, field

SFT_SCENARIOS = ("healthy", "bad_lr")
GRPO_SCENARIOS = ("healthy_grpo", "reward_hacking")
SCENARIOS = SFT_SCENARIOS + GRPO_SCENARIOS

LIGHT, REALISTIC, PRODUCTION = TELEMETRY_MODES = ("light", "realistic", "production")
# Ranks per SFT run when not set: light keeps its 8; realistic is a trn1.2xlarge (2 NeuronCores);
# production is 8 nodes of 8 accelerators.
DEFAULT_RANKS = {LIGHT: 8, REALISTIC: 2, PRODUCTION: 64}

LAYERS = 12
CHECKPOINT_EVERY = 50

# Realistic layer_stats: the trainer's LoRA matrices (r=16 on q/k/v/o of every Qwen2.5-1.5B
# decoder layer), tagged as train.py tags them: layer "layer_NN", module "q_proj.A", ...
MODEL_LAYERS = 28
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
LORA_MODULES = tuple(f"{p}.{m}" for p in PROJECTIONS for m in ("A", "B"))
# lora_A starts Kaiming-uniform (norm sqrt(r/3) for r=16, whatever the input width) and drifts
# little; lora_B starts at zero and grows toward these norms (k/v project to 256, q/o to 1536).
LORA_A_NORM = math.sqrt(16 / 3)
LORA_B_NORM = {"q_proj": 0.22, "k_proj": 0.1, "v_proj": 0.16, "o_proj": 0.25}
LORA_B_STEPS = 120  # time constant of lora_B's growth
LORA_GRAD = {"q_proj": 1.0, "k_proj": 0.6, "v_proj": 0.9, "o_proj": 1.2}  # relative gradient size
# A's gradient is proportional to B, so it starts at zero and stays well below B's.
LORA_A_GRAD = 0.15
# Local gradients (before the all-reduce averages ranks) are a little larger than the global norm.
LOCAL_GRAD = 1.1
# eval.window_questions: held-out questions over a short time window (15 minutes or 1 hour), 28
# of the trainer's 64 GRPO eval questions.
WINDOW_QUESTIONS = 28

# Production profile (see the module docstring). Environment overrides size a cardinality test.
PROD_LAYERS = 32
PROD_HEADS = 32
RANKS_PER_NODE = 8
DETAIL_EVERY = int(os.environ.get("SIM_DETAIL_EVERY") or 10)
PROMPTS_PER_STEP = int(os.environ.get("SIM_PROMPTS_PER_STEP") or 128)
PROMPT_POOL = int(os.environ.get("SIM_PROMPT_POOL") or 1_000_000)
PROMPT_SHAPES = 48
HELDOUT_QUESTIONS = int(os.environ.get("SIM_HELDOUT_QUESTIONS") or 512)
GRPO_GROUP_SIZE = 8


class LoraProfile:
    """Fixed per-run shape of realistic layer_stats: one gradient share and weight scale per
    (layer, LoRA matrix), so every series is smooth over steps. Gradients grow with layer index,
    much more steeply while bad_lr diverges, as the light mode's 12 layers do."""

    def __init__(self, rng: random.Random, layers: int = MODEL_LAYERS) -> None:
        self.keys = [(layer, module) for layer in range(layers) for module in LORA_MODULES]
        self.tags = [{"layer": f"layer_{layer:02d}", "module": module} for layer, module in self.keys]
        self.is_b = [module.endswith(".B") for _, module in self.keys]
        self.depth = [layer / (layers - 1) for layer, _ in self.keys]
        self.grad = [
            LORA_GRAD[module.split(".")[0]] * (1 + 0.55 * x) * rng.lognormvariate(0.0, 0.15)
            for (_, module), x in zip(self.keys, self.depth)
        ]
        # Light mode's divergence profile over 12 layers, (1 + 0.4 * layer), on a 0..1 depth scale.
        self.grad_diverging = [g * (1 + 4.4 * x) for g, x in zip(self.grad, self.depth)]
        self.weight = [
            LORA_B_NORM[module.split(".")[0]] * (1 + 0.1 * x) * rng.lognormvariate(0.0, 0.08)
            if b else LORA_A_NORM * rng.lognormvariate(0.0, 0.02)
            for (_, module), x, b in zip(self.keys, self.depth, self.is_b)
        ]
        # Sums of squared shares of the B and of the A matrices, to scale the per-matrix norms so
        # they add up (root sum of squares) to the step's gradient norm.
        self.norms = {
            diverging: (
                sum(g * g for g, b in zip(shares, self.is_b) if b),
                sum(g * g for g, b in zip(shares, self.is_b) if not b),
            )
            for diverging, shares in ((False, self.grad), (True, self.grad_diverging))
        }


@dataclass
class Run:
    scenario: str
    index: int
    # Distinguishes simulator starts ("1a2b-"), so run IDs never repeat across restarts: a reused
    # ID would inherit Flink's per-run state (moving averages, best eval) from an earlier run.
    prefix: str = ""
    max_steps: int = 360
    # Ranks of an SFT run; None: DEFAULT_RANKS for the telemetry mode.
    world_size: int | None = None
    step: int = 0
    attempt: int = 1
    done: bool = False
    faults_enabled: bool = True
    telemetry: str = LIGHT
    # Steps of the checkpoints this attempt's timeline saved, as the trainer keeps them.
    checkpoints: list[int] = field(default_factory=list)
    rng: random.Random = field(init=False)
    # Realistic mode's extra values only: never drawn from `rng`, so light output stays fixed.
    detail: random.Random = field(init=False)
    # rl_step exec_match noise (reward_hacking), in both modes: never drawn from `rng` or `detail`.
    grader: random.Random = field(init=False)
    lora: LoraProfile | None = field(init=False)
    # Fault onsets, randomized per run so repeated runs differ.
    spike_at: int = field(init=False)
    hack_at: int = field(init=False)

    def __post_init__(self) -> None:
        if self.telemetry not in TELEMETRY_MODES:
            raise ValueError(f"telemetry must be one of {TELEMETRY_MODES}, not {self.telemetry!r}")
        if self.world_size is None:
            self.world_size = DEFAULT_RANKS[self.telemetry]
        if self.world_size < 1:
            raise ValueError("world_size must be at least 1")
        self.rng = random.Random(zlib.crc32(self.run_id.encode()))
        self.spike_at = 120 + self.rng.randint(-10, 10)
        self.hack_at = 110 + self.rng.randint(-10, 10)
        self.detail = random.Random(zlib.crc32(f"{self.run_id}:detail".encode()))
        self.grader = random.Random(zlib.crc32(f"{self.run_id}:grader".encode()))
        self.lora = LoraProfile(self.detail) if self.realistic else None
        if self.production:
            self.lora = LoraProfile(self.detail, PROD_LAYERS)
            # Each head's resting attention entropy (nats): lower in deeper layers, varied by head.
            self.head_entropy = [
                [2.6 - 1.1 * layer / (PROD_LAYERS - 1) + self.detail.gauss(0.0, 0.25) for _ in range(PROD_HEADS)]
                for layer in range(PROD_LAYERS)
            ]
            self.prompt_offset = self.detail.randrange(PROMPT_POOL)

    @property
    def realistic(self) -> bool:
        return self.telemetry == REALISTIC

    @property
    def production(self) -> bool:
        return self.telemetry == PRODUCTION

    @property
    def detailed(self) -> bool:
        """realistic or production: the trainer's newer rl_step and eval fields."""
        return self.telemetry in (REALISTIC, PRODUCTION)

    @property
    def run_id(self) -> str:
        return f"{self.scenario}-{self.prefix}{self.index:03d}"

    @property
    def phase(self) -> str:
        return "grpo" if self.scenario in GRPO_SCENARIOS else "sft"

    # ------------------------------------------------------------------ control

    def on_command(self, command: dict) -> str | None:
        """Applies a watchdog control command; returns a description of what changed."""
        action = command.get("action")
        if action == "ROLLBACK":
            # Same checks as the trainer: a ROLLBACK in GRPO (its checkpoints are SFT's), an
            # incident from an attempt this run already left, or an argument that isn't a step,
            # changes nothing. The trainer's --max-rollbacks has no counterpart: a rollback turns
            # the fault off, so a simulated run never diverges twice.
            if self.phase != "sft":
                return f"ignored: ROLLBACK applies to SFT, and the run is in {self.phase}"
            if command.get("attempt") not in (None, self.attempt):
                return f"ignored: ROLLBACK is for attempt {command['attempt']}, and the run is on attempt {self.attempt}"
            if not str(command.get("arg") or 0).isdecimal():
                return f"ignored: ROLLBACK argument {command.get('arg')!r} is not a step"
            # The arg is the newest step to restore at or before; restore the newest checkpoint
            # saved by then (0 is the initial weights), as the trainer does.
            target = int(command.get("arg") or 0)
            checkpoint = max((c for c in self.checkpoints if c <= target), default=0)
            self.checkpoints = [c for c in self.checkpoints if c <= checkpoint]
            self.step = checkpoint
            self.attempt += 1
            self.faults_enabled = False
            return f"rolled back to checkpoint step {checkpoint} with a lower learning rate"
        if action == "STOP":
            self.done = True
            return f"stopped; best checkpoint at or before step {command.get('arg')}"
        if action == "ADVISE":
            # HELDOUT_DECLINE: likely overfitting, not a gamed grader. Advice only; keep training.
            return f"noted: held-out accuracy declining since GRPO step {command.get('arg')}; no action"
        return None

    # ------------------------------------------------------------------ events

    def tick(self, now_ms: int) -> list[dict]:
        """Advances one step and returns the telemetry it produced."""
        if self.done:
            return []
        self.step += 1
        if self.step > self.max_steps:
            self.done = True
            return []
        if self.phase == "sft":
            return self._sft_events(now_ms)
        return self._grpo_events(now_ms)

    def _base(self, kind: str, ts: int, **extra) -> dict:
        event = {
            "type": kind,
            "ts": ts,
            "run_id": self.run_id,
            "scenario": self.scenario,
            "phase": self.phase,
            "attempt": self.attempt,
            "step": self.step,
        }
        event.update(extra)
        return event

    def _noise(self, sigma: float) -> float:
        return self.rng.gauss(0.0, sigma)

    def _sft_events(self, now: int) -> list[dict]:
        s = self.step
        events: list[dict] = []
        diverging = self.scenario == "bad_lr" and self.faults_enabled and s >= self.spike_at
        k = s - self.spike_at

        lr_scale = 10.0 if self.scenario == "bad_lr" and self.faults_enabled else 1.0
        lr = 2e-4 * lr_scale * min(1.0, s / 20) * (0.5 + 0.5 * math.cos(math.pi * s / self.max_steps))
        loss = 0.35 + 2.2 * math.exp(-s / 60)
        grad = 1.0 + 0.5 * math.exp(-s / 40)
        if diverging:
            loss *= 2.0 + 0.5 * k
            if k >= 3:
                grad *= 8 + 4 * (k - 3)

        # The first step compiles the training graph (Neuron warm-up).
        compile_s = 45.0 + self._noise(3) if s == 1 else 0.0
        for rank in range(self.world_size):
            # Local compute time before the all-reduce barrier.
            step_ms = 950 + self._noise(25) + compile_s * 1000
            events.append(
                self._base(
                    "train_step",
                    now + self.rng.randint(0, 40),
                    rank=rank,
                    world_size=self.world_size,
                    **({"tags": self._topology(rank)} if self.production else {}),
                    metrics={
                        "loss": round(loss * (1 + self._noise(0.02)), 5),
                        "grad_norm": round(grad * (1 + self._noise(0.03)), 5),
                        "lr": lr,
                        "step_time_ms": round(step_ms, 1),
                        "tokens_per_s": round(8 * 2048 / (step_ms / 1000), 1),
                    },
                )
            )
            if s % 5 == 0:
                util = 88 + self._noise(3)
                events.append(
                    self._base(
                        "neuron_hw",
                        now,
                        rank=rank,
                        tags={"neuron_core": f"nc{rank:02d}"},
                        metrics={
                            "util_pct": round(max(0.0, min(100.0, util)), 1),
                            "mem_used_gb": round(26 + self._noise(0.8), 2),
                            "exec_latency_ms": round(step_ms * 0.9, 1),
                        },
                    )
                )

        if self.realistic:
            events.extend(self._lora_stats(now, grad, diverging, k))
        elif self.production:
            events.extend(self._production_detail(now, grad, diverging, k))
        elif s % 5 == 0:
            for layer in range(LAYERS):
                layer_grad = grad / math.sqrt(LAYERS) * (1 + 0.05 * layer)
                if diverging and k >= 3:
                    layer_grad *= 1 + 0.4 * layer
                events.append(
                    self._base(
                        "layer_stats",
                        now,
                        tags={"layer": f"layer_{layer:02d}"},
                        metrics={
                            "grad_norm": round(layer_grad * (1 + self._noise(0.05)), 5),
                            "update_ratio": round(1e-3 * lr_scale * (1 + self._noise(0.1)), 6),
                        },
                    )
                )

        if s % 25 == 0:
            accuracy = 0.15 + 0.55 * (1 - math.exp(-s / 90))
            if diverging:
                accuracy = 0.05
            events.append(
                self._base(
                    "eval",
                    now,
                    tags={"eval_set": "heldout_sql"},
                    metrics={"exec_accuracy": round(accuracy + self._noise(0.015), 4)},
                )
            )
        if s % CHECKPOINT_EVERY == 0 and not diverging:
            self.checkpoints.append(s)
            events.append(self._base("checkpoint", now))
        return events

    def _lora_stats(self, now: int, grad: float, diverging: bool, k: int) -> list[dict]:
        """Realistic layer_stats, as train.py reports them: each rank's local gradient norm for
        each LoRA matrix, and the matrix's weight norm (the same on every rank: data parallel)."""
        lora = self.lora
        progress = 1 - math.exp(-self.step / LORA_B_STEPS)  # how far lora_B has grown
        blowup = diverging and k >= 3  # when the step's gradient norm blows up
        shares = lora.grad_diverging if blowup else lora.grad
        b_sum, a_sum = lora.norms[blowup]
        a_share = LORA_A_GRAD * progress
        # Weights blow up a few steps behind the gradients, the deepest layers most.
        grow = 0.3 * max(0, k - 2) if diverging else 0.0
        weights = [
            w * progress * (1 + grow * (0.2 + x)) if b else w * (1 + 0.04 * progress) * (1 + 0.1 * grow * (0.2 + x))
            for w, b, x in zip(lora.weight, lora.is_b, lora.depth)
        ]
        factors = [share if b else share * a_share for share, b in zip(shares, lora.is_b)]
        scale = grad * LOCAL_GRAD / math.sqrt(b_sum + a_share * a_share * a_sum)
        gauss = self.detail.gauss
        events: list[dict] = []
        for rank in range(self.world_size):
            rank_scale = scale * (1 + gauss(0.0, 0.03))  # this rank's batch
            for tags, factor, weight in zip(lora.tags, factors, weights):
                g = rank_scale * factor * (1 + gauss(0.0, 0.05))
                events.append(
                    self._base(
                        "layer_stats",
                        now,
                        rank=rank,
                        tags=dict(tags),
                        metrics={"grad_norm": g, "weight_norm": weight, "grad_to_weight": g / weight if weight else 0.0},
                    )
                )
        return events

    def _grpo_events(self, now: int) -> list[dict]:
        s = self.step
        hacking = self.scenario == "reward_hacking" and s >= self.hack_at
        h = max(0, s - self.hack_at)
        if self.scenario == "reward_hacking":
            # Naive reward: "query executes and returns rows".
            reward = 0.2 + 0.75 * (1 - math.exp(-s / 80))
            accuracy = 0.55 + 0.15 * (1 - math.exp(-min(s, self.hack_at) / 60)) - 0.004 * h
            entropy = 1.2 - 0.3 * (1 - math.exp(-s / 150)) - (0.65 * (1 - math.exp(-h / 40)) if hacking else 0)
            resp_len = 60 - (42 * (1 - math.exp(-h / 30)) if hacking else 0)
            kl = 0.02 + 0.0002 * s + (0.0015 * h if hacking else 0)
            # Grader gap, reward - exec_match. Negative at first: correct answers about hosts with
            # no data return no rows, so the lenient grader does not pay them. It rises through zero
            # as hacking sets in and the policy is paid for wrong answers (dropped filters), and by
            # the time the held-out decline rule fires it has risen well past the detector's
            # hackGapRise (0.15) since the best evaluation. Shaped like a real seed-1 run, whose
            # 10-step means went -0.34, -0.54, -0.26, -0.10, +0.04 and then +0.2 to +0.7.
            gap = -0.35 * math.exp(-s / 50) + (0.6 * (1 - math.exp(-h / 40)) if hacking else 0.0)
        else:
            # Fixed reward: result-set match against the gold query.
            reward = 0.2 + 0.5 * (1 - math.exp(-s / 120))
            accuracy = 0.55 + 0.2 * (1 - math.exp(-s / 120))
            entropy = 1.2 - 0.3 * (1 - math.exp(-s / 150))
            resp_len = 60
            kl = 0.02 + 0.0002 * s
            gap = None  # the honest grader pays exactly the correct answers: exec_match == reward
        step_metrics = {
            "reward": round(reward + self._noise(0.03), 4),
            "kl": round(max(0.0, kl + self._noise(0.002)), 5),
            "entropy": round(max(0.05, entropy + self._noise(0.02)), 4),
            "resp_len": round(max(5.0, resp_len + self._noise(4)), 1),
        }
        if gap is None:
            step_metrics["exec_match"] = step_metrics["reward"]
        else:
            exec_match = step_metrics["reward"] - gap + self.grader.gauss(0.0, 0.02)
            step_metrics["exec_match"] = round(min(1.0, max(0.0, exec_match)), 4)
        if self.detailed:
            # entropy is the policy's entropy; sample_nll, the mean NLL of the sampled tokens, has
            # the same expectation (sampling at temperature 1) but is noisier: a mean over the
            # tokens of 4 candidates, fewer as responses get short.
            nll_sigma = 0.07 * math.sqrt(max(0.05, entropy) * 60 / max(5.0, resp_len))
            step_metrics["sample_nll"] = round(max(0.05, entropy + self.detail.gauss(0.0, nll_sigma)), 4)
        events = [self._base("rl_step", now, metrics=step_metrics)]
        if s % 20 == 0:
            eval_metrics = {"exec_accuracy": round(max(0.0, accuracy) + self._noise(0.015), 4)}
            if self.detailed:
                # Short windows hold the fewest readings, so they are a little harder, and their
                # correct answer is more often empty, which a hacking policy (paid for any rows)
                # abandons first: their accuracy falls faster.
                window = accuracy - 0.06 - (0.002 * h if hacking else 0.0) + self.detail.gauss(0.0, 0.04)
                hits = round(min(1.0, max(0.0, window)) * WINDOW_QUESTIONS)
                eval_metrics["exec_accuracy_window"] = hits / WINDOW_QUESTIONS
                eval_metrics["window_questions"] = float(WINDOW_QUESTIONS)
            events.append(self._base("eval", now, tags={"eval_set": "heldout_sql"}, metrics=eval_metrics))
            if self.production:
                events.extend(self._eval_questions(now, eval_metrics["exec_accuracy"], hacking))
        if self.production:
            events.extend(self._prompts(now, step_metrics["reward"], step_metrics["exec_match"]))
            events.extend(self._production_detail(now, 1.0, False, 0))
        if s % CHECKPOINT_EVERY == 0:
            self.checkpoints.append(s)
            events.append(self._base("checkpoint", now))
        return events

    # ------------------------------------------------------------------ production profile

    @staticmethod
    def _topology(rank: int) -> dict:
        return {"node": f"node-{rank // RANKS_PER_NODE:03d}", "local_rank": str(rank % RANKS_PER_NODE)}

    def _production_detail(self, now: int, grad: float, diverging: bool, k: int) -> list[dict]:
        """The ranks due this step (rank r at the steps where (step + r) % DETAIL_EVERY == 0):
        layer_stats for each layer and LoRA matrix, and attn_stats for each layer and head."""
        ranks = [r for r in range(self.world_size) if (self.step + r) % DETAIL_EVERY == 0]
        if not ranks:
            return []
        lora = self.lora
        progress = 1 - math.exp(-self.step / LORA_B_STEPS)
        blowup = diverging and k >= 3
        shares = lora.grad_diverging if blowup else lora.grad
        b_sum, a_sum = lora.norms[blowup]
        a_share = LORA_A_GRAD * progress
        factors = [share if b else share * a_share for share, b in zip(shares, lora.is_b)]
        weights = [w * progress if b else w for w, b in zip(lora.weight, lora.is_b)]
        scale = grad * LOCAL_GRAD / math.sqrt(b_sum + a_share * a_share * a_sum)
        # Attention entropy collapses in the deeper layers as a divergence takes hold.
        collapse = min(1.0, 0.15 * max(0, k)) if diverging else 0.0
        gauss = self.detail.gauss
        events: list[dict] = []
        for rank in ranks:
            topology = self._topology(rank)
            rank_scale = scale * (1 + gauss(0.0, 0.03))
            for tags, factor, weight in zip(lora.tags, factors, weights):
                g = rank_scale * factor * (1 + gauss(0.0, 0.05))
                events.append(self._base(
                    "layer_stats", now, rank=rank, tags={**tags, **topology},
                    metrics={"grad_norm": g, "weight_norm": weight, "grad_to_weight": g / weight if weight else 0.0},
                ))
            for layer in range(PROD_LAYERS):
                depth = layer / (PROD_LAYERS - 1)
                for head in range(PROD_HEADS):
                    entropy = self.head_entropy[layer][head] * (1 - collapse * depth) + gauss(0.0, 0.03)
                    events.append(self._base(
                        "attn_stats", now, rank=rank,
                        tags={"layer": f"layer_{layer:02d}", "head": f"h{head:02d}", **topology},
                        metrics={"entropy": round(max(0.0, entropy), 5),
                                 "max_prob": round(min(1.0, math.exp(-max(0.0, entropy)) * 1.6), 5)},
                    ))
        return events

    def _prompts(self, now: int, reward: float, exec_match: float) -> list[dict]:
        """One rl_prompt record per prompt this GRPO step: the group's mean reward and strict
        correctness. Prompts are consecutive from the run's offset in the pool, so a run of N
        steps sees N x PROMPTS_PER_STEP distinct prompts (wrapping around the pool)."""
        gauss = self.detail.gauss
        events = []
        start = self.prompt_offset + (self.step - 1) * PROMPTS_PER_STEP
        for i in range(PROMPTS_PER_STEP):
            prompt = (start + i) % PROMPT_POOL
            r = min(1.0, max(0.0, reward + gauss(0.0, 0.25)))
            m = min(1.0, max(0.0, exec_match + gauss(0.0, 0.25)))
            # Quantized like a group of GRPO_GROUP_SIZE pass/fail grades.
            r = round(r * GRPO_GROUP_SIZE) / GRPO_GROUP_SIZE
            m = round(m * GRPO_GROUP_SIZE) / GRPO_GROUP_SIZE
            events.append(self._base(
                "rl_prompt", now,
                tags={"prompt_id": f"p{prompt:07d}", "prompt_shape": f"shape_{prompt % PROMPT_SHAPES:02d}"},
                metrics={"reward_mean": r, "exec_match_mean": m, "grader_gap": r - m,
                         "group_size": float(GRPO_GROUP_SIZE)},
            ))
        return events

    def _eval_questions(self, now: int, accuracy: float, hacking: bool) -> list[dict]:
        """One eval_question record per held-out question: whether the model's answer matched."""
        uniform = self.detail.random
        events = []
        for q in range(HELDOUT_QUESTIONS):
            no_data = q % 3 == 0  # about a third of held-out answers are empty, as in the trainer
            p = accuracy - (0.15 if hacking and no_data else 0.0)
            events.append(self._base(
                "eval_question", now,
                tags={"eval_set": "heldout_sql", "question_id": f"h{q:04d}",
                      "question_type": "no_data" if no_data else "has_data",
                      "window": "short" if q % 7 < 3 else "long"},
                metrics={"correct": 1.0 if uniform() < p else 0.0},
            ))
        return events
