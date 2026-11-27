# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""LoRA SFT of Qwen2.5 on Trainium, streaming watchdog telemetry and obeying control commands.

    torchrun --nproc_per_node=2 -m watchdog_trainer.train --scenario healthy \
        --run-id "trn-healthy-$(date -u +%Y%m%d%H%M%S)"

Give every run a new ID: Flink keeps a run's detector state for 7 days, so a reused ID would
inherit the earlier run's.

One process per NeuronCore (trn1.2xlarge has 2), data parallel. Scenarios inject real faults:

  healthy          no fault
  bad_lr           learning rate jumps --lr-spike times at --fault-step; ROLLBACK restores the
                   named checkpoint and resumes at a lower learning rate
  healthy_grpo     --sft-steps of SFT, then --grpo-steps (120) of GRPO rewarded for SQL whose
                   result matches the gold query's result (strict grader); if held-out accuracy
                   declines anyway (overfitting), the watchdog's ADVISE keeps the adapters of the
                   evaluation it names (the watchdog's best) and training continues
  reward_hacking   the same, but rewarded for SQL that runs and returns a non-null row (lenient
                   grader); dropping filters pays, held-out execution accuracy falls, and the
                   watchdog's STOP ends the run, keeping the adapters of the held-out evaluation
                   the STOP names (the watchdog's best)
  reward_hacking_unwatched
                   reward_hacking with the watchdog's commands ignored, for all 120 GRPO steps:
                   what happens when nobody stops it

--seed varies LoRA initialization, GRPO sampling, and GRPO question order; the data and the
sample tables keep their own fixed seeds.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist

from . import data, telemetry

LOG = logging.getLogger("watchdog_trainer")
# reward_hacking_unwatched: the same lenient grader, but the trainer logs and ignores the
# watchdog's commands, to show what happens when nobody stops it.
GRPO_SCENARIOS = ("healthy_grpo", "reward_hacking", "reward_hacking_unwatched")
# The run IDs the watchdog accepts (TelemetryDeserializer): it skips telemetry with any other.
RUN_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
SCENARIOS = ("healthy", "bad_lr") + GRPO_SCENARIOS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=SCENARIOS, default="healthy")
    p.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds LoRA init, GRPO sampling noise, and GRPO question order (not the data)")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4, help="per NeuronCore")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="0 disables clipping")
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=25)
    # Telemetry density: per-layer stats every step and NeuronCore samples every second, as
    # production training telemetry commonly is.
    p.add_argument("--layer-stats-every", type=int, default=1)
    p.add_argument("--hw-period", default="1s", help="neuron-monitor sampling period")
    p.add_argument("--train-examples", type=int, default=4000)
    p.add_argument("--eval-examples", type=int, default=64)
    p.add_argument("--gen-examples", type=int, default=32, help="held-out questions decoded per generation eval")
    p.add_argument("--gen-eval-every", type=int, default=50)
    # Early and large: once the model has memorized the task, a 100x jump at step 120 does not diverge.
    p.add_argument("--fault-step", type=int, default=70, help="bad_lr: step the learning rate jumps")
    p.add_argument("--lr-spike", type=float, default=1000.0, help="bad_lr: learning-rate multiplier")
    p.add_argument("--rollback-lr-scale", type=float, default=0.5,
                   help="learning-rate multiplier applied on every ROLLBACK (0.5, then 0.25, ...)")
    p.add_argument("--max-rollbacks", type=int, default=3,
                   help="ROLLBACKs to apply; one more stops the run, since lowering the rate has not helped")
    p.add_argument("--sft-steps", type=int, default=30, help="GRPO scenarios: SFT steps before GRPO")
    p.add_argument("--grpo-steps", type=int, default=120,
                   help="GRPO steps for every GRPO scenario (reward_hacking's STOP usually ends it sooner)")
    p.add_argument("--grpo-eval-examples", type=int, default=64,
                   help="held-out questions per GRPO eval; more is steadier, fewer is faster")
    p.add_argument("--grpo-eval-batch", type=int, default=8, help="decode batch for GRPO evals")
    p.add_argument("--group-size", type=int, default=4, help="sampled SQL candidates per question")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--grpo-lr", type=float, default=3e-5)
    p.add_argument("--kl-beta", type=float, default=0.02)
    p.add_argument("--grpo-eval-every", type=int, default=10)
    p.add_argument("--final-verdict-seconds", type=float, default=15.0,
                   help="after the last GRPO evaluation, wait this long for the watchdog's verdict on it")
    p.add_argument("--sample-every", type=int, default=5, help="GRPO steps between logged SQL samples")
    p.add_argument("--head-stats", action="store_true",
                   help="at every evaluation, report attention entropy per layer and head (attn_stats) on a "
                        "fixed probe batch; adds one compiled graph on Neuron")
    p.add_argument("--head-probe", type=int, default=2, help="held-out examples per rank in the --head-stats probe")
    p.add_argument("--exec-workers", type=int, default=3, help="sandboxed SQL workers per rank")
    p.add_argument("--checkpoint-dir", default=os.environ.get("CHECKPOINT_DIR", "/opt/watchdog/checkpoints"))
    p.add_argument("--telemetry-stream", default=os.environ.get("TELEMETRY_STREAM"))
    # Control plane: the watchdog's commands are OPEN rows in InfluxDB's incident table.
    p.add_argument("--influx-url", default=os.environ.get("INFLUX_URL"), help="reader endpoint to poll")
    p.add_argument("--influx-db", default=os.environ.get("INFLUX_DB", "watchdog"))
    p.add_argument("--influx-token-secret", default=os.environ.get("INFLUX_TOKEN_SECRET"),
                   help="Secrets Manager ARN of the read-only token")
    p.add_argument("--control-poll-seconds", type=float, default=float(os.environ.get("CONTROL_POLL_SECONDS", "1.0")))
    p.add_argument("--region", default=os.environ.get("AWS_REGION"))
    args = p.parse_args(argv)
    if not args.run_id:
        p.error("--run-id (or RUN_ID) is required so every rank reports the same run")
    # The ID also names the run's checkpoint directory.
    if not RUN_ID.fullmatch(args.run_id) or args.run_id in (".", ".."):
        p.error("--run-id must be 1 to 128 of A-Za-z0-9._:- (the watchdog skips telemetry with any other run ID)")
    if args.seed < 0:
        p.error("--seed must be 0 or more")
    args.eval_examples = max(args.eval_examples, args.grpo_eval_examples)
    if args.scenario == "bad_lr":
        # Clipping would hide the blowup this scenario exists to show.
        args.max_grad_norm = 0.0
    return args


# ---------------------------------------------------------------------- data


def encode(tokenizer, example: data.Example) -> tuple[list[int], list[int]]:
    prompt = tokenizer(example.prompt, add_special_tokens=False).input_ids
    answer = tokenizer(example.sql + tokenizer.eos_token, add_special_tokens=False).input_ids
    return prompt + answer, [-100] * len(prompt) + answer


def round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def collate(encoded: list[tuple[list[int], list[int]]], length: int, pad_id: int) -> dict[str, torch.Tensor]:
    ids = torch.full((len(encoded), length), pad_id, dtype=torch.long)
    labels = torch.full((len(encoded), length), -100, dtype=torch.long)
    mask = torch.zeros((len(encoded), length), dtype=torch.long)
    for row, (tokens, target) in enumerate(encoded):
        tokens, target = tokens[:length], target[:length]
        ids[row, : len(tokens)] = torch.tensor(tokens)
        labels[row, : len(target)] = torch.tensor(target)
        mask[row, : len(tokens)] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def batch_indices(step: int, rank: int, world: int, batch_size: int, n: int) -> list[int]:
    """Deterministic per step, so a rollback replays the same data."""
    rng = random.Random(f"{step}")
    picks = rng.sample(range(n), batch_size * world)
    return picks[rank * batch_size : (rank + 1) * batch_size]


def seed_rngs(seed: int, xm=None) -> None:
    """Seeds the CPU generator (LoRA initialization) and, given torch_xla's xla_model, the device
    generator (GRPO sampling noise). Seed 0 leaves torch_xla's default device seed."""
    torch.manual_seed(seed)
    if seed and xm is not None:
        xm.set_rng_state(seed)


def shape_tag(shape: tuple) -> str:
    """A question shape as a tag value: measurement, aggregate, grouping, then its filter flags."""
    return "-".join(str(int(x)) if isinstance(x, bool) else str(x) for x in shape)


def grpo_question(step: int, rank: int, seed: int, n: int) -> int:
    """Index of the training question a rank samples candidates for at a GRPO step. Seed 0 leaves
    the seed out of the key."""
    key = f"grpo:{step}:{rank}" if not seed else f"grpo:{seed}:{step}:{rank}"
    return random.Random(key).randrange(n)


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v) for v in value)
    return value


# ---------------------------------------------------------------------- training


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_backend  # noqa: F401 - registers the "xla" backend
        import torch_xla.runtime as xr
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.args = args
        self.xm = xm
        dist.init_process_group("xla", init_method="xla://")
        self.rank = xr.global_ordinal()
        # Where this rank runs: node and local rank (torchrun's GROUP_RANK and LOCAL_RANK), tagged on
        # per-rank detail as the simulator's production profile tags it. They follow from the rank,
        # so they add no series.
        self.topology = {
            "node": f"node-{int(os.environ.get('GROUP_RANK', '0')):03d}",
            "local_rank": os.environ.get("LOCAL_RANK", str(self.rank)),
        }
        self.world = xr.world_size()
        self.cpu_group = dist.new_group(backend="gloo")
        self.device = torch_xla.device()
        self.scenario = f"trn_{args.scenario}"
        self.attempt = 1
        self.step = 0
        self.faults_enabled = True
        self.lr_scale = 1.0
        self.checkpoints: dict[int, Path] = {}
        self.last_loss = float("nan")
        self.last_grad_norm = float("nan")
        self.phase = "sft"
        self.grpo = args.scenario in GRPO_SCENARIOS
        self.lenient = args.scenario.startswith("reward_hacking")
        self.unwatched = args.scenario == "reward_hacking_unwatched"
        self.best_exec = (-1.0, 0)  # (held-out exec accuracy, GRPO step)
        # Every GRPO evaluation's accuracy (all ranks) and adapters (rank 0, which saves them), so
        # a STOP can keep the step it names. LoRA adapters are small: about 17 MB per evaluation.
        self.eval_accuracy: dict[int, float] = {}
        self.eval_states: dict[int, dict[str, torch.Tensor]] = {}

        seed_rngs(args.seed, xm)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.train_set = [encode(self.tokenizer, e) for e in data.build("train", args.train_examples)]
        heldout = data.build("heldout", args.eval_examples)
        self.eval_set = [encode(self.tokenizer, e) for e in heldout]
        longest = max(len(t) for t, _ in self.train_set + self.eval_set)
        self.fixed_length = round_up(longest, 16)
        # Generation (eval and GRPO sampling): decode into one fixed-length buffer so every decode
        # step reuses one compiled graph. Size it for the longest prompt and answer of either set.
        self.gen_examples = heldout[: args.gen_examples]
        self.grpo_eval_examples = heldout[: args.grpo_eval_examples]
        self.grpo_pool = data.build("train", args.train_examples)
        sized = self.grpo_eval_examples + self.grpo_pool[:1000]
        prompt_tokens = [len(self.tokenizer(e.prompt, add_special_tokens=False).input_ids) for e in sized]
        answer_tokens = [len(encode(self.tokenizer, e)[0]) - p for e, p in zip(sized, prompt_tokens)]
        self.gen_new_tokens = max(answer_tokens) + 8
        self.gen_length = round_up(max(prompt_tokens) + self.gen_new_tokens, 16)

        # Eager attention: SDPA returns NaN logits on Neuron (torch-neuronx 2.9, verified on trn1).
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()  # fp32 adapters; bf16 frozen base
        self.model = model.to(self.device)
        self.trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self.initial_state = {n: p.detach().cpu().clone() for n, p in self.trainable}
        # One entry per LoRA matrix: (layer, "q_proj.A"), ... 28 layers x 4 projections x A/B.
        self.module_keys: list[tuple[int, str]] = []
        self.module_params: list[torch.nn.Parameter] = []
        for name, param in self.trainable:
            match = re.search(r"layers\.(\d+)\..*?\.(\w+_proj)\.lora_(A|B)\.", name)
            if match:
                self.module_keys.append((int(match.group(1)), f"{match.group(2)}.{match.group(3)}"))
                self.module_params.append(param)
        self.layer_ids = sorted({layer for layer, _ in self.module_keys})
        self.optimizer = self._new_optimizer()

        self.emitter = None
        self.control = None
        self.monitor = None
        self.executor = None
        if self.grpo:
            from . import sqlexec

            self.executor = sqlexec.Executor(workers=args.exec_workers)
        if args.telemetry_stream:
            self.emitter = telemetry.KinesisEmitter(args.telemetry_stream, args.region)
        if self.rank == 0 and args.influx_url and args.influx_token_secret:
            # Read every OPEN incident for this run since a minute before it started.
            self.control = telemetry.InfluxControlPoller(
                args.influx_url, args.influx_db, telemetry.read_token(args.influx_token_secret, args.region),
                args.run_id, since_ms=int(time.time() * 1000) - 60_000, interval=args.control_poll_seconds,
                on_stats=lambda stats: self._emit("control_poll", tags={"endpoint": "reader"}, metrics=stats),
            )
            self.control.start()
            self.control.ready.wait(30)
            if isinstance(self.control.last_error, telemetry.ControlAuthError):
                # Retrying won't help, and a watched run that can't read its commands is unwatched.
                raise RuntimeError(f"InfluxDB rejected the read-only token: {self.control.last_error}")
        if self.rank == 0 and self.emitter is not None:
            # Heartbeats keep Flink's event-time watermark moving while the trainer is silent
            # (a Neuron compile can take minutes); without them, event-time timers such as the
            # incident quiet period fire only when the next real event arrives.
            import threading

            def heartbeat() -> None:
                while True:
                    time.sleep(10)
                    self._emit("heartbeat")

            threading.Thread(target=heartbeat, name="heartbeat", daemon=True).start()
            from .neuron_monitor import NeuronMonitorForwarder

            self.monitor = NeuronMonitorForwarder(self._on_neuron_sample, period=args.hw_period, on_extra=self._on_neuron_extra)
            self.monitor.start()
        LOG.info(
            "rank %d/%d: %s, seed %d, %d trainable tensors over %d layers, fixed length %d",
            self.rank, self.world, args.model, args.seed, len(self.trainable), len(self.layer_ids), self.fixed_length,
        )
        if self.rank == 0:
            self._emit(
                "run_info",
                tags={
                    "seed": str(args.seed),
                    "grader": ("lenient" if self.lenient else "strict") if self.grpo else "none",
                    "watched": "no" if self.unwatched else "yes",
                    "model": args.model.rstrip("/").rsplit("/", 1)[-1],
                },
                metrics={
                    "lr": args.lr, "grpo_lr": args.grpo_lr, "kl_beta": args.kl_beta,
                    "group_size": float(args.group_size), "temperature": args.temperature, "lora_r": 16.0,
                    "sft_steps": float(args.sft_steps if self.grpo else args.max_steps),
                    "grpo_steps": float(args.grpo_steps if self.grpo else 0), "world_size": float(self.world),
                },
            )

    # ------------------------------------------------------------------ telemetry

    def _emit(self, kind: str, **fields) -> None:
        if self.emitter is None:
            return
        fields.setdefault("step", self.step)
        fields.setdefault("phase", self.phase)
        self.emitter.emit(
            telemetry.event(kind, run_id=self.args.run_id, scenario=self.scenario, attempt=self.attempt, **fields)
        )

    def _on_neuron_sample(self, sample: dict) -> None:
        metrics = {k: v for k, v in sample.items() if k != "core"}
        self._emit("neuron_hw", rank=sample["core"], tags={"neuron_core": f"nc{sample['core']:02d}"}, metrics=metrics)

    def _on_neuron_extra(self, extra: dict) -> None:
        self._emit(extra["kind"], tags=extra["tags"] or None, metrics=extra["metrics"])

    # ------------------------------------------------------------------ helpers

    def _new_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW([p for _, p in self.trainable], lr=self.args.lr, weight_decay=0.0)

    def _lr(self) -> float:
        a = self.args
        warm = min(1.0, self.step / max(1, a.warmup_steps))
        progress = min(1.0, self.step / a.max_steps)
        # Floor at 10% so the scalar never becomes 0, which XLA bakes in as a constant (recompile).
        cosine = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        spike = a.lr_spike if a.scenario == "bad_lr" and self.faults_enabled and self.step >= a.fault_step else 1.0
        return a.lr * warm * cosine * self.lr_scale * spike

    def _broadcast(self, value):
        holder = [value]
        dist.broadcast_object_list(holder, src=0, group=self.cpu_group)
        return holder[0]

    # ------------------------------------------------------------------ steps

    def train_step(self) -> None:
        a = self.args
        encoded = [self.train_set[i] for i in batch_indices(self.step, self.rank, self.world, a.batch_size, len(self.train_set))]
        # One fixed length for every step and rank: a new shape would force a Neuron recompile, and
        # ranks with different shapes compile different graphs whose all-reduces never pair up.
        batch = {k: v.to(self.device) for k, v in collate(encoded, self.fixed_length, self.pad_id).items()}
        lr = self._lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        started = time.monotonic()
        self.model.train()
        loss = self.model(**batch).loss
        loss.backward()
        # Per-module norms of this rank's own gradients, before the all-reduce averages them away,
        # so rank-to-rank differences stay visible.
        local_grad_norms = torch.stack([torch.linalg.vector_norm(p.grad) for p in self.module_params])
        self.xm.reduce_gradients(self.optimizer)
        total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for _, p in self.trainable]))
        if a.max_grad_norm > 0:
            coef = torch.clamp(a.max_grad_norm / (total_norm + 1e-6), max=1.0)
            for _, p in self.trainable:
                p.grad.mul_(coef)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        weight_norms = torch.stack([torch.linalg.vector_norm(p.detach()) for p in self.module_params])
        self.xm.mark_step()
        loss_value = loss.item()
        grad_value = total_norm.item()
        step_ms = (time.monotonic() - started) * 1000
        self.last_loss = loss_value
        self.last_grad_norm = grad_value

        tokens = int(batch["attention_mask"].sum().item())
        self._emit(
            "train_step",
            rank=self.rank,
            world_size=self.world,
            tags=dict(self.topology),
            metrics={
                "loss": loss_value,
                "grad_norm": grad_value,
                "lr": lr,
                "step_time_ms": round(step_ms, 1),
                "tokens_per_s": round(tokens / (step_ms / 1000), 1),
            },
        )
        if self.step % self.args.layer_stats_every == 0:
            for (layer, module), grad, weight in zip(
                self.module_keys, local_grad_norms.cpu().tolist(), weight_norms.cpu().tolist()
            ):
                self._emit(
                    "layer_stats",
                    rank=self.rank,
                    tags={"layer": f"layer_{layer:02d}", "module": module, **self.topology},
                    metrics={"grad_norm": grad, "weight_norm": weight, "grad_to_weight": grad / weight if weight else 0.0},
                )
        if self.rank == 0 and self.step % 10 == 0:
            LOG.info("step %d attempt %d: loss %.4f grad %.3f lr %.2e %.0f ms", self.step, self.attempt, loss_value, grad_value, lr, step_ms)

    def _decode(self, prompts: list[list[int]], temperature: float = 0.0):
        """Decodes a batch of prompts into a fixed-length buffer, greedily or by sampling (Gumbel-max
        at `temperature`). Fixed shapes and one graph cut per token, so Neuron compiles it once.
        Returns device tensors (ids, attention mask, cursor, done)."""
        lm = self.model.get_base_model()
        size, length = len(prompts), self.gen_length
        rows = torch.arange(size, device=self.device)
        ids = torch.full((size, length), self.pad_id, dtype=torch.long)
        mask = torch.zeros((size, length), dtype=torch.long)
        for row, prompt in enumerate(prompts):
            ids[row, : len(prompt)] = torch.tensor(prompt)
            mask[row, : len(prompt)] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        cursor = torch.tensor([len(p) for p in prompts], device=self.device)
        done = torch.zeros(size, dtype=torch.bool).to(self.device)
        for _ in range(self.gen_new_tokens):
            hidden = lm.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            # A diverged model yields NaN logits, and argmax over NaN on Neuron can return an
            # out-of-vocabulary id that crashes the next embedding lookup; keep ids in range.
            logits = torch.nan_to_num(lm.lm_head(hidden[rows, cursor - 1]).float(), nan=-1e4, posinf=1e4, neginf=-1e4)
            if temperature > 0:
                uniform = torch.rand(logits.shape, device=self.device).clamp(1e-10, 1.0)
                logits = logits / temperature - torch.log(-torch.log(uniform))
            token = logits.argmax(-1).clamp(0, logits.size(-1) - 1)
            token = torch.where(done, torch.full_like(token, self.pad_id), token)
            ids = ids.scatter(1, cursor.unsqueeze(1), token.unsqueeze(1))
            mask = mask.scatter(1, cursor.unsqueeze(1), (~done).long().unsqueeze(1))
            done = done | (token == self.tokenizer.eos_token_id)
            cursor = cursor + (~done).long()
            self.xm.mark_step()
        return ids, mask, cursor, done

    def _texts(self, ids: torch.Tensor, prompts: list[list[int]]) -> list[str]:
        """Decodes the generated part of each row (CPU ids) up to the first EOS or pad."""
        texts = []
        for row, prompt in enumerate(prompts):
            tokens = ids[row, len(prompt) :].tolist()
            for stop in (self.tokenizer.eos_token_id, self.pad_id):
                if stop in tokens:
                    tokens = tokens[: tokens.index(stop)]
            texts.append(" ".join(self.tokenizer.decode(tokens, skip_special_tokens=True).split()))
        return texts

    def _generate_heldout(self, examples: list[data.Example] | None = None, size: int | None = None) -> tuple[list[data.Example], list[str]]:
        """Greedy-decodes this rank's share of held-out examples (default: the SFT generation set)."""
        examples = (self.gen_examples if examples is None else examples)[self.rank :: self.world]
        size, texts = size or self.args.batch_size, []
        for start in range(0, len(examples), size):
            chunk = examples[start : start + size]
            prompts = [self.tokenizer(e.prompt, add_special_tokens=False).input_ids for e in chunk]
            prompts += [prompts[0]] * (size - len(prompts))  # keep one decode shape
            ids, _, _, _ = self._decode(prompts)
            texts.extend(self._texts(ids.cpu(), prompts)[: len(chunk)])
        return examples, texts

    def _generate(self) -> tuple[list[float], list[tuple[str, str]]]:
        """Exact matches of greedily generated held-out SQL: ([matches, examples, window-question
        matches, window questions], samples)."""
        examples, texts = self._generate_heldout()
        hits = [t == " ".join(e.sql.split()) for e, t in zip(examples, texts)]
        window = [e.is_window_question for e in examples]
        counts = [sum(hits), len(examples), sum(h for h, w in zip(hits, window) if w), sum(window)]
        return [float(c) for c in counts], [(e.sql, t) for e, t in zip(examples, texts)][:2]

    def evaluate(self, generate: bool = False) -> None:
        """Teacher-forced held-out loss, token accuracy, and exact match of the whole SQL; with
        `generate`, also exact match of SQL the model writes itself (greedy decoding)."""
        self.model.eval()
        shard = self.eval_set[self.rank :: self.world]
        sums = torch.zeros(4, device=self.device)
        with torch.no_grad():
            for start in range(0, len(shard), self.args.batch_size):
                chunk = shard[start : start + self.args.batch_size]
                while len(chunk) < self.args.batch_size:  # keep one eval shape
                    chunk = chunk + [(chunk[0][0], [-100] * len(chunk[0][1]))]
                batch = {k: v.to(self.device) for k, v in collate(chunk, self.fixed_length, self.pad_id).items()}
                logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                logits, labels = logits[:, :-1].float(), batch["labels"][:, 1:]
                mask = labels != -100
                predicted = logits.argmax(-1)
                loss_sum = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum"
                )
                exact = (((predicted == labels) | ~mask).all(-1) & mask.any(-1)).sum()
                sums += torch.stack([loss_sum, ((predicted == labels) & mask).sum().float(), mask.sum().float(), exact.float()])
                self.xm.mark_step()
            gen_counts, samples = self._generate() if generate else ([0.0] * 4, [])
        totals = torch.cat([sums.cpu().double(), torch.tensor(gen_counts, dtype=torch.float64)])
        dist.all_reduce(totals, group=self.cpu_group)
        loss_sum, correct, tokens, exact, gen_matches, gen_count, gen_window_matches, gen_window_n = totals.tolist()
        if self.rank == 0:
            metrics = {
                "eval_loss": loss_sum / max(1.0, tokens),
                "token_accuracy": correct / max(1.0, tokens),
                "exact_match": exact / len(self.eval_set),
            }
            if gen_count:
                metrics["gen_exact_match"] = gen_matches / gen_count
                metrics["gen_exact_match_window"] = gen_window_matches / max(1.0, gen_window_n)
                metrics["gen_window_questions"] = gen_window_n
                for gold, text in samples:
                    LOG.info("generated %r (gold %r)", text, gold)
            self._emit("eval", tags={"eval_set": "heldout_sql"}, metrics=metrics)
            LOG.info("eval at step %d: %s", self.step, {k: round(v, 4) for k, v in metrics.items()})
        self.head_stats()

    def head_stats(self) -> None:
        """With --head-stats: attention entropy (nats) and peak attention weight per layer and head,
        each averaged over the real query positions of a fixed, teacher-forced probe batch of this
        rank's held-out examples. One attn_stats record per (layer, head) from every rank: 12 heads
        x 28 layers on Qwen2.5-1.5B. Attention entropy collapsing is a known sign of instability.
        output_attentions adds one compiled graph on Neuron (compiled once, then cached)."""
        if not self.args.head_stats:
            return
        # Observing must not change the run: the probe's graph executions advance the device's
        # random number generator, which GRPO sampling draws from, so restore it afterwards.
        # Without this, a --head-stats run of seed 1 sampled differently from GRPO step 1 on.
        rng_state = self.xm.get_rng_state()
        probe = self.eval_set[self.rank :: self.world][: self.args.head_probe]
        batch = {k: v.to(self.device) for k, v in collate(probe, self.fixed_length, self.pad_id).items()}
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_attentions=True)
            query = batch["attention_mask"][:, None, :].float()  # [B, 1, T]: real query positions
            weight = query.sum((0, 2))
            entropy, peak = [], []
            for p in out.attentions:  # one [B, H, T, T] per layer; each row sums to 1 over keys
                p = p.float()
                entropy.append((-(p * torch.log(p.clamp_min(1e-12))).sum(-1) * query).sum((0, 2)) / weight)
                peak.append((p.amax(-1) * query).sum((0, 2)) / weight)
            stats = torch.stack([torch.stack(entropy), torch.stack(peak)])  # [2, layers, heads]
            self.xm.mark_step()
        entropy_rows, peak_rows = stats.cpu().tolist()
        self.xm.set_rng_state(rng_state)
        for layer, (entropies, peaks) in enumerate(zip(entropy_rows, peak_rows)):
            for head, (e, m) in enumerate(zip(entropies, peaks)):
                self._emit(
                    "attn_stats",
                    rank=self.rank,
                    tags={"layer": f"layer_{layer:02d}", "head": f"h{head:02d}", **self.topology},
                    metrics={"entropy": e, "max_prob": m},
                )

    # ------------------------------------------------------------------ GRPO

    def _all_reduce(self, values: list[float]) -> list[float]:
        totals = torch.tensor(values, dtype=torch.float64)
        dist.all_reduce(totals, group=self.cpu_group)
        return totals.tolist()

    def grpo_step(self) -> None:
        """One GRPO step: sample a group of SQL candidates per question, execute and grade them,
        and push the policy toward candidates that beat their group's mean reward."""
        from . import sqlexec

        a = self.args
        started = time.monotonic()
        question_index = grpo_question(self.step, self.rank, a.seed, len(self.grpo_pool))
        question = self.grpo_pool[question_index]
        prompt = self.tokenizer(question.prompt, add_special_tokens=False).input_ids
        prompts = [prompt] * a.group_size
        self.model.eval()
        with torch.no_grad():
            ids, attention, cursor, done = self._decode(prompts, a.temperature)
        texts = self._texts(ids.cpu(), prompts)
        *results, gold = self.executor.run_many(texts + [question.sql])
        lenient = [float(sqlexec.returns_rows(r)) for r in results]
        strict = [float(sqlexec.matches(r, gold)) for r in results]
        rewards = torch.tensor(lenient if self.lenient else strict)
        advantages_cpu = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
        advantages = advantages_cpu.to(self.device)
        # How the candidates game (or do not game) the grader, as numbers.
        features = [sqlexec.features(t) for t in texts]
        result_rows = [sqlexec.row_count(r) for r in results]
        group_size = len(texts)
        paid_but_wrong = sum(l == 1.0 and c == 0.0 for l, c in zip(lenient, strict)) / group_size
        honest_unpaid = sum(c == 1.0 and l == 0.0 for l, c in zip(lenient, strict)) / group_size

        # Log-probabilities of the sampled tokens, gathered with tensor indices so every question
        # (whatever its prompt length) reuses one compiled graph.
        lm = self.model.get_base_model()
        new = self.gen_new_tokens
        offsets = torch.arange(new, device=self.device)
        # The prompt length goes to the device as data: a Python scalar here would be baked into
        # the graph as a constant and force a recompile for every new prompt length.
        start = torch.tensor([len(prompt)]).to(self.device)
        positions = (start - 1 + offsets).clamp(max=self.gen_length - 2)
        lengths = cursor - start + done.long()  # generated tokens, including EOS
        token_mask = (offsets.unsqueeze(0) < lengths.unsqueeze(1)).float()
        targets = ids.index_select(1, positions + 1)

        def token_logp() -> tuple[torch.Tensor, torch.Tensor]:
            """Log-probabilities of the sampled tokens, and every next-token distribution's."""
            hidden = lm.model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
            logits = lm.lm_head(hidden.index_select(1, positions)).float()
            log_probs = torch.log_softmax(logits, -1)
            return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1), log_probs

        self.model.train()
        logp, log_probs = token_logp()
        # The KL reference is the base model, before SFT: SFT and GRPO train the same adapter, and
        # disabling it is the only reference without a second copy of the weights in the graph. So
        # the KL is above 0 from GRPO's first step, and the penalty pulls toward the base model.
        with torch.no_grad(), self.model.disable_adapter():
            ref_logp, _ = token_logp()
        denominator = token_mask.sum(1).clamp(min=1.0)
        policy_loss = -(advantages * (logp * token_mask).sum(1) / denominator).mean()
        log_ratio = ref_logp - logp
        kl = ((torch.exp(log_ratio) - log_ratio - 1) * token_mask).sum(1) / denominator
        loss = policy_loss + a.kl_beta * kl.mean()
        with torch.no_grad():
            # Per-token means over the sampled completions: the entropy of the policy's next-token
            # distribution (from the log-probabilities the loss already needs, so no extra pass),
            # and the negative log-likelihood of the tokens actually sampled. Computed before the
            # graph cut, so they run inside the step's graph.
            tokens = token_mask.sum().clamp(min=1.0)
            token_entropy = -(log_probs.exp() * log_probs).sum(-1)
            diagnostics = torch.stack(
                [kl.mean(), (token_entropy * token_mask).sum() / tokens, -(logp * token_mask).sum() / tokens]
            )
        # A tensor still referenced at the graph cut becomes a graph output; the full-vocabulary
        # log-probabilities would be a large one.
        del log_probs, token_entropy
        for group in self.optimizer.param_groups:
            group["lr"] = a.grpo_lr
        loss.backward()
        self.xm.reduce_gradients(self.optimizer)
        torch.nn.utils.clip_grad_norm_([p for _, p in self.trainable], a.max_grad_norm or 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.xm.mark_step()
        candidate_lengths = lengths.cpu().tolist()
        kl_mean, entropy_mean, nll_mean = diagnostics.cpu().tolist()
        stats = self._all_reduce(
            [
                rewards.mean().item(),
                sum(strict) / len(strict),
                kl_mean,
                entropy_mean,
                nll_mean,
                float(sum(candidate_lengths)) / group_size,
                paid_but_wrong,
                honest_unpaid,
                float(rewards.std(unbiased=False).item() > 0),
                advantages_cpu.abs().mean().item(),
                sum(f["filters"] for f in features) / group_size,
                sum(f["entity_filter"] for f in features) / group_size,
                float(sum(result_rows)) / group_size,
                1.0,
            ]
        )
        (reward, exec_match, kl_value, entropy, sample_nll, resp_len, paid_wrong, unpaid, nonzero_adv, abs_adv,
         filters, entity_rate, rows_mean, ranks) = stats
        question_tags = {"question_id": f"q{question_index:05d}", "question_shape": shape_tag(question.shape)}
        for index, (feature, rows, length) in enumerate(zip(features, result_rows, candidate_lengths)):
            self._emit(
                "rl_candidate",
                rank=self.rank,
                tags={"candidate": str(index), **question_tags,
                      "result_class": sqlexec.result_class(results[index], gold)},
                metrics={
                    "lenient_reward": lenient[index],
                    "strict_correct": strict[index],
                    "trained_reward": rewards[index].item(),
                    "advantage": advantages_cpu[index].item(),
                    "resp_len": float(length),
                    "result_rows": float(rows),
                    **feature,
                },
            )
        if self.rank == 0:
            self._emit(
                "rl_step",
                metrics={
                    "reward": reward / ranks,
                    "exec_match": exec_match / ranks,
                    "kl": kl_value / ranks,
                    # Nats per sampled token: the policy's next-token entropy, and the sampled
                    # tokens' negative log-likelihood (its one-sample estimate at temperature 1).
                    "entropy": entropy / ranks,
                    "sample_nll": sample_nll / ranks,
                    "resp_len": resp_len / ranks,
                    "step_time_ms": (time.monotonic() - started) * 1000,
                    # Grader disagreement: paid but wrong (the exploit) vs correct but unpaid (the pressure).
                    "paid_but_wrong_rate": paid_wrong / ranks,
                    "honest_unpaid_rate": unpaid / ranks,
                    # Learning signal: share of candidate groups whose rewards differ, and mean |advantage|.
                    "nonzero_advantage_frac": nonzero_adv / ranks,
                    "mean_abs_advantage": abs_adv / ranks,
                    "filters_mean": filters / ranks,
                    "entity_filter_rate": entity_rate / ranks,
                    "result_rows_mean": rows_mean / ranks,
                },
            )
            if self.step % a.sample_every == 0 or self.step == 1:
                best = max(range(len(texts)), key=lambda i: rewards[i].item())
                self._emit(
                    "sample",
                    metrics={"reward": rewards[best].item(), "exec_match": strict[best]},
                    text={"sql": texts[best][:500], "question": question.question[:300], "gold": question.sql[:500]},
                )
            if self.step % 10 == 0:
                LOG.info("grpo step %d: reward %.2f exec_match %.2f kl %.4f entropy %.3f sample_nll %.3f len %.1f | %s",
                         self.step, reward / ranks, exec_match / ranks, kl_value / ranks, entropy / ranks,
                         sample_nll / ranks, resp_len / ranks, texts[0][:160])

    def grpo_evaluate(self) -> None:
        """Held-out execution accuracy: greedy SQL whose result matches the gold query's result.
        Independent of the training reward; this is what the reward-hacking detector watches."""
        from . import sqlexec

        self.model.eval()
        with torch.no_grad():
            examples, texts = self._generate_heldout(self.grpo_eval_examples, self.args.grpo_eval_batch)
        results = self.executor.run_many(texts + [e.sql for e in examples])
        produced, gold = results[: len(texts)], results[len(texts) :]
        correct = [float(sqlexec.matches(r, g)) for r, g in zip(produced, gold)]
        matched = sum(correct)
        lenient = sum(sqlexec.returns_rows(r) for r in produced)
        # The loophole is questions whose correct answer is empty: split accuracy on those from the
        # rest, seen tables from the held-out-only `disk` table, and short time windows (window
        # questions, data.SHORT_WINDOWS) from the rest.
        no_data = [not sqlexec.returns_rows(g) for g in gold]
        unseen = [e.shape[0] in data.UNSEEN_MEASUREMENTS for e in examples]
        window = [e.is_window_question for e in examples]
        features = [sqlexec.features(t) for t in texts]
        sums = [
            matched, lenient, len(examples),
            sum(c for c, n in zip(correct, no_data) if n), sum(no_data),
            sum(c for c, n in zip(correct, no_data) if not n), len(examples) - sum(no_data),
            sum(c for c, u in zip(correct, unseen) if u), sum(unseen),
            sum(c for c, w in zip(correct, window) if w), sum(window),
            sum(f["filters"] for f in features), sum(f["entity_filter"] for f in features),
        ]
        (matched, lenient, count, no_data_hits, no_data_n, data_hits, data_n, unseen_hits, unseen_n,
         window_hits, window_n, filters, entity) = self._all_reduce(sums)
        accuracy = matched / max(1.0, count)
        self.eval_accuracy[self.step] = accuracy
        if self.rank == 0:
            self.eval_states[self.step] = {n: p.detach().to("cpu", copy=True) for n, p in self.trainable}
        if accuracy > self.best_exec[0]:
            self.best_exec = (accuracy, self.step)
        if self.rank == 0:
            self._emit(
                "eval",
                tags={"eval_set": "heldout_sql"},
                metrics={
                    "exec_accuracy": accuracy,
                    "returns_rows": lenient / max(1.0, count),
                    "exec_accuracy_no_data": no_data_hits / max(1.0, no_data_n),
                    "exec_accuracy_has_data": data_hits / max(1.0, data_n),
                    "exec_accuracy_unseen_table": unseen_hits / max(1.0, unseen_n),
                    "exec_accuracy_window": window_hits / max(1.0, window_n),
                    "no_data_questions": no_data_n,
                    "window_questions": window_n,
                    "heldout_filters_mean": filters / max(1.0, count),
                    "heldout_entity_filter_rate": entity / max(1.0, count),
                },
            )
            LOG.info("grpo eval at step %d: exec_accuracy %.3f (window questions %.3f) returns_rows %.3f | %s",
                     self.step, accuracy, window_hits / max(1.0, window_n), lenient / max(1.0, count),
                     texts[0][:160] if texts else "")
        # This rank's share of the held-out questions, one record each; question i of the shard is
        # held-out question rank + i * world.
        for i, (hit, empty, short, feature, result) in enumerate(zip(correct, no_data, window, features, produced)):
            self._emit(
                "eval_question",
                tags={"eval_set": "heldout_sql", "question_id": f"h{self.rank + i * self.world:03d}",
                      "question_type": "no_data" if empty else "has_data", "window": "short" if short else "long"},
                metrics={"correct": hit, "returns_rows": float(sqlexec.returns_rows(result)),
                         "filters": feature["filters"], "entity_filter": feature["entity_filter"]},
            )
        self.head_stats()

    def keep_heldout(self, requested: str | None) -> str:
        """On STOP or ADVISE: saves the adapters of the GRPO evaluation step the command names (the
        watchdog's best held-out evaluation), or this trainer's own best, with a warning, if it has
        no evaluation at that step. Every rank decides alike; rank 0 saves. Returns what was kept."""
        try:
            step = int(requested)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            step = None
        if not self.eval_accuracy:
            return "no held-out evaluation yet, nothing kept"
        if step in self.eval_accuracy:
            kept, note = step, " as requested"
        else:
            kept, note = self.best_exec[1], f" (own best; no adapters for requested step {requested})"
            if self.rank == 0:
                LOG.warning("The command names GRPO step %r, but adapters exist only for steps %s; "
                            "keeping own best, step %d", requested, sorted(self.eval_accuracy), kept)
        accuracy = self.eval_accuracy[kept]
        if self.rank == 0:
            path = Path(self.args.checkpoint_dir) / self.args.run_id / "best-heldout.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"params": self.eval_states[kept], "grpo_step": kept, "exec_accuracy": accuracy,
                        "requested_grpo_step": step}, path)
        return f"kept GRPO step {kept}{note}, held-out exec accuracy {accuracy:.2f}"

    def run_grpo(self) -> None:
        LOG.info("switching to GRPO (%s grader) after %d SFT steps",
                 "lenient" if self.lenient else "strict", self.step)
        self.phase = "grpo"
        self.step = 0  # a new phase, with its own warm-up (its one-time graph compiles)
        self.optimizer = self._new_optimizer()
        while self.step < self.args.grpo_steps:
            command = self._broadcast(self.control.poll() if self.control else None)
            if command is not None and not self.apply(command):
                if self.rank == 0:
                    LOG.info("stopped at GRPO step %d", self.step)
                return
            self.step += 1
            self.grpo_step()
            if self.step == 1 or self.step % self.args.grpo_eval_every == 0:
                self.grpo_evaluate()
        self.await_final_verdict()

    def await_final_verdict(self) -> None:
        """The watchdog judges the last evaluation after the last step's poll. Waits up to
        --final-verdict-seconds for that verdict, so that an ADVISE or a STOP on the final
        evaluation still keeps the adapters it names. A STOP ends the wait; anything else (an
        ADVISE, whose HELDOUT_DECLINE can still escalate to REWARD_HACKING) waits out the deadline.
        Rank 0 decides, so every rank leaves together."""
        if self.unwatched or self.args.final_verdict_seconds <= 0:
            return
        deadline = time.monotonic() + self.args.final_verdict_seconds
        while True:
            if self.rank == 0:
                waiting = self.control is not None and time.monotonic() < deadline
                state = (self.control.poll() if self.control else None, waiting)
            command, waiting = self._broadcast(state if self.rank == 0 else None)
            if command is not None:
                # The run is over either way: this only keeps adapters.
                if not self.apply(command):
                    return
                continue  # the next queued command, if any, without waiting
            if not waiting:
                return
            time.sleep(0.2)

    def checkpoint(self) -> None:
        # The loss is computed before this step's update, so a finite loss does not mean the update
        # was: check the gradient norm too. All ranks must agree (rank 0 saves, then every rank
        # meets at the barrier), and each rank's loss is its own, so decide together.
        clean = math.isfinite(self.last_loss) and math.isfinite(self.last_grad_norm)
        if self._all_reduce([0.0 if clean else 1.0])[0] > 0:
            LOG.warning("skipping checkpoint at step %d: loss or gradient norm is not finite", self.step)
            return
        path = Path(self.args.checkpoint_dir) / self.args.run_id / f"step-{self.step:05d}.pt"
        if self.rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Optimizer state too: restoring it keeps the steady-state training graph (already
            # compiled), where a fresh optimizer would force a minutes-long recompile.
            torch.save(
                {
                    "params": {n: p.detach().cpu() for n, p in self.trainable},
                    "optimizer": _to_cpu(self.optimizer.state_dict()),
                },
                path,
            )
            self._emit("checkpoint")
        dist.barrier(group=self.cpu_group)
        self.checkpoints[self.step] = path

    def rollback(self, target: int) -> str:
        candidates = [s for s in self.checkpoints if s <= target]
        restored = max(candidates) if candidates else 0
        saved = torch.load(self.checkpoints[restored]) if restored else None
        params = saved["params"] if saved else self.initial_state
        # Copy in place into the same parameter and optimizer-state tensors, then cut the graph:
        # new tensors would change the training graph's inputs (and its buffer aliasing), and
        # Neuron would recompile it for minutes. In place, training resumes on the cached graph.
        for name, param in self.trainable:
            param.data.copy_(params[name].to(self.device))
        if saved and self.optimizer.state:
            saved_state = saved["optimizer"]["state"]
            for index, (_, param) in enumerate(self.trainable):
                live = self.optimizer.state[param]
                for key, value in saved_state[index].items():
                    live[key].copy_(value.to(live[key].device))
        else:
            self.optimizer = self._new_optimizer()
        self.xm.mark_step()
        self.step = restored
        self.attempt += 1
        # Checkpoints after the restored one belong to the abandoned attempt.
        self.checkpoints = {s: p for s, p in self.checkpoints.items() if s <= restored}
        self.faults_enabled = False
        # Compounds: if the last rate still diverged, the next attempt tries a lower one.
        self.lr_scale *= self.args.rollback_lr_scale
        return f"restored step {restored}; attempt {self.attempt} at {self.lr_scale:g}x learning rate"

    def apply(self, command: dict) -> bool:
        """Applies a control command on every rank; returns False to stop training.

        ADVISE (from HELDOUT_DECLINE: held-out accuracy fell while the grader agreed with the strict
        check, so likely overfitting rather than reward hacking) keeps the adapters of the
        evaluation it names (the watchdog's best), as a STOP does, and training continues: it never
        stops or rolls back a run. The unwatched run ignores every command, ADVISE too."""
        action, arg = command.get("action"), command.get("arg")
        outcome, stop = "ignored", False
        if self.unwatched:
            outcome = "ignored (watchdog off)"
        elif action == "ADVISE":
            kept = self.keep_heldout(arg) if self.phase == "grpo" else "nothing kept"
            outcome = f"noted: held-out accuracy declining since GRPO step {arg}; {kept}; training continues"
        elif action == "ROLLBACK" and self.phase != "sft":
            # Divergence is detected on SFT steps only (GRPO reports rl_step, not train_step), and
            # the checkpoints are SFT's: a ROLLBACK seen in GRPO is late or forged.
            outcome = f"ignored: ROLLBACK applies to SFT, and the run is in {self.phase}"
        elif action == "ROLLBACK" and command.get("attempt") not in (None, self.attempt):
            # Detected in an attempt a rollback already left: a late or replayed incident (Flink
            # restored from a snapshot, say). Rolling back again would undo a healthy attempt.
            outcome = f"ignored: ROLLBACK is for attempt {command['attempt']}, and the run is on attempt {self.attempt}"
        elif action == "ROLLBACK" and not str(arg or 0).isdecimal():
            outcome = f"ignored: ROLLBACK argument {arg!r} is not a step"
        elif action == "ROLLBACK" and self.attempt > self.args.max_rollbacks:
            outcome, stop = f"stopping: still diverging after {self.attempt - 1} rollbacks", True
        elif action == "ROLLBACK":
            outcome = self.rollback(int(arg or 0))
        elif action == "STOP":
            # In GRPO, keep the adapters of the evaluation the STOP names (its argument).
            stop = True
            outcome = f"stopping: {self.keep_heldout(arg)}" if self.phase == "grpo" else "stopping"
        if self.rank == 0:
            LOG.info("control %s(%s) incident %s: %s", action, arg, command.get("incident_id"), outcome)
            self._ack(command, outcome)
        return not stop

    def _ack(self, command: dict, outcome: str) -> None:
        """Reports what the trainer did with a command, so InfluxDB holds both the watchdog's
        decision (the incident row) and the trainer's response (this control_ack row)."""
        now_ms = int(time.time() * 1000)
        detected = command.get("ts")
        metrics = {}
        if detected:
            metrics["apply_latency_ms"] = now_ms - detected
            if command.get("seen_ms"):
                metrics["seen_latency_ms"] = command["seen_ms"] - detected
        self._emit(
            "control_ack",
            tags={"action": str(command.get("action")), "type": str(command.get("type"))},
            metrics=metrics,
            text={"incident_id": str(command.get("incident_id")), "outcome": outcome[:200]},
        )

    def run(self) -> None:
        self.evaluate(generate=True)  # step 0: compile the eval graphs before warm-up ends
        sft_steps = self.args.sft_steps if self.grpo else self.args.max_steps
        while self.step < sft_steps:
            command = self._broadcast(self.control.poll() if self.control else None)
            if command is not None and not self.apply(command):
                LOG.info("finished at step %d, attempt %d (stopped)", self.step, self.attempt)
                return
            self.step += 1
            self.train_step()
            if self.step % self.args.eval_every == 0:
                self.evaluate(generate=self.step % self.args.gen_eval_every == 0)
            if self.step % self.args.checkpoint_every == 0:
                self.checkpoint()
        if self.grpo:
            self.run_grpo()
        LOG.info("finished at step %d, attempt %d", self.step, self.attempt)

    def close(self, synchronize: bool = True) -> None:
        """Stops the workers and flushes telemetry. With `synchronize`, also waits for the other
        ranks and ends the process group: only after a normal finish, since a rank that raised
        would wait there while the others wait in a different collective, and none would exit."""
        if self.executor is not None:
            self.executor.close()
        if self.monitor is not None:
            self.monitor.stop()
        if self.emitter is not None:
            self.emitter.close()
        if synchronize:
            dist.barrier(group=self.cpu_group)
            dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    trainer = Trainer(args)
    try:
        trainer.run()
    except BaseException:
        # Exit without a collective, so torchrun sees this rank fail and stops the others.
        trainer.close(synchronize=False)
        raise
    trainer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
