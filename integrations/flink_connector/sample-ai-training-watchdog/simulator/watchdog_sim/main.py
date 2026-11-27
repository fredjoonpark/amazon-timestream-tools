# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Runs the demo scenarios continuously, or writes a deterministic fixture file.

  python -m watchdog_sim.main --sink kinesis      # live: TELEMETRY_STREAM; control from INFLUX_URL
  python -m watchdog_sim.main --sink kinesis --telemetry realistic --ranks 2 --step-seconds 0.5
  python -m watchdog_sim.main --sink kinesis --runs 10 --senders 3   # 3 concurrent PutRecords calls
  python -m watchdog_sim.main --sink file --out events.jsonl.gz --steps 240

--telemetry light (default) is the Flink test fixture's format; realistic writes per-rank,
per-LoRA-matrix layer_stats every step, like the Trainium trainer (see watchdog_sim.scenarios).

A fleet can be split over several ECS tasks (SIM_TASKS, set by the stack's -c simRunsPerTask):
each task then tags its run IDs with its ECS task ID, so tasks never emit the same run.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import secrets
import sys
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .control import InfluxControlPlane, ack_event
from .scenarios import DEFAULT_RANKS, LIGHT, SCENARIOS, TELEMETRY_MODES, Run

LOG = logging.getLogger("watchdog_sim")

# Fixed start for fixture files so they are reproducible.
FIXTURE_EPOCH_MS = 1_767_225_600_000  # 2026-01-01T00:00:00Z
RESTART_DELAY_STEPS = 30
# Steps per fleet run before it restarts under a new ID (SIM_MAX_STEPS); unset: Run's default.
MAX_STEPS = int(os.environ.get("SIM_MAX_STEPS") or 0)
# Staggered starts spread runs over this many steps, so a large fleet's faults do not align.
STAGGER_STEPS = 120
# A one-shot run reads commands this long after its last step: about Flink's 5 s event-time
# watermark, the sink's 1 s buffer, InfluxDB's WAL flush and replication, and a poll, with margin.
ONCE_LINGER_SECONDS = 30.0
# Read by Flink's per-run reward-hacking detector in order, so they share one partition key per
# run (one shard); the trainer keys them the same way.
ORDERED_TYPES = frozenset({"rl_step", "eval"})
# Tries per PutRecords call, backing off from 0.2 s to 5 s between them, as the trainer does.
PUT_ATTEMPTS = 10


def partition_key(event: dict) -> str:
    """The run ID for ORDERED_TYPES; otherwise run, rank, and tag values, as the trainer's emitter
    keys them, so high-volume streams (a point per layer and module) spread across shards."""
    if event["type"] in ORDERED_TYPES:
        return event["run_id"][:256]
    tags = sorted((event.get("tags") or {}).values())
    return "|".join([event["run_id"], str(event.get("rank", -1)), *tags])[:256]


def put_batches(events: list[dict]):
    """PutRecords calls for `events`, in order, up to 500 records each. A call holds at most one
    record per ORDERED_TYPES key: PutRecords does not keep a call's records in order (a throttled
    record is resent after the rest of its call went through), so a run's rl_step and eval of one
    step go in successive calls, and each call completes, retries included, before the next."""
    batch: list[dict] = []
    ordered: set[str] = set()
    for e in events:
        key = partition_key(e)
        is_ordered = e["type"] in ORDERED_TYPES
        if len(batch) == 500 or (is_ordered and key in ordered):
            yield batch
            batch, ordered = [], set()
        batch.append({"Data": json.dumps(e, separators=(",", ":")).encode(), "PartitionKey": key})
        if is_ordered:
            ordered.add(key)
    if batch:
        yield batch


def lanes(events: list[dict], count: int) -> list[list[dict]]:
    """Splits one write's events into at most `count` lanes, each run's events in a single lane and
    in order. Runs go to the least-loaded lane, largest first, so an SFT run's hundreds of
    layer_stats records and a GRPO run's few spread evenly."""
    by_run: dict[str, list[dict]] = {}
    for e in events:
        by_run.setdefault(e["run_id"], []).append(e)
    result: list[list[dict]] = [[] for _ in range(max(1, count))]
    for run_events in sorted(by_run.values(), key=len, reverse=True):
        min(result, key=len).extend(run_events)
    return [lane for lane in result if lane]


class KinesisSink:
    """Writes events with PutRecords. With `senders` > 1, a write goes out over that many threads
    at once (`lanes`): one run's events always travel in one lane, sequentially, and a write returns
    only when every lane is done, so a run's next tick never overtakes this one."""

    def __init__(self, stream: str, region: str | None, senders: int = 1, client=None) -> None:
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client("kinesis", region_name=region,
                                  config=Config(max_pool_connections=max(10, senders)))
        self.client = client
        self.stream = stream
        self.senders = max(1, senders)
        self.pool = ThreadPoolExecutor(self.senders, thread_name_prefix="kinesis-put") if self.senders > 1 else None

    def write(self, events: list[dict]) -> None:
        if self.pool is None:
            self._put(events)
            return
        for future in [self.pool.submit(self._put, lane) for lane in lanes(events, self.senders)]:
            future.result()

    def _put(self, events: list[dict]) -> None:
        for records in put_batches(events):
            for attempt in range(PUT_ATTEMPTS):
                response = self.client.put_records(StreamName=self.stream, Records=records)
                if not response.get("FailedRecordCount"):
                    break
                failed = [(r, res["ErrorCode"]) for r, res in zip(records, response["Records"]) if "ErrorCode" in res]
                codes = ", ".join(sorted({code for _, code in failed}))
                LOG.warning("put_records: %d of %d records failed (%s); retrying", len(failed), len(records), codes)
                records = [r for r, _ in failed]
                time.sleep(min(5.0, 0.2 * 2**attempt))
            else:
                # Fail loudly (ECS restarts the task) rather than drop records and skew a load test.
                raise RuntimeError(f"put_records: {len(records)} records still failing after {PUT_ATTEMPTS} attempts ({codes})")


class FileSink:
    def __init__(self, path: str) -> None:
        opener = gzip.open if path.endswith(".gz") else open
        self.file = opener(path, "wt")

    def write(self, events: list[dict]) -> None:
        for e in sorted(events, key=lambda e: e["ts"]):
            self.file.write(json.dumps(e, separators=(",", ":"), sort_keys=True) + "\n")

    def close(self) -> None:
        self.file.close()


def ecs_task_id() -> str | None:
    """This ECS task's ID (the end of its ARN), from the task metadata endpoint; None outside ECS
    or when the endpoint does not answer."""
    uri = os.environ.get("ECS_CONTAINER_METADATA_URI_V4")
    if not uri:
        return None
    try:
        with urllib.request.urlopen(uri.rstrip("/") + "/task", timeout=2) as response:
            return json.load(response)["TaskARN"].rsplit("/", 1)[-1] or None
    except Exception as error:  # noqa: BLE001 - fall back to a random tag
        LOG.warning("could not read the ECS task ID (%s); using a random run ID tag", error)
        return None


def fleet_prefix(now: float, tasks: int = 1, task_id: str | None = None) -> str:
    """A fleet's run ID prefix: its start time in seconds, as 8 hex digits, so a restarted
    simulator never reuses run IDs. One of several tasks (`tasks` > 1) adds 6 characters of its ECS task ID, or 6
    random hex digits without one: a service starts all its tasks within the same few seconds."""
    prefix = f"{int(now):08x}-"
    if tasks > 1:
        prefix += f"{(task_id or '')[:6] or secrets.token_hex(3)}-"
    return prefix


class Fleet:
    """`size` concurrent runs, scenarios assigned round-robin. Beyond the first one-of-each, runs
    start staggered so their faults do not all fire on the same second; a finished run is
    replaced by a fresh one of the same scenario after RESTART_DELAY_STEPS."""

    def __init__(self, size: int, stagger_steps: int = STAGGER_STEPS, prefix: str | None = None,
                 telemetry: str = LIGHT, ranks: int | None = None) -> None:
        self.prefix = fleet_prefix(time.time()) if prefix is None else prefix
        self.telemetry = telemetry
        self.ranks = ranks
        self.scenarios = [SCENARIOS[i % len(SCENARIOS)] for i in range(size)]
        self.generation: dict[str, int] = {}
        self.runs: list[Run | None] = [None] * size
        # Slot i starts after a delay; the first len(SCENARIOS) slots start immediately.
        self.wait = [0 if i < len(SCENARIOS) else (i * 7) % stagger_steps + 1 for i in range(size)]

    def _new_run(self, scenario: str) -> Run:
        self.generation[scenario] = self.generation.get(scenario, 0) + 1
        return Run(scenario, self.generation[scenario], prefix=self.prefix, telemetry=self.telemetry,
                   world_size=self.ranks, **({"max_steps": MAX_STEPS} if MAX_STEPS else {}))

    def on_command(self, command: dict) -> tuple[Run, str | None] | None:
        run = next((r for r in self.runs if r is not None and r.run_id == command.get("run_id")), None)
        if run is None:
            return None
        if run.done:
            # Polled while it ran, applied after it finished (its slot hasn't restarted yet).
            return run, f"ignored: the run had already finished at step {run.step}"
        return run, run.on_command(command)

    def active_run_ids(self) -> list[str]:
        return [r.run_id for r in list(self.runs) if r is not None and not r.done]

    def tick(self, now: int) -> list[dict]:
        events: list[dict] = []
        for slot, run in enumerate(self.runs):
            if run is None or run.done:
                self.wait[slot] -= 1
                if self.wait[slot] <= 0:
                    self.runs[slot] = self._new_run(self.scenarios[slot])
                    self.wait[slot] = RESTART_DELAY_STEPS
                    LOG.debug("starting %s", self.runs[slot].run_id)
                continue
            events.extend(run.tick(now))
        return events


def drain_control(control, apply) -> list[dict]:
    """Applies queued commands with `apply(command) -> (run, outcome) | None`; returns the
    control_ack events, plus any control_poll stats the poller queued."""
    events: list[dict] = []
    if control is None:
        return events
    while control.commands:
        command = control.commands.popleft()
        applied = apply(command)
        if applied is not None:
            run, outcome = applied
            LOG.info("control %s(%s) incident %s: %s", command.get("action"), command.get("arg"),
                     command.get("incident_id"), outcome or "ignored")
            events.append(ack_event(run, command, outcome or "ignored", int(time.time() * 1000)))
    while getattr(control, "events", None):
        events.append(control.events.popleft())
    return events


def run_live(sink: KinesisSink, control, step_seconds: float, size: int, fleet: Fleet | None = None) -> None:
    fleet = fleet or Fleet(size)
    LOG.info("fleet of %d concurrent runs", size)
    while True:
        started = time.monotonic()
        now = int(time.time() * 1000)
        events = drain_control(control, fleet.on_command)
        events.extend(fleet.tick(now))
        if events:
            sink.write(events)
        elapsed = time.monotonic() - started
        if elapsed > step_seconds:
            LOG.warning("tick took %.2f s for %d events; the fleet is too large for the step interval", elapsed, len(events))
        time.sleep(max(0.0, step_seconds - elapsed))


def run_once(sink: KinesisSink, control, step_seconds: float, scenario: str, prefix: str,
             run: Run | None = None, linger_seconds: float = ONCE_LINGER_SECONDS) -> None:
    """One run of one scenario from step 0, reacting to the watchdog's commands, then exit. Logs the
    same control and finish lines as the Trainium trainer, so scripts/demo.sh can check it.

    After the last step it keeps reading commands for `linger_seconds`: Flink can open an incident
    from the run's last telemetry after the run has ended, and the log (and so scripts/demo.sh)
    should still show it. Such commands are acknowledged and not applied."""
    run = run or Run(scenario, 1, prefix=prefix)
    LOG.info("started %s (scenario %s, simulated)", run.run_id, scenario)

    def apply(command: dict):
        if command.get("run_id") != run.run_id:
            return None
        if run.done:
            return run, f"ignored: the run had already finished at step {run.step}"
        return run, run.on_command(command)

    while not run.done:
        started = time.monotonic()
        events = drain_control(control, apply)
        events.extend(run.tick(int(time.time() * 1000)))
        if events:
            sink.write(events)
        time.sleep(max(0.0, step_seconds - (time.monotonic() - started)))
    events = drain_control(control, apply)  # acks and stats still queued when the run ended
    if events:
        sink.write(events)
    LOG.info("finished at step %d, attempt %d", run.step, run.attempt)
    if control is None:
        return
    for _ in range(math.ceil(linger_seconds)):
        time.sleep(1.0)
        events = drain_control(control, apply)
        if events:
            sink.write(events)


def control_plane(run_ids, prefix: str) -> InfluxControlPlane | None:
    """The InfluxDB control poller, if INFLUX_URL and INFLUX_TOKEN are set. Its poll stats become
    one `control_poll` event per 10 s for the whole process (run ID `<prefix>fleet`)."""
    url, token = os.environ.get("INFLUX_URL"), os.environ.get("INFLUX_TOKEN")
    if not url or not token:
        LOG.warning("INFLUX_URL or INFLUX_TOKEN not set; runs will not receive the watchdog's commands")
        return None
    control = InfluxControlPlane(
        url, os.environ.get("INFLUX_DB", "watchdog"), token, run_ids,
        interval=float(os.environ.get("CONTROL_POLL_SECONDS", "1.0")),
        workers=int(os.environ.get("CONTROL_POLL_WORKERS", "16")),
    )
    control.events = deque()
    fleet_id = f"{prefix}fleet"

    def on_stats(stats: dict) -> None:
        control.events.append({
            "type": "control_poll", "ts": int(time.time() * 1000), "run_id": fleet_id, "scenario": "fleet",
            "phase": "control", "attempt": 1, "step": 0, "tags": {"endpoint": "reader"}, "metrics": stats,
        })

    control.on_stats = on_stats
    control.start()
    return control


def write_fixture(path: str, steps: int, step_seconds: float, telemetry: str = LIGHT,
                  ranks: int | None = None) -> int:
    sink = FileSink(path)
    runs = [Run(s, 1, max_steps=steps, telemetry=telemetry, world_size=ranks) for s in SCENARIOS]
    count = 0
    for i in range(steps):
        now = FIXTURE_EPOCH_MS + int(i * step_seconds * 1000)
        events = [e for run in runs for e in run.tick(now)]
        sink.write(events)
        count += len(events)
    sink.close()
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sink", choices=("kinesis", "file"), default="kinesis")
    parser.add_argument("--out", help="output path for --sink file (.gz compresses)")
    parser.add_argument("--steps", type=int, default=240, help="steps per run for --sink file")
    parser.add_argument("--step-seconds", type=float, default=float(os.environ.get("STEP_SECONDS", "1.0")),
                        help="seconds per step (STEP_SECONDS); every run takes one step per interval, "
                             "so 0.5 doubles the event rate")
    parser.add_argument("--once", choices=SCENARIOS, help="run one scenario once from step 0, then exit")
    parser.add_argument("--run-prefix", default="", help="--once: run ID prefix, e.g. once-20261110180005-12345-")
    parser.add_argument("--runs", type=int, default=int(os.environ.get("SIM_RUNS", len(SCENARIOS))),
                        help="concurrent runs (SIM_RUNS); scenarios are assigned round-robin")
    parser.add_argument("--telemetry", choices=TELEMETRY_MODES, default=os.environ.get("WATCHDOG_SIM_TELEMETRY") or LIGHT,
                        help="light (default; the Flink test fixture's format) or realistic (per-rank, "
                             "per-LoRA-matrix layer_stats every step, like the trainer) "
                             "(WATCHDOG_SIM_TELEMETRY)")
    parser.add_argument("--ranks", type=int, default=os.environ.get("WATCHDOG_SIM_RANKS") or None,
                        help=f"ranks per SFT run (WATCHDOG_SIM_RANKS); default {DEFAULT_RANKS['realistic']} "
                             f"for realistic (a trn1.2xlarge), {DEFAULT_RANKS['light']} for light")
    parser.add_argument("--senders", type=int, default=int(os.environ.get("SIM_SENDERS") or 1),
                        help="concurrent PutRecords calls (SIM_SENDERS; default 1); each run's events "
                             "always go through one sender, in order")
    args = parser.parse_args(argv)
    if args.telemetry not in TELEMETRY_MODES:
        parser.error(f"--telemetry (WATCHDOG_SIM_TELEMETRY) must be one of {', '.join(TELEMETRY_MODES)}")
    if args.ranks is not None and args.ranks < 1:
        parser.error("--ranks (WATCHDOG_SIM_RANKS) must be at least 1")
    if args.senders < 1:
        parser.error("--senders (SIM_SENDERS) must be at least 1")
    # Tasks of one simulator service (the stack's -c simRunsPerTask); each runs its own fleet.
    tasks = int(os.environ.get("SIM_TASKS") or 1)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ranks = args.ranks or DEFAULT_RANKS[args.telemetry]

    if args.sink == "file":
        if not args.out:
            parser.error("--out is required with --sink file")
        count = write_fixture(args.out, args.steps, args.step_seconds, args.telemetry, args.ranks)
        LOG.info("wrote %d events to %s (%s telemetry)", count, args.out, args.telemetry)
        return 0

    region = os.environ.get("AWS_REGION")
    telemetry = os.environ.get("TELEMETRY_STREAM")
    if not telemetry:
        parser.error("TELEMETRY_STREAM must be set")
    LOG.info("%s telemetry, %d ranks per SFT run, %.3g s per step, %d sender(s)", args.telemetry, ranks,
             args.step_seconds, args.senders)
    sink = KinesisSink(telemetry, region, senders=args.senders)
    if args.once:
        run = Run(args.once, 1, prefix=args.run_prefix, telemetry=args.telemetry, world_size=args.ranks)
        # Polls for the run until the process exits, after run_once's linger.
        control = control_plane(lambda: [run.run_id], args.run_prefix)
        run_once(sink, control, args.step_seconds, args.once, args.run_prefix, run=run)
        return 0
    prefix = fleet_prefix(time.time(), tasks, ecs_task_id() if tasks > 1 else None)
    fleet = Fleet(args.runs, prefix=prefix, telemetry=args.telemetry, ranks=args.ranks)
    control = control_plane(fleet.active_run_ids, fleet.prefix)
    LOG.info("streaming %d runs of %s to %s, run IDs <scenario>-%sNNN%s", args.runs, ", ".join(SCENARIOS),
             telemetry, prefix, f" (one of {tasks} tasks)" if tasks > 1 else "")
    run_live(sink, control, args.step_seconds, args.runs, fleet=fleet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
