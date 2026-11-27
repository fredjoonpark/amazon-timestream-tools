# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Telemetry events in the watchdog wire format, a batching Kinesis emitter, and the control
poller that reads the watchdog's commands from InfluxDB 3. Same contract as the simulator (see
simulator/watchdog_sim)."""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from collections import deque
from typing import Callable

LOG = logging.getLogger(__name__)

# Non-finite values are not valid JSON; report them as this sentinel so divergence stays visible.
NONFINITE_SENTINEL = 1e6


def clean_metrics(metrics: dict[str, float]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for name, value in metrics.items():
        value = float(value)
        cleaned[name] = value if math.isfinite(value) else NONFINITE_SENTINEL
    return cleaned


def event(
    kind: str,
    *,
    run_id: str,
    scenario: str,
    attempt: int,
    step: int,
    metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    ts: int | None = None,
    phase: str = "sft",
    text: dict[str, str] | None = None,
) -> dict:
    record = {
        "type": kind,
        "ts": ts if ts is not None else int(time.time() * 1000),
        "run_id": run_id,
        "scenario": scenario,
        "phase": phase,
        "attempt": attempt,
        "step": step,
    }
    if rank is not None:
        record["rank"] = rank
    if world_size is not None:
        record["world_size"] = world_size
    if tags:
        record["tags"] = tags
    if metrics:
        record["metrics"] = clean_metrics(metrics)
    if text:
        record["text"] = text  # string fields, e.g. a sampled SQL query
    return record


# Events a run's detector reads in order (the reward-hacking detector joins each eval with the RL
# steps before it): keyed by run alone, so one run's events share a shard and keep their order.
RUN_ORDERED_TYPES = ("rl_step", "eval")


def partition_key(record: dict) -> str:
    """The Kinesis partition key for an event."""
    if record["type"] in RUN_ORDERED_TYPES:
        return record["run_id"][:256]
    # Include the tags so high-volume streams (one point per layer and module per step) spread
    # across shards; a key per rank alone would put a run on one or two shards.
    return "|".join([record["run_id"], str(record.get("rank", -1)), *sorted((record.get("tags") or {}).values())])[:256]


def put_batches(events: list[dict]):
    """PutRecords calls for `events`, in order, up to 500 records each. A call holds at most one
    record per RUN_ORDERED_TYPES key: PutRecords doesn't keep a call's records in order (a throttled
    record is resent after the rest of its call went through), so a run's rl_step and eval go in
    successive calls, and each call completes, retries included, before the next."""
    batch: list[dict] = []
    ordered: set[str] = set()
    for e in events:
        key = partition_key(e)
        is_ordered = e["type"] in RUN_ORDERED_TYPES
        if len(batch) == 500 or (is_ordered and key in ordered):
            yield batch
            batch, ordered = [], set()
        batch.append({"Data": json.dumps(e, separators=(",", ":"), allow_nan=False).encode(), "PartitionKey": key})
        if is_ordered:
            ordered.add(key)
    if batch:
        yield batch


class KinesisEmitter:
    """Queues events and writes them to Kinesis from a background thread in batches."""

    def __init__(self, stream: str, region: str | None, flush_seconds: float = 0.5) -> None:
        import boto3

        self.client = boto3.client("kinesis", region_name=region)
        self.stream = stream
        self.flush_seconds = flush_seconds
        self.queue: queue.Queue[dict | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="kinesis-emitter", daemon=True)
        self.thread.start()

    def emit(self, record: dict) -> None:
        self.queue.put(record)

    def close(self, timeout: float = 60.0) -> None:
        """Sends what is queued, waiting up to `timeout`: a throttled call can retry for ~30 s, and
        the queue's tail is the run's last eval, rl_step, and command acknowledgement."""
        self.queue.put(None)
        self.thread.join(timeout)
        if self.thread.is_alive():
            # A daemon thread: what it hasn't sent is lost when the process exits.
            LOG.error("telemetry still sending after %.0f s; dropping the batch in flight and %d queued records",
                      timeout, max(0, self.queue.qsize() - 1))

    def _run(self) -> None:
        stopping = False
        while not stopping:
            batch: list[dict] = []
            deadline = time.monotonic() + self.flush_seconds
            while len(batch) < 500:
                try:
                    item = self.queue.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if item is None:
                    stopping = True
                    break
                batch.append(item)
            if batch:
                self._put(batch)

    def _put(self, batch: list[dict]) -> None:
        for records in put_batches(batch):
            self._put_call(records)

    def _put_call(self, records: list[dict]) -> None:
        for attempt in range(10):
            try:
                response = self.client.put_records(StreamName=self.stream, Records=records)
            except Exception:  # noqa: BLE001 - telemetry must never stop training
                # The call may have gone through: resending can duplicate records, which Flink
                # drops (delivery is at least once), but never loses them.
                LOG.exception("put_records failed")
                time.sleep(min(5.0, 0.2 * 2**attempt))
                continue
            if not response.get("FailedRecordCount"):
                return
            failed = [(r, res["ErrorCode"]) for r, res in zip(records, response["Records"]) if "ErrorCode" in res]
            LOG.warning("put_records: %d of %d records failed (%s); retrying", len(failed), len(records),
                        ", ".join(sorted({code for _, code in failed})))
            records = [r for r, _ in failed]
            time.sleep(min(5.0, 0.2 * 2**attempt))
        LOG.error("dropped %d telemetry records after retries", len(records))


# Every OPEN incident Flink writes is a command: its action and argument are columns of the row.
# The run ID is a query parameter; `since` is an integer this module formats itself.
CONTROL_QUERY = (
    "SELECT incident_id, type, action, action_arg, attempt, step, onset_step, message, "
    "arrow_cast(time, 'Int64') AS time_ns FROM incident "
    "WHERE run_id = $run_id AND state = 'OPEN' AND time >= to_timestamp_millis({since}) ORDER BY time"
)


def read_token(secret_arn: str, region: str | None) -> str:
    """The read-only database token from its Secrets Manager secret ({"token": ...})."""
    import boto3

    secret = boto3.client("secretsmanager", region_name=region).get_secret_value(SecretId=secret_arn)["SecretString"]
    return json.loads(secret)["token"]


class ControlAuthError(RuntimeError):
    """InfluxDB rejected the read-only token (HTTP 401 or 403): retrying won't help."""


def incident_command(row: dict, seen_ms: int) -> dict:
    """An incident row as the command dict the trainer applies."""
    return {
        "incident_id": row["incident_id"],
        "type": row.get("type"),
        "action": row.get("action"),
        "arg": row.get("action_arg") or None,
        "attempt": row.get("attempt"),  # the attempt Flink detected the failure in
        "step": row.get("step"),
        "onset_step": row.get("onset_step"),
        "message": row.get("message"),
        "ts": int(row["time_ns"]) // 1_000_000,  # detection time (the triggering event's time)
        "seen_ms": seen_ms,
    }


class InfluxControlPoller(threading.Thread):
    """Polls InfluxDB 3 for the incidents the watchdog opened for one run, and queues each one
    once as a command. A command is state, not a message: every poll rereads all OPEN incidents
    since `since_ms`, so a command opened while a poll failed still arrives. The trainer sets
    `since_ms` to a minute before it started, so a restarted trainer misses incidents older than
    that, such as an earlier STOP (README, Production considerations).

    Every `stats_seconds`, `on_stats` gets the poll count, errors, rows, and client-side latency,
    which the trainer reports as `control_poll` telemetry (the dashboard's read-side load)."""

    def __init__(
        self,
        endpoint: str,
        database: str,
        token: str,
        run_id: str,
        since_ms: int,
        interval: float = 1.0,
        stats_seconds: float = 10.0,
        on_stats: Callable[[dict], None] | None = None,
    ) -> None:
        super().__init__(name="control-poller", daemon=True)
        import urllib3

        self.http = urllib3.PoolManager(num_pools=2, maxsize=2, retries=False, timeout=urllib3.Timeout(connect=3.0, read=5.0))
        self.url = endpoint.rstrip("/") + "/api/v3/query_sql"
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}
        self.body = json.dumps(
            {"db": database, "q": CONTROL_QUERY.format(since=int(since_ms)), "params": {"run_id": run_id}, "format": "json"}
        ).encode()
        self.interval = interval
        self.stats_seconds = stats_seconds
        self.on_stats = on_stats
        self.commands: deque[dict] = deque()
        self.seen: set[str] = set()
        self.ready = threading.Event()
        self._latencies: list[float] = []
        self._polls = self._errors = self._rows = 0
        self._last_error_log = 0.0
        self.last_error: Exception | None = None  # the latest poll's failure; None once one succeeds

    def query(self) -> list[dict]:
        response = self.http.request("POST", self.url, body=self.body, headers=self.headers)
        if response.status != 200:
            error = ControlAuthError if response.status in (401, 403) else RuntimeError
            raise error(f"HTTP {response.status}: {response.data[:300].decode(errors='replace')}")
        rows = json.loads(response.data) if response.data.strip() else []
        if not isinstance(rows, list):
            raise RuntimeError(f"unexpected response: {str(rows)[:300]}")
        return rows

    def poll_once(self) -> None:
        started = time.monotonic()
        try:
            rows = self.query()
        except Exception as error:  # noqa: BLE001 - keep polling through transient errors
            self._errors += 1
            self.last_error = error
            if time.monotonic() - self._last_error_log > 60:
                LOG.warning("control poll failed: %s", error)
                self._last_error_log = time.monotonic()
            return
        finally:
            self._polls += 1
        self.last_error = None
        self._latencies.append((time.monotonic() - started) * 1000)
        self._rows += len(rows)
        now_ms = int(time.time() * 1000)
        for row in rows:
            incident_id = row.get("incident_id") if isinstance(row, dict) else None
            if not incident_id or incident_id in self.seen:
                continue
            # Seen either way: a row that can't become a command is logged once, not every poll.
            self.seen.add(incident_id)
            try:
                self.commands.append(incident_command(row, now_ms))
            except Exception:  # noqa: BLE001 - one bad row must not stop the poller
                self._errors += 1
                LOG.exception("control poll: skipping incident %s, which is not a valid command", incident_id)

    def stats(self) -> dict:
        latencies = sorted(self._latencies)
        result = {
            "polls": float(self._polls),
            "errors": float(self._errors),
            "rows": float(self._rows),
            "latency_p50_ms": latencies[len(latencies) // 2] if latencies else 0.0,
            "latency_max_ms": latencies[-1] if latencies else 0.0,
        }
        self._latencies, self._polls, self._errors, self._rows = [], 0, 0, 0
        return result

    def run(self) -> None:
        next_stats = time.monotonic() + self.stats_seconds
        while True:
            started = time.monotonic()
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 - a dead poller would leave the run unwatched
                self._errors += 1
                LOG.exception("control poll failed unexpectedly")
            self.ready.set()
            if self.on_stats is not None and time.monotonic() >= next_stats:
                next_stats += self.stats_seconds
                try:
                    self.on_stats(self.stats())
                except Exception:  # noqa: BLE001 - stats must never stop polling
                    LOG.exception("control poll stats failed")
            time.sleep(max(0.0, self.interval - (time.monotonic() - started)))

    def poll(self) -> dict | None:
        return self.commands.popleft() if self.commands else None
