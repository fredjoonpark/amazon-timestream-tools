# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The control plane, read from InfluxDB 3: every OPEN incident Flink writes to the `incident`
table carries an action for the run it names. Each simulated run polls for its own run ID, as an
independent trainer would, so a fleet of N runs sends N queries per poll interval.

Same query and command shape as the Trainium trainer (trainer/watchdog_trainer/telemetry.py).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

LOG = logging.getLogger("watchdog_sim")

# The run ID is a query parameter; `since` is an integer this module formats itself.
CONTROL_QUERY = (
    "SELECT incident_id, type, action, action_arg, attempt, step, onset_step, message, "
    "arrow_cast(time, 'Int64') AS time_ns FROM incident "
    "WHERE run_id = $run_id AND state = 'OPEN' AND time >= to_timestamp_millis({since}) ORDER BY time"
)


def incident_command(run_id: str, row: dict, seen_ms: int) -> dict:
    """An incident row as the command dict a run applies."""
    return {
        "run_id": run_id,
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


def ack_event(run, command: dict, outcome: str, now_ms: int) -> dict:
    """What a run did with a command, as `control_ack` telemetry (same shape as the trainer's)."""
    metrics = {}
    if command.get("ts"):
        metrics["apply_latency_ms"] = float(now_ms - command["ts"])
        if command.get("seen_ms"):
            metrics["seen_latency_ms"] = float(command["seen_ms"] - command["ts"])
    event = {
        "type": "control_ack", "ts": now_ms, "run_id": run.run_id, "scenario": run.scenario,
        "phase": run.phase, "attempt": run.attempt, "step": run.step,
        "tags": {"action": str(command.get("action")), "type": str(command.get("type"))},
        "text": {"incident_id": str(command.get("incident_id")), "outcome": outcome[:200]},
    }
    if metrics:
        event["metrics"] = metrics
    return event


class InfluxControlPlane(threading.Thread):
    """Polls InfluxDB for every active run once per `interval`, one query per run, and queues each
    newly seen incident once in `commands`. `run_ids` returns the runs to poll for.

    Like a trainer's poller, a run's query reads incidents from `lookback_ms` before the poller
    first saw the run, so a fleet that runs for days doesn't scan back to when it started. A run
    that leaves `run_ids` (it finished) drops its window and the incidents it has seen.

    Every `stats_seconds`, `on_stats` gets the fleet's poll count, errors, rows, and client-side
    latency, reported as `control_poll` telemetry (the dashboard's read-side load)."""

    def __init__(
        self,
        endpoint: str,
        database: str,
        token: str,
        run_ids: Callable[[], list[str]],
        lookback_ms: int = 60_000,
        interval: float = 1.0,
        workers: int = 16,
        stats_seconds: float = 10.0,
        on_stats: Callable[[dict], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(name="control-plane", daemon=True)
        import urllib3

        self.http = urllib3.PoolManager(num_pools=2, maxsize=workers, retries=False,
                                        timeout=urllib3.Timeout(connect=3.0, read=5.0))
        self.url = endpoint.rstrip("/") + "/api/v3/query_sql"
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}
        self.database = database
        self.lookback_ms = lookback_ms
        self.clock = clock
        self.run_ids = run_ids
        self.interval = interval
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="control-poll")
        self.stats_seconds = stats_seconds
        self.on_stats = on_stats
        self.commands: deque[dict] = deque()
        self.since: dict[str, int] = {}  # per active run: the start of its query's window
        self.seen: dict[str, set[str]] = {}  # per active run: the incidents already queued
        self.lock = threading.Lock()
        self._latencies: list[float] = []
        self._polls = self._errors = self._rows = 0
        self._last_error_log = 0.0

    def _fetch(self, run_id: str, since_ms: int) -> list[dict]:
        query = CONTROL_QUERY.format(since=int(since_ms))
        body = json.dumps({"db": self.database, "q": query, "params": {"run_id": run_id}, "format": "json"})
        response = self.http.request("POST", self.url, body=body.encode(), headers=self.headers)
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.data[:300].decode(errors='replace')}")
        return json.loads(response.data) if response.data.strip() else []

    def _poll_run(self, run_id: str, since_ms: int) -> None:
        started = time.monotonic()
        try:
            rows = self._fetch(run_id, since_ms)
        except Exception as error:  # noqa: BLE001 - keep polling through transient errors
            with self.lock:
                self._polls += 1
                self._errors += 1
                if time.monotonic() - self._last_error_log > 60:
                    LOG.warning("control poll failed: %s", error)
                    self._last_error_log = time.monotonic()
            return
        latency = (time.monotonic() - started) * 1000
        now_ms = int(time.time() * 1000)
        with self.lock:
            self._polls += 1
            self._rows += len(rows)
            self._latencies.append(latency)
            seen = self.seen[run_id]
            for row in rows if isinstance(rows, list) else []:
                incident_id = row.get("incident_id") if isinstance(row, dict) else None
                if not incident_id or incident_id in seen:
                    continue
                # Seen either way: a row that can't become a command is logged once, not every poll.
                seen.add(incident_id)
                try:
                    self.commands.append(incident_command(run_id, row, now_ms))
                except Exception:  # noqa: BLE001 - one bad row must not stop the poller
                    self._errors += 1
                    LOG.exception("control poll: skipping incident %s, which is not a valid command", incident_id)

    def poll_once(self) -> None:
        run_ids = list(self.run_ids())
        start = int(self.clock() * 1000) - self.lookback_ms
        with self.lock:
            self.since = {r: self.since.get(r, start) for r in run_ids}
            self.seen = {r: self.seen.get(r, set()) for r in run_ids}
            since = [self.since[r] for r in run_ids]
        list(self.pool.map(self._poll_run, run_ids, since))

    def stats(self) -> dict:
        with self.lock:
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
            except Exception:  # noqa: BLE001 - a dead poller would leave every run unwatched
                with self.lock:
                    self._errors += 1
                LOG.exception("control poll failed unexpectedly")
            if self.on_stats is not None and time.monotonic() >= next_stats:
                next_stats += self.stats_seconds
                try:
                    self.on_stats(self.stats())
                except Exception:  # noqa: BLE001 - stats must never stop polling
                    LOG.exception("control poll stats failed")
            elapsed = time.monotonic() - started
            if elapsed > self.interval * 2:
                LOG.warning("control polls took %.2f s for one interval; the fleet is too large for it", elapsed)
            time.sleep(max(0.0, self.interval - elapsed))
