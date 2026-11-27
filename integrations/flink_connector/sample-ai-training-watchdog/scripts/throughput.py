# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Summarizes the pipeline for the last N minutes, for the throughput test (README, Load):
Kinesis, Managed Flink, and the InfluxDB 3 connector from CloudWatch (GetMetricData); the control
plane from InfluxDB (control_poll, incident, control_ack, through scripts/influx-query.sh); and the
simulator's overload warnings from its logs. Read-only: describe, list, and get calls, and one SQL
query with the read-only token. Run it through scripts/throughput.sh.

Every row is one metric: AVG and MAX are over per-minute values, TOTAL is the window's count where
counting makes sense. Rows by kind:
  per-second rates (IncomingRecords, ...)  per-minute Sum / 60
  counts (WriteProvisionedThroughputExceeded, Flink and connector counters)  per minute
  lag and busiest-task metrics  per-minute Maximum
  gauges (CPU, heap, latency, ...)  AVG of per-minute Average, MAX of per-minute Maximum
Connector metrics come from every sink subtask (Flink parallelism); rates and counters are summed
over subtasks (Average x reporting subtasks, counted against the job-level `uptime` metric).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

REGION = os.environ.get("AWS_REGION", "us-east-1")
STACK = "TrainingWatchdogStack"
CONNECTOR = "TimestreamInfluxDB3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINESIS, FLINK = "AWS/Kinesis", "AWS/KinesisAnalytics"
SOURCES = ("simulator", "trainer")


def aws(*args: str) -> dict:
    result = subprocess.run(["aws", *args, "--region", REGION, "--output", "json"],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def try_aws(*args: str) -> dict | None:
    try:
        return aws(*args)
    except subprocess.CalledProcessError as error:
        print(f"warning: aws {' '.join(args[:2])} failed: {error.stderr.strip()[:300]}", file=sys.stderr)
        return None


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1000 or float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:.1f}" if abs(value) >= 10 else f"{value:.2f}"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


class Context:
    """The stack's resources and their current configuration."""

    def __init__(self) -> None:
        stack = aws("cloudformation", "describe-stacks", "--stack-name", STACK)["Stacks"][0]
        self.outputs = {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}
        self.stream = self.outputs["TelemetryStreamName"]
        self.application = self.outputs["FlinkApplicationName"]
        summary = try_aws("kinesis", "describe-stream-summary", "--stream-name", self.stream)
        self.shards = summary and summary["StreamDescriptionSummary"].get("OpenShardCount")
        app = try_aws("kinesisanalyticsv2", "describe-application", "--application-name", self.application)
        detail = (app or {}).get("ApplicationDetail", {})
        self.flink_status = detail.get("ApplicationStatus", "?")
        self.parallelism = (detail.get("ApplicationConfigurationDescription", {})
                            .get("FlinkApplicationConfigurationDescription", {})
                            .get("ParallelismConfigurationDescription", {}).get("CurrentParallelism"))
        self.simulator = self._simulator()
        self.lag_id = self._lag_id()

    def _simulator(self) -> str:
        cluster = self.outputs.get("EcsClusterName")
        services = try_aws("ecs", "list-services", "--cluster", cluster) if cluster else None
        arn = next((s for s in (services or {}).get("serviceArns", []) if "SimulatorService" in s), None)
        if not arn:
            return "simulator service not found"
        described = try_aws("ecs", "describe-services", "--cluster", cluster, "--services", arn)
        if not (described or {}).get("services"):
            return "simulator service not described"
        service = described["services"][0]
        definition = try_aws("ecs", "describe-task-definition", "--task-definition", service["taskDefinition"])
        env = {}
        if definition:
            container = definition["taskDefinition"]["containerDefinitions"][0]
            env = {e["name"]: e["value"] for e in container.get("environment", [])}
            env["cpu/memory"] = f"{definition['taskDefinition'].get('cpu')}/{definition['taskDefinition'].get('memory')}"
        settings = ", ".join(f"{k}={env[k]}" for k in ("SIM_RUNS", "SIM_TASKS", "WATCHDOG_SIM_TELEMETRY",
                                                         "WATCHDOG_SIM_RANKS", "STEP_SECONDS", "SIM_SENDERS",
                                                         "cpu/memory") if k in env)
        return (f"simulator: {service['runningCount']} of {service['desiredCount']} tasks running"
                f"{'; ' + settings if settings else ''}")

    def _lag_id(self) -> str:
        """The Kinesis source's `Id` dimension of millisBehindLatest: the stream name, '-' as '_'."""
        expected = self.stream.replace("-", "_")
        listed = try_aws("cloudwatch", "list-metrics", "--namespace", FLINK, "--metric-name", "millisBehindLatest",
                         "--dimensions", f"Name=Application,Value={self.application}")
        for metric in (listed or {}).get("Metrics", []):
            dims = {d["Name"]: d["Value"] for d in metric["Dimensions"]}
            if dims.get("Id") == expected or dims.get("Id", "").replace("_", "-") == self.stream:
                return dims["Id"]
        return expected


# (label, unit, namespace, metric, dimension set, kind, scale)
def specs(ctx: Context) -> list[tuple]:
    stream = [("StreamName", ctx.stream)]
    app = [("Application", ctx.application)]
    connector = app + [("Connector", CONNECTOR)]
    lag = app + [("Flow", "Input"), ("Id", ctx.lag_id)]
    return [
        ("kinesis IncomingRecords", "records/s", KINESIS, "IncomingRecords", stream, "rate", 1),
        ("kinesis IncomingBytes", "KB/s", KINESIS, "IncomingBytes", stream, "rate", 1e-3),
        ("kinesis WriteProvisionedThroughputExceeded", "records/min", KINESIS,
         "WriteProvisionedThroughputExceeded", stream, "sum", 1),
        ("kinesis GetRecords.IteratorAgeMilliseconds", "ms", KINESIS, "GetRecords.IteratorAgeMilliseconds", stream,
         "max", 1),
        ("flink millisBehindLatest", "ms", FLINK, "millisBehindLatest", lag, "max", 1),
        ("flink cpuUtilization", "%", FLINK, "cpuUtilization", app, "gauge", 1),
        ("flink heapMemoryUtilization", "%", FLINK, "heapMemoryUtilization", app, "gauge", 1),
        ("flink backPressuredTimeMsPerSecond (max task)", "ms/s", FLINK, "backPressuredTimeMsPerSecond", app,
         "max", 1),
        ("flink busyTimeMsPerSecond (max task)", "ms/s", FLINK, "busyTimeMsPerSecond", app, "max", 1),
        ("flink lastCheckpointDuration", "ms", FLINK, "lastCheckpointDuration", app, "gauge", 1),
        ("flink numberOfFailedCheckpoints", "per min", FLINK, "numberOfFailedCheckpoints", app, "counter", 1),
        ("flink numRestarts", "per min", FLINK, "numRestarts", app, "counter", 1),
        ("flink downtime", "ms", FLINK, "downtime", app, "max", 1),
        ("connector numRecordsWrittenPerSecond", "points/s (all subtasks)", FLINK, "numRecordsWrittenPerSecond",
         connector, "meter", 1),
        ("connector writeLatencyP50Millis", "ms", FLINK, "writeLatencyP50Millis", connector, "gauge", 1),
        ("connector writeLatencyP99Millis", "ms", FLINK, "writeLatencyP99Millis", connector, "gauge", 1),
        ("connector batchSizeLines", "lines", FLINK, "batchSizeLines", connector, "gauge", 1),
        ("connector numInFlightWriteRequests", "per subtask", FLINK, "numInFlightWriteRequests", connector,
         "gauge", 1),
        ("connector numBufferedBytes", "KB/subtask", FLINK, "numBufferedBytes", connector, "gauge", 1e-3),
        ("connector numRecordsDropped", "per min", FLINK, "numRecordsDropped", connector, "counter", 1),
        ("connector numPartialWriteFailures", "per min", FLINK, "numPartialWriteFailures", connector, "counter", 1),
        ("connector numBatchesFailed", "per min", FLINK, "numBatchesFailed", connector, "counter", 1),
        ("connector numRetries", "per min", FLINK, "numRetries", connector, "counter", 1),
        ("connector numThrottledRequests", "per min", FLINK, "numThrottledRequests", connector, "counter", 1),
        ("connector numWriteTimeouts", "per min", FLINK, "numWriteTimeouts", connector, "counter", 1),
        ("connector numServerErrors", "per min", FLINK, "numServerErrors", connector, "counter", 1),
        ("connector numClientErrors", "per min", FLINK, "numClientErrors", connector, "counter", 1),
    ]


STATS = {"rate": ("Sum",), "sum": ("Sum",), "max": ("Maximum",), "gauge": ("Average", "Maximum"),
         "meter": ("Average", "SampleCount"), "counter": ("Average", "SampleCount")}


def cloudwatch(ctx: Context, start: datetime, end: datetime) -> list[tuple]:
    rows = specs(ctx)
    queries, ids = [], {}

    def add(namespace, metric, dims, stat) -> str:
        key = (namespace, metric, tuple(dims), stat)
        if key not in ids:
            ids[key] = f"q{len(ids)}"
            queries.append({"Id": ids[key], "ReturnData": True, "MetricStat": {
                "Metric": {"Namespace": namespace, "MetricName": metric,
                           "Dimensions": [{"Name": n, "Value": v} for n, v in dims]},
                "Period": 60, "Stat": stat}})
        return ids[key]

    # Job-level reports per minute (one reporter): the yardstick for how many subtasks reported.
    reference = add(FLINK, "uptime", [("Application", ctx.application)], "SampleCount")
    for _, _, namespace, metric, dims, kind, _ in rows:
        for stat in STATS[kind]:
            add(namespace, metric, dims, stat)
    # One minute before the window: the baseline for counters.
    request = {"MetricDataQueries": queries, "StartTime": (start - timedelta(minutes=1)).isoformat(),
               "EndTime": end.isoformat(), "ScanBy": "TimestampAscending"}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(request, f)
    try:
        response = aws("cloudwatch", "get-metric-data", "--cli-input-json", f"file://{f.name}")
    finally:
        os.unlink(f.name)
    series: dict[str, dict[datetime, float]] = {}
    for result in response.get("MetricDataResults", []):
        points = series.setdefault(result["Id"], {})
        for ts, value in zip(result["Timestamps"], result["Values"]):
            points[datetime.fromisoformat(ts)] = value
    in_window = lambda points: [v for t, v in sorted(points.items()) if t >= start]  # noqa: E731
    reporters_at = series.get(reference, {})

    table = []
    for label, unit, namespace, metric, dims, kind, scale in rows:
        stat = {s: series.get(ids[(namespace, metric, tuple(dims), s)], {}) for s in STATS[kind]}
        avg = peak = total = None
        if kind in ("rate", "sum"):
            values = [v * scale / (60 if kind == "rate" else 1) for v in in_window(stat["Sum"])]
            avg, peak = mean(values), max(values, default=None)
            total = sum(v * scale for v in in_window(stat["Sum"])) if values else None
        elif kind == "max":
            values = [v * scale for v in in_window(stat["Maximum"])]
            avg, peak = mean(values), max(values, default=None)
        elif kind == "gauge":
            averages, maxima = in_window(stat["Average"]), in_window(stat["Maximum"])
            avg = mean([v * scale for v in averages])
            peak = max((v * scale for v in maxima), default=None)
        else:  # meter, counter: summed over the subtasks that reported
            summed = {}
            for t, value in stat["Average"].items():
                reporters = stat["SampleCount"].get(t, 0) / reporters_at[t] if reporters_at.get(t) else 1
                summed[t] = value * max(1, round(reporters)) * scale
            if kind == "meter":
                values = [v for t, v in sorted(summed.items()) if t >= start]
                avg, peak = mean(values), max(values, default=None)
                total = sum(values) * 60 if values else None
            else:
                increases, previous = [], None
                for t, value in sorted(summed.items()):
                    if previous is not None and t >= start:
                        # A restarted job starts its counters over.
                        increases.append(value - previous if value >= previous else value)
                    previous = value
                if increases:
                    total = sum(increases)
                    avg, peak = total / len(increases), max(increases)
        table.append((label, avg, peak, total, unit))
    return table


def influx(start: datetime, end: datetime) -> list[tuple]:
    s, e = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    window = f"time >= to_timestamp_millis({s}) AND time < to_timestamp_millis({e})"
    source = "CASE WHEN scenario LIKE 'trn_%' THEN 'trainer' ELSE 'simulator' END AS src"
    text = "coalesce(CAST({} AS VARCHAR), '-')"
    line = lambda name, a, m, t, table, where="": (  # noqa: E731
        f"SELECT concat('{name}:', src) AS metric, {text.format(a)} AS avg_v, {text.format(m)} AS max_v, "
        f"{text.format(t)} AS total_v FROM {table} {where} GROUP BY src")
    sql = (
        f"WITH poll AS (SELECT {source}, date_bin(INTERVAL '1 minute', time) AS bucket, polls, errors, "
        f"latency_p50_ms, latency_max_ms FROM control_poll WHERE {window}), "
        "per_minute AS (SELECT src, bucket, sum(polls) / 60.0 AS polls_per_s, sum(errors) AS errors "
        "FROM poll GROUP BY src, bucket), "
        f"ack AS (SELECT {source}, seen_latency_ms, apply_latency_ms FROM control_ack WHERE {window}), "
        f"inc AS (SELECT {source}, incident_id FROM incident WHERE state = 'OPEN' AND {window}) "
        + " UNION ALL ".join([
            line("polls", "avg(polls_per_s)", "max(polls_per_s)", "sum(polls_per_s) * 60", "per_minute"),
            line("poll_errors", "avg(errors)", "max(errors)", "sum(errors)", "per_minute"),
            # Reports where every poll failed carry no latency.
            line("poll_p50", "avg(latency_p50_ms)", "max(latency_p50_ms)", "count(*)", "poll", "WHERE polls > errors"),
            line("poll_max", "avg(latency_max_ms)", "max(latency_max_ms)", "count(*)", "poll", "WHERE polls > errors"),
            line("incidents", "NULL", "NULL", "count(DISTINCT incident_id)", "inc"),
            line("seen", "avg(seen_latency_ms)", "max(seen_latency_ms)", "count(seen_latency_ms)", "ack"),
            line("apply", "avg(apply_latency_ms)", "max(apply_latency_ms)", "count(apply_latency_ms)", "ack"),
        ])
    )
    labels = {
        "polls": ("influx control_poll polls", "polls/s"),
        "poll_errors": ("influx control_poll errors", "per min"),
        "poll_p50": ("influx control_poll latency p50", "ms, client side (TOTAL: 10 s reports)"),
        "poll_max": ("influx control_poll latency max", "ms, client side (TOTAL: 10 s reports)"),
        "incidents": ("influx incidents opened", "count"),
        "seen": ("influx control_ack seen latency", "ms, detection to poll (TOTAL: acks)"),
        "apply": ("influx control_ack apply latency", "ms, detection to applied (TOTAL: acks)"),
    }
    values: dict[tuple[str, str], tuple] = {}
    try:
        output = subprocess.run([os.path.join(ROOT, "scripts", "influx-query.sh"), sql], check=True,
                                capture_output=True, text=True, env={**os.environ, "DEPLOY_REGION": REGION}).stdout
    except subprocess.CalledProcessError as error:
        print(f"warning: InfluxDB query failed (is the Trainium host running?): "
              f"{(error.stdout + error.stderr).strip()[-500:]}", file=sys.stderr)
        output = ""
    for row in output.splitlines():
        cells = row.split()
        if len(cells) == 4 and ":" in cells[0] and cells[0] != "metric":
            name, src = cells[0].split(":", 1)
            values[(name, src)] = tuple(None if c == "-" else float(c) for c in cells[1:])
    table = []
    for name, (label, unit) in labels.items():
        for src in SOURCES:
            avg, peak, total = values.get((name, src), (None, None, None))
            table.append((f"{label} ({src})", avg, peak, total, unit))
    return table


def simulator_warnings(ctx: Context, start: datetime, end: datetime) -> list[tuple]:
    group = ctx.outputs.get("SimulatorLogGroupName")
    rows = []
    for label, pattern in (("simulator slow ticks (log warnings)", '"the fleet is too large for the step interval"'),
                           ("simulator slow control-poll rounds", '"the fleet is too large for it"')):
        found = try_aws("logs", "filter-log-events", "--log-group-name", group, "--filter-pattern", pattern,
                        "--start-time", str(int(start.timestamp() * 1000)),
                        "--end-time", str(int(end.timestamp() * 1000))) if group else None
        count = None if found is None else float(len(found.get("events", [])))
        rows.append((label, None, None, count, "count"))
    return rows


def main(argv: list[str]) -> int:
    minutes = int(argv[1])
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(minutes=minutes)
    ctx = Context()
    rows = cloudwatch(ctx, start, end) + influx(start, end) + simulator_warnings(ctx, start, end)
    print(f"Pipeline throughput, last {minutes} min: {start:%Y-%m-%d %H:%M} to {end:%H:%M} UTC "
          f"(AVG and MAX of per-minute values; TOTAL over the window)")
    print(f"  stream {ctx.stream}: {ctx.shards if ctx.shards is not None else '?'} open shards")
    print(f"  flink {ctx.application}: {ctx.flink_status}, parallelism {ctx.parallelism or '?'}")
    print(f"  {ctx.simulator}")
    print()
    width = max(len(r[0]) for r in rows)
    print(f"{'METRIC':<{width}}  {'AVG':>10}  {'MAX':>10}  {'TOTAL':>10}  UNIT")
    for label, avg, peak, total, unit in rows:
        print(f"{label:<{width}}  {fmt(avg):>10}  {fmt(peak):>10}  {fmt(total):>10}  {unit}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
