# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Checks the InfluxDB control plane from inside the VPC (run on the Trainium host through SSM by
scripts/check-influx.sh). Prints one OK/FAIL line per check; exits non-zero on any FAIL.

  - the read-only token can run the trainers' control query on the reader endpoint
  - the read-only token is denied writes (on both endpoints)
  - the read-only token can read system.queries (the dashboard's read-latency panels)
  - which node answers the reader endpoint: it serves queries but rejects writes (query-only)
"""

import json
import sys
import time
import urllib.error
import urllib.request

import boto3

reader, writer, database, secret_arn, region = sys.argv[1:6]
token = json.loads(boto3.client("secretsmanager", region_name=region).get_secret_value(SecretId=secret_arn)["SecretString"])["token"]
failures = 0


def call(url, path, body, content_type="application/json"):
    request = urllib.request.Request(
        url.rstrip("/") + path, data=body, method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": content_type, "Accept": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode(errors="replace"), (time.monotonic() - started) * 1000
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")[:200], (time.monotonic() - started) * 1000


def check(ok, message):
    global failures
    print(("  OK    " if ok else "  FAIL  ") + message)
    failures += 0 if ok else 1


def query(url, sql, params=None):
    body = {"db": database, "q": sql, "format": "json"}
    if params:
        body["params"] = params
    return call(url, "/api/v3/query_sql", json.dumps(body).encode())


control_sql = ("SELECT incident_id, type, action, action_arg, attempt, step, onset_step, message, "
               "arrow_cast(time, 'Int64') AS time_ns FROM incident "
               "WHERE run_id = $run_id AND state = 'OPEN' AND time >= to_timestamp_millis(0) ORDER BY time")
status, body, ms = query(reader, control_sql, {"run_id": "_schema"})
check(status == 200, f"read token runs the control query on the reader endpoint (HTTP {status}, {ms:.0f} ms): {body[:120]}")

status, body, ms = query(reader, "SELECT count(*) AS n FROM system.queries")
check(status == 200, f"read token reads system.queries on the reader endpoint (HTTP {status}): {body[:120]}")

line = f"control_check,run_id=_check,scenario=_check step=0i {time.time_ns()}"
for name, url in (("reader", reader), ("writer", writer)):
    status, body, _ = call(url, f"/api/v3/write_lp?db={database}&precision=nanosecond", line.encode(), "text/plain")
    check(status in (401, 403), f"read token denied a write on the {name} endpoint (HTTP {status}): {body[:120]}")

status, body, ms = query(writer, "SELECT 1 AS one")
print(f"  INFO  writer endpoint answers queries with the read token: HTTP {status} ({ms:.0f} ms)")
latencies = []
for _ in range(20):
    status, _, ms = query(reader, control_sql, {"run_id": "_schema"})
    if status == 200:
        latencies.append(ms)
latencies.sort()
if latencies:
    print(f"  INFO  control query round trip from this host: p50 {latencies[len(latencies) // 2]:.1f} ms, "
          f"max {latencies[-1]:.1f} ms over {len(latencies)} queries")
sys.exit(1 if failures else 0)
