# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Runs one SQL query against the watchdog database with the read-only token and prints the rows
as a table. Runs on the Trainium host (InfluxDB is private); see scripts/influx-query.sh.
With --timed (INFLUX_QUERY_TIMED=1), allows 5 minutes instead of 30 s and ends with the query's
round trip from this host, excluding SSM: "-- N rows in X ms"."""

import json
import sys
import time
import urllib.error
import urllib.request

import boto3

endpoint, database, secret_arn, region, sql = sys.argv[1:6]
timed = sys.argv[6:7] == ["--timed"]
token = json.loads(boto3.client("secretsmanager", region_name=region).get_secret_value(SecretId=secret_arn)["SecretString"])["token"]
request = urllib.request.Request(
    endpoint.rstrip("/") + "/api/v3/query_sql",
    data=json.dumps({"db": database, "q": sql, "format": "json"}).encode(), method="POST",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"},
)
started = time.perf_counter()
try:
    with urllib.request.urlopen(request, timeout=300 if timed else 30) as response:
        rows = json.loads(response.read() or b"[]")
except urllib.error.HTTPError as error:
    sys.exit(f"HTTP {error.code}: {error.read().decode(errors='replace')[:500]}")
elapsed_ms = (time.perf_counter() - started) * 1000
if not rows:
    print("(no rows)")
    if timed:
        print(f"-- 0 rows in {elapsed_ms:.0f} ms")
    sys.exit(0)
columns = list(dict.fromkeys(k for r in rows for k in r))  # JSON rows omit null columns
cells = [[("" if r.get(c) is None else str(r.get(c)))[:80] for c in columns] for r in rows]
widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
for row in cells:
    print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
if timed:
    print(f"-- {len(rows)} rows in {elapsed_ms:.0f} ms")
