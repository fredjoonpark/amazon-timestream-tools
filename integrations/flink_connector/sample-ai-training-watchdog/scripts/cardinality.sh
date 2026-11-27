#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Cardinality test report (README, Load): how many series the database holds, and how long the
# queries that matter take while it holds them. Read-only: timed SQL through scripts/influx-query.sh
# (the Trainium host, read-only token, the query node). Run once per load stage, after
# scripts/throughput.sh measure.
# Usage: AWS_PROFILE=default scripts/cardinality.sh [window] [real-run-id]
#   window: how far back to count series, an SQL interval (default '1 day')
#   real-run-id: a real Trainium run whose control poll to time (default: the latest trn- run)
# Each query prints its rows, then "-- N rows in X ms": the round trip from the Trainium host.
set -euo pipefail
. "$(dirname "$0")/env.sh"   # optional settings from .env (see .env.example)
cd "$(dirname "$0")/.."
window=${1:-1 day}
# Only the reported queries are timed: an empty timed result ends with "-- 0 rows in N ms", which
# the run ID lookups below would read as a run ID.
unset INFLUX_QUERY_TIMED
# A query that fails (at tens of millions of series, the whole-database count can time out) is
# reported and the rest still run; the script then exits 1.
failures=0
q() {
  echo "== $1"
  if ! INFLUX_QUERY_TIMED=1 scripts/influx-query.sh "$2" | sed '/^[[:space:]]*$/d'; then
    echo "-- failed; continuing with the next query"
    failures=$((failures + 1))
  fi
  echo
}

# Tag keys per measurement: grafana/build_dashboard.py SERIES_TAGS (a series is one distinct tag
# combination).
series_sql=$(python3 - "$window" <<'PY'
import ast, sys
src = open("grafana/build_dashboard.py").read()
start = src.index("{", src.index("SERIES_TAGS = {"))
series_tags = ast.literal_eval(src[start : src.index("}\n", start) + 1])
window = sys.argv[1]
print(" UNION ALL ".join(
    f"SELECT '{m}' AS measurement, COUNT(*) AS points, COUNT(DISTINCT concat_ws('|', {tags})) AS series "
    f"FROM {m} WHERE time > now() - INTERVAL '{window}'"
    for m, tags in series_tags.items()
) + " ORDER BY series DESC")
PY
)
q "Series and points by measurement (last $window)" "$series_sql"

run=${2:-}
if [[ -z "$run" ]]; then
  run=$(scripts/influx-query.sh "SELECT run_id FROM run_info WHERE run_id LIKE 'trn-%' ORDER BY time DESC LIMIT 1" \
    | sed -n 2p | tr -d '[:space:]') || { run=; failures=$((failures + 1)); }
fi
sim=$(scripts/influx-query.sh "SELECT run_id FROM attn_stats WHERE time > now() - INTERVAL '15 minutes' AND run_id NOT LIKE 'trn-%' LIMIT 1" \
  | sed -n 2p | tr -d '[:space:]') || { sim=; failures=$((failures + 1)); }

if [[ -n "$run" ]]; then
  q "The trainer's control poll ($run)" \
    "SELECT incident_id, type, action, action_arg, attempt, step, onset_step, message, arrow_cast(time, 'Int64') AS time_ns FROM incident WHERE run_id = '$run' AND state = 'OPEN' AND time >= now() - INTERVAL '1 day' ORDER BY time"
  q "Control loop of $run: Flink, InfluxDB and the poll, the trainer" \
    "SELECT i.type, i.action, i.emitted_ms - arrow_cast(i.time, 'Int64') / 1000000 AS flink_ms, a.seen_latency_ms, a.apply_latency_ms FROM incident i JOIN control_ack a ON a.run_id = i.run_id AND a.incident_id = i.incident_id WHERE i.run_id = '$run' AND i.state = 'OPEN'"
  q "Control poll latency, trainer-side ($run)" \
    "SELECT ROUND(approx_percentile_cont(latency_p50_ms, 0.5), 1) AS p50_of_p50_ms, ROUND(MAX(latency_max_ms), 1) AS max_ms, SUM(errors) AS errors, SUM(polls) AS polls FROM control_poll WHERE run_id = '$run'"
fi
if [[ -n "$sim" ]]; then
  q "One run's attention entropy by layer and head, last 15 minutes ($sim)" \
    "SELECT layer, head, AVG(entropy) AS entropy FROM attn_stats WHERE run_id = '$sim' AND time > now() - INTERVAL '15 minutes' GROUP BY layer, head ORDER BY entropy LIMIT 5"
fi
q "Prompts the grader overpays most, whole fleet, last hour" \
  "SELECT prompt_id, AVG(grader_gap) AS gap, COUNT(*) AS steps FROM rl_prompt WHERE time > now() - INTERVAL '1 hour' GROUP BY prompt_id ORDER BY gap DESC, prompt_id LIMIT 5"
q "Distinct prompts, last hour" \
  "SELECT COUNT(DISTINCT prompt_id) AS prompts FROM rl_prompt WHERE time > now() - INTERVAL '1 hour'"
if (( failures > 0 )); then
  echo "$failures queries failed" >&2
  exit 1
fi
