#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Pipeline throughput test tooling (README, Load). Read-only: describe, list, and get calls, plus one
# SQL query through scripts/influx-query.sh (from the Trainium host, read-only token).
# Usage: AWS_PROFILE=default scripts/throughput.sh measure <minutes>
#   measure: one table for the last <minutes> whole minutes, avg and max of per-minute values (and
#   window totals for counts): Kinesis incoming records and bytes, write throttling, iterator age;
#   Flink lag (millisBehindLatest), CPU, heap, backpressure, busy time, checkpoints, restarts; the
#   InfluxDB 3 connector's write rate, latency, batching, drops, rejections, retries, and errors
#   (CloudWatch); control-poll rate, errors, and latency, incidents, and incident seen and apply
#   latency (InfluxDB control_poll, incident, control_ack); simulator overload warnings (its logs).
#   CloudWatch lags 1-3 minutes: measure a few minutes after a load step ends.
set -euo pipefail
. "$(dirname "$0")/env.sh"   # optional settings from .env (see .env.example)
cd "$(dirname "$0")/.."
export AWS_REGION=${DEPLOY_REGION:-us-east-1}
case "${1:-}" in
  measure)
    minutes=${2:?usage: throughput.sh measure <minutes>}
    if ! [[ "$minutes" =~ ^[1-9][0-9]*$ ]]; then
      echo "minutes must be a positive integer, not '$minutes'" >&2
      exit 1
    fi
    exec python3 -I scripts/throughput.py "$minutes"
    ;;
  *)
    echo "usage: throughput.sh measure <minutes>" >&2
    exit 1
    ;;
esac
