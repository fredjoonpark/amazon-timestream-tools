#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Runs one SQL query against the watchdog database (read-only token, reader endpoint) from the
# Trainium host through SSM, since InfluxDB is private. Prints a table.
# Usage: AWS_PROFILE=default scripts/influx-query.sh "SELECT ... FROM incident WHERE ..."
#        INFLUX_QUERY_TIMED=1 scripts/influx-query.sh "..."   # 5-minute limit; ends with the query's round trip
#        ENDPOINT_OUTPUT=InfluxDbEndpoint scripts/influx-query.sh "..."   # the writer (ingest) node
set -euo pipefail
. "$(dirname "$0")/env.sh"   # optional settings from .env (see .env.example)
cd "$(dirname "$0")/.."
sql=${1:?usage: influx-query.sh "<SQL>"}
export AWS_REGION=${DEPLOY_REGION:-us-east-1}
output() {
  aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name TrainingWatchdogStack \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
instance=$(output TrainiumInstanceId)
parameters=$(python3 - "$(gzip -9c scripts/influx_query.py | base64 | tr -d '\n')" "$(output "${ENDPOINT_OUTPUT:-InfluxDbReaderEndpoint}")" \
    "$(output InfluxDbDatabase)" "$(output InfluxDbReadTokenSecretArn)" "$AWS_REGION" "$sql" \
    ${INFLUX_QUERY_TIMED:+--timed} <<'PY'
import json, shlex, sys
script, *args = sys.argv[1:]
q = shlex.quote
# A file of its own (mktemp), so concurrent queries on the host don't overwrite each other's script.
command = ('f=$(mktemp) || exit 1; '
           f"echo {q(script)} | base64 -d | gunzip > \"$f\" && "
           "venv=$(ls -d /opt/aws_neuronx_venv_pytorch_2_* | grep -E '_pytorch_2_[0-9]+$' | sort -V | tail -1) && "
           f"$venv/bin/python -I \"$f\" {' '.join(q(a) for a in args)}; "
           'status=$?; rm -f "$f"; exit $status')
print(json.dumps({"commands": ["exec bash -c " + q(command)]}))
PY
)
id=$(aws ssm send-command --region "$AWS_REGION" --instance-ids "$instance" --document-name AWS-RunShellScript \
  --comment "watchdog influx query" --parameters "$parameters" --query Command.CommandId --output text)
polls=30
[[ -z "${INFLUX_QUERY_TIMED:-}" ]] || polls=160  # 2 s apart: past the timed query's 5-minute limit
for _ in $(seq 1 "$polls"); do
  status=$(aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$id" --instance-id "$instance" \
    --query Status --output text 2>/dev/null || echo Pending)
  [[ "$status" == Pending || "$status" == InProgress || "$status" == Delayed ]] || break
  sleep 2
done
aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$id" --instance-id "$instance" \
  --query '[StandardOutputContent, StandardErrorContent]' --output text
[[ "$status" == Success ]]
