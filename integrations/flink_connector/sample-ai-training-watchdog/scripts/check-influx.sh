#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Checks the InfluxDB control plane from inside the VPC: runs scripts/check_influx_control.py on
# the Trainium host through SSM (InfluxDB is private). Read-only except one write attempt that the
# read-only token must be denied.
# Usage: AWS_PROFILE=default scripts/check-influx.sh
set -euo pipefail
. "$(dirname "$0")/env.sh"   # optional settings from .env (see .env.example)
cd "$(dirname "$0")/.."
export AWS_REGION=${DEPLOY_REGION:-us-east-1}
stack=TrainingWatchdogStack
output() {
  aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name "$stack" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
instance=$(output TrainiumInstanceId)
args=("$(output InfluxDbReaderEndpoint)" "$(output InfluxDbEndpoint)" "$(output InfluxDbDatabase)"
      "$(output InfluxDbReadTokenSecretArn)" "$AWS_REGION")
parameters=$(python3 - "$(gzip -9c scripts/check_influx_control.py | base64 | tr -d '\n')" "${args[@]}" <<'EOF'
import json, shlex, sys
script, *args = sys.argv[1:]
q = shlex.quote
# A file of its own (mktemp), so a concurrent run on the host doesn't overwrite the script.
command = ('f=$(mktemp) || exit 1; '
           f"echo {q(script)} | base64 -d | gunzip > \"$f\" && "
           "venv=$(ls -d /opt/aws_neuronx_venv_pytorch_2_* | grep -E '_pytorch_2_[0-9]+$' | sort -V | tail -1) && "
           f"$venv/bin/python -I \"$f\" {' '.join(q(a) for a in args)}; "
           'status=$?; rm -f "$f"; exit $status')
print(json.dumps({"commands": ["exec bash -c " + q(command)]}))
EOF
)
id=$(aws ssm send-command --region "$AWS_REGION" --instance-ids "$instance" --document-name AWS-RunShellScript \
  --comment "watchdog influx check" --parameters "$parameters" --query Command.CommandId --output text)
for _ in $(seq 1 30); do
  status=$(aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$id" --instance-id "$instance" \
    --query Status --output text 2>/dev/null || echo Pending)
  [[ "$status" == Pending || "$status" == InProgress || "$status" == Delayed ]] || break
  sleep 2
done
aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$id" --instance-id "$instance" \
  --query '[StandardOutputContent, StandardErrorContent]' --output text
[[ "$status" == Success ]]
