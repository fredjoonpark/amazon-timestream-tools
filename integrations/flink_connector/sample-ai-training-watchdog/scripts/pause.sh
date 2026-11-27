#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Pauses the demo between sessions: stops the Trainium host once no run is active.
# Usage: AWS_PROFILE=default scripts/pause.sh [--force]
#   --force: cancel an in-progress run first (scripts/train.sh stop), then stop the host.
# Resume with scripts/preflight.sh --start-host. The rest of the stack keeps running; destroy it
# when finished for good (README: Tear down).
set -euo pipefail
. "$(dirname "$0")/env.sh"   # optional settings from .env (see .env.example)
cd "$(dirname "$0")/.."
export AWS_REGION=${DEPLOY_REGION:-us-east-1}
instance=$(aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name TrainingWatchdogStack \
  --query "Stacks[0].Outputs[?OutputKey=='TrainiumInstanceId'].OutputValue" --output text)
state=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$instance" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)
if [[ "$state" == stopped || "$state" == stopping ]]; then
  echo "Trainium host $instance is already $state."
  exit 0
fi
# Same check as train.sh: training runs only, JSON so the count spans every page.
busy=$(aws ssm list-commands --region "$AWS_REGION" --instance-id "$instance" --output json \
  --query "length(Commands[?(Status=='InProgress' || Status=='Pending') && starts_with(Comment || '', 'watchdog trn-')])")
if [[ "$busy" != 0 ]]; then
  if [[ "${1:-}" != --force ]]; then
    echo "A training run is in progress on $instance; wait for it, or rerun with --force to cancel it." >&2
    exit 1
  fi
  scripts/train.sh stop
fi
aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$instance" >/dev/null
echo "Stopping $instance..."
aws ec2 wait instance-stopped --region "$AWS_REGION" --instance-ids "$instance"
echo "Trainium host stopped. Resume with: scripts/preflight.sh --start-host"
