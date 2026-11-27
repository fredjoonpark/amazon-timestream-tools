#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Deletes the stack, with deploy.sh's settings: .env without overriding the environment, the Region
# pinned to DEPLOY_REGION, and the credentials checked against DEPLOY_ACCOUNT. cdk destroy asks
# before it deletes anything.
# Usage: AWS_PROFILE=<profile> DEPLOY_ACCOUNT=<account-id> [DEPLOY_REGION=<region>] \
#          scripts/destroy.sh [cdk destroy options]
set -euo pipefail
. "$(dirname "$0")/env.sh"   # optional settings from .env (see .env.example)
cd "$(dirname "$0")/.."

# Pin the Region explicitly; an ambient AWS_REGION must not redirect the delete.
export AWS_REGION=${DEPLOY_REGION:-us-east-1}
export AWS_DEFAULT_REGION=$AWS_REGION
export CDK_DEFAULT_REGION=$AWS_REGION
if ! CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text); then
  echo "Could not resolve AWS credentials for profile '${AWS_PROFILE:-default}'; refresh them and retry" >&2
  exit 1
fi
export CDK_DEFAULT_ACCOUNT
if [[ -z "${DEPLOY_ACCOUNT:-}" ]]; then
  echo "Set DEPLOY_ACCOUNT to the target account ID (credentials resolve to $CDK_DEFAULT_ACCOUNT)" >&2
  exit 1
fi
if [[ "$DEPLOY_ACCOUNT" != "$CDK_DEFAULT_ACCOUNT" ]]; then
  echo "Credentials resolve to account $CDK_DEFAULT_ACCOUNT, not DEPLOY_ACCOUNT=$DEPLOY_ACCOUNT; check AWS_PROFILE" >&2
  exit 1
fi
if ! status=$(aws cloudformation describe-stacks --stack-name TrainingWatchdogStack \
    --query 'Stacks[0].StackStatus' --output text 2>&1); then
  echo "No TrainingWatchdogStack in account $CDK_DEFAULT_ACCOUNT, $AWS_REGION: $status" >&2
  exit 1
fi

# cdk destroy synthesizes the app, which needs the built Flink job and the parameter group's ID.
pg_name=training-watchdog-enterprise-3node
pg_id=$(aws timestream-influxdb list-db-parameter-groups --query "items[?name=='$pg_name'].id | [0]" --output text)
if [[ -z "$pg_id" || "$pg_id" == None ]]; then
  echo "InfluxDB parameter group $pg_name not found in $AWS_REGION" >&2
  exit 1
fi
if [[ ! -f flink-job/target/training-watchdog-flink-0.1.0-SNAPSHOT-all.jar ]]; then
  (cd flink-job && mvn -q -B package -DskipTests)
fi

echo "Deleting TrainingWatchdogStack ($status) in account $CDK_DEFAULT_ACCOUNT, $AWS_REGION:"
echo "the InfluxDB cluster and its data, and the Trainium host and its compile cache."
(cd infra && npm ci --silent && npx cdk destroy TrainingWatchdogStack -c influxParameterGroupId="$pg_id" "$@")
