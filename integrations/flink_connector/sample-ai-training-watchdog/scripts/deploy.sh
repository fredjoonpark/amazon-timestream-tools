#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Builds the Flink job and deploys the stack.
# Usage: AWS_PROFILE=<profile> DEPLOY_ACCOUNT=<account-id> [DEPLOY_REGION=<region>] \
#          [DEPLOY_TAGS=Key=value,...] scripts/deploy.sh [cdk deploy options]
set -euo pipefail
. "$(dirname "$0")/env.sh"   # optional settings from .env (see .env.example)
cd "$(dirname "$0")/.."

# Pin the Region explicitly; an ambient AWS_REGION must not redirect the deploy.
export AWS_REGION=${DEPLOY_REGION:-us-east-1}
export AWS_DEFAULT_REGION=$AWS_REGION
export CDK_DEFAULT_REGION=$AWS_REGION
if ! CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text); then
  echo "Could not resolve AWS credentials for profile '${AWS_PROFILE:-default}'; refresh them and retry" >&2
  exit 1
fi
export CDK_DEFAULT_ACCOUNT
# Refuse to deploy anywhere but the account the operator names.
if [[ -z "${DEPLOY_ACCOUNT:-}" ]]; then
  echo "Set DEPLOY_ACCOUNT to the target account ID (credentials resolve to $CDK_DEFAULT_ACCOUNT)" >&2
  exit 1
fi
if [[ "$DEPLOY_ACCOUNT" != "$CDK_DEFAULT_ACCOUNT" ]]; then
  echo "Credentials resolve to account $CDK_DEFAULT_ACCOUNT, not DEPLOY_ACCOUNT=$DEPLOY_ACCOUNT; check AWS_PROFILE" >&2
  exit 1
fi
echo "Deploying to account $CDK_DEFAULT_ACCOUNT in $AWS_REGION"
# Keep the Trainium host's AMI, so a new DLAMI release never replaces the host (and its compile
# cache, checkpoints, and saved adapters), whichever checkout deploys. DEPLOY_TRAINIUM_AMI=latest
# takes the newest DLAMI and replaces the host; DEPLOY_TRAINIUM_AMI=<ami-id> names one.
if [[ -z "${DEPLOY_TRAINIUM_AMI:-}" ]]; then
  if ! host=$(aws cloudformation describe-stacks --stack-name TrainingWatchdogStack \
      --query "Stacks[0].Outputs[?OutputKey=='TrainiumInstanceId'].OutputValue | [0]" --output text 2>&1); then
    if [[ "$host" != *"does not exist"* ]]; then
      echo "Could not read the stack to find the Trainium host's AMI: $host" >&2
      exit 1
    fi
    host=  # first deploy
  fi
  if [[ "$host" == i-* ]]; then
    if ! DEPLOY_TRAINIUM_AMI=$(aws ec2 describe-instances --instance-ids "$host" \
        --query 'Reservations[0].Instances[0].ImageId' --output text) || [[ "$DEPLOY_TRAINIUM_AMI" != ami-* ]]; then
      echo "Could not read the AMI of Trainium host $host; set DEPLOY_TRAINIUM_AMI to keep it (or to latest)" >&2
      exit 1
    fi
    echo "Trainium host $host: keeping AMI $DEPLOY_TRAINIUM_AMI (DEPLOY_TRAINIUM_AMI=latest takes the newest DLAMI)"
    export DEPLOY_TRAINIUM_AMI
  fi
fi
# Always build with the same JDK so unchanged code yields an unchanged JAR (and Flink is not restarted).
# BUILD_JAVA_HOME wins; on macOS, java_home finds JDK 17; elsewhere, use JAVA_HOME if it is JDK 17.
if [[ -n "${BUILD_JAVA_HOME:-}" ]]; then
  JAVA_HOME=$BUILD_JAVA_HOME
elif [[ -x /usr/libexec/java_home ]]; then
  JAVA_HOME=$(/usr/libexec/java_home -v 17)
fi
if [[ -z "${JAVA_HOME:-}" ]] || ! "$JAVA_HOME/bin/java" -version 2>&1 | grep -q 'version "17'; then
  echo "Needs JDK 17: set BUILD_JAVA_HOME (or JAVA_HOME) to a JDK 17 installation" >&2
  exit 1
fi
export JAVA_HOME
# With the Finch docker context, drive Finch's engine with the docker CLI. `finch push` ignores
# `finch login` for ECR, and ~/.docker/config.json may name a credential helper (e.g. Docker
# Desktop's) that is not installed, so CDK's ECR login goes to a throwaway config deleted on exit.
if [[ -z "${CDK_DOCKER:-}" ]] && [[ "$(command docker context show 2>/dev/null)" == finch ]]; then
  DOCKER_HOST=$(command docker context inspect finch --format '{{.Endpoints.docker.Host}}')
  DOCKER_CONFIG=$(mktemp -d)
  trap 'rm -rf -- "${DOCKER_CONFIG:?}"' EXIT
  echo '{}' > "$DOCKER_CONFIG/config.json"
  export DOCKER_HOST DOCKER_CONFIG CDK_DOCKER=docker
fi

# InfluxDB 3 Enterprise node roles: 1 ingest+query node, 1 query-only node, a dedicated compactor.
# CloudFormation cannot provision parameter groups (the type is NON_PROVISIONABLE) and the service
# has no delete API for them, so create this one once, by name, and reuse it on every deploy.
# Parameter groups cannot be changed after creation: to change the node layout, use a new name.
pg_name=training-watchdog-enterprise-3node
pg_id=$(aws timestream-influxdb list-db-parameter-groups --query "items[?name=='$pg_name'].id | [0]" --output text)
if [[ -z "$pg_id" || "$pg_id" == None ]]; then
  pg_id=$(aws timestream-influxdb create-db-parameter-group --name "$pg_name" \
    --description "AI Training Watchdog: 1 ingest+query node, 1 query-only node, dedicated compactor" \
    --parameters '{"InfluxDBv3Enterprise":{"ingestQueryInstances":1,"queryOnlyInstances":1,"dedicatedCompactor":true,"queryLogSize":10000}}' \
    --query id --output text)
  echo "Created InfluxDB parameter group $pg_name ($pg_id)"
fi
echo "InfluxDB parameter group: $pg_name ($pg_id)"

(cd flink-job && mvn -q -B clean package)
# CDK asks before IAM or security-group changes that broaden access; for an unattended deploy,
# pass --require-approval never.
(cd infra && npm ci --silent && npx cdk deploy -c influxParameterGroupId="$pg_id" "$@")
