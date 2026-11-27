#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Opens Grafana at http://localhost:3000 through an ECS Exec port-forwarding session.
# Requires the AWS CLI and the Session Manager plugin. Nothing is exposed publicly.
set -euo pipefail
. "$(dirname "$0")/env.sh"   # optional settings from .env (see .env.example)

export AWS_REGION=${DEPLOY_REGION:-us-east-1}
STACK=${STACK:-TrainingWatchdogStack}
LOCAL_PORT=${LOCAL_PORT:-3000}

output() {
  aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}

cluster=$(output EcsClusterName)
service=$(output GrafanaServiceName)
task_arn=$(aws ecs list-tasks --region "$AWS_REGION" --cluster "$cluster" --service-name "$service" \
  --desired-status RUNNING --query 'taskArns[0]' --output text)
if [[ -z "$task_arn" || "$task_arn" == "None" ]]; then
  echo "No running Grafana task in $cluster/$service" >&2
  exit 1
fi
task_id=${task_arn##*/}
runtime_id=$(aws ecs describe-tasks --region "$AWS_REGION" --cluster "$cluster" --tasks "$task_arn" \
  --query "tasks[0].containers[?name=='grafana'].runtimeId | [0]" --output text)

echo "Grafana admin password: aws secretsmanager get-secret-value --region $AWS_REGION --secret-id $(output GrafanaAdminSecretArn) --query SecretString --output text"
echo "Forwarding http://localhost:$LOCAL_PORT -> Grafana (Ctrl-C to stop)"
aws ssm start-session --region "$AWS_REGION" \
  --target "ecs:${cluster}_${task_id}_${runtime_id}" \
  --document-name AWS-StartPortForwardingSession \
  --parameters "{\"portNumber\":[\"3000\"],\"localPortNumber\":[\"$LOCAL_PORT\"]}"
