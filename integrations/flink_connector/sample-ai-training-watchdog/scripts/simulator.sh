#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Turns the simulator on or off by scaling its ECS service (takes effect in about a minute).
# Usage: AWS_PROFILE=default scripts/simulator.sh on|off|status
#        AWS_PROFILE=default scripts/simulator.sh once <scenario>
#   once: one simulated run of one scenario from step 0, as its own ECS task, which exits when the
#   run ends (like a Trainium run, with no host). Scenarios: healthy, bad_lr, healthy_grpo,
#   reward_hacking. scripts/demo.sh --simulated wraps it.
# The setting lasts until changed here. A deploy applies -c simulator (on by default, off with
# -c simulator=false) only when it changes the simulator's service, not on every deploy.
# `on` starts the deployed task count: 1, or ceil(simRuns / simRunsPerTask) with -c simRunsPerTask.
set -euo pipefail
. "$(dirname "$0")/env.sh"   # optional settings from .env (see .env.example)
action=${1:?usage: simulator.sh on|off|status|once <scenario>}
export AWS_REGION=${DEPLOY_REGION:-us-east-1}
output() {
  aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name TrainingWatchdogStack \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
cluster=$(output EcsClusterName)
service=$(aws ecs list-services --region "$AWS_REGION" --cluster "$cluster" --query serviceArns --output text \
  | tr '\t' '\n' | grep SimulatorService) || true
if [[ -z "$service" ]]; then
  echo "No simulator service in cluster ${cluster:-?}; deploy without -c simulator=false" >&2
  exit 1
fi
if [[ "$action" == once ]]; then
  scenario=${2:?usage: simulator.sh once <healthy|bad_lr|healthy_grpo|reward_hacking>}
  # Unique per invocation: a reused run ID would inherit Flink's per-run state from the earlier run.
  prefix="once-$(date -u +%Y%m%d%H%M%S)-$RANDOM-"
  run_id="${scenario}-${prefix}001"
  overrides=$(python3 -c 'import json,sys; print(json.dumps({"containerOverrides": [{"name": "simulator",
    "command": ["--sink", "kinesis", "--once", sys.argv[1], "--run-prefix", sys.argv[2]]}]}))' "$scenario" "$prefix")
  if ! task=$(aws ecs run-task --region "$AWS_REGION" --cluster "$cluster" --launch-type FARGATE \
      --task-definition "$(output SimulatorTaskDefinitionArn)" --overrides "$overrides" \
      --network-configuration "awsvpcConfiguration={subnets=[$(output PrivateSubnetIds)],securityGroups=[$(output SimulatorSecurityGroupId)],assignPublicIp=DISABLED}" \
      --started-by "watchdog-once" --output text \
      --query 'tasks[0].taskArn || failures[0].reason || `"no task and no failure reason"`' 2>&1); then
    echo "Could not start the simulated run: $task" >&2
    if grep -q PassRole <<<"$task"; then
      echo "ECS RunTask needs iam:PassRole on the simulator's roles; run with credentials that allow it." >&2
    fi
    exit 1
  fi
  if [[ "$task" != arn:* ]]; then
    # RunTask can succeed and start nothing (no Fargate capacity, an ENI limit): the reason is in
    # its failures.
    echo "ECS started no task for the simulated run: $task" >&2
    exit 1
  fi
  echo "Started $run_id (scenario $scenario, simulated) as task ${task##*/}."
  echo "Logs: aws logs tail $(output SimulatorLogGroupName) --region $AWS_REGION --follow --log-stream-name-prefix Simulator/simulator/${task##*/}"
  exit 0
fi
case "$action" in
  on)
    # The deploy's task count: SIM_TASKS in the service's task definition (-c simRunsPerTask), else 1.
    task_definition=$(aws ecs describe-services --region "$AWS_REGION" --cluster "$cluster" --services "$service" \
      --query 'services[0].taskDefinition' --output text)
    count=$(aws ecs describe-task-definition --region "$AWS_REGION" --task-definition "$task_definition" \
      --query "taskDefinition.containerDefinitions[?name=='simulator'] | [0].environment[?name=='SIM_TASKS'].value | [0]" \
      --output text)
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || count=1
    ;;
  off) count=0 ;;
  status) count= ;;
  *) echo "usage: simulator.sh on|off|status|once <scenario>" >&2; exit 1 ;;
esac
if [[ -n "$count" ]]; then
  aws ecs update-service --region "$AWS_REGION" --cluster "$cluster" --service "$service" \
    --desired-count "$count" --query 'service.serviceName' --output text >/dev/null
  echo "Simulator set to $action (until changed here, or by a deploy that changes the simulator service)."
fi
aws ecs describe-services --region "$AWS_REGION" --cluster "$cluster" --services "$service" \
  --query 'services[0].[desiredCount,runningCount]' --output text \
  | awk '{ printf "Simulator tasks: desired %s, running %s\n", $1, $2 }'
