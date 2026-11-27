#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Checks that everything the real-Trainium demo needs is in place. Read-only unless --start-host.
# Usage: AWS_PROFILE=default scripts/preflight.sh [--start-host]
# Exits non-zero if any check fails.
set -uo pipefail
. "$(dirname "$0")/env.sh"   # optional settings from .env (see .env.example)
export AWS_REGION=${DEPLOY_REGION:-us-east-1}
account=${DEPLOY_ACCOUNT:?Set DEPLOY_ACCOUNT to the target account ID}
stack=TrainingWatchdogStack
start_host=false
[[ "${1:-}" == --start-host ]] && start_host=true
failures=0

ok() { printf '  OK    %s\n' "$1"; }
warn() { printf '  WARN  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; failures=$((failures + 1)); }
output() {
  aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name "$stack" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text 2>/dev/null
}

echo "Preflight for $stack in $account / $AWS_REGION"

actual=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
if [[ "$actual" == "$account" ]]; then ok "credentials resolve to $account"; else
  fail "credentials resolve to '${actual:-none}', not $account (check AWS_PROFILE; refresh credentials)"
  exit 1
fi

status=$(aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name "$stack" \
  --query 'Stacks[0].StackStatus' --output text 2>/dev/null)
case "$status" in
  *_COMPLETE) [[ "$status" == *ROLLBACK* ]] && warn "stack is $status (last update rolled back)" || ok "stack is $status" ;;
  "") fail "stack $stack not found; deploy it (README: Deploy)"; exit 1 ;;
  *) fail "stack is $status" ;;
esac

flink=$(aws kinesisanalyticsv2 describe-application --region "$AWS_REGION" --application-name training-watchdog \
  --query 'ApplicationDetail.ApplicationStatus' --output text 2>/dev/null)
[[ "$flink" == RUNNING ]] && ok "Flink application is RUNNING" || fail "Flink application is ${flink:-missing}"

# InfluxDB 3 Enterprise is the data store and the control plane: trainers poll it for commands.
influx_id=$(output InfluxDbClusterId)
read -r istatus iengine ingest_query query_only < <(aws timestream-influxdb get-db-cluster --region "$AWS_REGION" \
  --db-cluster-id "${influx_id:-none}" \
  --query '[status, engineType, clusterConfiguration.ingestQueryInstances, clusterConfiguration.queryOnlyInstances]' \
  --output text 2>/dev/null)
if [[ "$istatus" == AVAILABLE ]]; then
  ok "InfluxDB cluster $influx_id is AVAILABLE ($iengine: $ingest_query ingest+query, $query_only query-only node(s))"
else
  fail "InfluxDB cluster ${influx_id:-missing} is ${istatus:-not found}"
fi

cluster=$(output EcsClusterName)
while read -r name desired running; do
  case "$name" in
    *Grafana*) [[ "$running" == 1 ]] && ok "Grafana task running" || fail "Grafana tasks: desired $desired, running $running" ;;
    *Simulator*)
      if [[ "$desired" == 0 ]]; then ok "simulator off (real runs have the pipeline to themselves)"
      else warn "simulator on ($running running); turn off with scripts/simulator.sh off"; fi ;;
  esac
done < <(aws ecs describe-services --region "$AWS_REGION" --cluster "$cluster" \
  --services $(aws ecs list-services --region "$AWS_REGION" --cluster "$cluster" --query serviceArns --output text) \
  --query 'services[].[serviceName,desiredCount,runningCount]' --output text)

instance=$(output TrainiumInstanceId)
if [[ -z "$instance" || "$instance" == None ]]; then
  fail "no Trainium host in the stack (deploy without -c trainium=false)"
else
  # The optional DEPLOY_TAGS (infra/bin/app.ts), such as tags your account's policies require.
  if [[ -n "${DEPLOY_TAGS:-}" ]]; then
    have=$(aws ec2 describe-tags --region "$AWS_REGION" --filters "Name=resource-id,Values=$instance" \
      --query 'Tags[].[Key,Value]' --output text)
    missing=()
    IFS=',' read -ra wanted <<<"$DEPLOY_TAGS"
    trim() { local v=$1; v=${v#"${v%%[![:space:]]*}"}; printf '%s' "${v%"${v##*[![:space:]]}"}"; }
    for pair in "${wanted[@]}"; do
      # Trimmed as infra/bin/app.ts trims them before tagging: 'K1=v1, K2=v2' tags K2=v2.
      key=$(trim "${pair%%=*}")
      value=$(trim "${pair#*=}")
      [[ -z "$(trim "$pair")" ]] || grep -qxF "$key"$'\t'"$value" <<<"$have" || missing+=("$key=$value")
    done
    if [[ ${#missing[@]} -eq 0 ]]; then
      ok "host $instance has the DEPLOY_TAGS tags"
    else
      fail "host $instance is missing ${missing[*]} from DEPLOY_TAGS; deploy again with DEPLOY_TAGS set"
    fi
  fi
  state=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$instance" \
    --query 'Reservations[0].Instances[0].State.Name' --output text)
  if [[ "$state" != running ]] && $start_host; then
    echo "  ...   starting $instance"
    if aws ec2 start-instances --region "$AWS_REGION" --instance-ids "$instance" >/dev/null \
        && aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$instance"; then
      for _ in $(seq 1 40); do
        [[ $(aws ssm describe-instance-information --region "$AWS_REGION" --filters "Key=InstanceIds,Values=$instance" \
          --query 'InstanceInformationList[0].PingStatus' --output text) == Online ]] && break
        sleep 10
      done
      state=running
    else
      fail "could not start host $instance (see the error above)"
    fi
  fi
  if [[ "$state" != running ]]; then
    # With --start-host, the failed start was reported above.
    $start_host || fail "host $instance is $state (start it: scripts/preflight.sh --start-host)"
  else
    ok "host $instance is running"
    ping=$(aws ssm describe-instance-information --region "$AWS_REGION" --filters "Key=InstanceIds,Values=$instance" \
      --query 'InstanceInformationList[0].PingStatus' --output text)
    [[ "$ping" == Online ]] && ok "SSM agent online" || fail "SSM agent is ${ping:-not registered}; wait a minute and retry"
    # Same check as train.sh: training runs only, JSON so the count spans every page.
    busy=$(aws ssm list-commands --region "$AWS_REGION" --instance-id "$instance" --output json \
      --query "length(Commands[?(Status=='InProgress' || Status=='Pending') && starts_with(Comment || '', 'watchdog trn-')])")
    [[ "$busy" == 0 ]] && ok "no training run in progress" || warn "$busy run(s) in progress; scripts/train.sh stop to cancel"
    if [[ "$ping" == Online ]]; then
      id=$(aws ssm send-command --region "$AWS_REGION" --instance-ids "$instance" --document-name AWS-RunShellScript \
        --comment "watchdog preflight" --parameters '{"commands":["find /opt/watchdog/neuron-cache -name \"*.neff\" 2>/dev/null | wc -l"]}' \
        --query Command.CommandId --output text)
      for _ in $(seq 1 20); do
        s=$(aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$id" --instance-id "$instance" \
          --query Status --output text 2>/dev/null)
        [[ "$s" == Success || "$s" == Failed ]] && break
        sleep 2
      done
      neffs=$(aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$id" --instance-id "$instance" \
        --query StandardOutputContent --output text 2>/dev/null | tr -d '[:space:]')
      if [[ "${neffs:-0}" -ge 10 ]]; then ok "Neuron compile cache warm ($neffs compiled graphs)"
      else warn "Neuron compile cache has ${neffs:-0} graphs; run scripts/demo.sh --prewarm healthy (about 10 minutes)"; fi
    fi
  fi
fi

echo
if [[ $failures == 0 ]]; then echo "Preflight passed."; else echo "Preflight: $failures check(s) failed."; fi
exit $(( failures > 0 ))
