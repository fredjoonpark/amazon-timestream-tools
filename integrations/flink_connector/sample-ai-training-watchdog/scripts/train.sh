#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Starts a real Trainium training run on the stack's trn1.2xlarge through SSM Run Command.
# Usage: AWS_PROFILE=default scripts/train.sh <healthy|bad_lr|healthy_grpo|reward_hacking|reward_hacking_unwatched> [--seed N] [train.py args]
#        AWS_PROFILE=default scripts/train.sh stop    # cancel the run and free the NeuronCores
# The run ID is trn-<scenario>-<UTC time to the second>, or trn-<scenario>-s<N>-<UTC time> with
# --seed N (N > 0); --seed and every other argument go to train.py. Run IDs must not repeat: Flink
# keeps each run's detector state for 7 days.
# Output streams to CloudWatch Logs; the script prints the log group and stream.
set -euo pipefail
. "$(dirname "$0")/env.sh"   # optional settings from .env (see .env.example)
scenario=${1:?usage: train.sh <healthy|bad_lr|healthy_grpo|reward_hacking|reward_hacking_unwatched|stop> [--seed N] [train.py args]}
shift
export AWS_REGION=${DEPLOY_REGION:-us-east-1}

# train.py's --seed also names the run, so runs with different seeds never share an ID.
seed=0
previous=
for arg in "$@"; do
  [[ "$previous" == --seed ]] && seed=$arg
  [[ "$arg" == --seed=* ]] && seed=${arg#--seed=}
  previous=$arg
done
if [[ "$previous" == --seed || ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "--seed needs a whole number, 0 or more" >&2
  exit 1
fi
seed=$((10#$seed))
stack=TrainingWatchdogStack

output() {
  aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name "$stack" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
instance=$(output TrainiumInstanceId)
code=$(output TrainerCodeS3Url)
log_group=$(output TrainerLogGroupName)
telemetry=$(output TelemetryStreamName)
# Control plane: the trainer polls InfluxDB's incident table on the reader endpoint.
influx_url=$(output InfluxDbReaderEndpoint)
influx_db=$(output InfluxDbDatabase)
influx_token_secret=$(output InfluxDbReadTokenSecretArn)
if [[ -z "$instance" || "$instance" == None ]]; then
  echo "Stack $stack in $AWS_REGION has no Trainium host; deploy with -c trainium=true" >&2
  exit 1
fi

if [[ "$scenario" == stop ]]; then
  # Cancel only training runs (comment "watchdog trn-..."), not preflight or query commands.
  # Cancelling the Run Command leaves torchrun's workers holding the NeuronCores; stop them too.
  for id in $(aws ssm list-commands --region "$AWS_REGION" --instance-id "$instance" --output text \
      --query "Commands[?(Status=='InProgress' || Status=='Pending') && starts_with(Comment || '', 'watchdog trn-')].CommandId"); do
    aws ssm cancel-command --region "$AWS_REGION" --command-id "$id" && echo "Cancelled $id"
  done
  aws ssm send-command --region "$AWS_REGION" --instance-ids "$instance" --document-name AWS-RunShellScript \
    --comment "watchdog stop" --query Command.CommandId --output text \
    --parameters '{"commands":["pkill -f watchdog_trainer.train; pkill -f \"torchrun --standalone\"; pkill -x neuron-monitor; true"]}' \
    >/dev/null && echo "Stopped trainer processes on $instance"
  exit 0
fi

state=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$instance" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)
if [[ "$state" != running ]]; then
  echo "Trainium host $instance is $state; start it with: aws ec2 start-instances --instance-ids $instance" >&2
  exit 1
fi
# Count only training runs (comment "watchdog trn-..."), not preflight, query, or stop commands.
# --output json: with text output the CLI applies --query per 50-command page, printing one count
# per page ("0\n0"), which never equals 0 once the host has more than 50 commands.
busy=$(aws ssm list-commands --region "$AWS_REGION" --instance-id "$instance" --output json \
  --query "length(Commands[?(Status=='InProgress' || Status=='Pending') && starts_with(Comment || '', 'watchdog trn-')])")
if [[ "$busy" != 0 ]]; then
  echo "A training run is already in progress on $instance; one run at a time uses both NeuronCores" >&2
  echo "(scripts/train.sh stop cancels it)" >&2
  exit 1
fi

seed_part=
[[ "$seed" == 0 ]] || seed_part="-s$seed"
run_id="trn-${scenario//_/-}${seed_part}-$(date -u +%Y%m%d%H%M%S)"
parameters=$(python3 - "$code" "$telemetry" "$influx_url" "$influx_db" "$influx_token_secret" "$AWS_REGION" \
    "$scenario" "$run_id" "$@" <<'EOF'
import json, shlex, sys
code, telemetry, influx_url, influx_db, influx_token_secret, region, scenario, run_id, *extra = sys.argv[1:]
q = shlex.quote
commands = [
    "set -euo pipefail",
    "mkdir -p /opt/watchdog/code /opt/watchdog/logs",
    f"aws s3 cp {q(code)} /tmp/trainer.zip --region {q(region)} --only-show-errors",
    "rm -rf /opt/watchdog/code/trainer && python3 -m zipfile -e /tmp/trainer.zip /opt/watchdog/code/trainer",
    f"export TELEMETRY_STREAM={q(telemetry)} AWS_REGION={q(region)}",
    f"export INFLUX_URL={q(influx_url)} INFLUX_DB={q(influx_db)} INFLUX_TOKEN_SECRET={q(influx_token_secret)}",
    "bash /opt/watchdog/code/trainer/run.sh " + " ".join(q(a) for a in [scenario, run_id, *extra])
    + f" 2>&1 | tee /opt/watchdog/logs/{q(run_id)}.log",
]
# Run Command uses /bin/sh (dash on Ubuntu); run the script under bash for pipefail.
print(json.dumps({"commands": ["exec bash -c " + q("\n".join(commands))], "executionTimeout": ["10800"]}))
EOF
)
command_id=$(aws ssm send-command --region "$AWS_REGION" --instance-ids "$instance" \
  --document-name AWS-RunShellScript --comment "watchdog $run_id" \
  --parameters "$parameters" --timeout-seconds 600 \
  --cloud-watch-output-config "CloudWatchOutputEnabled=true,CloudWatchLogGroupName=$log_group" \
  --query Command.CommandId --output text)

cat <<EOF
Started $run_id (scenario trn_$scenario) on $instance, command $command_id.
Logs: aws logs tail $log_group --region $AWS_REGION --follow --log-stream-name-prefix $command_id
Stop: AWS_PROFILE=${AWS_PROFILE:-default} scripts/train.sh stop
EOF
