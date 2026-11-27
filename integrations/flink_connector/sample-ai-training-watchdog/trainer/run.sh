#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Runs one training scenario on the Trainium host (both NeuronCores of a trn1.2xlarge).
# Usage: run.sh <healthy|bad_lr|healthy_grpo|reward_hacking|reward_hacking_unwatched> <run-id> [extra train.py args]
# Env: TELEMETRY_STREAM, INFLUX_URL, INFLUX_DB, INFLUX_TOKEN_SECRET, AWS_REGION; optional MODEL.
set -euo pipefail
cd "$(dirname "$0")"
scenario=$1
run_id=$2
shift 2
model=${MODEL:-Qwen/Qwen2.5-1.5B}

# The base PyTorch venv, e.g. /opt/aws_neuronx_venv_pytorch_2_9 (not the _nxd_* variants).
venv=$(ls -d /opt/aws_neuronx_venv_pytorch_2_* 2>/dev/null | grep -E '_pytorch_2_[0-9]+$' | sort -V | tail -1) || true
if [[ -z "$venv" ]]; then
  echo "No Neuron PyTorch venv under /opt; is this the Neuron DLAMI?" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$venv/bin/activate"
python -m pip install -q -r requirements.txt \
  -c <(python -m pip freeze | grep -iE '^(torch|torch-xla|torch-neuronx|neuronx-cc|libneuronxla)==')

export HF_HOME=/opt/watchdog/hf
export NEURON_COMPILE_CACHE_URL=/opt/watchdog/neuron-cache
export NEURON_CC_FLAGS="--model-type=transformer"
export PATH="/opt/aws/neuron/bin:$PATH"
# Download once, before two ranks race for it.
python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "$model"

exec torchrun --standalone --nproc_per_node=2 -m watchdog_trainer.train \
  --scenario "$scenario" --run-id "$run_id" --model "$model" "$@"
