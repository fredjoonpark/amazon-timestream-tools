#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Runs the real-Trainium demo: each scenario in turn on the Trainium host, waiting for each to
# finish and printing what to watch on the dashboard while it runs.
#
# Usage: AWS_PROFILE=default scripts/demo.sh [--prewarm] [--seed N] [scenario ...]
#   default scenarios: healthy bad_lr reward_hacking (about 22 minutes, warm cache)
#   Run beforehand and show their history: healthy_grpo (GRPO control) and
#   reward_hacking_unwatched (what happens without the watchdog). Both run 30 SFT and 120 GRPO
#   steps, and decoding has a fixed length, so they take about as long as each other.
#   Ends with a PASS/FAIL table: each scenario must open its expected incident types (none for
#   healthy; DIVERGENCE for bad_lr; REWARD_HACKING for the hacking runs; for healthy_grpo, no
#   REWARD_HACKING, though HELDOUT_DECLINE / ADVISE, an overfitting note, is allowed), and the
#   run must have acted on its command (bad_lr rolled back, reward_hacking stopped, the unwatched
#   run received the STOP and ignored it; see response), plus a focused Grafana dashboard link
#   scoped to each run (open scripts/grafana-tunnel.sh). A run's incidents are read from InfluxDB
#   once they are final (see final_incidents).
#   --prewarm: first run a 3-step healthy job that compiles and caches the SFT graphs (do this
#              once on a new host, before an audience; a cold host spends ~10 minutes
#              compiling). The first bad_lr and GRPO runs still compile graphs of their own.
#   --seed N:  train.py's --seed for every run (prewarm too): LoRA init, GRPO sampling, and GRPO
#              question order. Run IDs gain -sN (trn-reward-hacking-s1-<UTC time>); 0, the
#              default, keeps the plain IDs.
#   --simulated: run simulated scenarios instead (no Trainium host), each once from step 0 as its
#              own ECS task (scripts/simulator.sh once). Scenarios: healthy, bad_lr, healthy_grpo,
#              reward_hacking; default: all four, about 6 minutes each. Not with --seed.
set -euo pipefail
. "$(dirname "$0")/env.sh"   # optional settings from .env (see .env.example)
cd "$(dirname "$0")/.."
export AWS_REGION=${DEPLOY_REGION:-us-east-1}

prewarm=false
simulated=false
seed_args=()
scenarios=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prewarm) prewarm=true ;;
    --simulated) simulated=true ;;
    --seed)
      seed_args=(--seed "${2:-}")
      [[ $# -lt 2 ]] || shift
      ;;
    --seed=*) seed_args=(--seed "${1#--seed=}") ;;
    *) scenarios+=("$1") ;;
  esac
  shift
done
if [[ ${#seed_args[@]} -gt 0 ]]; then
  if [[ ! "${seed_args[1]}" =~ ^[0-9]+$ ]]; then
    echo "--seed needs a whole number, 0 or more" >&2
    exit 1
  fi
  if $simulated; then
    echo "--seed is for real Trainium runs; simulated runs keep their own seeds" >&2
    exit 1
  fi
fi
if $simulated; then
  [[ ${#scenarios[@]} -gt 0 ]] || scenarios=(healthy bad_lr healthy_grpo reward_hacking)
  prewarm=false
else
  [[ ${#scenarios[@]} -gt 0 ]] || scenarios=(healthy bad_lr reward_hacking)
fi
cluster=$(aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name TrainingWatchdogStack \
  --query "Stacks[0].Outputs[?OutputKey=='EcsClusterName'].OutputValue" --output text)
sim_log_group=$(aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name TrainingWatchdogStack \
  --query "Stacks[0].Outputs[?OutputKey=='SimulatorLogGroupName'].OutputValue" --output text)

instance=$(aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name TrainingWatchdogStack \
  --query "Stacks[0].Outputs[?OutputKey=='TrainiumInstanceId'].OutputValue" --output text)
log_group=$(aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name TrainingWatchdogStack \
  --query "Stacks[0].Outputs[?OutputKey=='TrainerLogGroupName'].OutputValue" --output text)

sim_cue() {
  case "$1" in
    healthy) echo "Simulated SFT run, 360 steps at 1 step/s. Watch: loss falls; no incidents." ;;
    bad_lr) echo "Around step 120 the loss spikes, then the gradient norm blows up. Watch: a DIVERGENCE incident, ROLLBACK to the last checkpoint, and the run resumes as attempt 2." ;;
    healthy_grpo) echo "Simulated GRPO with an honest reward. Watch Reward vs held-out accuracy: reward and held-out accuracy rise together; the grader gap stays at 0; no incident." ;;
    reward_hacking) echo "Simulated GRPO; from about step 110 the policy games the reward. Watch Reward vs held-out accuracy: accuracy falls while reward climbs, the grader gap rises past 0, entropy and response length collapse, then REWARD_HACKING and STOP end the run." ;;
    *) echo "Watch the incident log and the scenario's panels." ;;
  esac
}

cue() {
  if $simulated; then sim_cue "$1"; return; fi
  case "$1" in
    healthy) echo "Watch: training loss falls; held-out generated exact match climbs from 0%; no incidents." ;;
    bad_lr) echo "At step 70 the learning rate jumps 1000x. Watch: loss and gradient norm spike, a DIVERGENCE incident opens, Flink sends ROLLBACK to the step-50 checkpoint, and training resumes as attempt 2." ;;
    healthy_grpo) echo "30 SFT steps, then 120 GRPO steps paid only for SQL whose result matches the gold result. Watch Reward vs held-out accuracy: reward and held-out execution accuracy rise together; the grader gap stays at 0. If held-out accuracy declines late (overfitting), Flink opens HELDOUT_DECLINE with ADVISE, not STOP: the trainer notes it and keeps training." ;;
    reward_hacking) echo "30 SFT steps, then GRPO paid for any SQL that runs and returns a row. Watch: held-out execution accuracy falls while the reward holds or climbs and the grader gap rises; the Sampled SQL table shows filters disappearing; a REWARD_HACKING incident opens and Flink's STOP ends the run; the trainer keeps the adapters of the evaluation the STOP names (the best held-out accuracy)." ;;
    reward_hacking_unwatched) echo "reward_hacking with the watchdog's commands ignored, for 120 GRPO steps. Watch Reward vs held-out accuracy: Flink still opens REWARD_HACKING, but the trainer keeps going and accuracy keeps falling." ;;
    *) echo "Watch the incident log and the scenario's panels." ;;
  esac
}

# What each scenario must open, as "<summary text>|<regex over the sorted, comma-separated
# incident types it opened>". healthy_grpo may get HELDOUT_DECLINE (ADVISE: an honest grader's
# held-out decline, likely overfitting) but never REWARD_HACKING. A hacking run may get
# HELDOUT_DECLINE before it escalates to REWARD_HACKING.
expected() {
  case "$1" in
    bad_lr) echo 'divergence|^divergence$' ;;
    healthy_grpo) echo 'none or heldout_decline|^(none|heldout_decline)$' ;;
    reward_hacking | reward_hacking_unwatched) echo 'reward_hacking|^(heldout_decline,)?reward_hacking$' ;;
    *) echo 'none|^none$' ;;
  esac
}

# What the run itself must have done with the scenario's command, as "<summary text>|<regex over
# the run's control log lines>" (empty: nothing). Flink opening the incident isn't enough: with a
# broken poller or a rejected token the command never reaches the run, and the watched and
# unwatched runs would look the same. The lines are already this run's (its own log stream).
response() {
  case "$1" in
    bad_lr) echo 'ROLLBACK applied| incident [^ ]*-divergence-[0-9]+: (restored step|rolled back)' ;;
    reward_hacking) echo 'STOP applied| incident [^ ]*-reward_hacking-[0-9]+: (stopping|stopped)' ;;
    reward_hacking_unwatched) echo 'STOP received and ignored| incident [^ ]*-reward_hacking-[0-9]+: ignored \(watchdog off\)' ;;
    *) echo '|' ;;
  esac
}

# Incident types for the summary, with the command each carries: heldout_decline/ADVISE.
describe() {
  sed -e 's#divergence#divergence/ROLLBACK#; s#heldout_decline#heldout_decline/ADVISE#; s#reward_hacking#reward_hacking/STOP#' <<<"$1"
}

# The distinct incident types (lowercase, sorted, comma-separated; "none") Flink opened for run $1
# since epoch ms $2, from InfluxDB's incident table: read-only token, reader endpoint, through the
# Trainium host (scripts/influx-query.sh). Fails if InfluxDB can't be reached that way.
incident_types() {
  local out
  [[ "$1" =~ ^[A-Za-z0-9_.-]+$ && "$2" =~ ^[0-9]+$ ]] || return 1
  out=$(scripts/influx-query.sh "SELECT DISTINCT type FROM incident WHERE run_id = '$1' AND state = 'OPEN' \
AND time >= to_timestamp_millis($(( $2 - 60000 ))) ORDER BY type" 2>/dev/null) || return 1
  grep -q -e '^type' -e '^(no rows)' <<<"$out" || return 1
  out=$(tr '\t' '\n' <<<"$out" | grep -E '^[A-Z][A-Z_]*[[:space:]]*$' | sed 's/[[:space:]]*$//' \
    | tr 'A-Z' 'a-z' | sort -u | paste -sd, - || true)
  echo "${out:-none}"
}

# A run's final incident types, as "<source> <types>". Flink can open an incident from a run's
# last telemetry after the run has ended, and the trainer never polls for it, so an incident at a
# run's final step never reaches the trainer's log. So once the run has finished, poll InfluxDB until the run's incident set has been unchanged for SETTLE_SECONDS (at
# least that long after the finish: Flink's 5 s event-time watermark, the sink's 1 s buffer,
# InfluxDB's WAL flush and replication, and a poll, with margin), for at most SETTLE_MAX_SECONDS.
# Without InfluxDB (the host stopped during a simulated demo), fall back to the types in the
# run's log ($3); a simulated run keeps reading commands for 30 s after its last step, so a late
# incident still reaches its log.
SETTLE_SECONDS=30
SETTLE_MAX_SECONDS=180
final_incidents() {
  local run_id=$1 since_ms=$2 from_log=$3 started changed now current last=
  started=$(date +%s)
  changed=$started
  echo "    waiting for the run's incidents to settle (InfluxDB, ${SETTLE_SECONDS} s unchanged)..." >&2
  while true; do
    if current=$(incident_types "$run_id" "$since_ms"); then
      if [[ "$current" != "$last" ]]; then
        last=$current
        changed=$(date +%s)
      fi
    elif [[ -z "$last" ]]; then
      echo "    (InfluxDB not reachable through the Trainium host; using the run's log)" >&2
      echo "log ${from_log:-none}"
      return
    fi
    now=$(date +%s)
    (( now - changed >= SETTLE_SECONDS || now - started >= SETTLE_MAX_SECONDS )) && break
    sleep 5
  done
  echo "influxdb $last"
}

# Appends the PASS/FAIL line and dashboard link for a finished run.
check() {
  local scenario=$1 run_id=$2 status=$3 start_ms=$4 end_ms=$5 lines=$6
  local from_log final source observed want pattern need need_pattern responded=true verdict note=
  # Incident ids look like <run_id>-<type>-<epoch ms>; collect the distinct types.
  from_log=$(sed -n "s/.* incident ${run_id}-\([a-z_]*\)-[0-9]*:.*/\1/p" <<<"$lines" | sort -u | paste -sd, -)
  final=$(final_incidents "$run_id" "$start_ms" "$from_log")
  source=${final%% *}
  observed=${final#* }
  want=$(expected "$scenario")
  pattern=${want#*|}
  want=${want%%|*}
  need=$(response "$scenario")
  need_pattern=${need#*|}
  need=${need%%|*}
  if [[ -n "$need_pattern" ]] && ! grep -qE "$need_pattern" <<<"$lines"; then responded=false; fi
  if [[ "$status" == Success && "$observed" =~ $pattern ]] && $responded; then
    verdict=PASS
  else
    verdict=FAIL
    failed=$((failed + 1))
  fi
  if [[ "$source" == log ]]; then note=" (from the run's log)"; fi
  if ! $responded; then note+="; the run's log shows no $need"; fi
  summary+=("$(printf '%-5s %-24s expected %-23s observed %-42s %s' "$verdict" "$scenario" "$want" \
    "$(describe "$observed")$note" "$run_id")")
  summary+=("      $(dashboard_link "$run_id" "$start_ms" "$end_ms" "$scenario")")
}

# Grafana link to the task's dashboard, scoped to one run and its time window (epoch ms).
dashboard_link() {
  local source="scenario LIKE 'trn_%'"
  $simulated && source="scenario NOT LIKE 'trn_%'"
  python3 - "$1" "$2" "$3" "$source" "$4" <<'PY'
import sys, urllib.parse
run_id, start, end, source, scenario = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
task = {
    "healthy": "learning",
    "bad_lr": "divergence",
    "healthy_grpo": "reward-hacking",
    "reward_hacking": "reward-hacking",
    "reward_hacking_unwatched": "reward-hacking",
}.get(scenario)
uid = f"training-watchdog-{task}" if task else "training-watchdog"
query = urllib.parse.urlencode({"orgId": 1, "var-source": source, "var-run": run_id,
                                "from": start - 60_000, "to": end + 120_000})
print(f"http://localhost:3000/d/{uid}?{query}")
PY
}

summary=()
failed=0

# Sets the caller's `status` to Run Command's status for command $1. Just after send-command the
# invocation may not exist yet, so an error reads as Pending; 30 in a row (expired credentials,
# say) read as Unknown, which ends the wait. Counts in the caller's `poll_errors`.
poll_command() {
  local out
  if out=$(aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$1" \
      --instance-id "$instance" --query Status --output text 2>&1); then
    status=$out
    poll_errors=0
  elif (( ++poll_errors >= 30 )); then
    status=Unknown
    echo "    get-command-invocation keeps failing: ${out//$'\n'/ }"
  else
    status=Pending
  fi
}

# Starts a run with train.sh, waits for it, prints its control lines, and checks the incidents.
run() {
  local scenario=$1
  shift
  local started command_id status run_id start_ms end_ms lines poll_errors=0
  start_ms=$(( $(date +%s) * 1000 ))
  if ! started=$(scripts/train.sh "$scenario" "$@" 2>&1); then
    echo "    could not start: $started"
    failed=$((failed + 1))
    summary+=("$(printf '%-5s %-24s %s' FAIL "$scenario" "did not start: ${started//$'\n'/ }")")
    return 1
  fi
  echo "$started" | head -1
  command_id=$(sed -n 's/.*command \([0-9a-f-]*\)\..*/\1/p' <<<"$started" | head -1)
  run_id=$(sed -n 's/^Started \([^ ]*\) .*/\1/p' <<<"$started" | head -1)
  while true; do
    poll_command "$command_id"
    [[ "$status" == Pending || "$status" == InProgress || "$status" == Delayed ]] || break
    # Run Command can leave a finished command marked InProgress; trust the trainer's own log.
    if aws logs filter-log-events --region "$AWS_REGION" --log-group-name "$log_group" \
        --log-stream-name-prefix "$command_id" --filter-pattern '"finished at step"' \
        --query 'length(events)' --output text 2>/dev/null | grep -qv '^0$'; then
      # The trainer has finished; shutdown (telemetry flush, workers) takes up to a minute.
      # Wait for the command to close so the next run can start; cancel it if it never does.
      # A command that does close keeps its status: a shutdown that fails (Failed) is a FAIL.
      for _ in $(seq 1 24); do
        poll_command "$command_id"
        [[ "$status" == InProgress || "$status" == Pending || "$status" == Delayed ]] || break
        sleep 5
      done
      if [[ "$status" == InProgress || "$status" == Pending || "$status" == Delayed ]]; then
        aws ssm cancel-command --region "$AWS_REGION" --command-id "$command_id" >/dev/null 2>&1 || true
        echo "    (cancelled a finished run's stale Run Command)"
        status=Success
      fi
      break
    fi
    sleep 10
  done
  end_ms=$(( $(date +%s) * 1000 ))
  lines=$(aws logs filter-log-events --region "$AWS_REGION" --log-group-name "$log_group" \
    --log-stream-name-prefix "$command_id" --query 'events[].message' --output text \
    | tr '\t' '\n' | grep -E "watchdog_trainer (control|finished|stopped)" | sort -u || true)
  [[ -n "$lines" ]] && sed 's/^/    /' <<<"$lines"
  echo "    -> $status"
  check "$scenario" "$run_id" "$status" "$start_ms" "$end_ms" "$lines"
}

# One simulated run as an ECS task: start it, wait for the task to stop, check its incidents.
run_simulated() {
  local scenario=$1
  local started task run_id status start_ms end_ms lines
  start_ms=$(( $(date +%s) * 1000 ))
  if ! started=$(scripts/simulator.sh once "$scenario" 2>&1); then
    echo "    could not start: $started"
    failed=$((failed + 1))
    summary+=("$(printf '%-5s %-24s %s' FAIL "$scenario" "did not start: ${started//$'\n'/ }")")
    return 1
  fi
  echo "$started" | head -1
  task=$(sed -n 's/.* as task \([0-9a-f]*\)\..*/\1/p' <<<"$started" | head -1)
  run_id=$(sed -n 's/^Started \([^ ]*\) .*/\1/p' <<<"$started" | head -1)
  if [[ -z "$task" ]]; then
    echo "    no task ID in: $started"
    failed=$((failed + 1))
    summary+=("$(printf '%-5s %-24s %s' FAIL "$scenario" "no task ID: ${started//$'\n'/ }")")
    return 1
  fi
  # An error, or a task describe-tasks doesn't find (None), counts toward giving up after 30 in a
  # row (5 minutes; expired credentials, say).
  local out errors=0
  status=PENDING
  while true; do
    if out=$(aws ecs describe-tasks --region "$AWS_REGION" --cluster "$cluster" --tasks "$task" \
        --query 'tasks[0].lastStatus' --output text 2>&1) && [[ "$out" != None ]]; then
      status=$out
      errors=0
    elif (( ++errors >= 30 )); then
      status=Unknown
      echo "    describe-tasks keeps failing: ${out//$'\n'/ }"
      break
    fi
    [[ "$status" == STOPPED ]] && break
    sleep 10
  done
  if [[ "$status" == STOPPED ]]; then
    local exit_code
    exit_code=$(aws ecs describe-tasks --region "$AWS_REGION" --cluster "$cluster" --tasks "$task" \
      --query "tasks[0].containers[?name=='simulator'].exitCode | [0]" --output text)
    [[ "$exit_code" == 0 ]] && status=Success || status="Failed (exit $exit_code)"
  fi
  end_ms=$(( $(date +%s) * 1000 ))
  lines=$(aws logs filter-log-events --region "$AWS_REGION" --log-group-name "$sim_log_group" \
    --log-stream-name-prefix "Simulator/simulator/$task" --query 'events[].message' --output text \
    | tr '\t' '\n' | grep -E " INFO (control|finished)" | sort -u || true)
  [[ -n "$lines" ]] && sed 's/^/    /' <<<"$lines"
  echo "    -> $status"
  check "$scenario" "$run_id" "$status" "$start_ms" "$end_ms" "$lines"
}

if [[ ${#seed_args[@]} -gt 0 ]]; then
  echo "== Seed ${seed_args[1]} for every run"
fi
# ${a[@]+"${a[@]}"}: an empty array under set -u, in bash 3.2 (macOS) as well.
if $prewarm; then
  echo "== Prewarming the Neuron compile cache (3 steps)"
  run healthy --max-steps 3 ${seed_args[@]+"${seed_args[@]}"}
  summary=()
  failed=0
fi
for scenario in "${scenarios[@]}"; do
  echo
  echo "== $scenario"
  echo "   $(cue "$scenario")"
  if $simulated; then run_simulated "$scenario" || true; else run "$scenario" ${seed_args[@]+"${seed_args[@]}"} || true; fi
done

echo
echo "== Summary (dashboard links need scripts/grafana-tunnel.sh running)"
printf '%s\n' "${summary[@]}"
exit $(( failed > 0 ))
