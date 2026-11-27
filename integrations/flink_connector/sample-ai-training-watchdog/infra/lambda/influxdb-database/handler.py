# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""CloudFormation custom resource that creates an InfluxDB 3 database if it does not exist,
creates every measurement the pipeline writes (so dashboard queries never fail on a missing
table), and creates a read-only and a write-only token for the database, each stored in its own
Secrets Manager secret.

The read-only token is what every reader uses: Grafana, and the trainer and simulator, which poll
the `incident` table for their run's control commands. Flink writes with the write-only token.
Only this bootstrap holds the admin token. An existing token in a secret is kept while it still
authenticates.

InfluxDB 3 creates a table on its first write. A query naming a table that does not exist yet
(for example `sample`, written only by real GRPO runs) fails outright, as does any UNION that
includes it. Each table is created by one placeholder point dated 2000-01-01, outside every
dashboard time range, whose only fields are integers the pipeline also writes as integers.
"""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

import boto3

# Every measurement the Flink job writes (sink/Points.java), with the tags and fields the dashboard
# queries, in the types the pipeline writes: telemetry metrics are always floats; step, attempt,
# and other counters are integers; incident text and sampled SQL are strings.
F = 0.0
I = "0i"
S = '""'
MEASUREMENTS = {
    "train_step": ({"phase": "_", "rank": "_", "node": "_", "local_rank": "_"},
                   {"loss": F, "grad_norm": F, "lr": F, "step_time_ms": F, "tokens_per_s": F}),
    "layer_stats": ({"phase": "_", "layer": "_", "module": "_", "rank": "_", "node": "_", "local_rank": "_"},
                    {"grad_norm": F, "update_ratio": F, "weight_norm": F, "grad_to_weight": F}),
    # Attention entropy per layer and head (nats), from each rank: the trainer with --head-stats,
    # and the simulator's production profile.
    "attn_stats": ({"phase": "_", "layer": "_", "head": "_", "rank": "_", "node": "_", "local_rank": "_"},
                   {"entropy": F, "max_prob": F}),
    "neuron_hw": ({"phase": "_", "rank": "_", "neuron_core": "_"}, {"util_pct": F, "mem_used_gb": F, "exec_latency_ms": F}),
    "eval": ({"phase": "_", "eval_set": "_"},
             {"eval_loss": F, "token_accuracy": F, "exact_match": F, "gen_exact_match": F, "exec_accuracy": F, "returns_rows": F,
              "exec_accuracy_no_data": F, "exec_accuracy_has_data": F, "exec_accuracy_unseen_table": F,
              "no_data_questions": F, "heldout_filters_mean": F, "heldout_entity_filter_rate": F,
              "exec_accuracy_window": F, "window_questions": F}),
    "rl_step": ({"phase": "_"}, {"reward": F, "exec_match": F, "kl": F, "entropy": F, "sample_nll": F, "resp_len": F,
                                 "step_time_ms": F, "paid_but_wrong_rate": F, "honest_unpaid_rate": F,
                                 "nonzero_advantage_frac": F, "mean_abs_advantage": F, "filters_mean": F,
                                 "entity_filter_rate": F, "result_rows_mean": F}),
    "rl_candidate": ({"phase": "_", "rank": "_", "candidate": "_", "question_id": "_", "question_shape": "_",
                      "result_class": "_"},
                     {"lenient_reward": F, "strict_correct": F, "trained_reward": F, "advantage": F, "resp_len": F,
                      "result_rows": F, "filters": F, "entity_filter": F, "group_filter": F, "threshold_filter": F,
                      "time_filter": F, "has_limit": F, "has_group_by": F, "agg_count": F, "agg_none": F}),
    # One record per GRPO prompt a step (the simulator's production profile): the group's mean
    # reward and strict correctness.
    "rl_prompt": ({"phase": "_", "prompt_id": "_", "prompt_shape": "_"},
                  {"reward_mean": F, "exec_match_mean": F, "grader_gap": F, "group_size": F}),
    # One record per held-out question per evaluation: whether the model's answer matched.
    "eval_question": ({"phase": "_", "eval_set": "_", "question_id": "_", "question_type": "_", "window": "_"},
                      {"correct": F, "returns_rows": F, "filters": F, "entity_filter": F}),
    # One record per run at its start: the run's settings.
    "run_info": ({"phase": "_", "seed": "_", "grader": "_", "watched": "_", "model": "_"},
                 {"lr": F, "grpo_lr": F, "kl_beta": F, "group_size": F, "temperature": F, "lora_r": F,
                  "sft_steps": F, "grpo_steps": F, "world_size": F}),
    "neuron_runtime": ({"phase": "_", "runtime": "_"},
                       {"completed": F, "errors": F, "latency_p50_ms": F, "latency_p99_ms": F, "vcpu_user_pct": F}),
    "host_cpu": ({"phase": "_", "cpu": "_"}, {"user_pct": F, "system_pct": F, "idle_pct": F}),
    "host_mem": ({"phase": "_"}, {"used_gb": F, "total_gb": F}),
    "sample": ({"phase": "_"}, {"reward": F, "exec_match": F, "sql": S, "question": S, "gold": S}),
    "checkpoint": ({"phase": "_"}, {}),
    "heartbeat": ({"phase": "_"}, {}),
    # Control plane telemetry from the trainers and simulator: what each poller saw and did.
    "control_poll": ({"phase": "_", "endpoint": "_"},
                     {"polls": F, "errors": F, "rows": F, "latency_p50_ms": F, "latency_max_ms": F}),
    "control_ack": ({"phase": "_", "action": "_", "type": "_"},
                    {"incident_id": S, "outcome": S, "seen_latency_ms": F, "apply_latency_ms": F}),
    "step_summary": ({"phase": "_"},
                     {"ranks_reporting": I, "loss": F, "grad_norm": F, "step_time_median_ms": F, "step_time_max_ms": F,
                      "slowest_rank": I, "skew": F, "loss_ema": F, "grad_norm_ema": F}),
    "incident": ({"type": "_", "state": "_", "severity": "_"},
                 {"incident_id": S, "onset_step": I, "detections": I, "message": S, "action": S, "action_arg": S,
                  "open_duration_s": F}),
}
PLACEHOLDER_NS = 946_684_800_000_000_000  # 2000-01-01T00:00:00Z

_LOGGER = logging.getLogger()
_LOGGER.setLevel(logging.INFO)


def _token(secret_arn, json_field):
    secret = boto3.client("secretsmanager").get_secret_value(SecretId=secret_arn)["SecretString"]
    return json.loads(secret)[json_field] if json_field else secret


def _post(url, token, body, content_type):
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": content_type, "Accept": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=30)


def _token_works(endpoint, database, token, action):
    """True if the token can still `action` the database: run a query ("read"), or rewrite the
    `heartbeat` placeholder point ("write"), which leaves the data as it was. Both go to the node
    that takes writes, which knows the database immediately; a query-only node learns of it at the
    next catalog sync."""
    if action == "read":
        url = f"{endpoint}/api/v3/query_sql"
        body, content_type = json.dumps({"db": database, "q": "SELECT 1", "format": "json"}).encode(), "application/json"
    else:
        url = f"{endpoint}/api/v3/write_lp?" + urllib.parse.urlencode({"db": database, "precision": "nanosecond"})
        body, content_type = _placeholder("heartbeat").encode(), "text/plain; charset=utf-8"
    try:
        with _post(url, token, body, content_type) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError as error:
        _LOGGER.info("stored %s token rejected: HTTP %s", action, error.code)
        return False


def _ensure_token(endpoint, database, admin_token, secret_arn, action):
    """Keeps the `action` ("read" or "write") token for the database in the secret: the stored one
    while it still works, else a new one."""
    secrets = boto3.client("secretsmanager")
    try:
        stored = json.loads(secrets.get_secret_value(SecretId=secret_arn)["SecretString"])
    except (ValueError, TypeError):
        stored = {}  # the secret's initial value is a generated string, not JSON
    if isinstance(stored, dict) and stored.get("token") and _token_works(endpoint, database, stored["token"], action):
        _LOGGER.info("%s token %s still valid; keeping it", action, stored.get("token_name"))
        return
    # Token names are unique per cluster; a timestamped name never collides with an earlier one.
    name = f"{database}-{action}-{int(time.time())}"
    body = json.dumps({
        "token_name": name,
        "permissions": [{"resource_type": "db", "resource_names": [database], "actions": [action]}],
    }).encode()
    try:
        with _post(f"{endpoint}/api/v3/enterprise/configure/token", admin_token, body, "application/json") as response:
            created = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        raise RuntimeError(f"creating {action} token for {database} failed: HTTP {error.code}: {detail}") from None
    secrets.put_secret_value(
        SecretId=secret_arn,
        SecretString=json.dumps({"token": created["token"], "token_name": name, "database": database}),
    )
    _LOGGER.info("created %s-only token %s for %s", action, name, database)


def _placeholder(name):
    """The placeholder point that creates measurement `name`."""
    tags, fields = MEASUREMENTS[name]
    tag_text = ",".join(f"{k}={v}" for k, v in {"run_id": "_schema", "scenario": "_schema", **tags}.items())
    field_text = ",".join(f"{k}={v}" for k, v in {"step": I, "attempt": I, **fields}.items())
    return f"{name},{tag_text} {field_text} {PLACEHOLDER_NS}"


def _create_measurements(endpoint, database, token):
    lines = "\n".join(_placeholder(name) for name in MEASUREMENTS)
    url = f"{endpoint}/api/v3/write_lp?" + urllib.parse.urlencode({"db": database, "precision": "nanosecond"})
    with _post(url, token, lines.encode(), "text/plain; charset=utf-8") as response:
        _LOGGER.info("created %d measurements in %s: HTTP %s", len(MEASUREMENTS), database, response.status)


def on_event(event, _context):
    props = event["ResourceProperties"]
    database = props["Database"]
    physical_id = f"{props['Endpoint']}/{database}"
    if event["RequestType"] == "Delete":
        # The database is removed with the cluster; keep data on stack updates.
        return {"PhysicalResourceId": event.get("PhysicalResourceId", physical_id)}

    endpoint = props["Endpoint"].rstrip("/")
    token = _token(props["SecretArn"], props.get("SecretJsonField"))
    try:
        with _post(f"{endpoint}/api/v3/configure/database", token, json.dumps({"db": database}).encode(),
                   "application/json") as response:
            _LOGGER.info("created database %s: HTTP %s", database, response.status)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")[:500]
        if error.code == 409 or "already exists" in body:
            _LOGGER.info("database %s already exists", database)
        else:
            raise RuntimeError(f"creating database {database} failed: HTTP {error.code}: {body}") from None
    try:
        _create_measurements(endpoint, database, token)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")[:500]
        raise RuntimeError(f"creating measurements in {database} failed: HTTP {error.code}: {body}") from None
    if props.get("ReadTokenSecretArn"):
        _ensure_token(endpoint, database, token, props["ReadTokenSecretArn"], "read")
    if props.get("WriteTokenSecretArn"):
        _ensure_token(endpoint, database, token, props["WriteTokenSecretArn"], "write")
    return {"PhysicalResourceId": physical_id}
