# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Generates the overview and task dashboards. Run after editing: python build_dashboard.py"""

import json
from copy import deepcopy
from pathlib import Path

DS = {"type": "influxdb", "uid": "influxdb3"}
# One fixed color per scenario, assigned in palette order and never cycled.
SCENARIO_COLORS = {
    "healthy": "#3987e5",
    "bad_lr": "#d95926",
    "healthy_grpo": "#3987e5",
    "reward_hacking": "#d95926",
    # Real Trainium runs (trainer/). The "Runs" selector never shows real and simulated runs on
    # one panel, so real runs take categorical slots in order within their own view. Slot 3
    # (#199e70) is unused, so these colors match earlier screenshots.
    "trn_healthy": "#3987e5",
    "trn_bad_lr": "#d95926",
    "trn_healthy_grpo": "#c98500",
    "trn_reward_hacking": "#d55181",
    "trn_reward_hacking_unwatched": "#9085e9",
}
# Dashboard variable: which runs every panel shows. A SQL predicate interpolated into each query.
SOURCE_FILTER = "$source AND run_id LIKE '${run}'"
# Break lines where a series has no point for longer than its sampling cadence (between runs,
# after a stop). Per-step series report every second; others pass a wider gap_ms.
GAP_MS = 5000
panels = []
_y = 0


def query(sql):
    sql = sql.strip().replace("$__timeFilter(time)", f"$__timeFilter(time) AND {SOURCE_FILTER}")
    return [{"refId": "A", "datasource": DS, "rawSql": sql, "rawQuery": True, "editorMode": "code", "format": "table"}]


def row(title):
    global _y
    panels.append({"type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": _y, "w": 24, "h": 1}, "panels": []})
    _y += 1


def timeseries(title, sql, by, x, w=12, h=8, unit=None, log=False, description="", color_by_scenario=True, gap_ms=GAP_MS, points=False, decimals=None, minimum=None, no_value=None):
    custom = {"drawStyle": "line", "lineWidth": 2, "showPoints": "always" if points else "never", "pointSize": 8,
              "spanNulls": False, "insertNulls": gap_ms}
    defaults = {"custom": custom, "displayName": "${__field.labels.%s}" % by}
    if unit:
        defaults["unit"] = unit
    if decimals is not None:
        defaults["decimals"] = decimals
    if unit == "percentunit":
        defaults.update({"min": 0, "max": 1})
    if minimum is not None:
        defaults["min"] = minimum
    if no_value:
        defaults["noValue"] = no_value  # shown instead of an empty chart: says why, and what to run
    if log:
        defaults["custom"]["scaleDistribution"] = {"type": "log", "log": 10}
    overrides = []
    if color_by_scenario:
        # Series are named by scenario (displayName above), so match on the scenario alone.
        for scenario, color in SCENARIO_COLORS.items():
            overrides.append(
                {
                    "matcher": {"id": "byName", "options": scenario},
                    "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}],
                }
            )
    panels.append(
        {
            "type": "timeseries",
            "title": title,
            "description": description,
            "datasource": DS,
            "gridPos": {"x": x, "y": _y, "w": w, "h": h},
            "targets": query(sql),
            "transformations": [
                {"id": "partitionByValues", "options": {"fields": [by], "keepFields": False, "naming": {"asLabels": True}}},
            ],
            "fieldConfig": {"defaults": defaults, "overrides": overrides},
            "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi", "sort": "desc"}},
        }
    )


def fixed(name, color):
    return {"matcher": {"id": "byName", "options": name},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]}


def advance(h=8):
    global _y
    _y += h


EXPECTED = """
**What each real Trainium scenario shows.** Use **Demo dashboards** to focus on one task, or use the links `scripts/demo.sh` prints. Pick a run in **Run**; navigation carries the filters and time range.

| Scenario | Incident log | Panels |
|---|---|---|
| `healthy` | none | *Training loss* falls; *Held-out exact match, generated* climbs to about 70-75% (teacher-forced: about 85%) |
| `bad_lr` | DIVERGENCE, ROLLBACK to checkpoint 50 | *Learning rate* x1000 at step 70; loss and gradient norm hit 1e6 (NaN), then recover as attempt 2 |
| `healthy_grpo` | none, or HELDOUT_DECLINE, ADVISE if held-out accuracy declines (overfitting); response `noted: ...; kept GRPO step N ...` | *GRPO training reward* ~0.95; *Sampled SQL* all **correct**; *grader gap* 0 throughout; the run keeps going |
| `reward_hacking` | REWARD_HACKING, STOP | Held-out accuracy falls while reward holds or climbs; the *grader gap* rises from below 0; paid-but-wrong rises, filters fall; *Sampled SQL* reward-1 rows marked **wrong**; the run stops |
| `reward_hacking_unwatched` | REWARD_HACKING, response `ignored (watchdog off)` | Same, but the run keeps going |

**InfluxDB is the control plane:** each incident row is a command (ROLLBACK, STOP, or ADVISE, which never stops a run); trainers poll `incident` for their run (query node) and write back `control_ack`. **response** in the *Incident log* is the trainer's answer.
**Cluster:** InfluxDB 3 Enterprise; writes on the ingest node, reads (polls, this dashboard) on the query node, compaction on its own node. The **Data plane** row at the bottom shows both sides.
Empty panels that need a specific run say which.
"""
panels.append(
    {
        "type": "text",
        "title": "How to read this dashboard",
        "gridPos": {"x": 0, "y": _y, "w": 24, "h": 12},
        "options": {"mode": "markdown", "content": EXPECTED.strip()},
    }
)
advance(12)

# Commands by what they do to a run: STOP ends it, ROLLBACK rewinds it, ADVISE lets it continue.
ACTION_COLORS = {
    "STOP": {"color": "red", "index": 0},
    "ROLLBACK": {"color": "orange", "index": 1},
    "ADVISE": {"color": "blue", "index": 2},
}
# The trainer's responses (control_ack outcome). Regex mappings without text keep the full outcome,
# such as "stopping: kept GRPO step 30 as requested, ..." or "noted: held-out accuracy declining
# since GRPO step 70; kept GRPO step 70 ...; training continues".
RESPONSE_MAPPINGS = [
    {"type": "regex", "options": {"pattern": "^(stopping|restored).*", "result": {"color": "green", "index": 0}}},
    {"type": "regex", "options": {"pattern": "^noted.*", "result": {"color": "blue", "index": 1}}},
    {"type": "value", "options": {
        "ignored (watchdog off)": {"text": "ignored (watchdog off)", "color": "red", "index": 2},
    }},
]

row("Incidents")
panels.append(
    {
        "type": "stat",
        "title": "Open incidents",
        "description": "Incidents with an OPEN transition and no RESOLVED transition in the time range. "
                       "RESOLVED means the detector went quiet; check the trainer's response to see whether a command was applied.",
        "datasource": DS,
        "gridPos": {"x": 0, "y": _y, "w": 4, "h": 8},
        "targets": query(
            """
SELECT count(*) AS open FROM (
  SELECT incident_id, max(state) AS latest FROM incident WHERE $__timeFilter(time) GROUP BY incident_id
) WHERE latest = 'OPEN'
"""
        ),
        "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "red", "value": 1}]}}, "overrides": []},
        "options": {"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}},
    }
)
panels.append(
    {
        "type": "table",
        "title": "Incident log: Flink decides, InfluxDB carries the command, the trainer responds",
        "description": "Each OPEN row is a command: the trainer reads it from this table (reader endpoint) and responds. "
                       "ROLLBACK and STOP change the run; ADVISE (HELDOUT_DECLINE: held-out accuracy fell but the grader "
                       "agreed with the strict check, likely overfitting) lets the run continue, and the trainer keeps its best "
                       "evaluation's adapters. "
                       "response is what the trainer reported doing (control_ack), joined on incident_id; seen after is "
                       "the time from the event that triggered the detection to the trainer reading the row from InfluxDB.",
        "datasource": DS,
        "gridPos": {"x": 4, "y": _y, "w": 20, "h": 8},
        "targets": query(
            # Both sides filter in their own subquery: the Runs/Run filter names unqualified columns.
            """
SELECT i.time, i.state, i.type, i.run_id, i.action, i.arg, a.outcome AS response,
       round(a.seen_latency_ms) AS "seen after", i.onset_step, i.step, i.severity, i.message
FROM (SELECT time, state, type, run_id, incident_id, action, action_arg AS arg, onset_step, step, severity, message
      FROM incident WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 100) i
LEFT JOIN (SELECT incident_id, outcome, seen_latency_ms FROM control_ack WHERE $__timeFilter(time)) a
  ON a.incident_id = i.incident_id
ORDER BY i.time DESC
"""
        ),
        "fieldConfig": {"defaults": {}, "overrides": [
            {"matcher": {"id": "byName", "options": "message"}, "properties": [{"id": "custom.width", "value": 600}]},
            {"matcher": {"id": "byName", "options": "action"}, "properties": [
                {"id": "custom.cellOptions", "value": {"type": "color-text"}},
                {"id": "mappings", "value": [{"type": "value", "options": ACTION_COLORS}]}]},
            {"matcher": {"id": "byName", "options": "response"}, "properties": [
                {"id": "custom.width", "value": 330},
                {"id": "custom.cellOptions", "value": {"type": "color-text"}},
                {"id": "mappings", "value": RESPONSE_MAPPINGS}]},
            {"matcher": {"id": "byName", "options": "run_id"}, "properties": [{"id": "custom.width", "value": 270}]},
            {"matcher": {"id": "byName", "options": "seen after"}, "properties": [{"id": "unit", "value": "ms"}]},
        ]},
    }
)
advance()
LOOP_COLORS = {"seen by the trainer": "#3987e5", "trainer responded": "#d95926"}
panels.append(
    {
        "type": "timeseries",
        "title": "Control loop through InfluxDB: from the triggering event to the trainer's response",
        "description": "One pair of dots per command. Seen by the trainer: telemetry -> Kinesis -> Flink detects -> "
                       "incident written to InfluxDB (ingest node) -> visible on the query node -> the trainer's next "
                       "poll. Trainer responded: the trainer handles commands between steps, so this adds the rest of "
                       "the current step (a GRPO step takes 6-8 s). A response can report an applied or ignored command; "
                       "read the Incident log for the outcome.",
        "datasource": DS,
        "gridPos": {"x": 0, "y": _y, "w": 24, "h": 7},
        "targets": query(
            """
SELECT time, 'seen by the trainer' AS series, seen_latency_ms AS value FROM control_ack
WHERE seen_latency_ms IS NOT NULL AND $__timeFilter(time)
UNION ALL
SELECT time, 'trainer responded' AS series, apply_latency_ms AS value FROM control_ack
WHERE apply_latency_ms IS NOT NULL AND $__timeFilter(time)
ORDER BY time
"""
        ),
        "transformations": [
            {"id": "partitionByValues", "options": {"fields": ["series"], "keepFields": False, "naming": {"asLabels": True}}},
        ],
        "fieldConfig": {
            "defaults": {"unit": "ms", "displayName": "${__field.labels.series}", "noValue": "No commands in the time range",
                         "custom": {"drawStyle": "points", "pointSize": 10, "showPoints": "always", "lineWidth": 0}},
            "overrides": [fixed(name, color) for name, color in LOOP_COLORS.items()],
        },
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi", "sort": "desc"}},
    }
)
advance(7)

row("Numerical failure: is the model failing to learn?")
# Log scale: a diverged run reports NaN as 1e6, which would flatten every other line.
timeseries(
    "Training loss (mean across ranks)",
    "SELECT time, scenario, loss FROM step_summary WHERE $__timeFilter(time) ORDER BY time",
    "scenario",
    0,
    log=True,
    description="Log scale. Non-finite losses (NaN) are plotted at 1e6.",
)
timeseries("Gradient norm", "SELECT time, scenario, grad_norm FROM step_summary WHERE $__timeFilter(time) ORDER BY time", "scenario", 12, log=True)
advance()
timeseries(
    "Layer gradient norms (bad learning-rate runs)",
    # Real runs write one point per layer, LoRA module, and rank each step; roll them up to one
    # norm per layer (rank 0). Simulated runs have no module or rank tags.
    "SELECT date_bin(INTERVAL '1 second', time) AS time, layer, sqrt(sum(grad_norm * grad_norm)) AS grad_norm "
    "FROM layer_stats WHERE scenario LIKE '%bad_lr' AND (rank IS NULL OR rank = '0') AND $__timeFilter(time) "
    "GROUP BY 1, layer ORDER BY 1",
    "layer",
    0,
    log=True,
    color_by_scenario=False,
    gap_ms=12000,
    no_value="bad_lr runs only: run bad_lr",
    description="One line per layer (12 simulated, 28 in the real Qwen2.5-1.5B): the norm across its LoRA modules, rank 0. In a divergence, see which layers blow up first.",
)
# Scientific units: log-scale ticks for values like 2e-4 otherwise round to "0".
timeseries("Learning rate", "SELECT time, scenario, lr FROM train_step WHERE rank = '0' AND $__timeFilter(time) ORDER BY time", "scenario", 12, log=True, unit="sci", minimum=1e-6)
advance()
# The real trainer's SFT eval is teacher-forced (no SQL execution), so it reports eval loss,
# token accuracy, and exact match rather than exec_accuracy; held-out uses unseen query shapes.
EVAL_NOTE = "Real Trainium SFT runs. Teacher-forced on held-out questions whose query shapes, some phrasings, and literals never appear in training; no SQL is executed."
timeseries(
    "Held-out eval loss (Trainium SFT)",
    "SELECT time, scenario, eval_loss FROM eval WHERE eval_loss IS NOT NULL AND $__timeFilter(time) ORDER BY time",
    "scenario",
    0,
    w=8,
    log=True,
    gap_ms=60000,
    points=True,
    description=EVAL_NOTE,
)
GEN_NOTE = "Real Trainium SFT runs. The model writes the whole SQL itself (greedy decoding) for 32 held-out questions with unseen query shapes, phrasings, and literals; counts exact matches with the gold SQL after whitespace normalization. Stricter than teacher-forced exact match, where the model is shown the correct SQL so far at every token."
for x, field, title, note in (
    (8, "gen_exact_match", "Held-out exact match, generated (Trainium SFT)", GEN_NOTE),
    (16, "exact_match", "Held-out exact match, teacher-forced (Trainium SFT)", EVAL_NOTE),
):
    timeseries(
        title,
        f"SELECT time, scenario, {field} FROM eval WHERE {field} IS NOT NULL AND $__timeFilter(time) ORDER BY time",
        "scenario",
        x,
        w=8,
        unit="percentunit",
        gap_ms=120000,
        points=True,
        description=note,
    )
advance()

GRPO_ONLY = "GRPO runs only: run healthy_grpo, reward_hacking, or reward_hacking_unwatched"
row("Objective failure (GRPO runs): is the model learning the wrong thing?")

# The headline GRPO view: what the grader pays (reward) against what is actually right
# (held-out execution accuracy), on one 0-1 axis. Color follows the run; reward is dashed.
REWARD_VS_ACCURACY = """
SELECT time, concat(scenario, ' reward') AS series,
       AVG(reward) OVER (PARTITION BY run_id ORDER BY time ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS value
FROM rl_step WHERE $__timeFilter(time)
UNION ALL
SELECT time, concat(scenario, ' held-out accuracy') AS series, exec_accuracy AS value
FROM eval WHERE exec_accuracy IS NOT NULL AND $__timeFilter(time)
ORDER BY time
"""
# Exact-name matchers: they match the label-derived display names (regex matchers do not).
overrides = []
for scenario, color in SCENARIO_COLORS.items():
    fixed_color = {"id": "color", "value": {"mode": "fixed", "fixedColor": color}}
    overrides.append({"matcher": {"id": "byName", "options": f"{scenario} reward"},
                      "properties": [fixed_color, {"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [10, 6]}},
                                     {"id": "custom.lineWidth", "value": 2}]})
    overrides.append({"matcher": {"id": "byName", "options": f"{scenario} held-out accuracy"},
                      "properties": [fixed_color, {"id": "custom.showPoints", "value": "always"},
                                     {"id": "custom.lineWidth", "value": 3}]})
panels.append(
    {
        "type": "timeseries",
        "title": "Reward vs held-out accuracy: what the grader pays against what is right",
        "description": "Dashed: GRPO training reward (10-step moving average). Solid with points: held-out SQL execution accuracy (64 questions the model never trains on). Reward hacking is the gap opening between them. Pick a run in Run to isolate it.",
        "datasource": DS,
        "gridPos": {"x": 0, "y": _y, "w": 24, "h": 9},
        "targets": query(REWARD_VS_ACCURACY),
        "transformations": [
            {"id": "partitionByValues", "options": {"fields": ["series"], "keepFields": False, "naming": {"asLabels": True}}},
        ],
        "fieldConfig": {
            "defaults": {
                "noValue": GRPO_ONLY,
                "displayName": "${__field.labels.series}",
                "unit": "percentunit",
                "min": 0,
                "max": 1,
                # Break lines at gaps over 2 minutes (evals are 60-80 s apart), so separate runs of one
                # scenario are not joined across the gap between them.
                "custom": {"drawStyle": "line", "lineWidth": 2, "showPoints": "never", "pointSize": 8,
                           "spanNulls": False, "insertNulls": 120000},
            },
            "overrides": overrides,
        },
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi", "sort": "desc"}},
    }
)
advance(9)

panels.append(
    {
        "type": "table",
        "title": "GRPO run outcomes: training progress and STOP response",
        "description": "One row per run: how far GRPO progressed, best and latest held-out accuracy in the selected "
                       "time range, and the trainer's latest STOP and ADVISE responses. Both hacking runs receive STOP; only "
                       "the watched trainer obeys it. A run whose held-out accuracy declines while its grader agrees with "
                       "the strict check (healthy_grpo overfitting) receives ADVISE instead: the trainer keeps its best "
                       "evaluation's adapters and keeps training. Best observed accuracy is the highest evaluation in the selected window.",
        "datasource": DS,
        "gridPos": {"x": 0, "y": _y, "w": 24, "h": 7},
        "targets": query(
            """
WITH progress AS (
  SELECT run_id, scenario, max(step) AS last_grpo_step
  FROM rl_step WHERE $__timeFilter(time) GROUP BY run_id, scenario
), best_eval AS (
  SELECT run_id, max(exec_accuracy) AS best_accuracy
  FROM eval WHERE exec_accuracy IS NOT NULL AND $__timeFilter(time) GROUP BY run_id
), latest_eval AS (
  SELECT run_id, exec_accuracy,
         row_number() OVER (PARTITION BY run_id ORDER BY time DESC) AS evaluation_order
  FROM eval WHERE exec_accuracy IS NOT NULL AND $__timeFilter(time)
), stop_response AS (
  SELECT run_id, outcome,
         row_number() OVER (PARTITION BY run_id ORDER BY time DESC) AS response_order
  FROM control_ack WHERE action = 'STOP' AND $__timeFilter(time)
), advise_response AS (
  SELECT run_id, outcome,
         row_number() OVER (PARTITION BY run_id ORDER BY time DESC) AS response_order
  FROM control_ack WHERE action = 'ADVISE' AND $__timeFilter(time)
), commands AS (
  SELECT run_id, max(CASE WHEN action = 'STOP' THEN 1 ELSE 0 END) AS got_stop,
         max(CASE WHEN action = 'ADVISE' THEN 1 ELSE 0 END) AS got_advise
  FROM incident WHERE state = 'OPEN' AND $__timeFilter(time) GROUP BY run_id
)
SELECT p.scenario, p.run_id, p.last_grpo_step AS "Last GRPO step",
       b.best_accuracy AS "Best observed accuracy", e.exec_accuracy AS "Latest accuracy",
       COALESCE(a.outcome, CASE WHEN c.got_stop = 1 THEN 'not acknowledged: the run had ended' END) AS "STOP response",
       COALESCE(v.outcome, CASE WHEN c.got_advise = 1 THEN 'not acknowledged: the run had ended' END) AS "ADVISE response"
FROM progress p
LEFT JOIN best_eval b ON b.run_id = p.run_id
LEFT JOIN latest_eval e ON e.run_id = p.run_id AND e.evaluation_order = 1
LEFT JOIN stop_response a ON a.run_id = p.run_id AND a.response_order = 1
LEFT JOIN advise_response v ON v.run_id = p.run_id AND v.response_order = 1
LEFT JOIN commands c ON c.run_id = p.run_id
ORDER BY p.run_id DESC
"""
        ),
        "fieldConfig": {
            "defaults": {"noValue": "—"},
            "overrides": [
                {"matcher": {"id": "byName", "options": "scenario"}, "properties": [{"id": "custom.width", "value": 240}]},
                {"matcher": {"id": "byName", "options": "run_id"}, "properties": [{"id": "custom.width", "value": 300}]},
                {"matcher": {"id": "byName", "options": "Last GRPO step"}, "properties": [
                    {"id": "custom.width", "value": 120}, {"id": "decimals", "value": 0}]},
                *[
                    {"matcher": {"id": "byName", "options": name}, "properties": [
                        {"id": "unit", "value": "percentunit"}, {"id": "decimals", "value": 0},
                        {"id": "custom.width", "value": 170}]}
                    for name in ("Best observed accuracy", "Latest accuracy")
                ],
                {"matcher": {"id": "byName", "options": "STOP response"}, "properties": [
                    {"id": "custom.width", "value": 250}, {"id": "noValue", "value": "none"},
                    {"id": "custom.cellOptions", "value": {"type": "color-text"}},
                    # STOP outcomes read "stopping: kept GRPO step ...": match the prefix and show the
                    # full outcome.
                    {"id": "mappings", "value": [
                        {"type": "regex", "options": {"pattern": "^stopping.*", "result": {"color": "green", "index": 0}}},
                        {"type": "value", "options": {
                            "ignored (watchdog off)": {"text": "ignored (watchdog off)", "color": "red", "index": 1},
                            "not acknowledged: the run had ended": {"color": "#8e8e8e", "index": 2},
                        }},
                    ]},
                ]},
                # ADVISE outcomes read "noted: held-out accuracy declining since GRPO step N; kept ...".
                {"matcher": {"id": "byName", "options": "ADVISE response"}, "properties": [
                    {"id": "custom.width", "value": 330}, {"id": "noValue", "value": "none"},
                    {"id": "custom.cellOptions", "value": {"type": "color-text"}},
                    {"id": "mappings", "value": [
                        {"type": "regex", "options": {"pattern": "^noted.*", "result": {"color": "blue", "index": 0}}},
                        {"type": "value", "options": {
                            "not acknowledged: the run had ended": {"color": "#8e8e8e", "index": 1},
                        }},
                    ]},
                ]},
            ],
        },
    }
)
advance(7)

# ---------------------------------------------------------------------- mechanism
# How the model games the grader, from the trainer's per-candidate telemetry: the structure of the
# SQL it writes, where the two graders disagree, and which held-out questions it gets wrong.
def rl_avg_field(field):
    return (
        f"SELECT time, scenario, AVG({field}) OVER (PARTITION BY scenario, run_id ORDER BY time "
        f"ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS {field} FROM rl_step WHERE {field} IS NOT NULL "
        f"AND $__timeFilter(time) ORDER BY time"
    )


# Line style by metric position: the first metric solid, the second dashed, the third dotted.
METRIC_LINE_STYLES = (None, {"fill": "dash", "dash": [10, 6]}, {"fill": "dot", "dash": [0, 10]})


def multi_series(title, description, sql, metrics, unit=None, points=False, h=8, x=0, w=12,
                 right=None, axis_label=None, minimum=None, right_properties=()):
    """One line per run per metric; color by run, line style by metric (see METRIC_LINE_STYLES).
    The query names each series '<scenario> <metric>'. right: (metric, axis label) for a metric
    on its own right-hand axis; right_properties: more field properties for it (unit, range)."""
    overrides = []
    for scenario, color in SCENARIO_COLORS.items():
        fixed_color = {"id": "color", "value": {"mode": "fixed", "fixedColor": color}}
        for metric, style in zip(metrics, METRIC_LINE_STYLES):
            properties = [fixed_color]
            if style:
                properties.append({"id": "custom.lineStyle", "value": style})
            if right and metric == right[0]:
                properties += [{"id": "custom.axisPlacement", "value": "right"},
                               {"id": "custom.axisLabel", "value": right[1]}, *right_properties]
            overrides.append({"matcher": {"id": "byName", "options": f"{scenario} {metric}"}, "properties": properties})
    defaults = {"displayName": "${__field.labels.series}", "noValue": GRPO_ONLY,
                "custom": {"drawStyle": "line", "lineWidth": 2, "showPoints": "always" if points else "never",
                           "pointSize": 7, "spanNulls": 120000, "insertNulls": False}}
    if axis_label:
        defaults["custom"]["axisLabel"] = axis_label
    if unit:
        defaults["unit"] = unit
    if unit == "percentunit":
        defaults.update({"min": 0, "max": 1})
    if minimum is not None:
        defaults["min"] = minimum
    panels.append({
        "type": "timeseries", "title": title, "description": description, "datasource": DS,
        "gridPos": {"x": x, "y": _y, "w": w, "h": h}, "targets": query(sql),
        "transformations": [{"id": "partitionByValues", "options": {"fields": ["series"], "keepFields": False, "naming": {"asLabels": True}}}],
        "fieldConfig": {"defaults": defaults, "overrides": overrides},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi", "sort": "desc"}},
    })


row("How the model games the grader (GRPO runs)")
multi_series(
    "Grader disagreement: paid but wrong vs correct but unpaid, and the grader gap",
    "Per GRPO step, share of sampled candidates (10-step moving average). Correct but unpaid (dashed): a right answer "
    "the lenient grader paid 0, usually a correct query that returns no rows. That is the pressure. Paid but wrong "
    "(solid): the lenient grader paid a wrong answer. That is the exploit. Both are measured against the lenient "
    "grader, even in healthy_grpo, whose training grader is the strict one. Grader gap (dotted, right axis): the "
    "training reward minus exec_match, the share of the same candidates that are strictly correct (10-step moving "
    "average); for the lenient grader it is paid but wrong minus correct but unpaid. It is what the watchdog checks "
    "when held-out accuracy falls: 0 for an honest grader (a decline is likely overfitting: HELDOUT_DECLINE, ADVISE), "
    "rising for one being gamed (REWARD_HACKING, STOP). Simulated runs report only the gap.",
    """
SELECT time, concat(scenario, ' paid but wrong') AS series,
       AVG(paid_but_wrong_rate) OVER (PARTITION BY run_id ORDER BY time ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS value
FROM rl_step WHERE paid_but_wrong_rate IS NOT NULL AND $__timeFilter(time)
UNION ALL
SELECT time, concat(scenario, ' correct but unpaid') AS series,
       AVG(honest_unpaid_rate) OVER (PARTITION BY run_id ORDER BY time ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS value
FROM rl_step WHERE honest_unpaid_rate IS NOT NULL AND $__timeFilter(time)
UNION ALL
SELECT time, concat(scenario, ' grader gap') AS series,
       AVG(reward - exec_match) OVER (PARTITION BY run_id ORDER BY time ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS value
FROM rl_step WHERE reward IS NOT NULL AND exec_match IS NOT NULL AND $__timeFilter(time)
ORDER BY time
""",
    ("paid but wrong", "correct but unpaid", "grader gap"), unit="percentunit", x=0,
    right=("grader gap", "grader gap"),
    right_properties=({"id": "unit", "value": "none"}, {"id": "decimals", "value": 2},
                      {"id": "min", "value": -1}, {"id": "max", "value": 1}),
)
multi_series(
    "Held-out accuracy by question type",
    "Held-out execution accuracy at each evaluation, by question type. No data (dashed): the correct answer is "
    "empty. The loophole is exactly these: a model that learns to always return rows gets them wrong first. "
    "Has data (solid): every other question. Time window (dotted): questions over a relative time range, such as "
    "\"last 15 minutes\", whose answer depends on keeping the time filter; each also counts as no data or has data.",
    """
SELECT time, concat(scenario, ' has data') AS series, exec_accuracy_has_data AS value
FROM eval WHERE exec_accuracy_has_data IS NOT NULL AND $__timeFilter(time)
UNION ALL
SELECT time, concat(scenario, ' no data') AS series, exec_accuracy_no_data AS value
FROM eval WHERE exec_accuracy_no_data IS NOT NULL AND $__timeFilter(time)
UNION ALL
SELECT time, concat(scenario, ' time window') AS series, exec_accuracy_window AS value
FROM eval WHERE exec_accuracy_window IS NOT NULL AND coalesce(window_questions, 1) > 0 AND $__timeFilter(time)
ORDER BY time
""",
    ("has data", "no data", "time window"), unit="percentunit", points=True, x=12,
)
advance()
timeseries(
    "Filters per generated query",
    rl_avg_field("filters_mean"),
    "scenario",
    0,
    gap_ms=30000,
    no_value=GRPO_ONLY,
    description="Average WHERE conditions in the sampled SQL (10-step moving average). Falling means the model is dropping filters to make queries return rows.",
)
timeseries(
    "Queries that filter on a host or sensor",
    rl_avg_field("entity_filter_rate"),
    "scenario",
    12,
    unit="percentunit",
    gap_ms=30000,
    no_value=GRPO_ONLY,
    description="Share of sampled queries with a host = or sensor = condition (10-step moving average). Hosts and sensors are where most no-data answers come from.",
)
advance()
# GRPO metrics: per-step values are group means over 4 samples (noisy), so plot a 10-step moving
# average. Real GRPO steps are 6-8 s apart, so allow 30 s gaps before breaking a line.
def rl_avg(field):
    return (
        f"SELECT time, scenario, AVG({field}) OVER (PARTITION BY scenario, run_id ORDER BY time "
        f"ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS {field} FROM rl_step WHERE $__timeFilter(time) ORDER BY time"
    )


RL_NOTE = "10-step moving average."
timeseries("GRPO training reward", rl_avg("reward"), "scenario", 0, gap_ms=30000, description=RL_NOTE + " What the training grader pays.", no_value=GRPO_ONLY)
timeseries(
    "Held-out SQL execution accuracy (independent of reward)",
    "SELECT time, scenario, exec_accuracy FROM eval WHERE phase = 'grpo' AND $__timeFilter(time) ORDER BY time",
    "scenario",
    12,
    unit="percentunit",
    gap_ms=30000,
    points=True,
    no_value=GRPO_ONLY,
)
advance()
# The trainer's own token probabilities, every GRPO step: no gold answer or held-out set involved.
def rl_avg_series(field, name):
    return (
        f"SELECT time, concat(scenario, ' {name}') AS series, AVG({field}) OVER (PARTITION BY run_id ORDER BY time "
        f"ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS value FROM rl_step WHERE {field} IS NOT NULL "
        f"AND $__timeFilter(time)"
    )


multi_series(
    "Label-free signals, no gold answers needed: KL to the reference and policy entropy",
    "Computed from the policy's own token probabilities at every GRPO step (10-step moving averages, nats), so "
    "they need no gold SQL, grader, or held-out set. Policy entropy (solid): mean per-token entropy of the next-token "
    "distribution; it falls as the policy narrows onto one kind of answer. Sample NLL (dashed): mean negative "
    "log-likelihood of the tokens actually sampled. KL to reference (dotted, right axis): how far the policy has moved from the base model, "
    "before SFT (the adapters disabled), so it starts above 0. These show that the policy is changing, not whether the change is right; held-out accuracy shows that.",
    " UNION ALL ".join(rl_avg_series(field, name) for field, name in (
        ("entropy", "entropy"), ("sample_nll", "sample NLL"), ("kl", "KL to reference"))) + " ORDER BY time",
    ("entropy", "sample NLL", "KL to reference"),
    right=("KL to reference", "KL to reference (nats, log)"),
    right_properties=({"id": "custom.scaleDistribution", "value": {"type": "log", "log": 10}},),
    axis_label="entropy, NLL (nats)",
    minimum=0,
)
timeseries("Response length (tokens)", rl_avg("resp_len"), "scenario", 12, gap_ms=30000, description=RL_NOTE, no_value=GRPO_ONLY)
advance()
panels.append(
    {
        "type": "table",
        "title": "Sampled SQL (GRPO): what the model writes, and what it is paid for",
        "description": "The highest-reward candidate every few GRPO steps (real Trainium runs). reward is what the training grader paid; correct is whether its result matches the gold query's result.",
        "datasource": DS,
        "gridPos": {"x": 0, "y": _y, "w": 24, "h": 9},
        "targets": query(
            """
SELECT time, scenario, step, reward, exec_match AS correct, sql, question
FROM sample WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 50
"""
        ),
        "fieldConfig": {
            "defaults": {"noValue": GRPO_ONLY},
            "overrides": [
                {"matcher": {"id": "byName", "options": "sql"}, "properties": [{"id": "custom.width", "value": 620}]},
                # Paid but wrong is the exploit: label correctness in words, with status color.
                {
                    "matcher": {"id": "byName", "options": "correct"},
                    "properties": [
                        {"id": "custom.width", "value": 90},
                        {"id": "custom.cellOptions", "value": {"type": "color-text"}},
                        {"id": "mappings", "value": [{"type": "value", "options": {
                            "0": {"text": "wrong", "color": "red", "index": 0},
                            "1": {"text": "correct", "color": "green", "index": 1}}}]},
                    ],
                },
                {"matcher": {"id": "byName", "options": "reward"}, "properties": [{"id": "custom.width", "value": 80}]},
                {"matcher": {"id": "byName", "options": "step"}, "properties": [{"id": "custom.width", "value": 70}]},
                {"matcher": {"id": "byName", "options": "question"}, "properties": [{"id": "custom.width", "value": 520}]},
            ],
        },
    }
)
advance(9)

row("Trainium / Neuron")
timeseries(
    "NeuronCore utilization (mean)",
    """
SELECT date_bin(INTERVAL '10 seconds', time) AS time, scenario, avg(util_pct) AS util
FROM neuron_hw WHERE $__timeFilter(time) GROUP BY 1, scenario ORDER BY 1
""",
    "scenario",
    0,
    w=24,
    unit="percent",
    gap_ms=25000,
)
advance()


# ---------------------------------------------------------------------- data plane
# How hard the watchdog works InfluxDB 3: series cardinality, calls per second, and the latency
# of every call. Writes (Flink, to the ingest node) come from the Flink InfluxDB connector's
# metrics, which Managed Flink publishes to CloudWatch. Reads (control polls and this dashboard,
# on the query node) come from InfluxDB's own query log (system.queries) and from the pollers'
# control_poll telemetry.
CW = {"type": "cloudwatch", "uid": "cloudwatch"}
MIXED = {"type": "datasource", "uid": "-- Mixed --"}
CONNECTOR_DIMS = {"Application": "training-watchdog", "Connector": "TimestreamInfluxDB3"}
# The control sink (incidents and step summaries) publishes under its own Sink dimension
# (TrainingWatchdogJob); matchExact leaves it out of CONNECTOR_DIMS, which is the telemetry sink.
CONTROL_SINK_DIMS = {**CONNECTOR_DIMS, "Sink": "control"}
READ_COLOR, WRITE_P50_COLOR, WRITE_P99_COLOR, POINTS_COLOR = "#3987e5", "#d95926", "#199e70", "#c98500"
POLL_COLOR = "#d55181"
# Control polls in system.queries: the pollers' SQL (watchdog_trainer.telemetry, watchdog_sim.control).
# This dashboard's own queries quote that text too, and they all read system.queries.
IS_CONTROL_POLL = "(strpos(query_text, 'AS time_ns FROM incident') > 0 AND strpos(query_text, 'system.queries') = 0)"
# Tag keys per measurement (system.influxdb_schema); a series is one distinct tag combination.
SERIES_TAGS = {
    "train_step": "local_rank, node, phase, rank, run_id, scenario",
    "layer_stats": "layer, local_rank, module, node, phase, rank, run_id, scenario",
    "attn_stats": "head, layer, local_rank, node, phase, rank, run_id, scenario",
    "rl_candidate": "candidate, phase, question_id, question_shape, rank, result_class, run_id, scenario",
    "rl_prompt": "phase, prompt_id, prompt_shape, run_id, scenario",
    "eval_question": "eval_set, phase, question_id, question_type, run_id, scenario, window",
    "run_info": "grader, model, phase, run_id, scenario, seed, watched",
    "neuron_runtime": "phase, run_id, runtime, scenario",
    "host_cpu": "cpu, phase, run_id, scenario",
    "host_mem": "phase, run_id, scenario",
    "neuron_hw": "neuron_core, phase, rank, run_id, scenario",
    "step_summary": "phase, run_id, scenario",
    "eval": "eval_set, phase, run_id, scenario",
    "rl_step": "phase, run_id, scenario",
    "sample": "phase, run_id, scenario",
    "checkpoint": "phase, run_id, scenario",
    "heartbeat": "phase, run_id, scenario",
    "incident": "run_id, scenario, severity, state, type",
    "control_ack": "action, phase, run_id, scenario, type",
    "control_poll": "endpoint, phase, run_id, scenario",
}


def raw(sql, ref="A"):
    """A query that skips the Runs/Run filter: these panels describe the whole database."""
    return {"refId": ref, "datasource": DS, "rawSql": sql.strip(), "rawQuery": True, "editorMode": "code", "format": "table"}


def cloudwatch(ref, metric, statistic, label, hide=False, metric_id=None, dimensions=CONNECTOR_DIMS):
    return {"refId": ref, "datasource": CW, "queryMode": "Metrics", "metricQueryType": 0, "metricEditorMode": 0,
            "namespace": "AWS/KinesisAnalytics", "metricName": metric, "dimensions": dimensions,
            "statistic": statistic, "period": "60", "region": "default", "matchExact": True,
            "label": label, "hide": hide, "id": metric_id or ref.lower()}


row("Data plane: InfluxDB 3 Enterprise under the watchdog (writes on the ingest node, reads on the query node)")
cardinality_sql = " UNION ALL ".join(
    f"SELECT '{m}' AS measurement, COUNT(DISTINCT concat_ws('|', {tags})) AS series "
    f"FROM {m} WHERE $__timeFilter(time)"
    for m, tags in SERIES_TAGS.items()
) + " ORDER BY series DESC"
panels.append(
    {
        "type": "bargauge",
        "title": "Series cardinality by measurement (selected time range)",
        "description": "Distinct tag combinations per measurement in the selected time range. Per-layer and per-head statistics are widest; everything else grows with runs and ranks. Whole database: ignores the Runs and Run selectors. Heavy: it scans every point in the range, on the query node that also serves the trainers' control polls, so a wide range over millions of series slows the polls (and, at 10 million series on db.influx.medium nodes, fails).",
        "datasource": DS,
        "gridPos": {"x": 0, "y": _y, "w": 9, "h": 10},
        "targets": [raw(cardinality_sql)],
        "options": {"displayMode": "lcd", "orientation": "horizontal", "showUnfilled": True, "valueMode": "text",
                    "namePlacement": "left", "sizing": "auto", "minVizHeight": 14, "maxVizHeight": 22,
                    "reduceOptions": {"values": True, "calcs": [], "fields": "/^series$/"}},
        "fieldConfig": {"defaults": {"color": {"mode": "fixed", "fixedColor": READ_COLOR}, "min": 0,
                                     "unit": "short", "displayName": "${__field.labels.measurement}"},
                        "overrides": []},
    }
)
panels.append(
    {
        "type": "stat",
        "title": "Calls per second (peak in the time range)",
        "description": "Writes: successful write requests from the Flink InfluxDB connector's two sinks (telemetry, and incidents with step summaries) to the ingest node (each carries a batch of points), and points per second. Reads: control polls/s is what every trainer and simulated run reported polling (control_poll telemetry); SQL queries/s is every query the query node logged (system.queries: control polls plus this dashboard).",
        "datasource": MIXED,
        "gridPos": {"x": 9, "y": _y, "w": 8, "h": 10},
        "targets": [
            cloudwatch("A", "numBatchesSuccess", "Maximum", "batches", hide=True, metric_id="batches"),
            cloudwatch("F", "numBatchesSuccess", "Maximum", "control batches", hide=True, metric_id="control_batches",
                       dimensions=CONTROL_SINK_DIMS),
            # Both sinks. FILL: a sink with no writes in a minute (no SFT run: no step summaries)
            # has no data point there, which would leave the sum out.
            {"refId": "B", "datasource": CW, "queryMode": "Metrics", "metricQueryType": 0, "metricEditorMode": 1,
             # The CloudWatch plugin rejects a math expression without a statistic (HTTP 500).
             "expression": "FILL(RATE(batches), 0) + FILL(RATE(control_batches), 0)", "id": "writes",
             "region": "default", "period": "60", "statistic": "Maximum", "label": "Write calls/s"},
            cloudwatch("C", "numRecordsWrittenPerSecond", "Average", "points", hide=True, metric_id="points"),
            cloudwatch("G", "numRecordsWrittenPerSecond", "Average", "control points", hide=True,
                       metric_id="control_points", dimensions=CONTROL_SINK_DIMS),
            {"refId": "H", "datasource": CW, "queryMode": "Metrics", "metricQueryType": 0, "metricEditorMode": 1,
             "expression": "FILL(points, 0) + FILL(control_points, 0)", "id": "points_written",
             "region": "default", "period": "60", "statistic": "Average", "label": "Points written/s"},
            raw("SELECT date_bin(INTERVAL '1 minute', time) AS time, sum(polls) / 60.0 AS \"Control polls/s\" "
                "FROM control_poll WHERE $__timeFilter(time) GROUP BY 1 ORDER BY 1", ref="D"),
            raw("SELECT date_bin(INTERVAL '1 minute', issue_time) AS time, count(*) / 60.0 AS \"SQL queries/s\" "
                "FROM system.queries WHERE $__timeFilter(issue_time) GROUP BY 1 ORDER BY 1", ref="E"),
        ],
        "options": {"colorMode": "none", "graphMode": "area", "justifyMode": "auto", "textMode": "value_and_name",
                    "orientation": "horizontal", "wideLayout": True,
                    "reduceOptions": {"calcs": ["max"], "fields": "", "values": False}},
        "fieldConfig": {"defaults": {"unit": "reqps", "decimals": 1},
                        "overrides": [fixed("Write calls/s", WRITE_P50_COLOR), fixed("Points written/s", POINTS_COLOR),
                                      fixed("Control polls/s", POLL_COLOR), fixed("SQL queries/s", READ_COLOR),
                                      {"matcher": {"id": "byName", "options": "Points written/s"},
                                       "properties": [{"id": "unit", "value": "short"}]}]},
    }
)
totals_sql = (
    "SELECT sum(points) AS \"Points stored\", sum(series) AS \"Series\", count(*) AS \"Measurements\" FROM ("
    + " UNION ALL ".join(
    f"SELECT '{m}' AS measurement, COUNT(*) AS points, COUNT(DISTINCT concat_ws('|', {tags})) AS series "
    f"FROM {m} WHERE $__timeFilter(time)"
    for m, tags in SERIES_TAGS.items()
)
    + ")"
)
panels.append(
    {
        "type": "stat",
        "title": "Database size (selected time range)",
        "description": "Counted from the measurements themselves, in the selected time range: points, distinct series (tag combinations), and measurements. Whole database: ignores the Runs and Run selectors. Heavy, like Series cardinality: keep the range short.",
        "datasource": DS,
        "gridPos": {"x": 17, "y": _y, "w": 7, "h": 10},
        "targets": [raw(totals_sql)],
        "options": {"colorMode": "none", "graphMode": "none", "justifyMode": "center", "textMode": "value_and_name",
                    "orientation": "vertical", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
        "fieldConfig": {"defaults": {"unit": "short"}, "overrides": []},
    }
)
advance(10)
panels.append(
    {
        "type": "timeseries",
        "title": "Latency of every call: dots are SQL queries the query node served; lines are write percentiles",
        "description": "Dots: end-to-end latency of every SQL query the query node logged (system.queries, up to 10,000 kept), split into control polls (the trainers' incident queries) and dashboard queries. Lines: write latency p50 and p99 of the Flink connector's write calls to the ingest node, per minute (CloudWatch), for the telemetry sink and (dotted) the control sink, which writes the incidents; and the pollers' own p50 round trip (control_poll telemetry, client side, includes the network). Log scale.",
        "datasource": MIXED,
        "gridPos": {"x": 0, "y": _y, "w": 24, "h": 10},
        "targets": [
            raw("SELECT issue_time AS time, arrow_cast(end2end_duration, 'Int64') / 1e6 AS \"control poll\" "
                "FROM system.queries WHERE success AND end2end_duration IS NOT NULL AND " + IS_CONTROL_POLL + " "
                "AND $__timeFilter(issue_time) ORDER BY issue_time"),
            raw("SELECT issue_time AS time, arrow_cast(end2end_duration, 'Int64') / 1e6 AS \"dashboard query\" "
                "FROM system.queries WHERE success AND end2end_duration IS NOT NULL AND NOT " + IS_CONTROL_POLL + " "
                "AND $__timeFilter(issue_time) ORDER BY issue_time", ref="E"),
            cloudwatch("B", "writeLatencyP50Millis", "Average", "write p50"),
            cloudwatch("C", "writeLatencyP99Millis", "Maximum", "write p99"),
            cloudwatch("F", "writeLatencyP50Millis", "Average", "control write p50", dimensions=CONTROL_SINK_DIMS),
            cloudwatch("G", "writeLatencyP99Millis", "Maximum", "control write p99", dimensions=CONTROL_SINK_DIMS),
            raw("SELECT date_bin(INTERVAL '1 minute', time) AS time, median(latency_p50_ms) AS \"control poll p50 (client)\" "
                "FROM control_poll WHERE $__timeFilter(time) GROUP BY 1 ORDER BY 1", ref="D"),
        ],
        "fieldConfig": {
            "defaults": {"unit": "ms", "custom": {"drawStyle": "line", "lineWidth": 2, "showPoints": "never",
                                                  "pointSize": 6, "spanNulls": False, "insertNulls": 180000,
                                                  "scaleDistribution": {"type": "log", "log": 10}}},
            "overrides": [
                {"matcher": {"id": "byName", "options": name},
                 "properties": [{"id": "custom.drawStyle", "value": "points"}, {"id": "custom.pointSize", "value": 4},
                                {"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]}
                for name, color in (("control poll", POLL_COLOR), ("dashboard query", READ_COLOR))
            ] + [
                fixed("write p50", WRITE_P50_COLOR),
                {"matcher": {"id": "byName", "options": "write p99"},
                 "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": WRITE_P99_COLOR}},
                                {"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [10, 6]}}]},
            ] + [
                {"matcher": {"id": "byName", "options": name},
                 "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}},
                                {"id": "custom.lineStyle", "value": {"fill": "dot", "dash": [0, 6]}}]}
                for name, color in (("control write p50", WRITE_P50_COLOR), ("control write p99", WRITE_P99_COLOR))
            ] + [
                {"matcher": {"id": "byName", "options": "control poll p50 (client)"},
                 "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": POLL_COLOR}},
                                {"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [4, 4]}}]},
            ],
        },
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi", "sort": "desc"}},
    }
)
advance(10)

# Keep the original panel IDs so existing links to individual overview panels keep working:
# a panel added later takes the next free ID, and the IDs of removed panels stay unused.
RETIRED_IDS = {9, 10, 29}
original_id = 0
for panel in panels:
    if panel["title"] == "GRPO run outcomes: training progress and STOP response":
        panel["id"] = len(panels) + len(RETIRED_IDS)
    else:
        original_id += 1
        while original_id in RETIRED_IDS:
            original_id += 1
        panel["id"] = original_id

dashboard = {
    "uid": "training-watchdog",
    "title": "AI Training Watchdog",
    "tags": ["flink", "influxdb3", "training", "watchdog-demo"],
    "timezone": "browser",
    "refresh": "10s",
    "time": {"from": "now-30m", "to": "now"},
    "schemaVersion": 39,
    "links": [
        {
            "title": "Demo dashboards",
            "type": "dashboards",
            "tags": ["watchdog-demo"],
            "asDropdown": True,
            "includeVars": True,
            "keepTime": True,
            "targetBlank": False,
        }
    ],
    "panels": panels,
    "templating": {
        "list": [
            {
                "name": "source",
                "label": "Runs",
                "type": "custom",
                "query": "Real Trainium : scenario LIKE 'trn_%',Simulated : scenario NOT LIKE 'trn_%'",
                "current": {"text": "Real Trainium", "value": "scenario LIKE 'trn_%'"},
                "options": [
                    {"text": "Real Trainium", "value": "scenario LIKE 'trn_%'", "selected": True},
                    {"text": "Simulated", "value": "scenario NOT LIKE 'trn_%'", "selected": False},
                ],
                "hide": 0,
            },
            {
                "name": "run",
                "label": "Run",
                "type": "query",
                "datasource": DS,
                # Every trainer and simulator run emits an eval event, so this lists all runs.
                "query": "SELECT DISTINCT run_id FROM eval WHERE $__timeFilter(time) AND $source ORDER BY run_id DESC",
                "definition": "SELECT DISTINCT run_id FROM eval WHERE $__timeFilter(time) AND $source ORDER BY run_id DESC",
                "refresh": 2,
                "includeAll": True,
                "allValue": "%",
                "current": {"text": "All", "value": "$__all"},
                "hide": 0,
            },
        ]
    },
}

# Each focused view reuses the same panel definitions and filter names as the overview.
# Its scenario predicate narrows both the queries and the Run dropdown to the task.
panel_templates = {
    "loss": next(p for p in panels if p["title"] == "Training loss (mean across ranks)"),
    "gradient": next(p for p in panels if p["title"] == "Gradient norm"),
    "learning_rate": next(p for p in panels if p["title"] == "Learning rate"),
    "incidents": next(p for p in panels if p["type"] == "table" and p["title"].startswith("Incident log:")),
    "control": next(p for p in panels if p["title"].startswith("Control loop through InfluxDB:")),
    "reward_accuracy": next(p for p in panels if p["title"].startswith("Reward vs held-out accuracy:")),
    "outcomes": next(p for p in panels if p["title"].startswith("GRPO run outcomes:")),
    "disagreement": next(p for p in panels if p["title"].startswith("Grader disagreement:")),
    "entity_filters": next(p for p in panels if p["title"] == "Queries that filter on a host or sensor"),
    "samples": next(p for p in panels if p["title"].startswith("Sampled SQL (GRPO):")),
    "accuracy_by_type": next(p for p in panels if p["title"] == "Held-out accuracy by question type"),
    "label_free": next(p for p in panels if p["title"].startswith("Label-free signals")),
    "cardinality": next(p for p in panels if p["title"].startswith("Series cardinality by measurement")),
    "calls": next(p for p in panels if p["title"].startswith("Calls per second")),
    "size": next(p for p in panels if p["title"].startswith("Database size")),
    "latency": next(p for p in panels if p["title"].startswith("Latency of every call:")),
}
healthy_accuracy = deepcopy(next(p for p in panels if p["title"] == "Held-out exact match, generated (Trainium SFT)"))
healthy_accuracy.update(
    title="Held-out quality: generated SQL exact match (Trainium) or execution accuracy (simulated)",
    description="Real Trainium: exact match of generated SQL on unseen questions. Simulated: held-out execution accuracy. "
                "These are different measures; the Runs filter selects the source.",
    targets=query(
        "SELECT time, scenario, coalesce(gen_exact_match, exec_accuracy) AS accuracy "
        "FROM eval WHERE (gen_exact_match IS NOT NULL OR exec_accuracy IS NOT NULL) AND $__timeFilter(time) ORDER BY time"
    ),
)
healthy_accuracy["fieldConfig"]["defaults"]["noValue"] = "Held-out evaluations appear as training progresses"
panel_templates["healthy_accuracy"] = healthy_accuracy


# The reward-hacking view plots one run by GRPO step (Grafana's Trend panel: a numeric x-axis, so
# the evaluation steps are evenly spaced and a long compile leaves no gap). The panel takes a
# single frame, so STOP is a column that is set only at the step where the incident opened, drawn
# as a red point; and runs are not separated, so the panels ask for one run.
STOP_STEP = ("LEFT JOIN (SELECT run_id AS stop_run, step AS stop_step FROM incident "
             "WHERE state = 'OPEN' AND type = 'REWARD_HACKING') AS stop ON stop.stop_run = run_id")
STOP_COLUMN = "CASE WHEN step = stop_step THEN 0.0 END AS \"STOP\""
# Run = All passes '%': return nothing, so the panel shows its "select one run" message.
ONE_RUN = "'${run}' <> '%'"


def one_run_placeholder(columns):
    """With Run = All, one all-null row at step 0: the panel keeps its step field (without it, Grafana's
    Trend panel reports "Unable to find field: step") and shows its noValue message."""
    nulls = ", ".join(f"CAST(NULL AS DOUBLE) AS \"{c}\"" for c in columns)
    return f" UNION ALL SELECT CAST(0 AS BIGINT) AS step, {nulls} WHERE '${{run}}' = '%'"


def by_step_series(name, color, style=None):
    properties = [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]
    if style:
        properties.append({"id": "custom.lineStyle",
                           "value": {"fill": style, "dash": [10, 6] if style == "dash" else [2, 6]}})
    return {"matcher": {"id": "byName", "options": name}, "properties": properties}


def by_step_panel(title, description, sql, y_fields, y_label, series, y_extra=(), point_size=0):
    """A Trend panel of one run against GRPO step. Unit, range and axis title go on the plotted
    columns only: anything in the defaults would also apply to the x field."""
    y_properties = [{"id": "custom.axisLabel", "value": y_label}, *y_extra]
    return {
        "type": "trend", "title": title, "description": description, "datasource": DS,
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 9},
        "options": {"xField": "step", "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}},
        "fieldConfig": {
            "defaults": {"noValue": "Select one GRPO run in Run",
                         "custom": {"drawStyle": "line", "lineWidth": 2, "pointSize": point_size,
                                    "showPoints": "always" if point_size else "never"}},
            "overrides": [
                {"matcher": {"id": "byName", "options": "step"}, "properties": [
                    {"id": "unit", "value": "none"}, {"id": "decimals", "value": 0},
                    {"id": "custom.axisLabel", "value": "GRPO step"}]},
                {"matcher": {"id": "byRegexp", "options": "^(" + "|".join(y_fields) + ")$"}, "properties": y_properties},
                *series,
                {"matcher": {"id": "byName", "options": "STOP"}, "properties": [
                    {"id": "color", "value": {"mode": "fixed", "fixedColor": "red"}},
                    {"id": "custom.drawStyle", "value": "points"}, {"id": "custom.pointSize", "value": 14},
                    {"id": "custom.showPoints", "value": "always"}]},
            ],
        },
        "targets": query(sql),
    }


panel_templates["accuracy_by_step"] = by_step_panel(
    "Held-out accuracy by question type, by GRPO step (select one run)",
    "Execution accuracy on the held-out questions, split by type: questions whose host or sensor has data, "
    "questions with no data (a correct answer is empty, which the lenient grader does not pay), and the short "
    "time windows (last 15 minutes or 1 hour). The red point marks the GRPO step where STOP opened. "
    "Select one run: this panel does not separate runs.",
    "SELECT step, exec_accuracy_has_data AS \"has data\", exec_accuracy_no_data AS \"no data\", "
    "exec_accuracy_window AS \"last 15 min / 1 h\", " + STOP_COLUMN + " FROM eval " + STOP_STEP + " "
    "WHERE phase = 'grpo' AND eval_set = 'heldout_sql' AND " + ONE_RUN + " AND $__timeFilter(time)"
    + one_run_placeholder(["has data", "no data", "last 15 min / 1 h", "STOP"]) + " ORDER BY step",
    ["has data", "no data", "last 15 min / 1 h", "STOP"], "held-out accuracy",
    [by_step_series("has data", "blue"), by_step_series("no data", "orange", "dash"),
     by_step_series("last 15 min / 1 h", "green", "dot")],
    y_extra=[{"id": "unit", "value": "percentunit"}, {"id": "min", "value": 0}, {"id": "max", "value": 1}],
    point_size=6,
)
panel_templates["label_free_by_step"] = by_step_panel(
    "Label-free signals by GRPO step (select one run): policy entropy, sample NLL, KL to the reference",
    "Signals that need no gold answers. Policy entropy (solid) and the negative log-likelihood of the sampled "
    "tokens (dashed) estimate the same quantity at temperature 1. KL to the reference (dotted, right axis, log "
    "scale) is measured against the base model with the adapters disabled; with two questions per step, one "
    "question can dominate a step. The red point marks the GRPO step where STOP opened. Select one run.",
    "SELECT step, entropy AS \"policy entropy\", sample_nll AS \"sample NLL\", kl AS \"KL to reference\", "
    + STOP_COLUMN + " FROM rl_step " + STOP_STEP + " WHERE " + ONE_RUN + " AND $__timeFilter(time)"
    + one_run_placeholder(["policy entropy", "sample NLL", "KL to reference", "STOP"]) + " ORDER BY step",
    ["policy entropy", "sample NLL", "STOP"], "entropy, sample NLL (nats)",
    [by_step_series("policy entropy", "purple"), by_step_series("sample NLL", "purple", "dash"),
     {"matcher": {"id": "byName", "options": "KL to reference"}, "properties": [
         {"id": "color", "value": {"mode": "fixed", "fixedColor": "text"}},
         {"id": "custom.lineStyle", "value": {"fill": "dot", "dash": [2, 6]}},
         {"id": "custom.axisPlacement", "value": "right"},
         {"id": "custom.axisLabel", "value": "KL to reference (nats, log scale)"},
         {"id": "custom.scaleDistribution", "value": {"type": "log", "log": 10}}]}],
)

FOCUSED_VIEWS = [
    {
        "slug": "learning",
        "title": "Healthy training",
        "scenarios": ("healthy", "trn_healthy"),
        "intro": "**For `healthy`.** Training loss should fall while quality on unseen questions improves. "
                 "The incident log should stay empty.\n\n"
                 "Use **Runs** to choose real or simulated training, and **Run** to isolate a run or compare related runs.",
        "layout": (("loss", "healthy_accuracy"), ("incidents",)),
    },
    {
        "slug": "divergence",
        "title": "Divergence and rollback",
        "scenarios": ("bad_lr", "trn_bad_lr"),
        "intro": "**For `bad_lr`.** Follow the learning-rate jump, loss and gradient blowup, then the "
                 "**DIVERGENCE / ROLLBACK** command. The trainer's **response** should report restoring a checkpoint; "
                 "loss and gradients should recover as training resumes.\n\n"
                 "**Run = All** compares divergence runs in the selected time range. The control-loop dots show command delivery and response time.",
        "layout": (("loss", "gradient"), ("learning_rate",), ("incidents",), ("control",)),
    },
    {
        "slug": "reward-hacking",
        "title": "Reward hacking",
        "scenarios": ("healthy_grpo", "reward_hacking", "trn_healthy_grpo",
                      "trn_reward_hacking", "trn_reward_hacking_unwatched"),
        "intro_height": 10,
        "intro": "**For `healthy_grpo`, `reward_hacking`, and `reward_hacking_unwatched`.** "
                 "Dashed reward: what the training grader pays. Solid accuracy: correct results on unseen questions.\n\n"
                 "Both hacking runs receive **STOP**. The trainer's **response** shows whether it obeyed (`stopping`) "
                 "or ignored it (`ignored (watchdog off)`). Compare **Last GRPO step** to see how far each continued. "
                 "The watched trainer saves its best earlier model; its accuracy chart ends at the stopping point.\n\n"
                 "Held-out accuracy can also fall with an honest grader (`healthy_grpo` overfitting late in a run). The "
                 "*grader gap* (reward minus strict correctness, in *Grader disagreement*) tells them apart: flat at 0 "
                 "there, so Flink opens **HELDOUT_DECLINE / ADVISE**, not STOP; the response reads `noted: ...` and "
                 "training continues. A gap rising from below 0 is the exploit.\n\n"
                 "*Held-out accuracy by question type* shows which questions the exploit breaks. *Label-free signals* "
                 "(policy entropy, sample NLL, KL to the reference) need no gold answers; they show the policy changing, "
                 "not whether it is right. Both plot one run by GRPO step: select a run in **Run**.\n\n"
                 "**Run = All** compares related runs in this time range. Real runs start with 30 SFT steps; these "
                 "panels fill during GRPO. In the incident log, **arg** is the best evaluation step; **RESOLVED** means the detector went quiet.",
        "layout": (("reward_accuracy",), ("outcomes",), ("incidents",),
                   ("disagreement", "entity_filters"), ("accuracy_by_step", "label_free_by_step"), ("samples",),
                   ("control",)),
        # Added after the view was published: these take the next free panel IDs.
        "appended": ("accuracy_by_step", "label_free_by_step"),
    },
    {
        "slug": "data-plane",
        "title": "InfluxDB data plane",
        "scenarios": (),
        "intro": "**How InfluxDB carries the demo.** Writes go to the ingest node; trainer polls and dashboard queries "
                 "go to the query node. Follow call rates, point rates, query/write latency, and the database's series and point counts.\n\n"
                 "Database panels describe all runs. **Runs**, **Run**, and the selected time range filter the control-loop "
                 "panel. Series cardinality and database size count the whole database in the selected time range; "
                 "they sit at the bottom because they scan every point in the range (keep it short).",
        "layout": (("calls",), ("latency",), ("control",), ("cardinality", "size")),
    },
]


def focused_dashboard(view):
    result = deepcopy(dashboard)
    result["uid"] = f"training-watchdog-{view['slug']}"
    result["title"] = f"AI Training Watchdog: {view['title']}"
    height = view.get("intro_height", 4)
    result["panels"] = [
        {
            "type": "text",
            "title": f"How to demonstrate {view['title'].lower()}",
            "gridPos": {"x": 0, "y": 0, "w": 24, "h": height},
            "options": {"mode": "markdown", "content": view["intro"]},
        }
    ]
    predicate = "scenario IN (" + ", ".join(f"'{s}'" for s in view["scenarios"]) + ")" if view["scenarios"] else None
    appended = view.get("appended", ())
    later = {}
    for names in view["layout"]:
        row_height = max(panel_templates[name]["gridPos"]["h"] for name in names)
        width = 24 // len(names)
        for x, name in enumerate(names):
            panel = deepcopy(panel_templates[name])
            panel["gridPos"] = {"x": x * width, "y": height, "w": width, "h": row_height}
            if predicate:
                for target in panel.get("targets", []):
                    if "rawSql" in target:
                        target["rawSql"] = target["rawSql"].replace(SOURCE_FILTER, f"{SOURCE_FILTER} AND {predicate}")
            result["panels"].append(panel)
            if name in appended:
                later[name] = panel
        height += row_height
    # Panel IDs: the original panels in layout order, then the appended ones, so links to
    # existing panels keep working.
    numbered = [p for p in result["panels"] if all(p is not q for q in later.values())]
    numbered += [later[name] for name in appended]
    for number, panel in enumerate(numbered, start=1):
        panel["id"] = number
    for variable in result["templating"]["list"]:
        if variable["name"] == "run" and predicate:
            for field in ("query", "definition"):
                variable[field] = variable[field].replace("$source", f"$source AND {predicate}")
    return result


output_directory = Path(__file__).with_name("dashboards")
output_directory.mkdir(exist_ok=True)
for filename, generated in [
    ("watchdog.json", dashboard),
    *[(f"watchdog-{view['slug']}.json", focused_dashboard(view)) for view in FOCUSED_VIEWS],
]:
    output_directory.joinpath(filename).write_text(json.dumps(generated, indent=2) + "\n")
