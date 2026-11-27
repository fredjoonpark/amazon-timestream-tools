# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Synthetic natural-language to InfluxDB 3 SQL pairs over a small IoT schema.

Questions are composed from independent parts (aggregate, filters, time window, grouping,
ranking), each with several phrasings. The held-out split tests generalization three ways:

- structure: about a quarter of the valid query shapes (part combinations) appear only in
  held-out, so the model must compose parts it has only seen separately;
- phrasing: some phrasings of each part appear only in held-out;
- literals: held-out hosts and sensors never appear in training;
- schema: questions about the `disk` table appear only in held-out, so the model must read the
  schema in the prompt rather than recall memorized columns.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

PROMPT_TEMPLATE = "### Schema: {schema}\n### Question: {question}\n### SQL: "
SCHEMA = (
    "cpu(time, host, region, usage_user, usage_system, usage_idle); "
    "mem(time, host, region, used_percent, available_bytes); "
    "temperature(time, sensor, location, value); "
    "disk(time, host, region, used_percent, free_bytes)"
)

# measurement -> (fields with readable names, entity tag, entity noun, group tag, group noun)
MEASUREMENTS = {
    "cpu": ({"usage_user": "user CPU", "usage_system": "system CPU", "usage_idle": "idle CPU"}, "host", "host", "region", "region"),
    "mem": ({"used_percent": "memory used percent", "available_bytes": "available memory"}, "host", "host", "region", "region"),
    "temperature": ({"value": "temperature"}, "sensor", "sensor", "location", "location"),
    "disk": ({"used_percent": "disk used percent", "free_bytes": "free disk space"}, "host", "host", "region", "region"),
}
UNSEEN_MEASUREMENTS = ("disk",)
UNSEEN_SHARE = 0.25
GROUP_VALUES = {"region": ("us-east", "us-west", "eu-central", "ap-south"), "location": ("lab", "warehouse", "datacenter", "office")}
AGGREGATES = ("AVG", "MAX", "MIN", "COUNT")
GROUPINGS = ("none", "time_bucket", "by_group")
WINDOWS = ("15 minutes", "1 hour", "6 hours", "1 day", "7 days")
# Every question has one of WINDOWS. Held-out evals report the short ones separately as "window
# questions": they hold the fewest readings, so they depend most on what now() is.
SHORT_WINDOWS = ("15 minutes", "1 hour")
BUCKETS = ("1 minute", "5 minutes", "1 hour", "1 day")
SECONDS = {"1 minute": 60, "5 minutes": 300, "15 minutes": 900, "1 hour": 3600, "6 hours": 21600, "1 day": 86400, "7 days": 604800}
# Thresholds in each field's own units.
THRESHOLDS = {
    "available_bytes": (500_000_000, 1_000_000_000, 4_000_000_000),
    "free_bytes": (10_000_000_000, 50_000_000_000, 100_000_000_000),
    "value": (20, 30, 40, 60),
}
PERCENT_THRESHOLDS = (10, 25, 50, 70, 80, 90, 95)

# Phrasings: index 0..n-2 are used in training; the last one only in held-out.
AGG_WORDS = {
    "AVG": ("average", "mean", "typical"),
    "MAX": ("maximum", "highest", "peak"),
    "MIN": ("minimum", "lowest", "smallest"),
}
INTROS = ("What is the {agg} {field}", "Show the {agg} {field}", "Give me the {agg} {field}", "I need the {agg} {field}")
COUNT_INTROS = ("How many {m} readings were recorded", "Count the {m} readings", "What is the number of {m} readings")
WINDOW_PHRASES = ("over the last {w}", "in the past {w}", "during the last {w}", "for the previous {w}")
ENTITY_PHRASES = ("for {noun} {value}", "on {noun} {value}", "from {noun} {value}")
GROUP_FILTER_PHRASES = ("in the {value} {noun}", "within {noun} {value}", "restricted to {noun} {value}")
THRESHOLD_PHRASES = ("where {field} is above {n}", "only counting readings with {field} over {n}", "considering readings where {field} exceeds {n}")
BUCKET_PHRASES = ("in {b} buckets", "per {b}", "bucketed by {b}", "at {b} resolution")
BY_GROUP_PHRASES = ("for each {noun}", "per {noun}", "broken down by {noun}")
TOPK_PHRASES = ("and list the top {k} by that value", "showing only the {k} highest", "and keep the {k} largest results")


@dataclass(frozen=True)
class Example:
    question: str
    sql: str
    shape: tuple

    @property
    def prompt(self) -> str:
        return PROMPT_TEMPLATE.format(schema=SCHEMA, question=self.question)

    @property
    def window(self) -> str:
        """The question's time window, e.g. "15 minutes" (read from the gold SQL)."""
        return self.sql.split("now() - INTERVAL '", 1)[1].split("'", 1)[0]

    @property
    def is_window_question(self) -> bool:
        return self.window in SHORT_WINDOWS


def shapes() -> list[tuple]:
    """Every valid (measurement, aggregate, grouping, entity, group filter, threshold, top-k)."""
    valid = []
    for m, agg, grouping, entity, group_filter, threshold, topk in itertools.product(
        MEASUREMENTS, AGGREGATES, GROUPINGS, (False, True), (False, True), (False, True), (False, True)
    ):
        if topk and grouping != "by_group":
            continue  # ranking needs one row per group
        if grouping == "by_group" and (entity or group_filter):
            continue  # grouping by the tag you filter to is degenerate
        if entity and group_filter:
            continue  # an entity already lives in one group
        valid.append((m, agg, grouping, entity, group_filter, threshold, topk))
    return valid


def heldout_shapes(seed: int = 7) -> set[tuple]:
    """A quarter of the seen-table shapes, plus every shape over an unseen table."""
    rng = random.Random(f"{seed}:shapes")
    seen = [s for s in shapes() if s[0] not in UNSEEN_MEASUREMENTS]
    return set(rng.sample(seen, len(seen) // 4)) | {s for s in shapes() if s[0] in UNSEEN_MEASUREMENTS}


def _pick(rng: random.Random, options: tuple, heldout: bool) -> str:
    return options[-1] if heldout and rng.random() < 0.5 else rng.choice(options[:-1])


def compose(rng: random.Random, shape: tuple, split: str) -> Example:
    m, agg, grouping, entity, group_filter, threshold, topk = shape
    fields, entity_tag, entity_noun, group_tag, group_noun = MEASUREMENTS[m]
    field, field_name = rng.choice(list(fields.items()))
    heldout = split == "heldout"
    offset = 0 if split == "train" else 500
    window = rng.choice(WINDOWS)

    if agg == "COUNT":
        parts = [_pick(rng, COUNT_INTROS, heldout).format(m=field_name if m == "temperature" else m)]
        select_value, value_alias = "COUNT(*)", "readings"
    else:
        agg_word = _pick(rng, AGG_WORDS[agg], heldout)
        parts = [_pick(rng, INTROS, heldout).format(agg=agg_word, field=field_name)]
        select_value, value_alias = f"{agg}({field})", f"{agg.lower()}_{field}"

    conditions = []
    if entity:
        value = f"{'server' if entity_tag == 'host' else 'sensor'}-{offset + rng.randrange(400):03d}"
        parts.append(_pick(rng, ENTITY_PHRASES, heldout).format(noun=entity_noun, value=value))
        conditions.append(f"{entity_tag} = '{value}'")
    if group_filter:
        value = rng.choice(GROUP_VALUES[group_tag])
        parts.append(_pick(rng, GROUP_FILTER_PHRASES, heldout).format(noun=group_noun, value=value))
        conditions.append(f"{group_tag} = '{value}'")
    if threshold:
        n = rng.choice(THRESHOLDS.get(field, PERCENT_THRESHOLDS))
        parts.append(_pick(rng, THRESHOLD_PHRASES, heldout).format(field=field_name, n=n))
        conditions.append(f"{field} > {n}")
    parts.append(_pick(rng, WINDOW_PHRASES, heldout).format(w=window))
    conditions.append(f"time >= now() - INTERVAL '{window}'")
    where = " AND ".join(conditions)

    if grouping == "none":
        sql = f"SELECT {select_value} FROM {m} WHERE {where}"
    elif grouping == "time_bucket":
        # At least a few buckets per window: no 1-day buckets over 15 minutes.
        bucket = rng.choice([b for b in BUCKETS if SECONDS[b] * 4 <= SECONDS[window]] or [BUCKETS[0]])
        parts.append(_pick(rng, BUCKET_PHRASES, heldout).format(b=bucket))
        sql = (
            f"SELECT date_bin(INTERVAL '{bucket}', time) AS bucket, {select_value} AS {value_alias} "
            f"FROM {m} WHERE {where} GROUP BY bucket ORDER BY bucket"
        )
    else:
        parts.append(_pick(rng, BY_GROUP_PHRASES, heldout).format(noun=group_noun))
        sql = f"SELECT {group_tag}, {select_value} AS {value_alias} FROM {m} WHERE {where} GROUP BY {group_tag}"
        if topk:
            k = rng.choice((3, 5, 10))
            parts.append(_pick(rng, TOPK_PHRASES, heldout).format(k=k))
            sql += f" ORDER BY {value_alias} DESC LIMIT {k}"
        else:
            sql += f" ORDER BY {group_tag}"
    question = " ".join(parts)
    question += "?" if question.startswith(("What", "How")) else "."
    return Example(question, sql, shape)


def build(split: str, count: int, seed: int = 7) -> list[Example]:
    """Returns `count` examples of the split; see the module docstring for what differs."""
    if split not in ("train", "heldout"):
        raise ValueError(f"unknown split {split!r}")
    held = heldout_shapes(seed)
    pool = [s for s in shapes() if (s in held) == (split == "heldout")]
    seen = [s for s in pool if s[0] not in UNSEEN_MEASUREMENTS]
    unseen = [s for s in pool if s[0] in UNSEEN_MEASUREMENTS]
    rng = random.Random(f"{seed}:{split}")
    # Unseen-table questions are about a quarter of held-out, so they inform but do not dominate.
    pick = lambda: rng.choice(unseen if unseen and rng.random() < UNSEEN_SHARE else seen)  # noqa: E731
    return [compose(rng, pick(), split) for _ in range(count)]
