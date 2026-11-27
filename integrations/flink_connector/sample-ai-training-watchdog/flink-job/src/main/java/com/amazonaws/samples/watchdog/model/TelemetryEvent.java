// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.model;

import java.util.HashMap;
import java.util.Map;

/**
 * One telemetry record emitted by a training run: a per-rank training step, an RL step, a
 * held-out evaluation, a checkpoint, or a Neuron hardware sample.
 */
public class TelemetryEvent {

    // The types the detectors read. Every other type (layer_stats, neuron_hw, sample, ...) is only
    // written through to InfluxDB, under its own name.
    public static final String TRAIN_STEP = "train_step";
    public static final String RL_STEP = "rl_step";
    public static final String EVAL = "eval";
    public static final String CHECKPOINT = "checkpoint";

    /** Producers report NaN and infinite metrics as this value (JSON has no NaN). */
    public static final double NONFINITE_SENTINEL = 1e6;

    public String type;
    /** Event time in epoch milliseconds. */
    public long ts;
    public String runId;
    public String scenario;
    public String phase;
    public int attempt;
    public long step;
    /** Data-parallel rank, or -1 for run-level events. */
    public int rank = -1;
    public int worldSize;
    /** Additional tag values, such as {@code layer}, {@code neuron_core}, or {@code eval_set}. */
    public Map<String, String> tags = new HashMap<>();
    public Map<String, Double> metrics = new HashMap<>();
    /** String fields, such as a sampled SQL query; stored as InfluxDB string fields. */
    public Map<String, String> text = new HashMap<>();

    public TelemetryEvent() {}

    public double metric(String name) {
        Double value = metrics.get(name);
        return value == null ? Double.NaN : value;
    }

    /** False for NaN, infinities, and {@link #NONFINITE_SENTINEL}. */
    public static boolean isFinite(double value) {
        return Double.isFinite(value) && Math.abs(value) < NONFINITE_SENTINEL;
    }

    public boolean is(String eventType) {
        return eventType.equals(type);
    }
}
