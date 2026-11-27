// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.sink;

import com.amazonaws.samples.watchdog.model.Incident;
import com.amazonaws.samples.watchdog.model.Point;
import com.amazonaws.samples.watchdog.model.StepSummary;
import com.amazonaws.samples.watchdog.model.TelemetryEvent;

import java.util.Map;
import java.util.Set;

/** Maps pipeline records to InfluxDB points. */
public final class Points {

    /**
     * The measurements only Flink writes. Trainers take their commands from {@code incident} rows,
     * so a producer record of one of these types is never written through: otherwise anyone who
     * can write to the stream could open an incident, and so issue a command, for any run.
     */
    public static final Set<String> FLINK_MEASUREMENTS = Set.of("incident", "step_summary");

    /**
     * Columns {@link #fromTelemetry} sets from the record's own fields ({@code time} is InfluxDB's
     * timestamp column). A producer tag, metric, or text field with one of these names is dropped,
     * so it can't relabel the point (another {@code run_id}) or change a column's type (a float
     * {@code step}).
     */
    static final Set<String> STANDARD_COLUMNS =
            Set.of("time", "run_id", "scenario", "phase", "rank", "step", "attempt");

    private Points() {}

    /** Whether a producer record is written through: every type but {@link #FLINK_MEASUREMENTS}. */
    public static boolean isWriteThrough(TelemetryEvent event) {
        return !FLINK_MEASUREMENTS.contains(event.type);
    }

    /**
     * Raw telemetry: the measurement is the event type, and {@code step} is an integer field. The
     * producer's tags, metrics, and text fields follow, except blank names, {@link
     * #STANDARD_COLUMNS}, and fields named like one of the point's tags.
     */
    public static Point fromTelemetry(TelemetryEvent event) {
        Point point =
                Point.of(event.type, event.ts)
                        .tag("run_id", event.runId)
                        .tag("scenario", event.scenario)
                        .tag("phase", event.phase)
                        .field("step", event.step)
                        .field("attempt", (long) event.attempt);
        if (event.rank >= 0) {
            point.tag("rank", Integer.toString(event.rank));
        }
        event.tags.forEach(
                (key, value) -> {
                    if (isProducerColumn(key)) {
                        point.tag(key, value);
                    }
                });
        for (Map.Entry<String, Double> metric : event.metrics.entrySet()) {
            if (isProducerField(point, metric.getKey())) {
                point.field(metric.getKey(), metric.getValue());
            }
        }
        event.text.forEach(
                (key, value) -> {
                    if (isProducerField(point, key)) {
                        point.field(key, value);
                    }
                });
        return point;
    }

    private static boolean isProducerColumn(String name) {
        return !name.isBlank() && !STANDARD_COLUMNS.contains(name);
    }

    private static boolean isProducerField(Point point, String name) {
        return isProducerColumn(name) && !point.tags.containsKey(name);
    }

    public static Point fromSummary(StepSummary s) {
        return Point.of("step_summary", s.ts)
                .tag("run_id", s.runId)
                .tag("scenario", s.scenario)
                .tag("phase", s.phase)
                .field("step", s.step)
                .field("attempt", (long) s.attempt)
                .field("ranks_reporting", (long) s.ranksReporting)
                .field("loss", s.loss)
                .field("grad_norm", s.gradNorm)
                .field("step_time_median_ms", s.stepTimeMedianMs)
                .field("step_time_max_ms", s.stepTimeMaxMs)
                .field("slowest_rank", (long) s.slowestRank)
                .field("skew", s.skew)
                .field("loss_ema", s.lossEma)
                .field("grad_norm_ema", s.gradNormEma);
    }

    public static Point fromIncident(Incident incident) {
        var d = incident.detection;
        return Point.of("incident", incident.ts)
                .tag("run_id", d.runId)
                .tag("scenario", d.scenario)
                .tag("type", d.type)
                .tag("state", incident.state)
                .tag("severity", d.severity)
                .field("incident_id", incident.incidentId)
                .field("step", d.step)
                .field("attempt", (long) d.attempt)
                .field("onset_step", d.onsetStep)
                .field("detections", (long) incident.detections)
                .field("message", d.message)
                .field("action", d.action)
                .field("action_arg", d.actionArg == null ? "" : d.actionArg)
                .field("open_duration_s", (incident.ts - incident.openedTs) / 1000.0)
                .field("emitted_ms", incident.emittedMs);
    }
}
