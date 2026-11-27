// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog;

import com.amazonaws.samples.watchdog.detect.DetectorConfig;
import com.amazonaws.samples.watchdog.detect.DivergenceDetector;
import com.amazonaws.samples.watchdog.detect.IncidentManager;
import com.amazonaws.samples.watchdog.detect.RewardHackingDetector;
import com.amazonaws.samples.watchdog.detect.StepAggregator;
import com.amazonaws.samples.watchdog.model.Detection;
import com.amazonaws.samples.watchdog.model.Incident;
import com.amazonaws.samples.watchdog.model.Point;
import com.amazonaws.samples.watchdog.model.StepSummary;
import com.amazonaws.samples.watchdog.model.TelemetryEvent;
import com.amazonaws.samples.watchdog.sink.Points;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.streaming.api.datastream.DataStream;

import java.time.Duration;

/** The source- and sink-independent part of the job, shared by the application and tests. */
public final class WatchdogTopology {

    /**
     * The points for InfluxDB, in two streams, and the incidents. {@code telemetry} is the
     * producers' records, written through. {@code control} is what Flink computes: step summaries
     * and incident rows, which carry the control actions. The job writes them through separate
     * sinks, so a producer line InfluxDB rejects can't stop the incident writes.
     */
    public record Outputs(
            DataStream<Point> telemetry, DataStream<Point> control, DataStream<Incident> incidents) {}

    /** How far behind the newest event time the watermark trails. */
    static final Duration OUT_OF_ORDERNESS = Duration.ofSeconds(5);

    private WatchdogTopology() {}

    public static WatermarkStrategy<TelemetryEvent> watermarks() {
        return WatermarkStrategy.<TelemetryEvent>forBoundedOutOfOrderness(OUT_OF_ORDERNESS)
                .withTimestampAssigner((event, recordTimestamp) -> event.ts)
                .withIdleness(Duration.ofSeconds(30));
    }

    public static Outputs build(DataStream<TelemetryEvent> events, DetectorConfig config) {
        DataStream<StepSummary> summaries =
                events.filter(
                                e -> e.is(TelemetryEvent.TRAIN_STEP) || e.is(TelemetryEvent.CHECKPOINT))
                        .name("train-steps-and-checkpoints")
                        .keyBy(e -> e.runId)
                        .process(new StepAggregator(config))
                        .name("cross-rank-step-summary")
                        .uid("cross-rank-step-summary");

        DataStream<Detection> divergence = DivergenceDetector.detect(summaries, config);
        DataStream<Detection> rewardHacking =
                events.filter(e -> e.is(TelemetryEvent.RL_STEP) || e.is(TelemetryEvent.EVAL))
                        .name("rl-steps-and-evals")
                        .keyBy(e -> e.runId)
                        .process(new RewardHackingDetector(config))
                        .name("reward-hacking-detector")
                        .uid("reward-hacking-detector");

        DataStream<Incident> incidents =
                divergence
                        .union(rewardHacking)
                        .keyBy(Detection::incidentKey)
                        .process(new IncidentManager(config))
                        .name("incident-manager")
                        .uid("incident-manager");

        DataStream<Point> telemetry =
                events.filter(Points::isWriteThrough)
                        .name("producer-measurements")
                        .map(Points::fromTelemetry)
                        .name("telemetry-points");
        DataStream<Point> control =
                summaries
                        .map(Points::fromSummary)
                        .name("summary-points")
                        .union(incidents.map(Points::fromIncident).name("incident-points"));
        return new Outputs(telemetry, control, incidents);
    }
}
