// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.detect;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;

import com.amazonaws.samples.watchdog.model.Detection;
import com.amazonaws.samples.watchdog.model.Incident;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.CloseableIterator;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

class IncidentManagerTest {

    /** Not a detection: the test source sleeps on it, so the wall clock moves but event time does not. */
    private static final String PAUSE = "PAUSE";

    private static Detection divergence(int attempt, long ts) {
        return Detection.of("run", "trn_bad_lr", attempt, Detection.DIVERGENCE, ts, ts / 1000, ts / 1000,
                "critical", "diverged", "ROLLBACK", "50");
    }

    private static List<Incident> incidents(DetectorConfig config, List<Detection> detections) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        List<Incident> incidents = new ArrayList<>();
        try (CloseableIterator<Incident> it =
                env.fromData(detections, TypeInformation.of(Detection.class))
                        .assignTimestampsAndWatermarks(
                                WatermarkStrategy.<Detection>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                                        .withTimestampAssigner((d, t) -> d.ts))
                        .filter(d -> {
                            if (PAUSE.equals(d.type)) {
                                Thread.sleep(3_000);
                                return false;
                            }
                            return true;
                        })
                        .keyBy(Detection::incidentKey)
                        .process(new IncidentManager(config))
                        .executeAndCollect()) {
            it.forEachRemaining(incidents::add);
        }
        return incidents;
    }

    @Test
    void opensANewIncidentWhenTheNextAttemptFails() throws Exception {
        List<Incident> incidents =
                incidents(
                        new DetectorConfig(),
                        List.of(
                                divergence(1, 1_000),
                                divergence(1, 2_000),
                                divergence(2, 5_000), // after the trainer rolled back
                                divergence(1, 5_500), // late, from the abandoned attempt
                                divergence(2, 6_000)));
        assertThat(incidents)
                .extracting(i -> i.state, i -> i.detection.attempt, i -> i.ts, i -> i.detections)
                .containsExactly(
                        tuple(Incident.OPEN, 1, 1_000L, 1),
                        tuple(Incident.RESOLVED, 1, 5_000L, 2),
                        tuple(Incident.OPEN, 2, 5_000L, 1),
                        tuple(Incident.RESOLVED, 2, 96_000L, 2));
        assertThat(incidents.get(0).incidentId).isNotEqualTo(incidents.get(2).incidentId);
    }

    @Test
    void recordsWhenEachTransitionWasEmitted() throws Exception {
        long before = System.currentTimeMillis();
        List<Incident> incidents =
                incidents(new DetectorConfig(), List.of(divergence(1, 1_000), divergence(2, 5_000)));
        long after = System.currentTimeMillis();
        assertThat(incidents).hasSize(4);
        assertThat(incidents).allSatisfy(i -> assertThat(i.emittedMs).isBetween(before, after));
    }

    @Test
    void resolvesOnTheWallClockWhenEventTimeStops() throws Exception {
        DetectorConfig config = new DetectorConfig();
        config.incidentQuietPeriod = Duration.ofSeconds(1);
        Detection pause = divergence(1, 1_500);
        pause.type = PAUSE;
        // Event time never passes the first quiet period (2 000), so only the wall-clock
        // fallback can resolve the first incident before the second firing arrives.
        List<Incident> incidents =
                incidents(config, List.of(divergence(1, 1_000), pause, divergence(1, 1_500)));
        assertThat(incidents)
                .extracting(i -> i.state, i -> i.ts)
                .containsExactly(
                        tuple(Incident.OPEN, 1_000L),
                        tuple(Incident.RESOLVED, 2_000L),
                        tuple(Incident.OPEN, 1_500L),
                        tuple(Incident.RESOLVED, 2_500L));
    }
}
