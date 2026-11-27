// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.detect;

import static org.assertj.core.api.Assertions.assertThat;

import com.amazonaws.samples.watchdog.WatchdogTopology;
import com.amazonaws.samples.watchdog.model.StepSummary;
import com.amazonaws.samples.watchdog.model.TelemetryEvent;

import org.apache.flink.api.common.eventtime.Watermark;
import org.apache.flink.api.common.eventtime.WatermarkGenerator;
import org.apache.flink.api.common.eventtime.WatermarkOutput;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.CloseableIterator;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** A rank that leaves a metric out must not hide the metric the other ranks reported. */
class StepAggregatorTest {

    private static final long T0 = 1_800_000_000_000L;
    private static final double SENTINEL = TelemetryEvent.NONFINITE_SENTINEL;

    /** One rank's event for {@code step}; a null metric is left out of the record. */
    private static TelemetryEvent rank(long step, int rank, Double loss, Double grad, Double time) {
        TelemetryEvent e = new TelemetryEvent();
        e.type = TelemetryEvent.TRAIN_STEP;
        e.runId = "r";
        e.scenario = "healthy";
        e.phase = "sft";
        e.attempt = 1;
        e.step = step;
        e.ts = T0 + step * 1000 + rank;
        e.rank = rank;
        e.worldSize = 3;
        if (loss != null) {
            e.metrics.put("loss", loss);
        }
        if (grad != null) {
            e.metrics.put("grad_norm", grad);
        }
        if (time != null) {
            e.metrics.put("step_time_ms", time);
        }
        return e;
    }

    private static Map<Long, StepSummary> summaries(List<TelemetryEvent> events) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        Map<Long, StepSummary> byStep = new HashMap<>();
        try (CloseableIterator<StepSummary> it =
                env.fromData(events, TypeInformation.of(TelemetryEvent.class))
                        .assignTimestampsAndWatermarks(WatchdogTopology.watermarks())
                        .keyBy(e -> e.runId)
                        .process(new StepAggregator(new DetectorConfig()))
                        .executeAndCollect()) {
            it.forEachRemaining(s -> byStep.put(s.step, s));
        }
        return byStep;
    }

    @Test
    void summarizesTheRanksThatReportedEachMetric() throws Exception {
        List<TelemetryEvent> events = new ArrayList<>();
        events.add(rank(1, 0, 2.0, 1.5, 300.0));
        events.add(rank(1, 1, 4.0, 2.5, 500.0));
        events.add(rank(1, 2, null, null, null));
        StepSummary s = summaries(events).get(1L);

        assertThat(s.ranksReporting).isEqualTo(3);
        assertThat(s.loss).isEqualTo(3.0);
        assertThat(s.gradNorm).isEqualTo(2.5);
        assertThat(s.stepTimeMedianMs).isEqualTo(400.0);
        assertThat(s.stepTimeMaxMs).isEqualTo(500.0);
        assertThat(s.slowestRank).isEqualTo(1);
    }

    @Test
    void keepsADivergedValueWhenAnotherRankLeftItOut() throws Exception {
        List<TelemetryEvent> events = new ArrayList<>();
        events.add(rank(1, 0, SENTINEL, SENTINEL, 300.0));
        events.add(rank(1, 1, null, null, 300.0));
        events.add(rank(1, 2, 2.0, 1.0, 300.0));
        StepSummary s = summaries(events).get(1L);

        assertThat(s.loss).isEqualTo(SENTINEL);
        assertThat(s.gradNorm).isEqualTo(SENTINEL);
    }

    @Test
    void aMetricNoRankReportedIsNaNAndStaysOutOfTheAverages() throws Exception {
        List<TelemetryEvent> events = new ArrayList<>();
        for (int r = 0; r < 3; r++) {
            events.add(rank(1, r, 2.0, 1.0, 300.0));
            events.add(rank(2, r, null, null, null));
            events.add(rank(3, r, 2.0, 1.0, 300.0));
        }
        Map<Long, StepSummary> s = summaries(events);

        assertThat(s.get(2L).loss).isNaN();
        assertThat(s.get(2L).gradNorm).isNaN();
        assertThat(s.get(2L).stepTimeMedianMs).isNaN();
        assertThat(s.get(2L).slowestRank).isEqualTo(-1);
        assertThat(s.get(3L).lossEma).isEqualTo(2.0);
        assertThat(s.get(3L).gradNormEma).isEqualTo(1.0);
    }

    @Test
    void aReportForAStepAlreadySummarizedIsDropped() throws Exception {
        List<TelemetryEvent> events = new ArrayList<>();
        // Step 1: rank 2 doesn't report in time, so the step goes out on its flush timeout.
        events.add(rank(1, 0, 2.0, 1.0, 300.0));
        events.add(rank(1, 1, 2.0, 1.0, 300.0));
        for (long step = 2; step <= 30; step++) {
            for (int r = 0; r < 3; r++) {
                events.add(rank(step, r, 2.0, 1.0, 300.0));
            }
        }
        // Then rank 2 reports step 1 late, and rank 0's report for step 5 arrives a second time.
        events.add(rank(1, 2, 9.0, 9.0, 300.0));
        events.add(rank(5, 0, 9.0, 9.0, 300.0));
        for (long step = 31; step <= 50; step++) {
            for (int r = 0; r < 3; r++) {
                events.add(rank(step, r, 2.0, 1.0, 300.0));
            }
        }
        List<StepSummary> out = summariesInOrder(events);

        assertThat(out).extracting(s -> s.step).doesNotHaveDuplicates().hasSize(50);
        StepSummary first = out.stream().filter(s -> s.step == 1).findFirst().orElseThrow();
        assertThat(first.ranksReporting).isEqualTo(2);
        assertThat(first.loss).isEqualTo(2.0);
        StepSummary fifth = out.stream().filter(s -> s.step == 5).findFirst().orElseThrow();
        assertThat(fifth.ranksReporting).isEqualTo(3);
        assertThat(fifth.loss).isEqualTo(2.0);
    }

    @Test
    void aStepThatArrivesAfterALaterOneIsStillSummarized() throws Exception {
        List<TelemetryEvent> events = new ArrayList<>();
        // Every rank's step-2 report arrives after step 3 went out (resent after a throttle, say).
        for (long step : new long[] {1, 3, 2}) {
            for (int r = 0; r < 3; r++) {
                events.add(rank(step, r, 2.0, 1.0, 300.0));
            }
        }
        for (long step = 4; step <= 30; step++) {
            for (int r = 0; r < 3; r++) {
                events.add(rank(step, r, 2.0, 1.0, 300.0));
            }
        }
        List<StepSummary> out = summariesInOrder(events);

        assertThat(out).extracting(s -> s.step).doesNotHaveDuplicates().hasSize(30);
        StepSummary second = out.stream().filter(s -> s.step == 2).findFirst().orElseThrow();
        assertThat(second.ranksReporting).isEqualTo(3);
    }

    /**
     * Every summary, in emission order, with a watermark after every event (5 s behind it), so a
     * step's flush timeout passes while the small test source is still running.
     */
    private static List<StepSummary> summariesInOrder(List<TelemetryEvent> events) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        WatermarkStrategy<TelemetryEvent> perEvent =
                WatermarkStrategy.<TelemetryEvent>forGenerator(
                                ctx ->
                                        new WatermarkGenerator<>() {
                                            @Override
                                            public void onEvent(
                                                    TelemetryEvent event, long ts, WatermarkOutput output) {
                                                output.emitWatermark(new Watermark(ts - 5_000));
                                            }

                                            @Override
                                            public void onPeriodicEmit(WatermarkOutput output) {}
                                        })
                        .withTimestampAssigner((event, ts) -> event.ts);
        List<StepSummary> out = new ArrayList<>();
        try (CloseableIterator<StepSummary> it =
                env.fromData(events, TypeInformation.of(TelemetryEvent.class))
                        .assignTimestampsAndWatermarks(perEvent)
                        .keyBy(e -> e.runId)
                        .process(new StepAggregator(new DetectorConfig()))
                        .executeAndCollect()) {
            it.forEachRemaining(out::add);
        }
        return out;
    }
}
