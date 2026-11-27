// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;

import com.amazonaws.samples.watchdog.detect.DetectorConfig;
import com.amazonaws.samples.watchdog.detect.StepAggregator;
import com.amazonaws.samples.watchdog.model.Incident;
import com.amazonaws.samples.watchdog.model.StepSummary;
import com.amazonaws.samples.watchdog.model.TelemetryEvent;

import org.apache.flink.api.common.eventtime.Watermark;
import org.apache.flink.api.common.eventtime.WatermarkGenerator;
import org.apache.flink.api.common.eventtime.WatermarkOutput;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.CloseableIterator;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A run that diverges, rolls back, and resumes as attempt 2: the watchdog's history (moving
 * averages, last checkpoint) must follow the resumed timeline, so attempt 2 is watched as
 * closely as attempt 1 was.
 */
class RollbackTest {

    private static final String RUN = "trn-rollback";
    private static final long T0 = 1_800_000_000_000L;
    private static final double NAN = TelemetryEvent.NONFINITE_SENTINEL;

    private static double healthyLoss(long step) {
        return 2.0 / Math.sqrt(step);
    }

    private static TelemetryEvent trainStep(int attempt, long step, long ts, int rank, double loss, double grad) {
        TelemetryEvent e = event(TelemetryEvent.TRAIN_STEP, attempt, step, ts);
        e.rank = rank;
        e.worldSize = 2;
        e.metrics.put("loss", loss);
        e.metrics.put("grad_norm", grad);
        e.metrics.put("step_time_ms", 370.0);
        return e;
    }

    private static TelemetryEvent event(String type, int attempt, long step, long ts) {
        TelemetryEvent e = new TelemetryEvent();
        e.type = type;
        e.runId = RUN;
        e.scenario = "trn_bad_lr";
        e.phase = "sft";
        e.attempt = attempt;
        e.step = step;
        e.ts = ts;
        return e;
    }

    /** One step a second on two ranks, with a checkpoint every 50 steps. */
    private static long run(List<TelemetryEvent> events, int attempt, long from, long to, long ts,
            Map<Long, double[]> faults) {
        for (long step = from; step <= to; step++, ts += 1000) {
            double[] v = faults.getOrDefault(step, new double[] {healthyLoss(step), 1.0});
            events.add(trainStep(attempt, step, ts, 0, v[0], v[1]));
            events.add(trainStep(attempt, step, ts + 10, 1, v[0], v[1]));
            if (step % 50 == 0) {
                events.add(event(TelemetryEvent.CHECKPOINT, attempt, step, ts + 500));
            }
        }
        return ts;
    }

    /**
     * Attempt 1 diverges at step 71 (NaN from step 72) and is rolled back to step 50. Attempt 2
     * diverges again, more mildly, at step 60: 10 s after it resumed, while the first incident is
     * still inside its quiet period.
     */
    private static List<TelemetryEvent> divergesTwice() {
        List<TelemetryEvent> events = new ArrayList<>();
        Map<Long, double[]> attempt1 = new HashMap<>(Map.of(71L, new double[] {11.0, 300.0}));
        for (long step = 72; step <= 80; step++) {
            attempt1.put(step, new double[] {NAN, NAN});
        }
        long ts = run(events, 1, 1, 80, T0, attempt1);
        // Rank 0 reported step 81 before the rollback; rank 1 never does, so it flushes late.
        events.add(trainStep(1, 81, ts, 0, NAN, NAN));
        run(events, 2, 51, 200, ts + 5000,
                Map.of(60L, new double[] {2.0, 1.0}, 61L, new double[] {healthyLoss(61), 50.0}));
        return events;
    }

    private static DataStream<TelemetryEvent> source(StreamExecutionEnvironment env, List<TelemetryEvent> events) {
        return source(env, events, WatchdogTopology.watermarks());
    }

    private static DataStream<TelemetryEvent> source(
            StreamExecutionEnvironment env, List<TelemetryEvent> events, WatermarkStrategy<TelemetryEvent> watermarks) {
        return env.fromData(events, TypeInformation.of(TelemetryEvent.class)).assignTimestampsAndWatermarks(watermarks);
    }

    /** The production strategy's 5 s bound, with a watermark after every event. */
    private static WatermarkStrategy<TelemetryEvent> watermarkPerEvent() {
        return WatermarkStrategy.<TelemetryEvent>forGenerator(ctx -> new WatermarkGenerator<>() {
                    @Override
                    public void onEvent(TelemetryEvent event, long ts, WatermarkOutput output) {
                        output.emitWatermark(new Watermark(ts - 5_000));
                    }

                    @Override
                    public void onPeriodicEmit(WatermarkOutput output) {}
                })
                .withTimestampAssigner((event, recordTimestamp) -> event.ts);
    }

    /** Moves attempt 1's checkpoint 50 to just after its step {@code after}: its shard lagged. */
    private static List<TelemetryEvent> checkpoint50ArrivesAfter(List<TelemetryEvent> events, long after) {
        List<TelemetryEvent> moved = new ArrayList<>(events);
        TelemetryEvent checkpoint = moved.stream()
                .filter(e -> e.is(TelemetryEvent.CHECKPOINT) && e.attempt == 1 && e.step == 50)
                .findFirst()
                .orElseThrow();
        moved.remove(checkpoint);
        int last = 0;
        for (int i = 0; i < moved.size(); i++) {
            if (moved.get(i).attempt == 1 && moved.get(i).step == after) {
                last = i;
            }
        }
        moved.add(last + 1, checkpoint);
        return moved;
    }

    private static Map<String, StepSummary> summaries(List<TelemetryEvent> events) throws Exception {
        return summaries(events, WatchdogTopology.watermarks());
    }

    private static Map<String, StepSummary> summaries(
            List<TelemetryEvent> events, WatermarkStrategy<TelemetryEvent> watermarks) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        Map<String, StepSummary> byStep = new HashMap<>();
        try (CloseableIterator<StepSummary> it =
                source(env, events, watermarks).keyBy(e -> e.runId).process(new StepAggregator(new DetectorConfig()))
                        .executeAndCollect()) {
            it.forEachRemaining(s -> byStep.put(s.attempt + ":" + s.step, s));
        }
        return byStep;
    }

    /** Incident transitions, in order (one task, so detections arrive in event-time order). */
    private static List<Incident> incidents(List<TelemetryEvent> events) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        List<Incident> incidents = new ArrayList<>();
        try (CloseableIterator<Incident> it =
                WatchdogTopology.build(source(env, events), new DetectorConfig()).incidents().executeAndCollect()) {
            it.forEachRemaining(incidents::add);
        }
        return incidents;
    }

    @Test
    void resumedAttemptStartsFromTheCheckpointsHistory() throws Exception {
        Map<String, StepSummary> s = summaries(divergesTwice());

        // Attempt 2 is judged against the averages saved with checkpoint 50, as attempt 1 was.
        assertThat(s.get("2:51").lossEma).isEqualTo(s.get("1:51").lossEma);
        assertThat(s.get("2:51").gradNormEma).isEqualTo(s.get("1:51").gradNormEma);

        // NaN steps are reported as NaN but never enter the averages.
        assertThat(s.get("1:75").loss).isEqualTo(NAN);
        assertThat(s.get("1:80").lossEma).isLessThan(1.0);
        assertThat(s.get("1:80").gradNormEma).isLessThan(20.0);

        // The late step from the abandoned attempt is not judged against attempt 2's history.
        assertThat(s.get("1:81").ranksReporting).isEqualTo(1);
        assertThat(s.get("1:81").lossEma).isNaN();
    }

    @Test
    void aCheckpointThatArrivesAfterTheDivergenceStillRestoresItsStepsAverages() throws Exception {
        // Checkpoint 50's shard lags until after step 80, past the spike at 71. That shard holds
        // the watermark back, so step 50's averages are still kept when the checkpoint arrives.
        Map<String, StepSummary> s =
                summaries(checkpoint50ArrivesAfter(divergesTwice(), 80), WatermarkStrategy.noWatermarks());
        assertThat(s.get("2:51").lossEma).isEqualTo(s.get("1:51").lossEma);
        assertThat(s.get("2:51").gradNormEma).isEqualTo(s.get("1:51").gradNormEma);
    }

    @Test
    void aCheckpointReportedLongAfterItsStepDoesNotSaveDivergedAverages() throws Exception {
        // Checkpoint 50 reached Kinesis late, after the watermark passed step 50: the averages
        // now include the spike at 71, so the resumed attempt starts its averages over instead.
        Map<String, StepSummary> s = summaries(checkpoint50ArrivesAfter(divergesTwice(), 80), watermarkPerEvent());
        assertThat(s.get("2:51").lossEma).isEqualTo(s.get("2:51").loss);
        assertThat(s.get("2:52").lossEma).isLessThan(s.get("1:80").lossEma);
    }

    @Test
    void forgetsHistoryTheAbandonedAttemptSavedAfterTheOneRestored() throws Exception {
        // Attempt 1 checkpointed at 50 and 100 and rolled back to 50; attempt 2 rolled back to 50
        // again before reaching 100. Attempt 3 starts from checkpoint 50's averages, as attempt 2 did.
        List<TelemetryEvent> events = new ArrayList<>();
        long ts = run(events, 1, 1, 110, T0, Map.of());
        ts = run(events, 2, 51, 60, ts + 5000, Map.of(55L, new double[] {11.0, 300.0}));
        run(events, 3, 51, 60, ts + 5000, Map.of());

        Map<String, StepSummary> s = summaries(events);
        assertThat(s.get("2:51").lossEma).isEqualTo(s.get("1:51").lossEma);
        assertThat(s.get("3:51").lossEma).isEqualTo(s.get("1:51").lossEma);
        assertThat(s.get("3:51").gradNormEma).isEqualTo(s.get("1:51").gradNormEma);
    }

    @Test
    void rollsBackPastACheckpointThatIncludesTheBadUpdate() throws Exception {
        // The learning rate spikes in step 100's update: step 100's loss is still fine, so the
        // trainer saves checkpoint 100, and the damage shows from step 101. ROLLBACK 99 makes the
        // trainer restore checkpoint 50.
        Map<Long, double[]> faults = new HashMap<>(Map.of(101L, new double[] {11.0, 300.0}));
        for (long step = 102; step <= 110; step++) {
            faults.put(step, new double[] {NAN, NAN});
        }
        List<TelemetryEvent> events = new ArrayList<>();
        run(events, 1, 1, 110, T0, faults);

        assertThat(incidents(events))
                .filteredOn(i -> Incident.OPEN.equals(i.state))
                .extracting(i -> i.detection.onsetStep, i -> i.detection.actionArg)
                .containsExactly(tuple(101L, "99"));
    }

    @Test
    void secondDivergenceOpensASecondIncident() throws Exception {
        assertThat(incidents(divergesTwice()))
                .extracting(i -> i.state, i -> i.detection.attempt, i -> i.detection.onsetStep, i -> i.detection.actionArg)
                .containsExactly(
                        // Attempt 1: onset at the step-71 spike, back to checkpoint 50 (or before 69).
                        tuple(Incident.OPEN, 1, 71L, "69"),
                        // Attempt 2 fails while that incident is still open: it closes, and a new
                        // one carries a new ROLLBACK.
                        tuple(Incident.RESOLVED, 1, 71L, "69"),
                        tuple(Incident.OPEN, 2, 60L, "58"),
                        tuple(Incident.RESOLVED, 2, 60L, "58"));
    }
}
