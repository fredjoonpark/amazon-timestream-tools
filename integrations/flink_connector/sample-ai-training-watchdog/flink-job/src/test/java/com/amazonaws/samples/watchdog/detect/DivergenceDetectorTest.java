// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.detect;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;

import com.amazonaws.samples.watchdog.WatchdogTopology;
import com.amazonaws.samples.watchdog.model.Detection;
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
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.util.CloseableIterator;
import org.apache.flink.util.Collector;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

class DivergenceDetectorTest {

    private final DetectorConfig config = new DetectorConfig();

    private static StepSummary summary(long step, double loss, double lossEma, double grad, double gradEma) {
        StepSummary s = new StepSummary();
        s.step = step;
        s.loss = loss;
        s.lossEma = lossEma;
        s.gradNorm = grad;
        s.gradNormEma = gradEma;
        return s;
    }

    @Test
    void ignoresLargeRatiosOfTinyValuesNearConvergence() {
        // At loss ~1e-4, 3x the average is only 2e-4 in absolute terms.
        assertThat(DivergenceDetector.isLossSpike(summary(200, 3e-4, 1e-4, 0.01, 0.002), config)).isFalse();
        assertThat(DivergenceDetector.isGradBlowup(summary(200, 3e-4, 1e-4, 0.05, 0.002), config)).isFalse();
    }

    @Test
    void flagsRealDivergence() {
        // A Trainium bad_lr run: NaN reported as 1e6.
        assertThat(DivergenceDetector.isLossSpike(summary(72, 1e6, 0.01, 1e6, 0.1), config)).isTrue();
        assertThat(DivergenceDetector.isGradBlowup(summary(73, 1e6, 0.01, 1e6, 0.1), config)).isTrue();
        // Simulated bad_lr: loss doubles from ~0.4, gradient norm x8 from ~1.
        assertThat(DivergenceDetector.isLossSpike(summary(121, 0.9, 0.4, 1.1, 1.0), config)).isTrue();
        assertThat(DivergenceDetector.isGradBlowup(summary(124, 1.5, 0.5, 8.5, 1.0), config)).isTrue();
    }

    @Test
    void respectsWarmup() {
        assertThat(DivergenceDetector.isLossSpike(summary(10, 5.0, 1.0, 1.0, 1.0), config)).isFalse();
    }

    // ------------------------------------------------------------------ matching

    private static final long T0 = 1_800_000_000_000L;

    /** Step {@code step} at {@code T0 + step} seconds: a spike, a blowup, or both. */
    private static StepSummary flagged(long step, boolean spike, boolean blowup) {
        StepSummary s = summary(step, spike ? 3.0 : 0.5, 0.5, blowup ? 50.0 : 1.0, 1.0);
        s.runId = "r";
        s.attempt = 1;
        s.ts = T0 + step * 1000;
        return s;
    }

    /** The detections, as "onset->step", from summaries in the order given. */
    private List<String> matches(StepSummary... summaries) {
        DivergenceDetector.Window window = new DivergenceDetector.Window();
        List<String> found = new ArrayList<>();
        for (StepSummary s : summaries) {
            Detection d =
                    window.add(s, DivergenceDetector.isLossSpike(s, config), DivergenceDetector.isGradBlowup(s, config), config);
            if (d != null) {
                found.add(d.onsetStep + "->" + d.step);
            }
        }
        return found;
    }

    @Test
    void aSpikeThenABlowupAtALaterStep() {
        assertThat(matches(flagged(71, true, false), flagged(73, false, true))).containsExactly("71->73");
        // A blowup before the spike, or at the same step, is not one.
        assertThat(matches(flagged(70, false, true), flagged(71, true, false))).isEmpty();
        assertThat(matches(flagged(71, true, true))).isEmpty();
    }

    @Test
    void summariesArrivingOutOfStepOrderStillMatch() {
        // The spike's summary arrives after the blowup's (a rank reported it late).
        assertThat(matches(flagged(73, false, true), flagged(71, true, false))).containsExactly("71->73");
    }

    @Test
    void aBlowupPairsWithTheEarliestSpikeAndASpikeWithTheFirstBlowup() {
        assertThat(matches(flagged(71, true, false), flagged(72, true, false), flagged(74, false, true)))
                .containsExactly("71->74");
        assertThat(matches(flagged(75, false, true), flagged(73, false, true), flagged(71, true, false)))
                .containsExactly("71->73");
    }

    @Test
    void theBlowupMustComeWithinTheWindow() {
        StepSummary spike = flagged(71, true, false);
        StepSummary atWindow = flagged(72, false, true);
        atWindow.ts = spike.ts + config.divergenceWithin.toMillis();
        assertThat(matches(spike, atWindow)).isEmpty();
        atWindow.ts -= 1;
        assertThat(matches(spike, atWindow)).containsExactly("71->72");
    }

    @Test
    void afterAMatchOnlyALaterSpikeStartsAnother() {
        assertThat(matches(
                        flagged(71, true, true), flagged(72, true, true), flagged(73, true, true),
                        flagged(74, false, true), flagged(80, true, false), flagged(82, false, true)))
                .containsExactly("71->72", "73->74", "80->82");
        // A late spike at or before the last match's blowup starts nothing.
        assertThat(matches(flagged(71, true, false), flagged(73, false, true), flagged(72, true, false),
                        flagged(75, false, true)))
                .containsExactly("71->73");
    }

    @Test
    void unmatchedSpikesAndBlowupsExpire() {
        DivergenceDetector.Window window = new DivergenceDetector.Window();
        window.add(flagged(71, true, false), true, false, config);
        window.add(flagged(500, false, false), false, false, config);
        assertThat(window.spikes).isEmpty();
    }

    // ------------------------------------------------------------------ late records

    private static final String RUN = "trn-late-rank";

    private static TelemetryEvent trainStep(long step, int rank, double loss, double grad) {
        TelemetryEvent e = new TelemetryEvent();
        e.type = TelemetryEvent.TRAIN_STEP;
        e.runId = RUN;
        e.scenario = "trn_bad_lr";
        e.phase = "sft";
        e.attempt = 1;
        e.step = step;
        e.ts = T0 + step * 1000 + rank * 10;
        e.rank = rank;
        e.worldSize = 2;
        e.metrics.put("loss", loss);
        e.metrics.put("grad_norm", grad);
        e.metrics.put("step_time_ms", 370.0);
        return e;
    }

    /**
     * Two ranks, one step a second, a checkpoint every 50 steps: a loss spike at step 71 and a
     * gradient blowup at step 73, and nothing else. Rank 1's reports for steps 71-73 reach the
     * stream 10 s late, after step 83, so the summaries they complete are behind the watermark.
     */
    private static List<TelemetryEvent> lateRankRun() {
        List<TelemetryEvent> events = new ArrayList<>();
        List<TelemetryEvent> late = new ArrayList<>();
        for (long step = 1; step <= 120; step++) {
            double loss = step == 71 ? 11.0 : 2.0 / Math.sqrt(step);
            double grad = step == 73 ? 300.0 : 1.0;
            events.add(trainStep(step, 0, loss, grad));
            (step >= 71 && step <= 73 ? late : events).add(trainStep(step, 1, loss, grad));
            if (step == 83) {
                events.addAll(late);
            }
            if (step % 50 == 0) {
                TelemetryEvent checkpoint = trainStep(step, 0, Double.NaN, Double.NaN);
                checkpoint.type = TelemetryEvent.CHECKPOINT;
                checkpoint.ts += 500;
                events.add(checkpoint);
            }
        }
        return events;
    }

    /** The production strategy's 5 s bound, with a watermark after every event. */
    private static WatermarkStrategy<TelemetryEvent> watermarkPerEvent() {
        return WatermarkStrategy.<TelemetryEvent>forGenerator(
                        ctx ->
                                new WatermarkGenerator<>() {
                                    @Override
                                    public void onEvent(TelemetryEvent event, long ts, WatermarkOutput output) {
                                        output.emitWatermark(new Watermark(ts - 5_000));
                                    }

                                    @Override
                                    public void onPeriodicEmit(WatermarkOutput output) {}
                                })
                .withTimestampAssigner((event, ts) -> event.ts);
    }

    private static DataStream<TelemetryEvent> source(StreamExecutionEnvironment env) {
        env.setParallelism(1);
        return env.fromData(lateRankRun(), TypeInformation.of(TelemetryEvent.class))
                .assignTimestampsAndWatermarks(watermarkPerEvent());
    }

    @Test
    void theLateRankRunSummarizesItsDivergenceBehindTheWatermark() throws Exception {
        // The precondition of the next test: an event-time operator would drop these summaries.
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        List<Long> lateSteps = new ArrayList<>();
        try (CloseableIterator<Long> it =
                source(env).keyBy(e -> e.runId)
                        .process(new StepAggregator(new DetectorConfig()))
                        .process(
                                new ProcessFunction<StepSummary, Long>() {
                                    @Override
                                    public void processElement(StepSummary s, Context ctx, Collector<Long> out) {
                                        if (ctx.timestamp() <= ctx.timerService().currentWatermark()) {
                                            out.collect(s.step);
                                        }
                                    }
                                })
                        .executeAndCollect()) {
            it.forEachRemaining(lateSteps::add);
        }
        assertThat(lateSteps).containsExactly(71L, 72L, 73L);
    }

    @Test
    void aDivergenceInLateSummariesStillOpensAnIncident() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        List<Incident> incidents = new ArrayList<>();
        try (CloseableIterator<Incident> it =
                WatchdogTopology.build(source(env), new DetectorConfig()).incidents().executeAndCollect()) {
            it.forEachRemaining(incidents::add);
        }
        assertThat(incidents)
                .extracting(i -> i.state, i -> i.detection.type, i -> i.detection.onsetStep, i -> i.detection.step,
                        i -> i.detection.actionArg)
                .containsExactly(
                        tuple(Incident.OPEN, Detection.DIVERGENCE, 71L, 73L, "69"),
                        tuple(Incident.RESOLVED, Detection.DIVERGENCE, 71L, 73L, "69"));
    }
}
