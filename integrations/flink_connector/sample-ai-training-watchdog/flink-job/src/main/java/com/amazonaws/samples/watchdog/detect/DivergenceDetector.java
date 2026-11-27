// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.detect;

import com.amazonaws.samples.watchdog.model.Detection;
import com.amazonaws.samples.watchdog.model.StepSummary;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.function.Predicate;

/**
 * Numerical divergence: a loss spike above its moving average, followed within {@code
 * divergenceWithin} by a gradient-norm blowup at a later step, within one attempt. Recommends
 * rolling back to the newest checkpoint that predates the update that produced the spike (see
 * {@link #rollbackBefore}).
 *
 * <p>Keyed by run and attempt: a spike before a rollback must not pair with a blowup after it. A
 * spike pairs with the first blowup after it, and a blowup with the earliest spike before it. After
 * a match, only a spike after its blowup starts another.
 *
 * <p>Matching uses each summary's own step and time, not the order summaries arrive in or the
 * watermark, so a late summary (completed by a rank whose records reached Kinesis late) is still
 * judged, and a late spike still pairs with a blowup that arrived before it. Unmatched spikes and
 * blowups are kept until they are {@code 2 * divergenceWithin} older than the newest summary.
 */
public class DivergenceDetector extends KeyedProcessFunction<String, StepSummary, Detection> {

    private final DetectorConfig config;
    private transient ListState<StepSummary> spikes;
    private transient ListState<StepSummary> blowups;
    /** Step of the last match's blowup; nothing at or before it starts or completes a match. */
    private transient ValueState<Long> matchedThrough;
    /** Time of the newest summary seen. */
    private transient ValueState<Long> newestTs;

    public DivergenceDetector(DetectorConfig config) {
        this.config = config;
    }

    public static DataStream<Detection> detect(DataStream<StepSummary> summaries, DetectorConfig config) {
        return summaries
                .keyBy(s -> s.runId + "/" + s.attempt)
                .process(new DivergenceDetector(config))
                .name("divergence-detector")
                .uid("divergence-detector");
    }

    /** Loss above its moving average by a ratio and by an absolute amount, after warm-up. */
    static boolean isLossSpike(StepSummary s, DetectorConfig config) {
        return s.step > config.divergenceWarmupSteps
                && s.lossEma > 0
                && s.loss > config.lossSpikeRatio * s.lossEma
                && s.loss - s.lossEma > config.lossSpikeMinDelta;
    }

    /** Gradient norm above its moving average by a ratio, and large in absolute terms. */
    static boolean isGradBlowup(StepSummary s, DetectorConfig config) {
        return s.gradNormEma > 0
                && s.gradNorm > config.gradBlowupRatio * s.gradNormEma
                && s.gradNorm > config.gradBlowupMinNorm;
    }

    @Override
    public void open(OpenContext openContext) {
        ListStateDescriptor<StepSummary> spikesDescriptor =
                new ListStateDescriptor<>("divergence-spikes", StepSummary.class);
        spikesDescriptor.enableTimeToLive(config.runStateTtlConfig());
        spikes = getRuntimeContext().getListState(spikesDescriptor);
        ListStateDescriptor<StepSummary> blowupsDescriptor =
                new ListStateDescriptor<>("divergence-blowups", StepSummary.class);
        blowupsDescriptor.enableTimeToLive(config.runStateTtlConfig());
        blowups = getRuntimeContext().getListState(blowupsDescriptor);
        ValueStateDescriptor<Long> matchedDescriptor =
                new ValueStateDescriptor<>("divergence-matched-through", Long.class);
        matchedDescriptor.enableTimeToLive(config.runStateTtlConfig());
        matchedThrough = getRuntimeContext().getState(matchedDescriptor);
        ValueStateDescriptor<Long> newestDescriptor =
                new ValueStateDescriptor<>("divergence-newest-ts", Long.class);
        newestDescriptor.enableTimeToLive(config.runStateTtlConfig());
        newestTs = getRuntimeContext().getState(newestDescriptor);
    }

    @Override
    public void processElement(StepSummary s, Context ctx, Collector<Detection> out) throws Exception {
        boolean spike = isLossSpike(s, config);
        boolean blowup = isGradBlowup(s, config);
        Long newest = newestTs.value();
        if (newest == null || s.ts > newest) {
            newestTs.update(s.ts);
        }
        if (!spike && !blowup) {
            return;
        }
        Window window = new Window();
        spikes.get().forEach(window.spikes::add);
        blowups.get().forEach(window.blowups::add);
        Long through = matchedThrough.value();
        window.matchedThrough = through == null ? Long.MIN_VALUE : through;
        window.newestTs = newestTs.value();
        Detection detection = window.add(s, spike, blowup, config);
        if (detection != null) {
            out.collect(detection);
        }
        spikes.update(window.spikes);
        blowups.update(window.blowups);
        matchedThrough.update(window.matchedThrough);
    }

    /** One run attempt's unmatched spikes and blowups, and the last match. */
    static final class Window {
        final List<StepSummary> spikes = new ArrayList<>();
        final List<StepSummary> blowups = new ArrayList<>();
        long matchedThrough = Long.MIN_VALUE;
        long newestTs = Long.MIN_VALUE;

        /**
         * Applies a summary that is a spike, a blowup, or both, and returns the detection it
         * completes, or null.
         */
        Detection add(StepSummary s, boolean spike, boolean blowup, DetectorConfig config) {
            newestTs = Math.max(newestTs, s.ts);
            long within = config.divergenceWithin.toMillis();
            Detection detection = null;
            if (s.step > matchedThrough) {
                StepSummary earlier =
                        blowup ? first(spikes, x -> x.step < s.step && s.ts - x.ts < within) : null;
                StepSummary later =
                        spike && earlier == null
                                ? first(blowups, x -> x.step > s.step && x.ts - s.ts < within)
                                : null;
                if (earlier != null) {
                    detection = detection(earlier, s);
                    matchedThrough = s.step;
                } else if (later != null) {
                    detection = detection(s, later);
                    matchedThrough = later.step;
                } else {
                    if (spike) {
                        spikes.add(s);
                    }
                    if (blowup) {
                        blowups.add(s);
                    }
                }
            }
            long horizon = newestTs - 2 * within;
            spikes.removeIf(x -> x.step <= matchedThrough || x.ts < horizon);
            blowups.removeIf(x -> x.step <= matchedThrough || x.ts < horizon);
            return detection;
        }

        /** The lowest-step summary in {@code list} that {@code test} accepts, or null. */
        private static StepSummary first(
                List<StepSummary> list, Predicate<StepSummary> test) {
            return list.stream().filter(test).min(Comparator.comparingLong(x -> x.step)).orElse(null);
        }
    }

    static Detection detection(StepSummary spike, StepSummary blowup) {
        return Detection.of(
                blowup.runId,
                blowup.scenario,
                blowup.attempt,
                Detection.DIVERGENCE,
                blowup.ts,
                blowup.step,
                spike.step,
                "critical",
                String.format(
                        Locale.ROOT,
                        "Loss spiked %.1fx its average at step %d, then gradient norm"
                                + " reached %.1fx its average at step %d",
                        spike.loss / spike.lossEma,
                        spike.step,
                        blowup.gradNorm / blowup.gradNormEma,
                        blowup.step),
                "ROLLBACK",
                Long.toString(rollbackBefore(spike)));
    }

    /**
     * The ROLLBACK argument: the trainer restores its newest checkpoint at or before this step.
     * The spike's loss comes from the weights after step - 1's update, so a checkpoint saved at
     * step - 1 already includes the update that may have caused it. The trainer picks the
     * checkpoint because it knows which it saved: Flink may not have seen the checkpoint yet,
     * since checkpoint and step reports travel on different shards. 0 means the initial weights.
     */
    static long rollbackBefore(StepSummary spike) {
        return Math.max(0, spike.step - 2);
    }
}
