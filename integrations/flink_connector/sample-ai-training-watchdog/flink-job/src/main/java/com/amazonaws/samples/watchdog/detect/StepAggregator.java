// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.detect;

import com.amazonaws.samples.watchdog.model.StepSummary;
import com.amazonaws.samples.watchdog.model.TelemetryEvent;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.state.MapState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Collects every rank's report for a step and emits one {@link StepSummary} when all ranks
 * have reported, or after {@link DetectorConfig#stepFlushTimeout} for ranks that never do.
 * Keyed by run. Also tracks moving averages so detectors downstream stay stateless about history.
 *
 * <p>Each step is summarized once. A report for a step already summarized (delivered twice, or a
 * rank reporting after the flush timeout) is dropped: a second, partial summary would be judged
 * again and overwrite the first in InfluxDB. Steps can arrive out of order (a producer resends a
 * throttled record after the rest of its batch), so the summarized steps are remembered until the
 * watermark is {@link #SUMMARIZED_HORIZON_FLUSHES} flush timeouts past them. A report older than
 * that, for a step at or below the highest one summarized in its attempt, is dropped too.
 *
 * <p>That history follows the attempt's timeline. A rollback resumes from a checkpoint, so the
 * new attempt starts from the averages as they were after the checkpoint's step (not the diverged
 * values that triggered the rollback), and the abandoned attempt's later history is forgotten.
 * Non-finite values never enter an average.
 *
 * <p>Checkpoint and step reports travel on different shards, so a checkpoint can arrive steps
 * after the step it saved. Each step's averages are therefore kept, by step, until the watermark
 * is {@link DetectorConfig#stepFlushTimeout} past it: the watermark trails the slowest shard, so
 * until then a checkpoint for that step can still arrive and take the averages of its own step.
 */
public class StepAggregator extends KeyedProcessFunction<String, TelemetryEvent, StepSummary> {

    private final DetectorConfig config;
    private transient MapState<String, StepAccumulator> pending;
    private transient ValueState<Averages> averages;
    /**
     * The current timeline's checkpoints: step to the {loss, gradNorm} averages after that step.
     */
    private transient MapState<Long, double[]> checkpointAverages;
    /**
     * The current timeline's recent steps, until the watermark passes them: step to the {loss,
     * gradNorm} averages after that step, and the step's time.
     */
    private transient MapState<Long, double[]> recentAverages;
    /** Attempt to the highest step summarized in it. */
    private transient MapState<Integer, Long> summarizedThrough;
    /**
     * Recently summarized steps ("attempt:step"), to their summary's time, until the watermark is
     * {@link #SUMMARIZED_HORIZON_FLUSHES} flush timeouts past it. Longer than the one flush timeout
     * after which an unknown report counts as old, so a rank whose report is timestamped later than
     * the summary's ranks (by up to the difference) is still recognized.
     */
    private transient MapState<String, Long> recentlySummarized;

    static final int SUMMARIZED_HORIZON_FLUSHES = 3;

    public StepAggregator(DetectorConfig config) {
        this.config = config;
    }

    /** Rank reports for one step of one attempt. */
    public static class StepAccumulator {
        public String scenario;
        public String phase;
        public int attempt;
        public long step;
        public int worldSize;
        public long maxTs;
        public long deadline;
        /** rank to {loss, gradNorm, stepTimeMs} */
        public Map<Integer, double[]> ranks = new HashMap<>();

        public StepAccumulator() {}
    }

    /** The current attempt's moving averages; NaN until the first finite value. */
    public static class Averages {
        public int attempt;
        public double loss = Double.NaN;
        public double gradNorm = Double.NaN;

        public Averages() {}

        Averages(int attempt) {
            this.attempt = attempt;
        }
    }

    @Override
    public void open(OpenContext openContext) {
        pending =
                getRuntimeContext()
                        .getMapState(
                                new MapStateDescriptor<>(
                                        "pending-steps", String.class, StepAccumulator.class));
        // Pending steps are cleared by their flush timers; the rest outlives the run, so expires.
        ValueStateDescriptor<Averages> averagesDescriptor =
                new ValueStateDescriptor<>("attempt-averages", Averages.class);
        averagesDescriptor.enableTimeToLive(config.runStateTtlConfig());
        averages = getRuntimeContext().getState(averagesDescriptor);
        MapStateDescriptor<Long, double[]> checkpointsDescriptor =
                new MapStateDescriptor<>("checkpoint-averages", Long.class, double[].class);
        checkpointsDescriptor.enableTimeToLive(config.runStateTtlConfig());
        checkpointAverages = getRuntimeContext().getMapState(checkpointsDescriptor);
        MapStateDescriptor<Long, double[]> recentDescriptor =
                new MapStateDescriptor<>("recent-step-averages", Long.class, double[].class);
        recentDescriptor.enableTimeToLive(config.runStateTtlConfig());
        recentAverages = getRuntimeContext().getMapState(recentDescriptor);
        MapStateDescriptor<Integer, Long> summarizedDescriptor =
                new MapStateDescriptor<>("summarized-through", Integer.class, Long.class);
        summarizedDescriptor.enableTimeToLive(config.runStateTtlConfig());
        summarizedThrough = getRuntimeContext().getMapState(summarizedDescriptor);
        // Cleared as the watermark passes; TTL covers the last steps of a run that has ended.
        MapStateDescriptor<String, Long> recentlySummarizedDescriptor =
                new MapStateDescriptor<>("recently-summarized", String.class, Long.class);
        recentlySummarizedDescriptor.enableTimeToLive(config.runStateTtlConfig());
        recentlySummarized = getRuntimeContext().getMapState(recentlySummarizedDescriptor);
    }

    @Override
    public void processElement(TelemetryEvent event, Context ctx, Collector<StepSummary> out)
            throws Exception {
        if (event.is(TelemetryEvent.CHECKPOINT)) {
            recordCheckpoint(event);
            return;
        }
        String key = event.attempt + ":" + event.step;
        StepAccumulator acc = pending.get(key);
        if (acc == null) {
            if (alreadySummarized(key, event, ctx.timerService().currentWatermark())) {
                return;
            }
            acc = new StepAccumulator();
            acc.scenario = event.scenario;
            acc.phase = event.phase;
            acc.attempt = event.attempt;
            acc.step = event.step;
            acc.deadline = event.ts + config.stepFlushTimeout.toMillis();
            ctx.timerService().registerEventTimeTimer(acc.deadline);
        }
        acc.worldSize = Math.max(acc.worldSize, event.worldSize);
        acc.maxTs = Math.max(acc.maxTs, event.ts);
        acc.ranks.put(
                event.rank,
                new double[] {
                    event.metric("loss"), event.metric("grad_norm"), event.metric("step_time_ms")
                });
        if (acc.worldSize > 0 && acc.ranks.size() >= acc.worldSize) {
            pending.remove(key);
            emit(ctx.getCurrentKey(), acc, out);
        } else {
            pending.put(key, acc);
        }
    }

    @Override
    public void onTimer(long timestamp, OnTimerContext ctx, Collector<StepSummary> out)
            throws Exception {
        List<StepAccumulator> due = new ArrayList<>();
        for (Iterator<Map.Entry<String, StepAccumulator>> it = pending.iterator(); it.hasNext(); ) {
            StepAccumulator acc = it.next().getValue();
            if (acc.deadline <= timestamp) {
                due.add(acc);
                it.remove();
            }
        }
        due.sort((a, b) -> Long.compare(a.step, b.step));
        for (StepAccumulator acc : due) {
            emit(ctx.getCurrentKey(), acc, out);
        }
        // Each step registers a flush timer, so these fire as the run's steps go by.
        long flushMillis = config.stepFlushTimeout.toMillis();
        long keepAfter = timestamp - flushMillis;
        for (Iterator<double[]> it = recentAverages.values().iterator(); it.hasNext(); ) {
            if (it.next()[2] < keepAfter) {
                it.remove();
            }
        }
        long rememberAfter = timestamp - SUMMARIZED_HORIZON_FLUSHES * flushMillis;
        for (Iterator<Long> it = recentlySummarized.values().iterator(); it.hasNext(); ) {
            if (it.next() < rememberAfter) {
                it.remove();
            }
        }
    }

    /**
     * Whether a report for a step not pending is for a step already summarized: one still
     * remembered, or, for a report a flush timeout behind the watermark, any step at or below the
     * highest summarized in its attempt (it would have been remembered, had it been summarized).
     */
    private boolean alreadySummarized(String key, TelemetryEvent event, long watermark)
            throws Exception {
        if (recentlySummarized.contains(key)) {
            return true;
        }
        Long through = summarizedThrough.get(event.attempt);
        return through != null
                && event.step <= through
                && event.ts + config.stepFlushTimeout.toMillis() < watermark;
    }

    private void emit(String runId, StepAccumulator acc, Collector<StepSummary> out) throws Exception {
        Long through = summarizedThrough.get(acc.attempt);
        if (through == null || acc.step > through) {
            summarizedThrough.put(acc.attempt, acc.step);
        }
        recentlySummarized.put(acc.attempt + ":" + acc.step, acc.maxTs);
        out.collect(summarize(runId, acc));
    }

    private void recordCheckpoint(TelemetryEvent event) throws Exception {
        Averages current = averages.value();
        if (current != null && event.attempt < current.attempt) {
            return; // saved by an abandoned attempt, reported late
        }
        double[] recent = recentAverages.get(event.step);
        if (recent != null) {
            checkpointAverages.put(event.step, new double[] {recent[0], recent[1]});
            return;
        }
        Long through = summarizedThrough.get(event.attempt);
        if (through == null || event.step > through || pending.contains(event.attempt + ":" + event.step)) {
            // Its step isn't summarized yet: a placeholder, replaced with the step's averages then.
            checkpointAverages.put(
                    event.step,
                    current == null
                            ? new double[] {Double.NaN, Double.NaN}
                            : new double[] {current.loss, current.gradNorm});
        }
        // Otherwise the watermark passed its step long ago, and the averages now may include a
        // divergence after it: a rollback to it restores an earlier checkpoint's averages instead.
    }

    private StepSummary summarize(String runId, StepAccumulator acc) throws Exception {
        int n = acc.ranks.size();
        double lossSum = 0;
        int lossCount = 0;
        boolean lossNonFinite = false;
        double gradMax = Double.NaN;
        double[] times = new double[n];
        int timeCount = 0;
        int slowestRank = -1;
        double slowest = Double.NaN;
        // NaN means the rank did not report that metric: leave it out of the summary rather than
        // letting it turn the mean, max, and median NaN. Infinities and the sentinel are reported
        // values (the loss or gradient diverged) and stay in.
        for (Map.Entry<Integer, double[]> rank : acc.ranks.entrySet()) {
            double[] v = rank.getValue();
            if (!Double.isNaN(v[0])) {
                if (TelemetryEvent.isFinite(v[0])) {
                    lossSum += v[0];
                    lossCount++;
                } else {
                    lossNonFinite = true;
                }
            }
            if (!Double.isNaN(v[1]) && (Double.isNaN(gradMax) || v[1] > gradMax)) {
                gradMax = v[1];
            }
            if (!Double.isNaN(v[2])) {
                times[timeCount++] = v[2];
                if (Double.isNaN(slowest) || v[2] > slowest) {
                    slowest = v[2];
                    slowestRank = rank.getKey();
                }
            }
        }
        Arrays.sort(times, 0, timeCount);
        int mid = timeCount / 2;
        double median =
                timeCount == 0
                        ? Double.NaN
                        : timeCount % 2 == 1 ? times[mid] : (times[mid - 1] + times[mid]) / 2.0;

        StepSummary s = new StepSummary();
        s.runId = runId;
        s.scenario = acc.scenario;
        s.phase = acc.phase;
        s.attempt = acc.attempt;
        s.step = acc.step;
        s.ts = acc.maxTs;
        s.ranksReporting = n;
        // A mean that includes a non-finite loss is not finite either; NaN if no rank reported one.
        s.loss =
                lossNonFinite
                        ? TelemetryEvent.NONFINITE_SENTINEL
                        : lossCount > 0 ? lossSum / lossCount : Double.NaN;
        // Gradients are all-reduced, so ranks agree; max guards against one rank reporting late.
        s.gradNorm = gradMax;
        s.stepTimeMedianMs = median;
        s.stepTimeMaxMs = slowest;
        s.slowestRank = slowestRank;
        s.skew = median > 0 ? slowest / median : 1.0;
        annotateHistory(s);
        return s;
    }

    /** Sets the averages the step is judged against, and folds the step into them. */
    private void annotateHistory(StepSummary s) throws Exception {
        Averages current = averages.value();
        if (current == null) {
            current = new Averages(s.attempt);
        } else if (s.attempt > current.attempt) {
            current = resume(s.attempt, s.step - 1);
        }
        if (s.attempt < current.attempt) {
            // A late step from an abandoned attempt (flushed with a rank missing): report it,
            // but neither judge it against nor fold it into the current attempt's averages.
            s.lossEma = Double.NaN;
            s.gradNormEma = Double.NaN;
            return;
        }
        s.lossEma = seed(current.loss, s.loss);
        s.gradNormEma = seed(current.gradNorm, s.gradNorm);
        current.loss = fold(s.lossEma, s.loss);
        current.gradNorm = fold(s.gradNormEma, s.gradNorm);
        averages.update(current);
        double[] after = {current.loss, current.gradNorm};
        recentAverages.put(s.step, new double[] {after[0], after[1], s.ts});
        if (checkpointAverages.contains(s.step)) {
            checkpointAverages.put(s.step, after); // the checkpoint arrived first
        }
    }

    /**
     * A new attempt resumed after step {@code resumedFrom}: restore the averages after that step
     * (from its checkpoint, or from recent history if the checkpoint hasn't arrived yet), else
     * the newest earlier checkpoint's, and drop the abandoned attempt's later history. Without
     * either (restarted from the initial weights) the averages start over.
     */
    private Averages resume(int attempt, long resumedFrom) throws Exception {
        Long restored = null;
        double[] saved = null;
        for (Iterator<Map.Entry<Long, double[]>> it = checkpointAverages.iterator(); it.hasNext(); ) {
            Map.Entry<Long, double[]> entry = it.next();
            if (entry.getKey() > resumedFrom) {
                it.remove();
            } else if (restored == null || entry.getKey() > restored) {
                restored = entry.getKey();
                saved = entry.getValue();
            }
        }
        for (Iterator<Long> it = recentAverages.keys().iterator(); it.hasNext(); ) {
            if (it.next() > resumedFrom) {
                it.remove();
            }
        }
        double[] recent = recentAverages.get(resumedFrom);
        if (recent != null && (restored == null || restored < resumedFrom)) {
            saved = recent;
        }
        Averages resumed = new Averages(attempt);
        if (saved != null) {
            resumed.loss = saved[0];
            resumed.gradNorm = saved[1];
        }
        return resumed;
    }

    /** The average to judge a value against: the value itself if there is no average yet. */
    private static double seed(double average, double value) {
        return Double.isNaN(average) && TelemetryEvent.isFinite(value) ? value : average;
    }

    private double fold(double average, double value) {
        if (!TelemetryEvent.isFinite(value)) {
            return average;
        }
        double a = config.emaAlpha;
        return a * value + (1 - a) * average;
    }
}
