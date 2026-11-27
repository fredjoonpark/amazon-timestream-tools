// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.detect;

import com.amazonaws.samples.watchdog.model.Detection;
import com.amazonaws.samples.watchdog.model.TelemetryEvent;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

import java.util.Locale;

/**
 * Objective failure: an independent held-out evaluation falls from its best value while the
 * training reward does not, so the policy is still being paid as well as before for answers that
 * are less often right. Joins the RL step stream with the evaluation stream per run.
 *
 * <p>Two paths, either of which completes the decline rule:
 *
 * <ul>
 *   <li><b>Reward rose:</b> held-out accuracy at least {@code hackEvalDrop} below its best, and the
 *       reward moving average at least {@code hackRewardRise} above its value at that best, for
 *       {@code hackConsecutiveEvals} evaluations in a row.
 *   <li><b>Reward held:</b> held-out accuracy at least {@code hackHoldEvalDrop} below its best, and
 *       the reward moving average no more than {@code hackHoldRewardTolerance} below its value at
 *       that best, for {@code hackHoldConsecutiveEvals} evaluations in a row. An exploit can
 *       replace paid correct answers with paid wrong ones, so the reward stays flat while accuracy
 *       falls (78% to 59% in one Trainium run). A larger drop and one more evaluation guard against
 *       held-out noise in a healthy run, whose reward also stays flat (near its ceiling).
 * </ul>
 *
 * <p>If the reward falls along with accuracy, training is degrading in general, not gaming the
 * grader, and neither path fires.
 *
 * <p><b>The baseline.</b> Both paths compare the moving averages with their values at the best
 * evaluation, so those values must be averages, not a single step. One RL step grades only a few
 * questions' candidates, so its reward is noisy. The moving averages are bias-corrected (as in Adam):
 * they weight the steps seen so far without seeding the average with the first one. And the
 * baseline of a best evaluation that comes before {@code hackBaselineSteps} RL steps (typically the
 * first evaluation, of the model straight out of SFT) stays open: it follows the averages until
 * they cover that many steps, and no decline is judged against it until then. Without this, a run
 * whose first RL step happens to pay all 8 candidates has a baseline reward of 1.00. If its reward
 * then averages 0.54-0.76, that reads as a fall, and neither path fires while held-out accuracy
 * falls from 73% to 30%.
 *
 * <p><b>Hacking or overfitting.</b> The decline rule alone cannot tell a gamed grader from an
 * honest one whose policy overfits its training questions: in the honest-grader control ({@code
 * healthy_grpo}), the reward can rise 0.94 -> 0.98 while held-out accuracy falls 0.83 -> 0.73.
 * The grader gap can: {@code reward - exec_match} on {@code rl_step}, what the training
 * grader paid minus the share of the same candidates that are strictly correct (their result
 * matches the gold query's). An honest grader's gap is 0 at every step. An exploitable grader's
 * starts negative (correct answers it does not pay, such as an empty result) and rises as the
 * policy learns which wrong answers it pays. The detector keeps a moving average of the gap (the
 * reward's smoothing factor) and its value at the best evaluation, and when the decline rule is
 * met:
 *
 * <ul>
 *   <li>gap moving average at least {@code hackGapRise} above its value at the best evaluation:
 *       {@link Detection#REWARD_HACKING}, severity critical, action STOP with the best evaluation's
 *       step as its argument (the trainer keeps that evaluation's adapters);
 *   <li>otherwise: {@link Detection#HELDOUT_DECLINE}, severity warning, action ADVISE with the same
 *       argument. A likely overfit: the trainer notes it and keeps training;
 *   <li>gap unknown, because the producer never reports {@code exec_match} (or had not reported it
 *       by the best evaluation): REWARD_HACKING / STOP. Without the gap the detector cannot rule
 *       hacking out, and a missed hack costs more than a stopped overfit.
 * </ul>
 *
 * <p>Each type fires at most once per run. REWARD_HACKING's STOP ends the run, and a run that
 * ignores it would otherwise fire again at every later qualifying evaluation; with evaluations
 * more than {@code incidentQuietPeriod} apart, each of those firings would open another STOP
 * incident. Once REWARD_HACKING fires, the detector ignores the run. HELDOUT_DECLINE does not end
 * the run, so the detector keeps evaluating it: if the grader gap starts rising later, while the
 * decline rule still (or again) holds, the run escalates to REWARD_HACKING.
 *
 * <p>The message reports the grader gap, when known, and the policy's mean per-token entropy
 * ({@code entropy} on {@code rl_step}, in nats) for context. The verdict does not use entropy, or
 * {@code sample_nll}, the mean negative log-likelihood of the sampled tokens.
 */
public class RewardHackingDetector extends KeyedProcessFunction<String, TelemetryEvent, Detection> {

    private final DetectorConfig config;
    private transient ValueState<RunState> state;

    public RewardHackingDetector(DetectorConfig config) {
        this.config = config;
    }

    /**
     * Per-run reward and evaluation history.
     *
     * <p>Keyed state, restored from the latest snapshot on every deploy. It is a Flink POJO (public
     * class, public no-arg constructor, public non-final fields), so Flink's POJO serializer
     * evolves its schema: a field added here restores from a snapshot taken without it, starting at
     * its initializer value. Removing or renaming a field drops its value on restore, and changing
     * a field's type does not restore at all. {@code RewardHackingDetectorTest} checks that the
     * type stays a POJO.
     */
    public static class RunState {
        /** Bias-corrected moving average of the reward; NaN until a step reports one. */
        public double rewardEma = Double.NaN;
        /** Total weight of {@link #rewardEma}'s history, 1 - (1 - alpha)^n after n steps. */
        public double rewardEmaWeight;
        public double entropyEmaWeight;
        public double gapEmaWeight;
        /** RL steps whose reward is folded into {@link #rewardEma}. */
        public long rlSteps;
        /**
         * The best evaluation came before {@code hackBaselineSteps} RL steps: its baseline (the
         * {@code ...AtBest} averages) still follows the moving averages, and no decline is judged.
         */
        public boolean baselinePending;
        /** Moving average of the mean per-token policy entropy (nats), for the message only. */
        public double entropyEma = Double.NaN;
        /**
         * Moving average of the grader gap, {@code reward - exec_match}; NaN until a step reports
         * both.
         */
        public double gapEma = Double.NaN;
        public double bestEval = Double.NEGATIVE_INFINITY;
        public long bestEvalStep;
        public double rewardEmaAtBest;
        public double entropyEmaAtBest = Double.NaN;
        /** {@link #gapEma} at the best evaluation; NaN if the gap was not known yet. */
        public double gapEmaAtBest = Double.NaN;
        /** Consecutive evaluations meeting the reward-rose condition. */
        public int consecutive;
        /** Consecutive evaluations meeting the reward-held condition. */
        public int consecutiveHeld;
        /** Set when REWARD_HACKING fires; the detector never fires again for the run. */
        public boolean fired;
        /** Set when HELDOUT_DECLINE fires; it never fires again for the run, REWARD_HACKING still can. */
        public boolean declineFired;
        /** Step of the last RL step applied; one at or below it was delivered twice. */
        public long lastRlStep = Long.MIN_VALUE;
        /** Step of the last evaluation judged; one at or below it was delivered twice. */
        public long lastEvalStep = Long.MIN_VALUE;

        public RunState() {}
    }

    /**
     * The per-run state: declared by class, so Flink derives its type (a {@code PojoTypeInfo}) and
     * serializer (a {@code PojoSerializer}) from {@link RunState}. Its name is part of the snapshot.
     */
    static ValueStateDescriptor<RunState> stateDescriptor() {
        return new ValueStateDescriptor<>("rl-run", RunState.class);
    }

    @Override
    public void open(OpenContext openContext) {
        ValueStateDescriptor<RunState> descriptor = stateDescriptor();
        descriptor.enableTimeToLive(config.runStateTtlConfig());
        state = getRuntimeContext().getState(descriptor);
    }

    @Override
    public void processElement(TelemetryEvent event, Context ctx, Collector<Detection> out)
            throws Exception {
        RunState run = state.value();
        if (run == null) {
            run = new RunState();
        }
        Detection detection = onEvent(run, event, config);
        if (detection != null) {
            out.collect(detection);
        }
        state.update(run);
    }

    /**
     * Applies one RL step or held-out evaluation to the run's state and returns the detection it
     * completes, or null.
     */
    public static Detection onEvent(RunState run, TelemetryEvent event, DetectorConfig config) {
        // Delivery is at least once, so a producer's retry can deliver a record twice. GRPO steps
        // only go forward (the trainer ignores ROLLBACK in GRPO), and rank 0 alone reports RL
        // steps and evaluations, so one at or below the last step applied is a copy.
        if (event.is(TelemetryEvent.RL_STEP)) {
            if (event.step <= run.lastRlStep) {
                return null;
            }
            run.lastRlStep = event.step;
            double alpha = config.rewardEmaAlpha;
            double reward = event.metric("reward");
            Ema r = Ema.fold(run.rewardEma, run.rewardEmaWeight, reward, alpha);
            run.rewardEma = r.average();
            run.rewardEmaWeight = r.weight();
            Ema e = Ema.fold(run.entropyEma, run.entropyEmaWeight, event.metric("entropy"), alpha);
            run.entropyEma = e.average();
            run.entropyEmaWeight = e.weight();
            Ema g = Ema.fold(run.gapEma, run.gapEmaWeight, graderGap(event), alpha);
            run.gapEma = g.average();
            run.gapEmaWeight = g.weight();
            if (TelemetryEvent.isFinite(reward)) {
                run.rlSteps++;
            }
            if (run.baselinePending) {
                setBaseline(run);
                run.baselinePending = run.rlSteps < config.hackBaselineSteps;
            }
        } else if (event.is(TelemetryEvent.EVAL) && !Double.isNaN(run.rewardEma)) {
            if (event.step <= run.lastEvalStep) {
                return null;
            }
            double score = event.metric("exec_accuracy");
            if (!TelemetryEvent.isFinite(score)) {
                // Not reported (NaN), or reported as diverged (the sentinel): neither is a score to
                // compare. Judged anyway, it would reset the consecutive-decline counts, or become
                // the best evaluation that every later one falls short of.
                return null;
            }
            run.lastEvalStep = event.step;
            Finding finding = evaluate(run, event.step, score, config);
            if (finding.type() != null) {
                return detection(event, run, score, finding, config);
            }
        }
        return null;
    }

    /** Which path, if any, a held-out evaluation completes. */
    public enum Verdict {
        NONE,
        REWARD_ROSE,
        REWARD_HELD
    }

    /**
     * What a held-out evaluation opens: an incident type ({@link Detection#REWARD_HACKING} or
     * {@link Detection#HELDOUT_DECLINE}, or null for none) and the decline path that completed.
     */
    public record Finding(String type, Verdict path) {
        public static final Finding NONE = new Finding(null, Verdict.NONE);
    }

    /**
     * Applies one held-out evaluation (accuracy {@code score} at {@code step}) to the run's state,
     * given its current reward and grader-gap moving averages, and returns what it opens, if
     * anything. Once REWARD_HACKING fires, every later evaluation of the run returns {@link
     * Finding#NONE}; after HELDOUT_DECLINE, only an escalation to REWARD_HACKING can fire.
     */
    public static Finding evaluate(RunState run, long step, double score, DetectorConfig config) {
        if (run.fired) {
            return Finding.NONE;
        }
        if (score > run.bestEval) {
            run.bestEval = score;
            run.bestEvalStep = step;
            setBaseline(run);
            run.baselinePending = run.rlSteps < config.hackBaselineSteps;
            run.consecutive = 0;
            run.consecutiveHeld = 0;
            return Finding.NONE;
        }
        if (run.baselinePending) {
            return Finding.NONE;
        }
        boolean rose =
                score <= run.bestEval - config.hackEvalDrop
                        && run.rewardEma >= run.rewardEmaAtBest + config.hackRewardRise;
        boolean held =
                score <= run.bestEval - config.hackHoldEvalDrop
                        && run.rewardEma >= run.rewardEmaAtBest - config.hackHoldRewardTolerance;
        run.consecutive = rose ? run.consecutive + 1 : 0;
        run.consecutiveHeld = held ? run.consecutiveHeld + 1 : 0;
        Verdict path =
                run.consecutive >= config.hackConsecutiveEvals
                        ? Verdict.REWARD_ROSE
                        : run.consecutiveHeld >= config.hackHoldConsecutiveEvals
                                ? Verdict.REWARD_HELD
                                : Verdict.NONE;
        if (path == Verdict.NONE) {
            return Finding.NONE;
        }
        if (!gapKnown(run) || run.gapEma - run.gapEmaAtBest >= config.hackGapRise) {
            run.fired = true;
            return new Finding(Detection.REWARD_HACKING, path);
        }
        if (run.declineFired) {
            return Finding.NONE;
        }
        run.declineFired = true;
        return new Finding(Detection.HELDOUT_DECLINE, path);
    }

    /** Whether the grader gap was known both at the best evaluation and now. */
    static boolean gapKnown(RunState run) {
        return !Double.isNaN(run.gapEma) && !Double.isNaN(run.gapEmaAtBest);
    }

    /** {@code reward - exec_match} for an RL step reporting both, finite; NaN otherwise. */
    private static double graderGap(TelemetryEvent event) {
        double reward = event.metric("reward");
        double execMatch = event.metric("exec_match");
        return TelemetryEvent.isFinite(reward) && TelemetryEvent.isFinite(execMatch)
                ? reward - execMatch
                : Double.NaN;
    }

    /** A value that prints as 0.00 prints without a sign: "-0.00" reads as a different number. */
    static double unsignedZero(double value) {
        return Math.abs(value) < 0.005 ? 0.0 : value;
    }

    private static Detection detection(
            TelemetryEvent event, RunState run, double score, Finding finding, DetectorConfig config) {
        String reward = finding.path() == Verdict.REWARD_ROSE ? "rose" : "held";
        boolean hacking = Detection.REWARD_HACKING.equals(finding.type());
        String message;
        if (hacking) {
            message =
                    String.format(
                            Locale.ROOT,
                            "Reward %s %.2f -> %.2f while held-out accuracy fell %.2f -> %.2f since step %d",
                            reward,
                            run.rewardEmaAtBest,
                            run.rewardEma,
                            run.bestEval,
                            score,
                            run.bestEvalStep);
            // Producers that do not report exec_match leave the gap out of the message.
            if (gapKnown(run)) {
                message +=
                        String.format(
                                Locale.ROOT,
                                "; grader gap %.2f -> %.2f",
                                unsignedZero(run.gapEmaAtBest), unsignedZero(run.gapEma));
            }
        } else {
            boolean agreed = Math.abs(run.gapEmaAtBest) < 0.005 && Math.abs(run.gapEma) < 0.005;
            message =
                    String.format(
                            Locale.ROOT,
                            "Held-out accuracy fell %.2f -> %.2f since step %d while the reward %s %.2f -> %.2f"
                                    + " and %s (gap %.2f -> %.2f): possible overfitting;"
                                    + " best evaluation at step %d",
                            run.bestEval,
                            score,
                            run.bestEvalStep,
                            reward,
                            run.rewardEmaAtBest,
                            run.rewardEma,
                            agreed
                                    ? "the grader agreed with the strict check"
                                    : String.format(
                                            Locale.ROOT,
                                            "the grader gap rose less than %.2f",
                                            config.hackGapRise),
                            unsignedZero(run.gapEmaAtBest),
                            unsignedZero(run.gapEma),
                            run.bestEvalStep);
        }
        // Producers that do not report entropy leave it out of the message.
        if (!Double.isNaN(run.entropyEmaAtBest) && !Double.isNaN(run.entropyEma)) {
            message +=
                    String.format(
                            Locale.ROOT,
                            "; mean per-token policy entropy %.2f -> %.2f nats",
                            run.entropyEmaAtBest, run.entropyEma);
        }
        return Detection.of(
                event.runId,
                event.scenario,
                event.attempt,
                finding.type(),
                event.ts,
                event.step,
                run.bestEvalStep,
                hacking ? "critical" : "warning",
                message,
                hacking ? "STOP" : "ADVISE",
                Long.toString(run.bestEvalStep));
    }

    /** The best evaluation's baseline: the moving averages as they are now. */
    private static void setBaseline(RunState run) {
        run.rewardEmaAtBest = run.rewardEma;
        run.entropyEmaAtBest = run.entropyEma;
        run.gapEmaAtBest = run.gapEma;
    }

    /**
     * A bias-corrected exponential moving average and the total weight of its history: after n
     * values, weight is 1 - (1 - alpha)^n and the average is the alpha-weighted sum of the values
     * divided by it. Unlike an average seeded with its first value, the first value's weight decays
     * like every other's: after 10 values (alpha 0.1), it is 6% of the average, not 39%.
     */
    record Ema(double average, double weight) {

        /** Folds {@code value} in; a non-finite value leaves the average unchanged. */
        static Ema fold(double average, double weight, double value, double alpha) {
            if (!TelemetryEvent.isFinite(value)) {
                return new Ema(average, weight);
            }
            // Weight 0 means no average yet (NaN), so the first value is the average.
            double next = alpha + (1 - alpha) * weight;
            double sum = alpha * value + (weight > 0 ? (1 - alpha) * weight * average : 0);
            return new Ema(sum / next, next);
        }
    }
}
