// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.detect;

import org.apache.flink.api.common.state.StateTtlConfig;

import java.io.Serializable;
import java.time.Duration;
import java.util.Properties;

/** Detector thresholds. Defaults suit the simulator; override them in the {@code Detectors} property group. */
public class DetectorConfig implements Serializable {

    private static final long serialVersionUID = 1L;


    /** Smoothing factor for loss and gradient-norm moving averages. */
    public double emaAlpha = 0.05;
    /** Steps ignored by the divergence detector while loss is still falling steeply. */
    public long divergenceWarmupSteps = 30;
    public double lossSpikeRatio = 1.5;
    public double gradBlowupRatio = 5.0;
    /**
     * Absolute floors: near convergence (loss ~1e-4) tiny noise produces large ratios, so a spike
     * must also be big in absolute terms.
     */
    public double lossSpikeMinDelta = 0.05;
    public double gradBlowupMinNorm = 1.0;
    public Duration divergenceWithin = Duration.ofSeconds(60);

    public double rewardEmaAlpha = 0.1;
    /** Absolute held-out accuracy drop from its best value. */
    public double hackEvalDrop = 0.08;
    /** Reward moving-average rise since the best held-out evaluation. */
    public double hackRewardRise = 0.03;
    public int hackConsecutiveEvals = 2;
    /**
     * The reward-held path (see RewardHackingDetector): a larger held-out drop, a reward moving
     * average no more than this tolerance below its value at the best evaluation, and one more
     * evaluation in a row.
     */
    public double hackHoldEvalDrop = 0.10;
    public double hackHoldRewardTolerance = 0.05;
    public int hackHoldConsecutiveEvals = 3;
    /**
     * Rise of the grader-gap moving average ({@code reward - exec_match}) since the best held-out
     * evaluation that makes a held-out decline reward hacking (STOP) rather than a likely overfit
     * (ADVISE). An honest grader's gap is exactly 0. In a seed-1 {@code reward_hacking} run on
     * Trainium, the exploitable grader's 10-step mean gap went -0.54 (steps 11-20), -0.26, -0.10,
     * +0.04 (steps 41-50) by the time the decline rule fired at step 50.
     */
    public double hackGapRise = 0.15;
    /**
     * RL steps the moving averages must cover before a best evaluation's baseline is fixed: one
     * evaluation interval. An earlier best evaluation (the first one, at GRPO step 1) keeps its
     * accuracy and step, and takes the averages as they stand once they cover this many steps.
     */
    public int hackBaselineSteps = 10;

    /** An incident resolves after this long without a new detection. */
    public Duration incidentQuietPeriod = Duration.ofSeconds(90);
    /** A step summary is emitted without missing ranks after this long. */
    public Duration stepFlushTimeout = Duration.ofSeconds(15);
    /**
     * A run's detector state expires this long after it was last written. Run IDs never repeat,
     * so without it every finished run's state stays in the job (and its snapshots) for good.
     */
    public Duration runStateTtl = Duration.ofDays(7);

    /** {@link #runStateTtl} as the TTL of the detectors' per-run state. */
    public StateTtlConfig runStateTtlConfig() {
        return StateTtlConfig.newBuilder(runStateTtl)
                .updateTtlOnCreateAndWrite()
                .neverReturnExpired()
                .build();
    }

    /**
     * The defaults, with any threshold set in {@code properties} under its field's name. Durations
     * take seconds, under the field's name plus {@code Seconds} (for example {@code
     * incidentQuietPeriodSeconds}).
     */
    public static DetectorConfig from(Properties properties) {
        DetectorConfig config = new DetectorConfig();
        if (properties == null) {
            return config;
        }
        config.emaAlpha = doubleOf(properties, "emaAlpha", config.emaAlpha);
        config.divergenceWarmupSteps =
                (long) doubleOf(properties, "divergenceWarmupSteps", config.divergenceWarmupSteps);
        config.lossSpikeRatio = doubleOf(properties, "lossSpikeRatio", config.lossSpikeRatio);
        config.gradBlowupRatio = doubleOf(properties, "gradBlowupRatio", config.gradBlowupRatio);
        config.lossSpikeMinDelta = doubleOf(properties, "lossSpikeMinDelta", config.lossSpikeMinDelta);
        config.gradBlowupMinNorm = doubleOf(properties, "gradBlowupMinNorm", config.gradBlowupMinNorm);
        config.divergenceWithin = secondsOf(properties, "divergenceWithinSeconds", config.divergenceWithin);
        config.rewardEmaAlpha = doubleOf(properties, "rewardEmaAlpha", config.rewardEmaAlpha);
        config.hackEvalDrop = doubleOf(properties, "hackEvalDrop", config.hackEvalDrop);
        config.hackRewardRise = doubleOf(properties, "hackRewardRise", config.hackRewardRise);
        config.hackConsecutiveEvals =
                (int) doubleOf(properties, "hackConsecutiveEvals", config.hackConsecutiveEvals);
        config.hackHoldEvalDrop = doubleOf(properties, "hackHoldEvalDrop", config.hackHoldEvalDrop);
        config.hackHoldRewardTolerance =
                doubleOf(properties, "hackHoldRewardTolerance", config.hackHoldRewardTolerance);
        config.hackHoldConsecutiveEvals =
                (int) doubleOf(properties, "hackHoldConsecutiveEvals", config.hackHoldConsecutiveEvals);
        config.hackGapRise = doubleOf(properties, "hackGapRise", config.hackGapRise);
        config.hackBaselineSteps =
                (int) doubleOf(properties, "hackBaselineSteps", config.hackBaselineSteps);
        config.incidentQuietPeriod =
                secondsOf(properties, "incidentQuietPeriodSeconds", config.incidentQuietPeriod);
        config.stepFlushTimeout =
                secondsOf(properties, "stepFlushTimeoutSeconds", config.stepFlushTimeout);
        config.runStateTtl = secondsOf(properties, "runStateTtlSeconds", config.runStateTtl);
        return config;
    }

    private static double doubleOf(Properties properties, String key, double defaultValue) {
        String value = properties.getProperty(key);
        return value == null || value.isBlank() ? defaultValue : Double.parseDouble(value.trim());
    }

    private static Duration secondsOf(Properties properties, String key, Duration defaultValue) {
        String value = properties.getProperty(key);
        return value == null || value.isBlank()
                ? defaultValue
                : Duration.ofMillis(Math.round(Double.parseDouble(value.trim()) * 1000));
    }
}
