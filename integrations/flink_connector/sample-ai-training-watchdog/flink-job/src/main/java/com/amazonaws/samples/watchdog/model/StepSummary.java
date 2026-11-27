// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.model;

/** Cross-rank summary of one training step, annotated with run history for detectors. */
public class StepSummary {

    public String runId;
    public String scenario;
    public String phase;
    public int attempt;
    public long step;
    /** Event time of the slowest rank's report, in epoch milliseconds. */
    public long ts;
    public int ranksReporting;
    public double loss;
    public double gradNorm;
    public double stepTimeMedianMs;
    public double stepTimeMaxMs;
    public int slowestRank;
    /** Slowest rank's step time divided by the median. */
    public double skew;
    /** Exponential moving averages before this step was folded in. */
    public double lossEma;
    public double gradNormEma;

    public StepSummary() {}
}
