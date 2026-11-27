// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.model;

/** A single detector firing. The incident manager turns repeated firings into one incident. */
public class Detection {

    public static final String DIVERGENCE = "DIVERGENCE";
    public static final String REWARD_HACKING = "REWARD_HACKING";
    /**
     * Held-out accuracy fell while the reward did not, but the training grader agreed with the
     * strict check: likely overfitting rather than a gamed grader. Advisory (action ADVISE).
     */
    public static final String HELDOUT_DECLINE = "HELDOUT_DECLINE";

    public String runId;
    public String scenario;
    /** Training attempt the firing came from; a rollback starts the next attempt. */
    public int attempt;
    public String type;
    public long ts;
    public long step;
    /** Step where the failure began, which can precede the detection step. */
    public long onsetStep;
    public String severity;
    public String message;
    /**
     * Control action for the trainer: ROLLBACK, STOP, or ADVISE (the trainer keeps the adapters of
     * the evaluation it names and keeps training).
     */
    public String action;
    /**
     * Action argument: for ROLLBACK, the SFT checkpoint step to restore; for STOP and ADVISE, the
     * best held-out evaluation's step, whose adapters the trainer keeps.
     */
    public String actionArg;

    public Detection() {}

    public static Detection of(
            String runId,
            String scenario,
            int attempt,
            String type,
            long ts,
            long step,
            long onsetStep,
            String severity,
            String message,
            String action,
            String actionArg) {
        Detection d = new Detection();
        d.runId = runId;
        d.scenario = scenario;
        d.attempt = attempt;
        d.type = type;
        d.ts = ts;
        d.step = step;
        d.onsetStep = onsetStep;
        d.severity = severity;
        d.message = message;
        d.action = action;
        d.actionArg = actionArg;
        return d;
    }

    public String incidentKey() {
        return runId + "|" + type;
    }
}
