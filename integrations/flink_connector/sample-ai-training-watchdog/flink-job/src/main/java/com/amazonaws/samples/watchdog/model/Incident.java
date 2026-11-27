// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.model;

/** An incident lifecycle transition: OPEN when first detected, RESOLVED after a quiet period. */
public class Incident {

    public static final String OPEN = "OPEN";
    public static final String RESOLVED = "RESOLVED";

    public String incidentId;
    public String state;
    /** Event time of this transition, in epoch milliseconds. */
    public long ts;
    public long openedTs;
    /**
     * Wall-clock time Flink emitted this transition, in epoch milliseconds. Against {@link #ts}
     * (event time), it splits the control loop's latency into Flink's share and InfluxDB's.
     */
    public long emittedMs;
    public int detections;
    public Detection detection;

    public Incident() {}
}
