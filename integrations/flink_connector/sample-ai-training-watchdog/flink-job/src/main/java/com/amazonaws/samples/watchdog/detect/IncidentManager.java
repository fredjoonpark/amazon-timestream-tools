// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.detect;

import com.amazonaws.samples.watchdog.model.Detection;
import com.amazonaws.samples.watchdog.model.Incident;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.state.StateTtlConfig;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.streaming.api.TimeDomain;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

import java.util.Locale;

/**
 * Turns repeated detector firings into one incident per run, failure type, and attempt: OPEN on
 * the first firing, RESOLVED once no firing arrives for {@link DetectorConfig#incidentQuietPeriod}.
 * Keyed by {@link Detection#incidentKey()}.
 *
 * <p>Each OPEN incident is a command the trainer applies once. So when the trainer has acted
 * (a rollback starts a new attempt) and the failure comes back, that is a new incident with a new
 * command, even while the old one is still open; firings from attempts the trainer has left are
 * ignored.
 *
 * <p>Quiet periods are measured in event time, and also on the wall clock: event time only
 * advances while telemetry arrives, so after a run ends its incident would otherwise stay open
 * until some other run starts.
 */
public class IncidentManager extends KeyedProcessFunction<String, Detection, Incident> {

    private final long quietMillis;
    private transient ValueState<Incident> open;
    private transient ValueState<Long> lastSeen;
    /** Processing time of the open incident's last firing. */
    private transient ValueState<Long> lastSeenWall;
    /** The newest attempt any firing came from; kept after its incident resolves. */
    private transient ValueState<Integer> latestAttempt;

    private final StateTtlConfig runStateTtl;

    public IncidentManager(DetectorConfig config) {
        this.quietMillis = config.incidentQuietPeriod.toMillis();
        this.runStateTtl = config.runStateTtlConfig();
    }

    @Override
    public void open(OpenContext openContext) {
        open = getRuntimeContext().getState(new ValueStateDescriptor<>("open", Incident.class));
        lastSeen = getRuntimeContext().getState(new ValueStateDescriptor<>("last-seen", Long.class));
        lastSeenWall =
                getRuntimeContext().getState(new ValueStateDescriptor<>("last-seen-wall", Long.class));
        // The other three are cleared when the incident resolves; this one is kept, so expires.
        ValueStateDescriptor<Integer> latestAttemptDescriptor =
                new ValueStateDescriptor<>("latest-attempt", Integer.class);
        latestAttemptDescriptor.enableTimeToLive(runStateTtl);
        latestAttempt = getRuntimeContext().getState(latestAttemptDescriptor);
    }

    @Override
    public void processElement(Detection d, Context ctx, Collector<Incident> out) throws Exception {
        Integer latest = latestAttempt.value();
        if (latest != null && d.attempt < latest) {
            return; // the trainer already rolled back past this attempt
        }
        latestAttempt.update(d.attempt);
        long now = ctx.timerService().currentProcessingTime();
        Incident incident = open.value();
        if (incident != null && d.attempt > incident.detection.attempt) {
            // The trainer applied this incident's command and the failure came back.
            out.collect(resolved(incident, Math.min(lastSeen.value() + quietMillis, d.ts), now));
            incident = null;
            lastSeen.clear();
        }
        if (incident == null) {
            incident = new Incident();
            incident.incidentId = d.runId + "-" + d.type.toLowerCase(Locale.ROOT) + "-" + d.ts;
            incident.state = Incident.OPEN;
            incident.ts = d.ts;
            incident.openedTs = d.ts;
            incident.detection = d;
            incident.detections = 1;
            incident.emittedMs = now;
            out.collect(incident);
        } else {
            incident.detections++;
        }
        open.update(incident);
        Long previous = lastSeen.value();
        long seen = previous == null ? d.ts : Math.max(previous, d.ts);
        lastSeen.update(seen);
        ctx.timerService().registerEventTimeTimer(seen + quietMillis);
        lastSeenWall.update(now);
        ctx.timerService().registerProcessingTimeTimer(now + quietMillis);
    }

    @Override
    public void onTimer(long timestamp, OnTimerContext ctx, Collector<Incident> out)
            throws Exception {
        Incident incident = open.value();
        Long seen = lastSeen.value();
        if (incident == null || seen == null) {
            return;
        }
        Long quietSince = ctx.timeDomain() == TimeDomain.EVENT_TIME ? seen : lastSeenWall.value();
        if (quietSince == null || timestamp < quietSince + quietMillis) {
            return;
        }
        out.collect(resolved(incident, seen + quietMillis, ctx.timerService().currentProcessingTime()));
        open.clear();
        lastSeen.clear();
        lastSeenWall.clear();
    }

    private static Incident resolved(Incident incident, long ts, long emittedMs) {
        Incident resolved = new Incident();
        resolved.incidentId = incident.incidentId;
        resolved.state = Incident.RESOLVED;
        resolved.ts = ts;
        resolved.emittedMs = emittedMs;
        resolved.openedTs = incident.openedTs;
        resolved.detections = incident.detections;
        resolved.detection = incident.detection;
        return resolved;
    }
}
