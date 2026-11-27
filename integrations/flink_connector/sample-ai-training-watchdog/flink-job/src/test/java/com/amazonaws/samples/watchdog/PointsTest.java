// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.entry;

import com.amazonaws.samples.watchdog.model.Point;
import com.amazonaws.samples.watchdog.model.TelemetryEvent;
import com.amazonaws.samples.watchdog.sink.LineProtocol;
import com.amazonaws.samples.watchdog.sink.Points;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.junit.jupiter.api.Test;

class PointsTest {

    private static TelemetryEvent parse(String json) throws Exception {
        return TelemetryDeserializer.parse(new ObjectMapper().readTree(json));
    }

    @Test
    void writesThroughEveryTypeButFlinksOwn() throws Exception {
        assertThat(Points.isWriteThrough(parse("{\"type\":\"incident\",\"ts\":1,\"run_id\":\"r\"}"))).isFalse();
        assertThat(Points.isWriteThrough(parse("{\"type\":\"step_summary\",\"ts\":1,\"run_id\":\"r\"}"))).isFalse();
        assertThat(Points.isWriteThrough(parse("{\"type\":\"control_ack\",\"ts\":1,\"run_id\":\"r\"}"))).isTrue();
        assertThat(Points.isWriteThrough(parse("{\"type\":\"train_step\",\"ts\":1,\"run_id\":\"r\"}"))).isTrue();
    }

    /**
     * A producer's tags, metrics, and text can't replace the record's own columns (relabeling the
     * point, or turning the integer {@code step} into a float that InfluxDB rejects), add a blank
     * or {@code time} column, or reuse a tag's name as a field.
     */
    @Test
    void producerColumnsCannotReplaceTheStandardOnes() throws Exception {
        Point point =
                Points.fromTelemetry(
                        parse(
                                "{\"type\":\"control_ack\",\"ts\":5,\"run_id\":\"r1\",\"scenario\":\"healthy\","
                                        + "\"phase\":\"sft\",\"step\":7,\"rank\":0,"
                                        + "\"tags\":{\"run_id\":\"victim\",\"rank\":\"9\",\"time\":\"0\",\"\":\"x\","
                                        + "\"action\":\"STOP\"},"
                                        + "\"metrics\":{\"step\":3.5,\"attempt\":2,\"action\":1,\"latency_ms\":4},"
                                        + "\"text\":{\"phase\":\"grpo\",\"time\":\"t\",\"\":\"y\",\"outcome\":\"ok\"}}"));
        assertThat(point.tags)
                .containsOnly(
                        entry("run_id", "r1"),
                        entry("scenario", "healthy"),
                        entry("phase", "sft"),
                        entry("rank", "0"),
                        entry("action", "STOP"));
        assertThat(point.fields).containsOnlyKeys("step", "attempt", "latency_ms", "outcome");
        assertThat(LineProtocol.format(point))
                .isEqualTo(
                        "control_ack,action=STOP,phase=sft,rank=0,run_id=r1,scenario=healthy"
                                + " step=7i,attempt=1i,latency_ms=4.0,outcome=\"ok\" 5");
    }
}
