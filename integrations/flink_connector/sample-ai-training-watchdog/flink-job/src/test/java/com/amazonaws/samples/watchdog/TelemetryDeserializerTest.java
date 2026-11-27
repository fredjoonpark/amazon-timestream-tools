// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.amazonaws.samples.watchdog.model.TelemetryEvent;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

class TelemetryDeserializerTest {

    private final TelemetryDeserializer deserializer = new TelemetryDeserializer();

    private List<TelemetryEvent> parse(String json) {
        List<TelemetryEvent> out = new ArrayList<>();
        deserializer.deserialize(
                json.getBytes(StandardCharsets.UTF_8),
                new org.apache.flink.util.Collector<>() {
                    @Override
                    public void collect(TelemetryEvent record) {
                        out.add(record);
                    }

                    @Override
                    public void close() {}
                });
        return out;
    }

    @Test
    void parsesRankEvent() {
        List<TelemetryEvent> events =
                parse(
                        "{\"type\":\"train_step\",\"ts\":5,\"run_id\":\"r\",\"scenario\":\"healthy\","
                                + "\"phase\":\"sft\",\"attempt\":2,\"step\":7,\"rank\":3,\"world_size\":8,"
                                + "\"tags\":{\"x\":\"y\",\"n\":null},\"metrics\":{\"loss\":0.4,\"bad\":\"text\"}}");
        assertThat(events).hasSize(1);
        TelemetryEvent e = events.get(0);
        assertThat(e.type).isEqualTo("train_step");
        assertThat(e.attempt).isEqualTo(2);
        assertThat(e.rank).isEqualTo(3);
        assertThat(e.worldSize).isEqualTo(8);
        assertThat(e.tags).containsOnlyKeys("x").containsEntry("x", "y");
        assertThat(e.metrics).containsOnlyKeys("loss");
    }

    @Test
    void parsesTextFieldsAndWritesThemAsStringFields() {
        List<TelemetryEvent> events =
                parse(
                        "{\"type\":\"sample\",\"ts\":5,\"run_id\":\"r\",\"phase\":\"grpo\","
                                + "\"metrics\":{\"reward\":1.0},"
                                + "\"text\":{\"sql\":\"SELECT * FROM cpu LIMIT 1\",\"bad\":3}}");
        assertThat(events).hasSize(1);
        assertThat(events.get(0).text).containsOnlyKeys("sql");
        String line =
                com.amazonaws.samples.watchdog.sink.LineProtocol.format(
                        com.amazonaws.samples.watchdog.sink.Points.fromTelemetry(events.get(0)));
        assertThat(line).contains("sql=\"SELECT * FROM cpu LIMIT 1\"").contains("reward=1.0");
    }

    @Test
    void defaultsRunLevelFields() {
        TelemetryEvent e = parse("{\"type\":\"eval\",\"ts\":1,\"run_id\":\"r\"}").get(0);
        assertThat(e.rank).isEqualTo(-1);
        assertThat(e.attempt).isEqualTo(1);
    }

    @Test
    void skipsMalformedRecords() {
        assertThat(parse("not json")).isEmpty();
        assertThat(parse("{\"ts\":1,\"run_id\":\"r\"}")).isEmpty();
    }

    @Test
    void propagatesFailuresFromDownstreamOperators() {
        // collect() runs the operators chained to the source, including the telemetry sink; their
        // failures must fail the task, not be logged as a malformed record and dropped.
        org.apache.flink.util.Collector<TelemetryEvent> failing =
                new org.apache.flink.util.Collector<>() {
                    @Override
                    public void collect(TelemetryEvent record) {
                        throw new IllegalStateException("sink failed");
                    }

                    @Override
                    public void close() {}
                };
        byte[] valid = train("{\"ts\":5").getBytes(StandardCharsets.UTF_8);
        assertThatThrownBy(() -> deserializer.deserialize(valid, failing))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("sink failed");
    }

    @Test
    void acceptsTheRunIdsTrainersAndTheSimulatorWrite() {
        assertThat(parse(event("trn-reward-hacking-s1-202611202021", "trn_reward_hacking", "grpo")))
                .hasSize(1);
        assertThat(parse(event("healthy-sim001", "healthy", "sft"))).hasSize(1);
    }

    @Test
    void skipsRecordsWhoseIdsCantBeTags() {
        // About 600 KB of commas escapes to a step_summary line over the sink's size limit.
        String commas = ",".repeat(600_000);
        assertThat(parse(event(commas, "healthy", "sft"))).isEmpty();
        assertThat(parse(event("r", commas, "sft"))).isEmpty();
        assertThat(parse(event("r", "healthy", commas))).isEmpty();
        assertThat(parse(event("x".repeat(129), "healthy", "sft"))).isEmpty();
        assertThat(parse(event("r\tx", "healthy", "sft"))).isEmpty();
        assertThat(parse(event("r x", "healthy", "sft"))).isEmpty();
        assertThat(parse(event("r", "", "sft"))).isEmpty();
    }

    @Test
    void skipsRecordsWhoseTimeCantBeARowTime() {
        // InfluxDB 3 rejects these as a row's time ("later than the latest supported timestamp",
        // "out of range for precision: Millisecond"), which would fail the control sink.
        for (String ts : List.of("9223372036854", "32503680000000", "9223372036854775807", "-1")) {
            assertThat(parse(train("{\"ts\":" + ts))).as(ts).isEmpty();
        }
        assertThat(parse(train("{\"ts\":" + (TelemetryDeserializer.MAX_TS - 1)))).isEmpty(); // in the future
        assertThat(parse(train("{\"ts\":1795000000000"))).hasSize(1);
    }

    @Test
    void skipsRecordsFromTheFuture() {
        // One would move event time ahead for every run: allow clock skew, up to the watermark's
        // out-of-orderness (5 s), so the watermark never passes Flink's clock.
        long now = System.currentTimeMillis();
        assertThat(parse(train("{\"ts\":" + (now + 2_000)))).hasSize(1);
        assertThat(parse(train("{\"ts\":" + (now + 30_000)))).isEmpty();
    }

    @Test
    void skipsRecordsWhoseTimeIsNotANumber() {
        // JsonNode.asLong() reads these as 0, which would pass the range check as a 1970 time.
        for (String ts : List.of("\"garbage\"", "\"1795000000000\"", "{}", "[]", "true")) {
            assertThat(parse(train("{\"ts\":" + ts))).as(ts).isEmpty();
        }
    }

    private static String train(String head) {
        return head + ",\"type\":\"train_step\",\"run_id\":\"r\",\"step\":1,\"rank\":0,\"world_size\":1,"
                + "\"metrics\":{\"loss\":0.4}}";
    }

    private static String event(String runId, String scenario, String phase) {
        return "{\"type\":\"train_step\",\"ts\":5,\"run_id\":\"" + runId.replace("\t", "\\t")
                + "\",\"scenario\":\"" + scenario + "\",\"phase\":\"" + phase
                + "\",\"step\":1,\"rank\":0,\"world_size\":1,\"metrics\":{\"loss\":0.4}}";
    }
}
