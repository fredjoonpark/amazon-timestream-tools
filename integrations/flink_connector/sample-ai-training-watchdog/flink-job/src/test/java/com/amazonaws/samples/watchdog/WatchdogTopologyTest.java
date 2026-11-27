// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog;

import static org.assertj.core.api.Assertions.assertThat;

import com.amazonaws.samples.watchdog.detect.DetectorConfig;
import com.amazonaws.samples.watchdog.model.Detection;
import com.amazonaws.samples.watchdog.model.Incident;
import com.amazonaws.samples.watchdog.model.Point;
import com.amazonaws.samples.watchdog.model.TelemetryEvent;
import com.amazonaws.samples.watchdog.sink.LineProtocol;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.CloseableIterator;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;

/**
 * Replays a simulator fixture (all four scenarios, 240 steps each) through the detection
 * topology and checks that each failure scenario opens exactly the expected incident, and the
 * healthy runs open none: simulated healthy_grpo's grader is honest (exec_match equals the reward)
 * and its held-out accuracy never falls, so it opens neither REWARD_HACKING nor HELDOUT_DECLINE.
 * Regenerate the fixture (from {@code simulator/}) with {@code python -m watchdog_sim.main --sink
 * file --telemetry light --steps 240 --out ../flink-job/src/test/resources/scenarios.jsonl.gz}.
 */
class WatchdogTopologyTest {

    private static List<TelemetryEvent> fixture;

    @BeforeAll
    static void loadFixture() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        fixture = new ArrayList<>();
        try (var in =
                new BufferedReader(
                        new InputStreamReader(
                                new GZIPInputStream(
                                        WatchdogTopologyTest.class.getResourceAsStream(
                                                "/scenarios.jsonl.gz")),
                                StandardCharsets.UTF_8))) {
            String line;
            while ((line = in.readLine()) != null) {
                fixture.add(TelemetryDeserializer.parse(mapper.readTree(line)));
            }
        }
    }

    /**
     * The fixture, each run's events in order, as in production: a run's rl_step and eval events
     * share one Kinesis partition key, so one source subtask reads them in order, and the
     * reward-hacking detector relies on that. {@code fromData} is a single-subtask source. Any
     * operator after it at the job's parallelism (2) would receive the events round-robin, and the
     * keyed detectors downstream would see each run's events in whatever order the two subtasks
     * interleaved them (an RL step after a later evaluation, for example). So the timestamp
     * assigner runs at the source's parallelism (chained to it), and the stream is partitioned by
     * run before the topology's first operators.
     */
    private static DataStream<TelemetryEvent> events(StreamExecutionEnvironment env) {
        return env.fromData(fixture, TypeInformation.of(TelemetryEvent.class))
                .assignTimestampsAndWatermarks(WatchdogTopology.watermarks())
                .setParallelism(1)
                .keyBy(e -> e.runId);
    }

    @Test
    void eachFailureScenarioOpensItsIncident() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);
        List<Incident> incidents = new ArrayList<>();
        try (CloseableIterator<Incident> it =
                WatchdogTopology.build(events(env), new DetectorConfig()).incidents().executeAndCollect()) {
            it.forEachRemaining(incidents::add);
        }

        Map<String, List<String>> opened =
                incidents.stream()
                        .filter(i -> Incident.OPEN.equals(i.state))
                        .collect(
                                Collectors.groupingBy(
                                        i -> i.detection.scenario,
                                        Collectors.mapping(i -> i.detection.type, Collectors.toList())));
        assertThat(opened)
                .containsOnly(
                        Map.entry("bad_lr", List.of(Detection.DIVERGENCE)),
                        Map.entry("reward_hacking", List.of(Detection.REWARD_HACKING)));

        Map<String, Detection> byType =
                incidents.stream()
                        .filter(i -> Incident.OPEN.equals(i.state))
                        .collect(Collectors.toMap(i -> i.detection.type, i -> i.detection));
        Detection divergence = byType.get(Detection.DIVERGENCE);
        assertThat(divergence.action).isEqualTo("ROLLBACK");
        assertThat(divergence.onsetStep).isBetween(110L, 130L);
        // The trainer restores its newest checkpoint at or before this: 100 here.
        assertThat(divergence.actionArg).isEqualTo(Long.toString(divergence.onsetStep - 2));
        Detection hacking = byType.get(Detection.REWARD_HACKING);
        assertThat(hacking.action).isEqualTo("STOP");
        assertThat(hacking.onsetStep).isBetween(80L, 140L);
        assertThat(hacking.actionArg).isEqualTo(Long.toString(hacking.onsetStep));
        // STOP rather than HELDOUT_DECLINE / ADVISE: the simulated lenient grader's gap
        // (reward - exec_match) rose well past hackGapRise (0.15) since the best evaluation.
        assertThat(hacking.message)
                .startsWith("Reward rose 0.71 -> 0.84 while held-out accuracy fell 0.66 -> 0.47 since step 100;")
                .contains("; grader gap -0.06 -> 0.39;");
        assertThat(hacking.step).isEqualTo(160);

        // The fixture ends, so the final watermark resolves every open incident.
        assertThat(incidents.stream().filter(i -> Incident.RESOLVED.equals(i.state)).count())
                .isEqualTo(2);
    }

    @Test
    void everyPointFormatsAsLineProtocol() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);
        long count = 0;
        WatchdogTopology.Outputs outputs = WatchdogTopology.build(events(env), new DetectorConfig());
        try (CloseableIterator<String> it =
                outputs.telemetry().union(outputs.control()).map(LineProtocol::format).executeAndCollect()) {
            while (it.hasNext()) {
                assertThat(it.next()).doesNotContain("\n").matches("^[a-z_]+,.* \\d{13}$");
                count++;
            }
        }
        // Raw telemetry, one summary per SFT step, and incident transitions.
        assertThat(count).isGreaterThan(fixture.size() + 2 * 240);
    }

    /** A producer can't write the rows trainers take commands from, or step summaries. */
    @Test
    void producerRecordsDoNotWriteFlinkMeasurements() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        List<TelemetryEvent> forged = new ArrayList<>();
        for (String json :
                List.of(
                        "{\"type\":\"incident\",\"ts\":1000,\"run_id\":\"victim\","
                                + "\"tags\":{\"state\":\"OPEN\",\"type\":\"REWARD_HACKING\"},"
                                + "\"text\":{\"action\":\"STOP\",\"action_arg\":\"0\",\"incident_id\":\"x\"}}",
                        "{\"type\":\"step_summary\",\"ts\":1000,\"run_id\":\"victim\",\"metrics\":{\"loss\":9.0}}",
                        "{\"type\":\"heartbeat\",\"ts\":1000,\"run_id\":\"victim\"}")) {
            forged.add(TelemetryDeserializer.parse(mapper.readTree(json)));
        }
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        DataStream<TelemetryEvent> events =
                env.fromData(forged, TypeInformation.of(TelemetryEvent.class))
                        .assignTimestampsAndWatermarks(WatchdogTopology.watermarks());
        WatchdogTopology.Outputs outputs = WatchdogTopology.build(events, new DetectorConfig());
        List<String> measurements = new ArrayList<>();
        try (CloseableIterator<Point> it =
                outputs.telemetry().union(outputs.control()).executeAndCollect()) {
            it.forEachRemaining(point -> measurements.add(point.measurement));
        }
        assertThat(measurements).containsExactly("heartbeat");
    }
}
