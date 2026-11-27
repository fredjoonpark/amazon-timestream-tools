// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog;

import com.amazonaws.samples.watchdog.model.TelemetryEvent;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.apache.flink.api.common.serialization.DeserializationSchema;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.util.Collector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Iterator;
import java.util.Map;
import java.util.regex.Pattern;

/** Parses simulator and trainer JSON records. Malformed records are logged and skipped. */
public class TelemetryDeserializer implements DeserializationSchema<TelemetryEvent> {

    private static final Logger LOG = LoggerFactory.getLogger(TelemetryDeserializer.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    /**
     * {@code run_id}, {@code scenario}, and {@code phase} become tags on the {@code step_summary}
     * and {@code incident} rows Flink writes. The sink for those fails the job on a rejected
     * point, so a record whose values can't make a valid point (an oversized or unparseable tag)
     * is skipped here instead of failing the job on every restart.
     */
    private static final Pattern ID = Pattern.compile("[A-Za-z0-9._:-]{1,128}");
    /**
     * {@code ts} (epoch milliseconds) becomes the time of those rows too, and InfluxDB rejects a
     * time after 2262-04-11 or before 1677-09-21. The accepted range, 1970 to 2200, leaves room for
     * the times Flink derives from it (an incident resolves up to its quiet period later).
     */
    static final long MIN_TS = 0L;
    static final long MAX_TS = Instant.parse("2200-01-01T00:00:00Z").toEpochMilli();
    /**
     * How far past this host's clock {@code ts} may be. Producers stamp records with their own
     * synced clocks, so a later time is a bad clock or a bad record, and it would move event time
     * ahead for every run (the watermark is shared): incidents would resolve as they open, steps
     * would flush without their late ranks, and the run's divergence detection would discard
     * every later spike as too old. At most the watermark's out-of-orderness, so no record can
     * move the watermark past this host's clock.
     */
    static final Duration MAX_AHEAD = WatchdogTopology.OUT_OF_ORDERNESS;

    @Override
    public TelemetryEvent deserialize(byte[] message) throws IOException {
        return parse(MAPPER.readTree(message));
    }

    @Override
    public void deserialize(byte[] message, Collector<TelemetryEvent> out) {
        TelemetryEvent event;
        try {
            event = deserialize(message);
        } catch (IOException | RuntimeException e) {
            LOG.warn("Skipping malformed telemetry record: {}", e.getMessage());
            return;
        }
        // Outside the try: collect() runs the operators chained to the source, including the
        // telemetry sink, so an exception from them must fail the task, not drop the record.
        out.collect(event);
    }

    static TelemetryEvent parse(JsonNode node) {
        TelemetryEvent event = new TelemetryEvent();
        event.type = requireText(node, "type");
        JsonNode ts = require(node, "ts");
        // asLong() reads anything that isn't a number as 0, which is inside the range below.
        if (!ts.isNumber()) {
            throw new IllegalArgumentException("'ts' is not a number");
        }
        event.ts = ts.asLong();
        if (event.ts < MIN_TS || event.ts >= MAX_TS) {
            throw new IllegalArgumentException("'ts' out of range: " + event.ts);
        }
        if (event.ts > System.currentTimeMillis() + MAX_AHEAD.toMillis()) {
            throw new IllegalArgumentException(
                    "'ts' is more than " + MAX_AHEAD.toSeconds() + " s ahead of the clock: " + event.ts);
        }
        event.runId = id(requireText(node, "run_id"), "run_id");
        event.scenario = id(node.path("scenario").asText("unknown"), "scenario");
        event.phase = id(node.path("phase").asText("unknown"), "phase");
        event.attempt = node.path("attempt").asInt(1);
        event.step = node.path("step").asLong(0);
        event.rank = node.path("rank").asInt(-1);
        event.worldSize = node.path("world_size").asInt(0);
        for (Iterator<Map.Entry<String, JsonNode>> it = node.path("tags").fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> tag = it.next();
            // asText() reads a JSON null as the string "null"; leave the tag out instead.
            if (!tag.getValue().isNull()) {
                event.tags.put(tag.getKey(), tag.getValue().asText());
            }
        }
        for (Iterator<Map.Entry<String, JsonNode>> it = node.path("text").fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> field = it.next();
            if (field.getValue().isTextual()) {
                event.text.put(field.getKey(), field.getValue().asText());
            }
        }
        for (Iterator<Map.Entry<String, JsonNode>> it = node.path("metrics").fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> metric = it.next();
            if (metric.getValue().isNumber()) {
                event.metrics.put(metric.getKey(), metric.getValue().asDouble());
            }
        }
        return event;
    }

    private static JsonNode require(JsonNode node, String field) {
        JsonNode value = node.get(field);
        if (value == null || value.isNull()) {
            throw new IllegalArgumentException("missing field '" + field + "'");
        }
        return value;
    }

    private static String requireText(JsonNode node, String field) {
        String value = require(node, field).asText();
        if (value.isBlank()) {
            throw new IllegalArgumentException("blank field '" + field + "'");
        }
        return value;
    }

    private static String id(String value, String field) {
        if (!ID.matcher(value).matches()) {
            // Leave the value out of the message: it may be the oversized string itself.
            throw new IllegalArgumentException("invalid '" + field + "'");
        }
        return value;
    }

    @Override
    public boolean isEndOfStream(TelemetryEvent nextElement) {
        return false;
    }

    @Override
    public TypeInformation<TelemetryEvent> getProducedType() {
        return TypeInformation.of(TelemetryEvent.class);
    }
}
