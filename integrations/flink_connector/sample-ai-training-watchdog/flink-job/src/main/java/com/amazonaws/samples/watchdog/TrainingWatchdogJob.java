// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog;

import com.amazonaws.samples.watchdog.detect.DetectorConfig;
import com.amazonaws.samples.watchdog.model.Point;
import com.amazonaws.samples.watchdog.model.TelemetryEvent;
import com.amazonaws.samples.watchdog.sink.LineProtocol;

import com.amazonaws.services.kinesisanalytics.runtime.KinesisAnalyticsRuntime;
import com.influxdb.v3.client.write.WritePrecision;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.aws.config.AWSConfigOptions;
import org.apache.flink.connector.kinesis.source.KinesisStreamsSource;
import org.apache.flink.connector.kinesis.source.config.KinesisSourceConfigOptions;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import software.amazon.timestream.influxdb3.flink.InfluxDB3Sink;
import software.amazon.timestream.influxdb3.flink.auth.SecretsManagerTokenProvider;

import java.io.IOException;
import java.util.Map;
import java.util.Properties;

/**
 * Managed Flink entry point: Kinesis telemetry in, InfluxDB 3 points out. Incidents are points
 * too, and they are the control plane: each OPEN incident row carries the action for the trainer
 * (ROLLBACK, STOP, or ADVISE, which never stops a run), and every trainer polls the {@code
 * incident} table for its own run. There is no separate control channel.
 *
 * <p>Property groups: {@code Telemetry} (streamArn, region, initialPosition), {@code InfluxDB3}
 * (endpoint, database, secretId, secretJsonField, region), and optionally {@code Detectors} (see
 * {@link DetectorConfig#from}).
 */
public final class TrainingWatchdogJob {

    private TrainingWatchdogJob() {}

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        build(env, loadProperties());
        env.execute("AI Training Watchdog");
    }

    static void build(StreamExecutionEnvironment env, Map<String, Properties> groups) {
        Properties telemetry = require(groups, "Telemetry");
        Properties influx = require(groups, "InfluxDB3");
        DetectorConfig detectors = DetectorConfig.from(groups.get("Detectors"));

        DataStream<TelemetryEvent> events =
                env.fromSource(
                                telemetrySource(telemetry),
                                WatchdogTopology.watermarks(),
                                "train-telemetry",
                                // The Kinesis source does not expose its produced type.
                                TypeInformation.of(TelemetryEvent.class))
                        .uid("train-telemetry-source");

        WatchdogTopology.Outputs outputs = WatchdogTopology.build(events, detectors);
        // The producers' records.
        outputs.telemetry()
                .sinkTo(influxSink(influx, false))
                .name("timestream-influxdb3")
                .uid("timestream-influxdb3-sink");
        // Step summaries and incidents, which carry the commands.
        outputs.control()
                .sinkTo(influxSink(influx, true))
                .name("timestream-influxdb3-control")
                .uid("timestream-influxdb3-control-sink");
    }

    static KinesisStreamsSource<TelemetryEvent> telemetrySource(Properties p) {
        Configuration config = new Configuration();
        config.set(AWSConfigOptions.AWS_REGION_OPTION, requireProperty(p, "region"));
        config.set(
                KinesisSourceConfigOptions.STREAM_INITIAL_POSITION,
                KinesisSourceConfigOptions.InitialPosition.valueOf(
                        p.getProperty("initialPosition", "LATEST")));
        return KinesisStreamsSource.<TelemetryEvent>builder()
                .setStreamArn(requireProperty(p, "streamArn"))
                .setSourceConfig(config)
                .setDeserializationSchema(new TelemetryDeserializer())
                .build();
    }

    /**
     * An InfluxDB sink. InfluxDB can reject single lines of a batch (a field-type conflict, for
     * example) or a whole batch (HTTP 400 or 422). What the sink does then depends on whose lines
     * they are:
     *
     * <ul>
     *   <li>{@code control} (step summaries and incidents, whose schema the job fixes): fail, and
     *       restart from the last checkpoint. A dropped incident row would be a silently lost
     *       command; for a control plane a visible outage is better. The accepted lines of the
     *       batch are rewritten on replay, which is harmless: a replayed point overwrites itself.
     *   <li>Telemetry (whatever the producers send): drop the line and count it in {@code
     *       numRecordsDropped}. Failing would replay the same line, which InfluxDB would reject
     *       again, for as long as the stream keeps the record (24 hours), and stop every write,
     *       incidents included.
     * </ul>
     */
    static InfluxDB3Sink<Point> influxSink(Properties p, boolean control) {
        SecretsManagerTokenProvider.Builder token =
                SecretsManagerTokenProvider.builder()
                        .setSecretId(requireProperty(p, "secretId"))
                        .setRegion(requireProperty(p, "region"));
        String jsonField = p.getProperty("secretJsonField");
        if (jsonField != null && !jsonField.isBlank()) {
            token.setJsonField(jsonField);
        }
        var sink =
                InfluxDB3Sink.<Point>builder()
                        .setEndpoint(requireProperty(p, "endpoint"))
                        .setDatabase(requireProperty(p, "database"))
                        .setTokenProvider(token.build())
                        .setConverter(LineProtocol::format)
                        .setWritePrecision(WritePrecision.MS)
                        .setMaxTimeInBufferMS(1_000)
                        // Rejected lines in a partly accepted batch.
                        .setFailOnRejected(control)
                        // A whole batch rejected as invalid, or a point the converter can't format.
                        .setFailOnValidation(control)
                        // Log each failed line with InfluxDB's reason. Lines hold telemetry, run
                        // IDs, and sampled SQL; never the token.
                        .setPrintFailedRequests(true)
                        .setManagedFlinkMetricsEnabled(true);
        if (control) {
            // Its own CloudWatch dimension (Sink=control). The telemetry sink, nearly every
            // point, publishes without one.
            sink.setManagedFlinkMetricsSinkName("control");
        }
        return sink.build();
    }

    private static Map<String, Properties> loadProperties() throws IOException {
        String localFile = System.getenv("APPLICATION_PROPERTIES_FILE");
        return localFile != null && !localFile.isBlank()
                ? KinesisAnalyticsRuntime.getApplicationProperties(localFile)
                : KinesisAnalyticsRuntime.getApplicationProperties();
    }

    private static Properties require(Map<String, Properties> groups, String name) {
        Properties group = groups.get(name);
        if (group == null) {
            throw new IllegalArgumentException("Missing property group '" + name + "'");
        }
        return group;
    }

    private static String requireProperty(Properties p, String key) {
        String value = p.getProperty(key);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("Missing property '" + key + "'");
        }
        return value.trim();
    }
}
