// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog;

import static org.assertj.core.api.Assertions.assertThat;

import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.PipelineOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Properties;

/** Builds the production job graph (sources, sinks, detectors) without connecting to AWS. */
class TrainingWatchdogJobTest {

    static Map<String, Properties> properties() {
        Properties telemetry = new Properties();
        telemetry.setProperty("streamArn", "arn:aws:kinesis:us-east-1:123456789012:stream/telemetry");
        telemetry.setProperty("region", "us-east-1");
        telemetry.setProperty("initialPosition", "LATEST");
        Properties influx = new Properties();
        influx.setProperty("endpoint", "https://example.timestream-influxdb.us-east-1.on.aws:8181");
        influx.setProperty("database", "watchdog");
        influx.setProperty(
                "secretId", "arn:aws:secretsmanager:us-east-1:123456789012:secret:influx-AbCdEf");
        influx.setProperty("secretJsonField", "token");
        influx.setProperty("region", "us-east-1");
        return Map.of("Telemetry", telemetry, "InfluxDB3", influx);
    }

    @Test
    void buildsJobGraph() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        TrainingWatchdogJob.build(env, properties());
        assertThat(env.getStreamGraph().getJobGraph().getNumberOfVertices()).isPositive();
    }

    @Test
    void buildsJobGraphWithGenericTypesDisabled() {
        Configuration config = new Configuration();
        config.set(PipelineOptions.GENERIC_TYPES, false);
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        TrainingWatchdogJob.build(env, properties());
        assertThat(env.getStreamGraph().getJobGraph().getNumberOfVertices()).isPositive();
    }
}
