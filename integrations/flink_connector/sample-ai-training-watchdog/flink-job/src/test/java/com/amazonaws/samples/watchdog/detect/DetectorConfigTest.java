// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.detect;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.time.Duration;
import java.util.Properties;

class DetectorConfigTest {

    @Test
    void everyThresholdCanBeSetFromTheDetectorsGroup() throws Exception {
        DetectorConfig defaults = new DetectorConfig();
        int thresholds = 0;
        for (Field field : DetectorConfig.class.getFields()) {
            if (Modifier.isStatic(field.getModifiers())) {
                continue;
            }
            thresholds++;
            Properties properties = new Properties();
            Object expected;
            if (field.getType() == Duration.class) {
                properties.setProperty(field.getName() + "Seconds", "7.5");
                expected = Duration.ofMillis(7_500);
            } else if (field.getType() == double.class) {
                properties.setProperty(field.getName(), "0.123");
                expected = 0.123;
            } else if (field.getType() == int.class) {
                properties.setProperty(field.getName(), "7");
                expected = 7;
            } else {
                assertThat(field.getType()).as(field.getName()).isEqualTo(long.class);
                properties.setProperty(field.getName(), "7");
                expected = 7L;
            }
            assertThat(field.get(defaults)).as(field.getName()).isNotEqualTo(expected);
            assertThat(field.get(DetectorConfig.from(properties))).as(field.getName()).isEqualTo(expected);
        }
        assertThat(thresholds).isEqualTo(19);
    }

    @Test
    void unsetAndBlankPropertiesKeepTheDefaults() {
        Properties properties = new Properties();
        properties.setProperty("hackGapRise", " ");
        DetectorConfig config = DetectorConfig.from(properties);
        assertThat(config.hackGapRise).isEqualTo(new DetectorConfig().hackGapRise);
        assertThat(DetectorConfig.from(null).incidentQuietPeriod).isEqualTo(Duration.ofSeconds(90));
    }
}
