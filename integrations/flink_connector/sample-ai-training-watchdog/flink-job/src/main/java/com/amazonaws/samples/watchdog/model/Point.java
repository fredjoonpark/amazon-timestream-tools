// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.model;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * One InfluxDB point. Field values are stored already encoded as line protocol values
 * ({@code 42i}, {@code 0.5}, {@code "text"}) so the type is serializable
 * without Flink's generic (Kryo) fallback.
 */
public class Point {

    public String measurement;
    public Map<String, String> tags = new TreeMap<>();
    public Map<String, String> fields = new LinkedHashMap<>();
    /** Epoch milliseconds; the sink writes with millisecond precision. */
    public long timestampMs;

    public Point() {}

    public static Point of(String measurement, long timestampMs) {
        Point point = new Point();
        point.measurement = measurement;
        point.timestampMs = timestampMs;
        return point;
    }

    public Point tag(String key, String value) {
        if (value != null && !value.isEmpty()) {
            tags.put(key, value);
        }
        return this;
    }

    public Point field(String key, double value) {
        if (!Double.isNaN(value) && !Double.isInfinite(value)) {
            fields.put(key, Double.toString(value));
        }
        return this;
    }

    public Point field(String key, Double value) {
        return value == null ? this : field(key, value.doubleValue());
    }

    public Point field(String key, long value) {
        fields.put(key, value + "i");
        return this;
    }

    public Point field(String key, String value) {
        if (value == null) {
            return this;
        }
        StringBuilder quoted = new StringBuilder(value.length() + 2).append('"');
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (c == '\n' || c == '\r') {
                c = ' ';
            } else if (c == '"' || c == '\\') {
                quoted.append('\\');
            }
            quoted.append(c);
        }
        fields.put(key, quoted.append('"').toString());
        return this;
    }
}
