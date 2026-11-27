// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog.sink;

import com.amazonaws.samples.watchdog.model.Point;

import java.util.Map;

/** Formats a {@link Point} as one escaped InfluxDB line protocol row with a millisecond timestamp. */
public final class LineProtocol {

    private LineProtocol() {}

    public static String format(Point point) {
        if (point.fields.isEmpty()) {
            throw new IllegalArgumentException("Point " + point.measurement + " has no fields");
        }
        StringBuilder line = new StringBuilder(128);
        line.append(escape(point.measurement, ", "));
        for (Map.Entry<String, String> tag : point.tags.entrySet()) {
            line.append(',')
                    .append(escape(tag.getKey(), ",= "))
                    .append('=')
                    .append(escape(tag.getValue(), ",= "));
        }
        char separator = ' ';
        for (Map.Entry<String, String> field : point.fields.entrySet()) {
            line.append(separator)
                    .append(escape(field.getKey(), ",= "))
                    .append('=')
                    .append(field.getValue());
            separator = ',';
        }
        return line.append(' ').append(point.timestampMs).toString();
    }

    private static String escape(String value, String special) {
        String text = singleLine(value);
        StringBuilder escaped = new StringBuilder(text.length() + 8);
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (special.indexOf(c) >= 0 || c == '\\') {
                escaped.append('\\');
            }
            escaped.append(c);
        }
        return escaped.toString();
    }

    private static String singleLine(String value) {
        return value.replace('\n', ' ').replace('\r', ' ');
    }
}
