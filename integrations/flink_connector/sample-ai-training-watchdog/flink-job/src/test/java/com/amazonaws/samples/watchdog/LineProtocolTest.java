// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package com.amazonaws.samples.watchdog;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.amazonaws.samples.watchdog.model.Point;
import com.amazonaws.samples.watchdog.sink.LineProtocol;

import org.junit.jupiter.api.Test;

class LineProtocolTest {

    @Test
    void formatsTypedFieldsWithSortedTags() {
        Point point =
                Point.of("train_step", 1_700_000_000_123L)
                        .tag("scenario", "bad_lr")
                        .tag("run_id", "bad_lr-001")
                        .field("step", 42L)
                        .field("loss", 0.5);
        assertThat(LineProtocol.format(point))
                .isEqualTo(
                        "train_step,run_id=bad_lr-001,scenario=bad_lr step=42i,loss=0.5"
                                + " 1700000000123");
    }

    @Test
    void escapesSpecialCharactersAndNewlines() {
        Point point =
                Point.of("my measurement", 1L)
                        .tag("k=1", "a,b c")
                        .field("message", "say \"hi\"\\\nnext");
        assertThat(LineProtocol.format(point))
                .isEqualTo("my\\ measurement,k\\=1=a\\,b\\ c message=\"say \\\"hi\\\"\\\\ next\" 1");
    }

    @Test
    void dropsNonFiniteFieldsAndEmptyTags() {
        Point point = Point.of("m", 1L).tag("empty", "").field("nan", Double.NaN).field("v", 1.0);
        assertThat(LineProtocol.format(point)).isEqualTo("m v=1.0 1");
    }

    @Test
    void rejectsPointsWithoutFields() {
        assertThatThrownBy(() -> LineProtocol.format(Point.of("m", 1L).field("nan", Double.NaN)))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
