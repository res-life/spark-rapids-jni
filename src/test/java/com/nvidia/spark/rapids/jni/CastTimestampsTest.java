/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.jni;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;

import org.junit.jupiter.api.Test;

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;

public class CastTimestampsTest {

  private static long toMicros(int year, int month, int day,
      int hour, int minute, int second, int microsecond) {
    Instant instant = LocalDateTime.of(
        year, month, day, hour, minute, second, /* ns */ microsecond * 1000)
        .atZone(ZoneOffset.UTC).toInstant();
    return instant.getEpochSecond() * 1_000_000 + instant.getNano() / 1000;
  }

  @Test
  void castDateToString() {
    long ts1 = toMicros(2025, 1, 12, 23, 22, 59, 123456);
    long ts2 = toMicros(2025, 1, 1, 0, 0, 0, 0);
    long ts3 = toMicros(2025, 1, 1, 0, 0, 0, 1);
    long ts4 = toMicros(-2025, 1, 1, 0, 0, 0, 0);
    long ts5 = toMicros(-2025, 1, 1, 0, 0, 0, 1);
    long ts6 = toMicros(123456, 1, 1, 0, 0, 0, 0);
    long ts7 = toMicros(-123456, 1, 1, 0, 0, 0, 0);
    long ts8 = toMicros(123456, 12, 3, 1, 2, 3, 300);
    long ts9 = toMicros(-123456, 12, 3, 1, 2, 3, 30);
    long ts10 = toMicros(12, 1, 1, 12, 33, 44, 0);
    long ts11 = toMicros(0, 1, 1, 12, 33, 44, 0);

    try (ColumnVector inputCv = ColumnVector.timestampMicroSecondsFromBoxedLongs(
        null,
        ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9, ts10, ts11);
        ColumnVector actual = CastTimestamps.toString(inputCv);
        ColumnVector expected = ColumnVector.fromStrings(
            null,
            "2025-01-12 23:22:59.123456",
            "2025-01-01 00:00:00",
            "2025-01-01 00:00:00.000001",
            "-2025-01-01 00:00:00",
            "-2025-01-01 00:00:00.000001",
            "123456-01-01 00:00:00",
            "-123456-01-01 00:00:00",
            "123456-12-03 01:02:03.0003",
            "-123456-12-03 01:02:03.00003",
            "0012-01-01 12:33:44",
            "0000-01-01 12:33:44")) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }
  }
}
