/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

import java.time.LocalDate;

import org.junit.jupiter.api.Test;

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;

public class CastDatesTest {

  @Test
  void castDateToString() {
    int days1 = (int) LocalDate.of(2025, 1, 3).toEpochDay();
    int days2 = (int) LocalDate.of(123456, 11, 2).toEpochDay();
    int days3 = (int) LocalDate.of(-123456, 3, 31).toEpochDay();
    int days4 = (int) LocalDate.of(1, 3, 31).toEpochDay();

    try (ColumnVector inputCv = ColumnVector.timestampDaysFromBoxedInts(
        null, days1, days2, days3, days4);
        ColumnVector actual = CastDates.toString(inputCv);
        ColumnVector expected = ColumnVector.fromStrings(
            null,
            "2025-01-03",
            "123456-11-02",
            "-123456-03-31",
            "0001-03-31")) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }
  }
}
