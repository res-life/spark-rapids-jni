/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ai.rapids.cudf.*;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.zone.ZoneOffsetTransition;
import java.time.zone.ZoneOffsetTransitionRule;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.TimeZone;
import java.util.concurrent.TimeUnit;

public class GpuTimeZoneDBTest {

  private static final long microsPerMillis = TimeUnit.MILLISECONDS.toMicros(1);
  private static final long MICROS_PER_SECOND = TimeUnit.SECONDS.toMicros(1);

  private static TimeZone getTimeZoneForOrc(String timezoneId) {
    return TimeZone.getTimeZone(GpuTimeZoneDB.getZoneId(timezoneId));
  }

  private static long orc2015YearBaseOffsetUs(String timezoneId) {
    OrcTimezoneInfo info = OrcTimezoneInfo.get(timezoneId);
    if (info.transitions == null && info.dstRule == null) {
      return TimeUnit.MILLISECONDS.toMicros(info.rawOffset);
    }
    TimeZone tz = getTimeZoneForOrc(timezoneId);
    return TimeUnit.MILLISECONDS.toMicros(
        tz.getOffset(OrcTimezoneInfo.utcMillisForDate(2015, 1, 1)));
  }

  private static long applyOrcBaseOffsetOnCPU(long decodedUs, long baseOffsetUs) {
    if (baseOffsetUs == 0) {
      return decodedUs;
    }

    // ORC timezone base offsets are second-aligned. For an arbitrary microsecond offset, the
    // original nanos field cannot be reconstructed reliably, so retain the plain offset behavior.
    if (baseOffsetUs % MICROS_PER_SECOND != 0) {
      return decodedUs - baseOffsetUs;
    }

    long fractionalUs = Math.floorMod(decodedUs, MICROS_PER_SECOND);
    boolean hasBorrowableFraction = fractionalUs >= microsPerMillis;
    boolean cudfAppliedBorrow = decodedUs < 0 && hasBorrowableFraction;

    long unborrowedUs = decodedUs + (cudfAppliedBorrow ? MICROS_PER_SECOND : 0L);
    long adjustedUnborrowedUs = unborrowedUs - baseOffsetUs;
    boolean apacheAppliesBorrow = adjustedUnborrowedUs < 0 && hasBorrowableFraction;

    return adjustedUnborrowedUs - (apacheAppliesBorrow ? MICROS_PER_SECOND : 0L);
  }

  private static long[] getFutureDstBoundaryMicros(String timezoneId) {
    List<ZoneOffsetTransitionRule> rules =
        ZoneId.of(timezoneId).getRules().getTransitionRules();
    assertEquals(2, rules.size(), "expected two recurring DST rules for " + timezoneId);
    long[] microseconds = new long[rules.size() * 3];
    int index = 0;
    for (ZoneOffsetTransitionRule rule : rules) {
      ZoneOffsetTransition transition = rule.createTransition(9999);
      long transitionMillis = transition.getInstant().toEpochMilli();
      microseconds[index++] = (transitionMillis - 1) * microsPerMillis;
      microseconds[index++] = transitionMillis * microsPerMillis;
      microseconds[index++] = (transitionMillis + 1) * microsPerMillis;
    }
    return microseconds;
  }

  /**
   * Java implementation of timezone conversion to compare against the GPU
   * results.
   * Refer to https://github.com/apache/orc/blob/rel/release-1.9.1/java/core/
   * src/java/org/apache/orc/impl/SerializationUtils.java#L1440
   */
  private static ColumnVector convertOrcTimezonesOnCPU(
      long[] microseconds,
      String writeTzId,
      String readerTzId) {
    long[] results = new long[microseconds.length];
    TimeZone writeTz = getTimeZoneForOrc(writeTzId);
    TimeZone readerTz = getTimeZoneForOrc(readerTzId);
    long writer2015YearBaseOffsetUs = orc2015YearBaseOffsetUs(writeTzId);
    for (int i = 0; i < microseconds.length; ++i) {
      long adjustedUs = applyOrcBaseOffsetOnCPU(microseconds[i], writer2015YearBaseOffsetUs);
      // Floor-divide µs to ms (and floor-mod for the sub-ms remainder) so reconstruction
      // round-trips for negative timestamps with a non-zero sub-millisecond component. Truncation
      // toward zero would round such an input up by one ms; at a DST gap transition that lands on
      // the post-transition offset, producing a 1-hour off-by-one. Must match the GPU kernel's
      // floor-divide in convert_timestamp_between_timezones.
      long millis = Math.floorDiv(adjustedUs, microsPerMillis);
      long writerOffset = writeTz.getOffset(millis);
      long readerOffset = readerTz.getOffset(millis);
      long adjustedMillis = millis + writerOffset - readerOffset;
      long adjustedReader = readerTz.getOffset(adjustedMillis);
      long finalDiffs = writerOffset - adjustedReader;
      results[i] =
          (millis + finalDiffs) * microsPerMillis + Math.floorMod(adjustedUs, microsPerMillis);
    }
    return ColumnVector.timestampMicroSecondsFromLongs(results);
  }

  private static long convertOrcTimezonesOnCPU(
      long microseconds, String writerTzId, String readerTzId) {
    TimeZone writerTz = getTimeZoneForOrc(writerTzId);
    TimeZone readerTz = getTimeZoneForOrc(readerTzId);
    long adjustedUs = applyOrcBaseOffsetOnCPU(
        microseconds, orc2015YearBaseOffsetUs(writerTzId));
    long millis = Math.floorDiv(adjustedUs, microsPerMillis);
    long writerOffset = writerTz.getOffset(millis);
    long readerOffset = readerTz.getOffset(millis);
    long adjustedMillis = millis + writerOffset - readerOffset;
    long adjustedReaderOffset = readerTz.getOffset(adjustedMillis);
    return adjustedUs + (writerOffset - adjustedReaderOffset) * microsPerMillis;
  }

  private static long convertJavaTimeLocalToUtc(long localMicros, String readerTzId) {
    long epochSeconds = Math.floorDiv(localMicros, MICROS_PER_SECOND);
    int nanos = Math.toIntExact(
        Math.floorMod(localMicros, MICROS_PER_SECOND) * TimeUnit.MICROSECONDS.toNanos(1));
    long utcSeconds = LocalDateTime.ofEpochSecond(epochSeconds, nanos, ZoneOffset.UTC)
        .atZone(GpuTimeZoneDB.getZoneId(readerTzId))
        .toEpochSecond();
    return utcSeconds * MICROS_PER_SECOND + Math.floorMod(localMicros, MICROS_PER_SECOND);
  }

  private static long convertPhysicalOrcTimestampToSparkOnCPU(
      long microseconds, String writerTzId, String readerTzId) {
    long orcInstant = convertOrcTimezonesOnCPU(microseconds, writerTzId, readerTzId);
    TimeZone readerTz = getTimeZoneForOrc(readerTzId);
    long readerLocal = orcInstant
        + readerTz.getOffset(Math.floorDiv(orcInstant, microsPerMillis)) * microsPerMillis;
    return convertJavaTimeLocalToUtc(readerLocal, readerTzId);
  }

  private static ColumnVector convertOrcFromUtcOnCPU(
      Long[] microseconds,
      String readerTzId) {
    Long[] results = new Long[microseconds.length];
    TimeZone readerTz = getTimeZoneForOrc(readerTzId);
    for (int i = 0; i < microseconds.length; ++i) {
      Long valueUs = microseconds[i];
      if (valueUs != null) {
        long valueMillis = Math.floorDiv(valueUs, microsPerMillis);
        int offsetMillis = readerTz.getOffset(valueMillis - readerTz.getRawOffset());
        results[i] = (valueMillis - offsetMillis) * microsPerMillis
            + Math.floorMod(valueUs, microsPerMillis);
      }
    }
    return ColumnVector.timestampMicroSecondsFromBoxedLongs(results);
  }

  private static Long[] getOrcFromUtcBoundaryMicros(String readerTzId) {
    long minSupportedUs = LocalDateTime.of(1, 1, 1, 0, 0)
        .toEpochSecond(ZoneOffset.UTC) * MICROS_PER_SECOND;
    long maxSupportedUs = LocalDateTime.of(9999, 12, 31, 23, 59, 59)
        .toEpochSecond(ZoneOffset.UTC) * MICROS_PER_SECOND + 999_999L;
    List<Long> values = new ArrayList<>(Arrays.asList(
        null,
        minSupportedUs,
        minSupportedUs + 1,
        -3_649_379_812_521_628L,
        -2_957_649_381_472_612L,
        -1_501L,
        -1_001L,
        -999L,
        -1L,
        0L,
        1L,
        999L,
        1_001L,
        514_952_012L,
        maxSupportedUs - 1,
        maxSupportedUs));

    OrcTimezoneInfo readerInfo = OrcTimezoneInfo.get(readerTzId);
    if (readerInfo.transitions != null) {
      for (long transitionMillis : readerInfo.transitions) {
        long localTransitionUs =
            TimeUnit.MILLISECONDS.toMicros(transitionMillis + readerInfo.rawOffset);
        values.add(localTransitionUs - 1);
        values.add(localTransitionUs);
        values.add(localTransitionUs + 1);
      }
    }

    for (ZoneOffsetTransitionRule rule :
        GpuTimeZoneDB.getZoneId(readerTzId).getRules().getTransitionRules()) {
      long transitionMillis = rule.createTransition(2099).getInstant().toEpochMilli();
      long localTransitionUs =
          TimeUnit.MILLISECONDS.toMicros(transitionMillis + readerInfo.rawOffset);
      values.add(localTransitionUs - 1);
      values.add(localTransitionUs);
      values.add(localTransitionUs + 1);
    }
    return values.toArray(new Long[0]);
  }

  @Test
  void testIsSupportedTimeZone() {
    // Named zones with ZoneRules.
    assertTrue(GpuTimeZoneDB.isSupportedTimeZone("UTC"));
    assertTrue(GpuTimeZoneDB.isSupportedTimeZone("Asia/Shanghai"));

    // Unknown id.
    assertFalse(GpuTimeZoneDB.isSupportedTimeZone("Invalid/Zone"));

    // Offset-style ids: "+05:30" must be accepted; malformed offsets must be
    // rejected even when the parser throws DateTimeException rather than the
    // narrower ZoneRulesException. This is the regression the widened catch in
    // isSupportedTimeZone guards against.
    assertTrue(GpuTimeZoneDB.isSupportedTimeZone("+05:30"));
    assertFalse(GpuTimeZoneDB.isSupportedTimeZone("+25:00"));
  }

  @Test
  void testConvertOrcTimezonesRejectsInvalidId() {
    // Invalid timezone IDs must surface an exception rather than silently
    // falling back to GMT. We assert the broad RuntimeException type so this
    // stays a regression guard even if the exact wrapping is refactored later.
    GpuTimeZoneDB.cacheDatabase();
    try (ColumnVector input =
        ColumnVector.timestampMicroSecondsFromLongs(new long[] {0L})) {
      assertThrows(RuntimeException.class,
          () -> GpuTimeZoneDB.convertOrcTimezones(input, "Invalid/Zone", "UTC"));
    }
  }

  @Test
  void testConvertOrcTimezonesPreservesEmptyAndNulls() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    try (ColumnVector input =
            ColumnVector.timestampMicroSecondsFromBoxedLongs(new Long[] {});
        ColumnVector actual = GpuTimeZoneDB.convertOrcTimezones(input, "UTC", "UTC")) {
      assertColumnsAreEqual(input, actual);
    }

    try (ColumnVector input =
            ColumnVector.timestampMicroSecondsFromBoxedLongs(null, 0L, null);
        ColumnVector actual = GpuTimeZoneDB.convertOrcTimezones(input, "UTC", "UTC")) {
      assertColumnsAreEqual(input, actual);
    }
  }

  @Test
  void testConvertOrcTimezonesCorrectsIgnoredWriterTimezoneEpochBorrow() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    try (ColumnVector input =
            ColumnVector.timestampMicroSecondsFromLongs(new long[] {21_087_883_873L});
        ColumnVector expected =
            ColumnVector.timestampMicroSecondsFromLongs(new long[] {-7_713_116_127L});
        ColumnVector actual =
            GpuTimeZoneDB.convertOrcTimezones(input, "Asia/Shanghai", "Asia/Shanghai")) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testConvertOrcTimezonesFixedOffsetIds() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    long[] microseconds = {0L, -1L, 1L, -2_957_649_381_472_612L};
    String[][] cases = {
        {"UTC", "+05:30"},
        {"+05:30", "UTC"},
        {"UTC", "EST"},
        {"EST", "UTC"}
    };

    for (String[] timezones : cases) {
      try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
          ColumnVector expected =
              convertOrcTimezonesOnCPU(microseconds, timezones[0], timezones[1]);
          ColumnVector actual =
              GpuTimeZoneDB.convertOrcTimezones(input, timezones[0], timezones[1])) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  @Test
  void testConvertOrcFromUtcAllTimezones() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    List<String> timezones = Arrays.asList(
        "UTC",
        "America/New_York",
        "America/Vancouver",
        "America/Los_Angeles",
        "Europe/Paris",
        "Asia/Shanghai",
        "Australia/Sydney",
        "US/Pacific",
        "PST",
        "EST",
        "+05:30");

    for (String readerTzId : timezones) {
      Long[] values = getOrcFromUtcBoundaryMicros(readerTzId);
      Long[] padded = new Long[values.length + 2];
      padded[0] = 123L;
      System.arraycopy(values, 0, padded, 1, values.length);
      padded[padded.length - 1] = 456L;

      try (ColumnVector full = ColumnVector.timestampMicroSecondsFromBoxedLongs(padded);
          ColumnVector input = full.subVector(1, values.length + 1);
          ColumnVector expected = convertOrcFromUtcOnCPU(values, readerTzId);
          GpuTimeZoneDB.OrcTimezoneContext context =
              GpuTimeZoneDB.buildOrcTimezoneContext("UTC", readerTzId);
          ColumnVector fromContext = GpuTimeZoneDB.convertOrcFromUtc(input, context);
          ColumnVector fromTimezone = GpuTimeZoneDB.convertOrcFromUtc(input, readerTzId)) {
        assertColumnsAreEqual(expected, fromContext);
        assertColumnsAreEqual(expected, fromTimezone);
      }
    }

    try (ColumnVector empty =
            ColumnVector.timestampMicroSecondsFromBoxedLongs(new Long[] {});
        ColumnVector actual = GpuTimeZoneDB.convertOrcFromUtc(empty, "UTC")) {
      assertColumnsAreEqual(empty, actual);
    }
  }

  @Test
  void testOrcTimezoneContextConversionFailures() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(0L)) {
      GpuTimeZoneDB.OrcTimezoneContext closed =
          GpuTimeZoneDB.buildOrcTimezoneContext("UTC", "UTC");
      closed.close();
      assertThrows(IllegalStateException.class,
          () -> GpuTimeZoneDB.convertOrcTimezones(input, closed));
      assertThrows(IllegalStateException.class,
          () -> GpuTimeZoneDB.convertOrcFromUtc(input, closed));
      assertThrows(IllegalStateException.class,
          () -> GpuTimeZoneDB.convertOrcTimestampToSpark(input, closed));
      assertThrows(IllegalStateException.class,
          () -> GpuTimeZoneDB.convertOrcIntegerTimestampToSpark(input, closed));
    }

    try (ColumnVector input = ColumnVector.timestampSecondsFromLongs(0L);
        GpuTimeZoneDB.OrcTimezoneContext context =
            GpuTimeZoneDB.buildOrcTimezoneContext("UTC", "UTC")) {
      assertThrows(CudfException.class,
          () -> GpuTimeZoneDB.convertOrcTimezones(input, context));
      assertThrows(CudfException.class,
          () -> GpuTimeZoneDB.convertOrcFromUtc(input, context));
      assertThrows(CudfException.class,
          () -> GpuTimeZoneDB.convertOrcTimestampToSpark(input, context));
      assertThrows(CudfException.class,
          () -> GpuTimeZoneDB.convertOrcIntegerTimestampToSpark(input, context));
    }
  }

  @Test
  void testConvertPhysicalOrcTimestampToSparkAcrossNewYorkLmtTransition() {
    GpuTimeZoneDB.cacheDatabase();
    String timezoneId = "America/New_York";
    long orcInstant = -2_717_655_100_076_025L;
    long decodedMicros = orcInstant + orc2015YearBaseOffsetUs(timezoneId);
    long expectedMicros = convertPhysicalOrcTimestampToSparkOnCPU(
        decodedMicros, timezoneId, timezoneId);
    assertEquals(-2_717_655_338_076_025L, expectedMicros);

    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(decodedMicros);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromLongs(expectedMicros);
        GpuTimeZoneDB.OrcTimezoneContext context =
            GpuTimeZoneDB.buildOrcTimezoneContext(timezoneId, timezoneId);
        ColumnVector actual = GpuTimeZoneDB.convertOrcTimestampToSpark(input, context)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testConvertIntegerOrcTimestampToSparkBeforeShanghaiFirstTransition() {
    GpuTimeZoneDB.cacheDatabase();
    String timezoneId = "Asia/Shanghai";
    long localMicros = -2_208_988_800L * MICROS_PER_SECOND;
    long expectedMicros = convertJavaTimeLocalToUtc(localMicros, timezoneId);

    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(localMicros);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromLongs(expectedMicros);
        GpuTimeZoneDB.OrcTimezoneContext context =
            GpuTimeZoneDB.buildOrcTimezoneContext(timezoneId, timezoneId);
        ColumnVector actual = GpuTimeZoneDB.convertOrcIntegerTimestampToSpark(input, context)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testHistoricalRebaseWithoutTimeZoneTransitions() {
    GpuTimeZoneDB.cacheDatabase();
    String timezoneId = "Pacific/Port_Moresby";
    OrcTimezoneInfo timezoneInfo = OrcTimezoneInfo.get(timezoneId);
    assertTrue(timezoneInfo.transitions == null);

    long orcInstant = Instant.parse("1870-01-01T00:00:00Z").getEpochSecond()
        * MICROS_PER_SECOND;
    long decodedMicros = orcInstant + orc2015YearBaseOffsetUs(timezoneId);
    long expectedPhysicalMicros = convertPhysicalOrcTimestampToSparkOnCPU(
        decodedMicros, timezoneId, timezoneId);
    long localMicros = Instant.parse("1870-01-01T10:00:00Z").getEpochSecond()
        * MICROS_PER_SECOND;
    long expectedIntegerMicros = convertJavaTimeLocalToUtc(localMicros, timezoneId);

    try (ColumnVector physicalInput = ColumnVector.timestampMicroSecondsFromLongs(decodedMicros);
        ColumnVector expectedPhysical =
            ColumnVector.timestampMicroSecondsFromLongs(expectedPhysicalMicros);
        ColumnVector integerInput = ColumnVector.timestampMicroSecondsFromLongs(localMicros);
        ColumnVector expectedInteger =
            ColumnVector.timestampMicroSecondsFromLongs(expectedIntegerMicros);
        GpuTimeZoneDB.OrcTimezoneContext context =
            GpuTimeZoneDB.buildOrcTimezoneContext(timezoneId, timezoneId);
        ColumnVector actualPhysical =
            GpuTimeZoneDB.convertOrcTimestampToSpark(physicalInput, context);
        ColumnVector actualInteger =
            GpuTimeZoneDB.convertOrcIntegerTimestampToSpark(integerInput, context)) {
      assertColumnsAreEqual(expectedPhysical, actualPhysical);
      assertColumnsAreEqual(expectedInteger, actualInteger);
    }
  }

  @Test
  void testHistoricalDifferenceCutoffs() {
    try (GpuTimeZoneDB.OrcTimezoneContext context =
        GpuTimeZoneDB.buildOrcTimezoneContext("UTC", "America/New_York")) {
      assertEquals(Instant.parse("1883-11-18T17:00:00Z").getEpochSecond()
              * MICROS_PER_SECOND,
          context.getReaderHistoricalDifferenceEndUtcUs());
      assertEquals(LocalDateTime.of(1883, 11, 18, 12, 3, 58)
              .toEpochSecond(ZoneOffset.UTC) * MICROS_PER_SECOND,
          context.getReaderHistoricalDifferenceEndLocalUs());
    }

    try (GpuTimeZoneDB.OrcTimezoneContext context =
        GpuTimeZoneDB.buildOrcTimezoneContext("UTC", "UTC")) {
      assertEquals(Long.MIN_VALUE, context.getReaderHistoricalDifferenceEndUtcUs());
      assertEquals(Long.MIN_VALUE, context.getReaderHistoricalDifferenceEndLocalUs());
    }
  }

  @Test
  void testReaderFirstTransitionUs() {
    String transitionTzId = "Europe/Paris";
    OrcTimezoneInfo transitionInfo = OrcTimezoneInfo.get(transitionTzId);
    assertTrue(transitionInfo.rawOffset > 0);
    try (GpuTimeZoneDB.OrcTimezoneContext context =
        GpuTimeZoneDB.buildOrcTimezoneContext("UTC", transitionTzId)) {
      assertEquals(TimeUnit.MILLISECONDS.toMicros(
              transitionInfo.transitions[0] + transitionInfo.rawOffset),
          context.getReaderFirstTransitionUs());
    }

    try (GpuTimeZoneDB.OrcTimezoneContext context =
        GpuTimeZoneDB.buildOrcTimezoneContext("UTC", "+05:30")) {
      assertEquals(Long.MIN_VALUE, context.getReaderFirstTransitionUs());
    }

    GpuTimeZoneDB.OrcTimezoneContext closed =
        GpuTimeZoneDB.buildOrcTimezoneContext("UTC", "UTC");
    closed.close();
    assertThrows(IllegalStateException.class, closed::getReaderFirstTransitionUs);
  }

  @Test
  void testConvertOrcTimezones() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    // test time range: (0001-01-01 00:00:00, 9999-12-31 23:59:59)
    long min = LocalDateTime.of(1, 1, 1, 0, 0, 0)
        .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1);
    long max = LocalDateTime.of(9999, 12, 31, 23, 59, 59)
        .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1);

    // Keep the DST matrix deterministic so failures are reproducible.
    Random rng = new Random(42L);

    List<String> timezones = Arrays.asList(
        "America/Los_Angeles",
        "America/Vancouver",
        "America/Cancun",
        "Asia/Shanghai",
        "Antarctica/DumontDUrville",
        "Etc/GMT-12",
        "CNT",
        "Australia/Sydney",
        "Asia/Tokyo");

    for (String writerTz : timezones) {
      for (String readerTz : timezones) {
        // Use 1024 as a reasonable batch size for testing timezone conversions.
        long[] microseconds = new long[1024];
        for (int i = 0; i < microseconds.length; ++i) {
          // range is years from 0001 to 9999
          microseconds[i] = min + (long) (rng.nextDouble() * (max - min));
        }

        try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
            // Convert on CPU
            ColumnVector expected = convertOrcTimezonesOnCPU(microseconds, writerTz, readerTz);
            // Convert on GPU
            ColumnVector actual = GpuTimeZoneDB.convertOrcTimezones(input, writerTz, readerTz)) {
          assertColumnsAreEqual(expected, actual);
        }
      }
    }
  }

  @Test
  void testConvertOrcTimezonesFutureDstRuleFallback() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    for (String timezoneId : Arrays.asList("America/Los_Angeles", "Australia/Sydney")) {
      long[] microseconds = getFutureDstBoundaryMicros(timezoneId);
      String[][] cases = {
          {timezoneId, "UTC"},
          {"UTC", timezoneId}
      };

      for (String[] timezones : cases) {
        try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
            ColumnVector expected =
                convertOrcTimezonesOnCPU(microseconds, timezones[0], timezones[1]);
            ColumnVector actual =
                GpuTimeZoneDB.convertOrcTimezones(input, timezones[0], timezones[1])) {
          assertColumnsAreEqual(expected, actual);
        }
      }
    }
  }

  @Test
  void testConvertOrcTimezonesAsiaGazaPairedTransitions() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    long[] microseconds = {
        LocalDateTime.of(2037, 10, 15, 0, 0)
            .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1)
    };
    String[][] cases = {
        {"Asia/Gaza", "UTC"},
        {"UTC", "Asia/Gaza"}
    };

    for (String[] timezones : cases) {
      try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
          ColumnVector expected =
              convertOrcTimezonesOnCPU(microseconds, timezones[0], timezones[1]);
          ColumnVector actual =
              GpuTimeZoneDB.convertOrcTimezones(input, timezones[0], timezones[1])) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }
}
