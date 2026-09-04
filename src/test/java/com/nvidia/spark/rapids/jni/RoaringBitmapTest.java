/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * See the License for the specific language governing permissions and limitations under
 * the License.
 */

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DefaultHostMemoryAllocator;
import ai.rapids.cudf.HostMemoryAllocator;
import ai.rapids.cudf.HostMemoryBuffer;

import org.junit.jupiter.api.Test;
import org.roaringbitmap.longlong.Roaring64NavigableMap;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RoaringBitmapTest {
  private static byte[] getBytes(SerializedBitmap bitmap) {
    long length = bitmap.getSerializedSizeInBytes();
    assertTrue(length <= Integer.MAX_VALUE);
    byte[] bytes = new byte[(int) length];
    bitmap.getBuffer().getBytes(bytes, 0, 0, length);
    return bytes;
  }

  private static Roaring64NavigableMap deserialize(byte[] bytes) throws IOException {
    Roaring64NavigableMap bitmap = new Roaring64NavigableMap();
    try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes))) {
      bitmap.deserializePortable(in);
      assertEquals(0, in.available());
    }
    return bitmap;
  }

  private static HostMemoryBuffer serializePortable(boolean optimize, long... positions)
      throws IOException {
    Roaring64NavigableMap bitmap = new Roaring64NavigableMap();
    bitmap.add(positions);
    if (optimize) {
      bitmap.runOptimize();
    }
    byte[] bytes;
    try (ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
         DataOutputStream out = new DataOutputStream(byteStream)) {
      bitmap.serializePortable(out);
      out.flush();
      bytes = byteStream.toByteArray();
    }
    HostMemoryBuffer result = HostMemoryBuffer.allocate(bytes.length);
    result.setBytes(0, bytes, 0, bytes.length);
    return result;
  }

  @Test
  void emptyProducesValidPortableBitmapAndOwnsBuffer() throws IOException {
    SerializedBitmap result;
    try (ColumnVector positions = ColumnVector.fromLongs()) {
      result = RoaringBitmap.buildAndSerialize64(positions);
    }

    assertEquals(0, result.getCardinality());
    assertEquals(8, result.getSerializedSizeInBytes());
    assertArrayEquals(new byte[8], getBytes(result));
    assertTrue(deserialize(getBytes(result)).isEmpty());
    result.close();
    assertThrows(IllegalStateException.class, result::getBuffer);
    result.close();
  }

  @Test
  void unionsNewAndExistingBitmapsDeterministicallyOnNonDefaultStream() throws IOException {
    long[] runValues = LongStream.range(1000, 7000).toArray();
    long[] bitsetValues = LongStream.range(0, 5000).map(value -> value * 2).toArray();
    long[] newValues = {
        RoaringBitmap.MAX_POSITION, 1L << 32, 7, 7, 1000, (1L << 32) + 3, 0
    };
    long[] expected = LongStream.concat(LongStream.concat(
            Arrays.stream(new long[] {0, 5, 7, 65537, 1L << 32, (1L << 32) + 2,
                (1L << 32) + 3, RoaringBitmap.MAX_POSITION}),
            Arrays.stream(runValues)), Arrays.stream(bitsetValues))
        .distinct().sorted().toArray();

    try (HostMemoryBuffer existingSparse = serializePortable(false,
             5, 7, 65537, (1L << 32) + 2);
         HostMemoryBuffer existingRun = serializePortable(true, runValues);
         HostMemoryBuffer existingBitset = serializePortable(false, bitsetValues);
         ColumnVector positions = ColumnVector.fromLongs(newValues);
         Cuda.Stream stream = new Cuda.Stream(true);
         SerializedBitmap first = RoaringBitmap.buildAndSerialize64(
             positions, new HostMemoryBuffer[] {existingSparse, existingRun, existingBitset},
             DefaultHostMemoryAllocator.get(), stream);
         SerializedBitmap second = RoaringBitmap.buildAndSerialize64(
             positions, new HostMemoryBuffer[] {existingSparse, existingRun, existingBitset},
             DefaultHostMemoryAllocator.get(), stream)) {
      assertEquals(expected.length, first.getCardinality());
      assertArrayEquals(expected, deserialize(getBytes(first)).toArray());
      assertArrayEquals(getBytes(first), getBytes(second));
    }
  }

  @Test
  void consecutiveRangeUsesCompactRunEncoding() throws IOException {
    long[] values = LongStream.range(1234, 11234).toArray();
    try (ColumnVector positions = ColumnVector.fromLongs(values);
         SerializedBitmap result = RoaringBitmap.buildAndSerialize64(positions)) {
      assertEquals(values.length, result.getCardinality());
      assertTrue(result.getSerializedSizeInBytes() < 128);
      assertArrayEquals(values, deserialize(getBytes(result)).toArray());
    }
  }

  @Test
  void acceptsJavaBitmapWithOnlyMaximumPosition() throws IOException {
    try (HostMemoryBuffer existing = serializePortable(false, RoaringBitmap.MAX_POSITION);
         ColumnVector positions = ColumnVector.fromLongs();
         SerializedBitmap result = RoaringBitmap.buildAndSerialize64(
             positions, new HostMemoryBuffer[] {existing}, DefaultHostMemoryAllocator.get())) {
      assertEquals(1, result.getCardinality());
      assertArrayEquals(new long[] {RoaringBitmap.MAX_POSITION},
          deserialize(getBytes(result)).toArray());
    }
  }

  @Test
  void choosesArrayAndBitsetEncodings() throws IOException {
    try (ColumnVector arrayPositions = ColumnVector.fromLongs(1, 100, 1000);
         SerializedBitmap arrayResult = RoaringBitmap.buildAndSerialize64(arrayPositions)) {
      assertEquals(34, arrayResult.getSerializedSizeInBytes());
      assertArrayEquals(new long[] {1, 100, 1000}, deserialize(getBytes(arrayResult)).toArray());
    }

    long[] bitsetValues = LongStream.range(0, 5000).map(value -> value * 2).toArray();
    try (ColumnVector bitsetPositions = ColumnVector.fromLongs(bitsetValues);
         SerializedBitmap bitsetResult = RoaringBitmap.buildAndSerialize64(bitsetPositions)) {
      assertEquals(8220, bitsetResult.getSerializedSizeInBytes());
      assertArrayEquals(bitsetValues, deserialize(getBytes(bitsetResult)).toArray());
    }
  }

  @Test
  void usesRequestedHostAllocator() {
    AtomicBoolean calledWithPinnedPreference = new AtomicBoolean();
    HostMemoryAllocator allocator = new HostMemoryAllocator() {
      @Override
      public HostMemoryBuffer allocate(long bytes, boolean preferPinned) {
        calledWithPinnedPreference.set(preferPinned);
        return HostMemoryBuffer.allocate(bytes, preferPinned);
      }

      @Override
      public HostMemoryBuffer allocate(long bytes) {
        return HostMemoryBuffer.allocate(bytes);
      }
    };

    try (ColumnVector positions = ColumnVector.fromLongs(1, 2, 3);
         SerializedBitmap ignored = RoaringBitmap.buildAndSerialize64(
             positions, new HostMemoryBuffer[0], allocator)) {
      assertTrue(calledWithPinnedPreference.get());
    }
  }

  @Test
  void rejectsInvalidPositionsAndColumns() {
    try (ColumnVector wrongType = ColumnVector.fromInts(1, 2, 3)) {
      assertThrows(IllegalArgumentException.class,
          () -> RoaringBitmap.buildAndSerialize64(wrongType));
    }
    try (ColumnVector withNull = ColumnVector.fromBoxedLongs(1L, null, 2L)) {
      assertThrows(IllegalArgumentException.class,
          () -> RoaringBitmap.buildAndSerialize64(withNull));
    }
    try (ColumnVector negative = ColumnVector.fromLongs(-1)) {
      assertThrows(CudfException.class, () -> RoaringBitmap.buildAndSerialize64(negative));
    }
    try (ColumnVector tooLarge = ColumnVector.fromLongs(RoaringBitmap.MAX_POSITION + 1)) {
      assertThrows(CudfException.class, () -> RoaringBitmap.buildAndSerialize64(tooLarge));
    }
  }

  @Test
  void rejectsMalformedExistingBitmap() throws IOException {
    try (ColumnVector positions = ColumnVector.fromLongs(1);
         HostMemoryBuffer truncated = HostMemoryBuffer.allocate(7)) {
      truncated.setMemory(0, 7, (byte) 0);
      assertThrows(CudfException.class, () -> RoaringBitmap.buildAndSerialize64(
          positions, new HostMemoryBuffer[] {truncated}, DefaultHostMemoryAllocator.get()));
    }

    try (ColumnVector positions = ColumnVector.fromLongs();
         HostMemoryBuffer tooLarge = serializePortable(false, RoaringBitmap.MAX_POSITION + 1)) {
      assertThrows(CudfException.class, () -> RoaringBitmap.buildAndSerialize64(
          positions, new HostMemoryBuffer[] {tooLarge}, DefaultHostMemoryAllocator.get()));
    }
  }

  @Test
  void validatesJavaArguments() {
    try (ColumnVector positions = ColumnVector.fromLongs(1)) {
      assertThrows(IllegalArgumentException.class,
          () -> RoaringBitmap.buildAndSerialize64(positions, null,
              DefaultHostMemoryAllocator.get()));
      assertThrows(IllegalArgumentException.class,
          () -> RoaringBitmap.buildAndSerialize64(positions,
              new HostMemoryBuffer[] {null}, DefaultHostMemoryAllocator.get()));
      assertThrows(IllegalArgumentException.class,
          () -> RoaringBitmap.buildAndSerialize64(positions, new HostMemoryBuffer[0], null));
    }
    assertThrows(IllegalArgumentException.class,
        () -> RoaringBitmap.buildAndSerialize64(null));
  }
}
