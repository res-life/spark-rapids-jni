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

import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.DefaultHostMemoryAllocator;
import ai.rapids.cudf.HostMemoryAllocator;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.NativeDepsLoader;

/** GPU primitives for portable 64-bit Roaring bitmaps used by Iceberg deletion vectors. */
public final class Roaring64Bitmap {
  /** Maximum row position supported by Iceberg deletion vectors. */
  public static final long MAX_POSITION = 9223372030412324864L;

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private Roaring64Bitmap() {}

  /**
   * Builds a portable Roaring64 bitmap from {@code positions} using the default host allocator and
   * CUDA stream.
   *
   * @param positions non-nullable INT64 row positions in the range
   *                  {@code [0, MAX_POSITION]}; values may be unsorted, duplicated, or empty
   * @return a serialized bitmap that owns its host buffer and must be closed
   * @throws IllegalArgumentException if {@code positions} is null, has the wrong type, or has nulls
   * @throws CudfException if a position is out of range or native processing fails
   */
  public static SerializedRoaring64Bitmap buildAndSerialize64(ColumnView positions) {
    return buildAndSerialize64(positions, new HostMemoryBuffer[0],
        DefaultHostMemoryAllocator.get(), Cuda.DEFAULT_STREAM);
  }

  /**
   * Builds a portable Roaring64 bitmap containing the distinct union of new positions and all
   * existing raw portable Roaring64 payloads, using the default CUDA stream.
   *
   * @param positions non-nullable INT64 row positions in the range
   *                  {@code [0, MAX_POSITION]}; values may be unsorted, duplicated, or empty
   * @param existingPortableBitmaps zero or more raw portable Roaring64 payloads; the array and its
   *                                elements must not be null
   * @param allocator allocator for the returned host buffer, invoked with pinned memory preferred
   * @return a raw portable Roaring64 payload and its cardinality
   * @throws IllegalArgumentException if an argument is null, {@code positions} has the wrong type,
   *                                  or {@code positions} has nulls
   * @throws CudfException if a position is out of range, an existing payload is malformed, or
   *                       native processing fails
   */
  public static SerializedRoaring64Bitmap buildAndSerialize64(
      ColumnView positions,
      HostMemoryBuffer[] existingPortableBitmaps,
      HostMemoryAllocator allocator) {
    return buildAndSerialize64(
        positions, existingPortableBitmaps, allocator, Cuda.DEFAULT_STREAM);
  }

  /**
   * Builds a portable Roaring64 bitmap on the supplied CUDA stream.
   *
   * <p>The result contains {@code distinct(positions UNION all existingPortableBitmaps)}. Each
   * existing buffer must contain only a little-endian portable Roaring64 payload, without the
   * Iceberg length, magic number, or CRC envelope. This method does not take ownership of any input
   * buffer, and all inputs must remain valid until it returns.
   *
   * <p>This method is synchronous: it returns only after native GPU work and the final
   * device-to-host copy on {@code stream} have completed. The returned object owns the output host
   * buffer and must be closed by the caller.
   *
   * @param positions non-nullable INT64 row positions in the range
   *                  {@code [0, MAX_POSITION]}; values may be unsorted, duplicated, or empty
   * @param existingPortableBitmaps zero or more raw portable Roaring64 payloads; the array and its
   *                                elements must not be null
   * @param allocator allocator for the returned host buffer, invoked with pinned memory preferred
   * @param stream CUDA stream used for all native GPU work
   * @return a raw portable Roaring64 payload and its cardinality
   * @throws IllegalArgumentException if an argument is null, {@code positions} has the wrong type,
   *                                  or {@code positions} has nulls
   * @throws CudfException if a position is out of range, an existing payload is malformed, or
   *                       native processing fails
   */
  public static SerializedRoaring64Bitmap buildAndSerialize64(
      ColumnView positions,
      HostMemoryBuffer[] existingPortableBitmaps,
      HostMemoryAllocator allocator,
      Cuda.Stream stream) {
    if (positions == null) {
      throw new IllegalArgumentException("positions must not be null");
    }
    if (!DType.INT64.equals(positions.getType())) {
      throw new IllegalArgumentException("positions must have INT64 type");
    }
    if (positions.getNullCount() != 0) {
      throw new IllegalArgumentException("positions must not contain nulls");
    }
    if (existingPortableBitmaps == null) {
      throw new IllegalArgumentException("existingPortableBitmaps must not be null");
    }
    if (allocator == null) {
      throw new IllegalArgumentException("allocator must not be null");
    }
    if (stream == null) {
      throw new IllegalArgumentException("stream must not be null");
    }

    long[] addresses = new long[existingPortableBitmaps.length];
    long[] lengths = new long[existingPortableBitmaps.length];
    for (int i = 0; i < existingPortableBitmaps.length; i++) {
      HostMemoryBuffer bitmap = existingPortableBitmaps[i];
      if (bitmap == null) {
        throw new IllegalArgumentException("existingPortableBitmaps must not contain null");
      }
      addresses[i] = bitmap.getAddress();
      lengths[i] = bitmap.getLength();
    }

    long[] cardinality = new long[1];
    HostMemoryBuffer buffer = buildAndSerialize64Native(positions.getNativeView(), addresses,
        lengths, allocator, stream.getStream(), cardinality);
    return new SerializedRoaring64Bitmap(buffer, cardinality[0]);
  }

  private static native HostMemoryBuffer buildAndSerialize64Native(
      long positions,
      long[] existingAddresses,
      long[] existingLengths,
      HostMemoryAllocator allocator,
      long stream,
      long[] cardinality) throws CudfException;
}
