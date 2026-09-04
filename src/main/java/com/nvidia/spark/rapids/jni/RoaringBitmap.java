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

/** GPU primitives for portable Roaring bitmaps. */
public final class RoaringBitmap {
  /** Maximum row position supported by Iceberg deletion vectors. */
  public static final long MAX_POSITION = 9223372030412324864L;

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private RoaringBitmap() {}

  /** Builds a portable Roaring64 bitmap from {@code positions}. */
  public static SerializedBitmap buildAndSerialize64(ColumnView positions) {
    return buildAndSerialize64(positions, new HostMemoryBuffer[0],
        DefaultHostMemoryAllocator.get(), Cuda.DEFAULT_STREAM);
  }

  /**
   * Builds a portable Roaring64 bitmap containing the distinct union of new positions and all
   * existing raw portable Roaring64 payloads.
   */
  public static SerializedBitmap buildAndSerialize64(
      ColumnView positions,
      HostMemoryBuffer[] existingPortableBitmaps,
      HostMemoryAllocator allocator) {
    return buildAndSerialize64(
        positions, existingPortableBitmaps, allocator, Cuda.DEFAULT_STREAM);
  }

  /**
   * Builds a portable Roaring64 bitmap on the supplied CUDA stream.
   *
   * @param positions non-nullable INT64 row positions
   * @param existingPortableBitmaps existing raw portable Roaring64 payloads
   * @param allocator allocator for the returned host buffer
   * @param stream CUDA stream used for all native GPU work
   * @return serialized bitmap owned by the caller
   */
  public static SerializedBitmap buildAndSerialize64(
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
    return new SerializedBitmap(buffer, cardinality[0]);
  }

  private static native HostMemoryBuffer buildAndSerialize64Native(
      long positions,
      long[] existingAddresses,
      long[] existingLengths,
      HostMemoryAllocator allocator,
      long stream,
      long[] cardinality) throws CudfException;
}
