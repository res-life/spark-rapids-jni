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

import ai.rapids.cudf.HostMemoryBuffer;

/** A portable serialized bitmap and its cardinality. */
public final class SerializedBitmap implements AutoCloseable {
  private HostMemoryBuffer buffer;
  private final long cardinality;
  private final long serializedSizeInBytes;

  SerializedBitmap(HostMemoryBuffer buffer, long cardinality) {
    this.buffer = buffer;
    this.cardinality = cardinality;
    this.serializedSizeInBytes = buffer.getLength();
  }

  /** Returns the raw portable bitmap payload. */
  public HostMemoryBuffer getBuffer() {
    if (buffer == null) {
      throw new IllegalStateException("SerializedBitmap is closed");
    }
    return buffer;
  }

  /** Returns the number of distinct values in the bitmap. */
  public long getCardinality() {
    return cardinality;
  }

  /** Returns the serialized payload size. */
  public long getSerializedSizeInBytes() {
    return serializedSizeInBytes;
  }

  @Override
  public void close() {
    if (buffer != null) {
      buffer.close();
      buffer = null;
    }
  }
}
