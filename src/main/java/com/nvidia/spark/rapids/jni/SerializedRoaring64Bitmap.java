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

/**
 * An owning host-memory representation of a portable Roaring64 bitmap and its cardinality.
 *
 * <p>Close this object to release its host buffer. The buffer returned by {@link #getBuffer()} is a
 * borrowed reference and must not be closed separately or retained after this object is closed.
 */
public final class SerializedRoaring64Bitmap implements AutoCloseable {
  private HostMemoryBuffer buffer;
  private final long cardinality;
  private final long serializedSizeInBytes;

  SerializedRoaring64Bitmap(HostMemoryBuffer buffer, long cardinality) {
    this.buffer = buffer;
    this.cardinality = cardinality;
    this.serializedSizeInBytes = buffer.getLength();
  }

  /**
   * Returns a borrowed reference to the raw portable Roaring64 payload.
   *
   * <p>The caller must not close the returned buffer or use it after this object is closed.
   */
  public HostMemoryBuffer getBuffer() {
    if (buffer == null) {
      throw new IllegalStateException("SerializedRoaring64Bitmap is closed");
    }
    return buffer;
  }

  /** Returns the number of distinct positions in the bitmap. */
  public long getCardinality() {
    return cardinality;
  }

  /** Returns the raw portable payload size in bytes. */
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
