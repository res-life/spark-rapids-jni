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

import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.GatherMap;
import ai.rapids.cudf.HashJoin;
import ai.rapids.cudf.Table;

/**
 * A reusable hash join that builds the hash table once and can be probed multiple times.
 * <p>
 * This class wraps cuDF's {@link HashJoin} to enable efficient join operations when the
 * build side (typically the smaller table) remains constant while the probe side changes.
 * This is particularly useful in Spark's hash join implementation where the build side
 * is broadcast and joined with multiple stream batches.
 * </p>
 * <p>
 * Performance benefit: Building the hash table is expensive (O(n) with hash computation).
 * By reusing the hash table across multiple probes, we avoid rebuilding it for each
 * stream batch, which can provide significant speedup.
 * </p>
 * <p>
 * Thread safety: This class is NOT thread-safe. Each thread should have its own instance.
 * </p>
 * <p>
 * Memory management: The hash table consumes GPU memory. The caller is responsible for
 * closing this object when done to release GPU resources.
 * </p>
 *
 * @see HashJoin
 */
public class ReusableHashJoin implements AutoCloseable {

  private final HashJoin hashJoin;
  private final boolean compareNullsEqual;
  private final long buildRowCount;
  private boolean isClosed = false;

  /**
   * Build a reusable hash table from the given build keys.
   * <p>
   * The hash table is built immediately upon construction. The build keys table
   * is copied internally, so the caller can close their reference after construction.
   * </p>
   *
   * @param buildKeys Table containing the join key columns for the build side
   * @param compareNullsEqual true if null key values should match, false otherwise
   */
  public ReusableHashJoin(Table buildKeys, boolean compareNullsEqual) {
    this.compareNullsEqual = compareNullsEqual;
    this.buildRowCount = buildKeys.getRowCount();
    // HashJoin copies the table internally, so it's safe for caller to close buildKeys after
    this.hashJoin = new HashJoin(buildKeys, compareNullsEqual);
  }

  /**
   * Probe the hash table with the given probe keys to compute inner join gather maps.
   * <p>
   * Returns gather maps that can be used to manifest the result of an inner equi-join.
   * The left gather map corresponds to the probe (stream) side, and the right gather map
   * corresponds to the build side.
   * </p>
   *
   * @param probeKeys Table containing the join key columns for the probe (stream) side
   * @return Array of two GatherMaps: [probe_map, build_map]
   */
  public GatherMap[] innerJoinProbe(Table probeKeys) {
    checkNotClosed();
    // probeKeys.innerJoinGatherMaps(HashJoin) returns [left_map, right_map]
    // where left is probe side and right is build side
    return probeKeys.innerJoinGatherMaps(hashJoin);
  }

  /**
   * Probe the hash table with the given probe keys to compute left outer join gather maps.
   * <p>
   * Returns gather maps that can be used to manifest the result of a left outer equi-join.
   * All rows from the probe (left) side are included. Unmatched probe rows have the
   * build (right) index set to the null sentinel value.
   * </p>
   *
   * @param probeKeys Table containing the join key columns for the probe (left) side
   * @return Array of two GatherMaps: [probe_map, build_map]
   */
  public GatherMap[] leftOuterJoinProbe(Table probeKeys) {
    checkNotClosed();
    return probeKeys.leftJoinGatherMaps(hashJoin);
  }

  /**
   * Probe the hash table with the given probe keys to compute full outer join gather maps.
   * <p>
   * Returns gather maps that can be used to manifest the result of a full outer equi-join.
   * All rows from both sides are included. Unmatched rows have the corresponding
   * index set to the null sentinel value.
   * </p>
   *
   * @param probeKeys Table containing the join key columns for the probe side
   * @return Array of two GatherMaps: [probe_map, build_map]
   */
  public GatherMap[] fullOuterJoinProbe(Table probeKeys) {
    checkNotClosed();
    return probeKeys.fullJoinGatherMaps(hashJoin);
  }

  /**
   * Probe the hash table to get the output row count for an inner join.
   * <p>
   * This can be used to check the join size before actually computing gather maps,
   * which is useful for memory estimation and avoiding OOM situations.
   * </p>
   *
   * @param probeKeys Table containing the join key columns for the probe side
   * @return Number of rows in the inner join result
   */
  public long innerJoinRowCount(Table probeKeys) {
    checkNotClosed();
    return probeKeys.innerJoinRowCount(hashJoin);
  }

  /**
   * Probe the hash table to get the output row count for a left outer join.
   *
   * @param probeKeys Table containing the join key columns for the probe side
   * @return Number of rows in the left outer join result
   */
  public long leftJoinRowCount(Table probeKeys) {
    checkNotClosed();
    return probeKeys.leftJoinRowCount(hashJoin);
  }

  /**
   * Probe the hash table with the given probe keys and pre-computed output size.
   * <p>
   * This is more efficient when the output size was already computed via
   * {@link #innerJoinRowCount(Table)}.
   * </p>
   * <p>
   * WARNING: Passing an outputRowCount smaller than the actual result size will
   * result in undefined behavior.
   * </p>
   *
   * @param probeKeys Table containing the join key columns for the probe side
   * @param outputRowCount Pre-computed number of output rows
   * @return Array of two GatherMaps: [probe_map, build_map]
   */
  public GatherMap[] innerJoinProbeWithSize(Table probeKeys, long outputRowCount) {
    checkNotClosed();
    return probeKeys.innerJoinGatherMaps(hashJoin, outputRowCount);
  }

  /**
   * Get the number of columns in the build keys.
   *
   * @return Number of key columns
   */
  public long getNumberOfColumns() {
    checkNotClosed();
    return hashJoin.getNumberOfColumns();
  }

  /**
   * Get the number of rows in the build table.
   *
   * @return Number of rows in the build side
   */
  public long getBuildRowCount() {
    return buildRowCount;
  }

  /**
   * Returns true if the hash table was built to match on nulls.
   *
   * @return true if nulls are compared as equal, false otherwise
   */
  public boolean getCompareNullsEqual() {
    return compareNullsEqual;
  }

  /**
   * Check if this hash join has been closed.
   *
   * @return true if closed, false otherwise
   */
  public boolean isClosed() {
    return isClosed;
  }

  private void checkNotClosed() {
    if (isClosed) {
      throw new IllegalStateException("ReusableHashJoin has been closed");
    }
  }

  @Override
  public void close() {
    if (!isClosed) {
      hashJoin.close();
      isClosed = true;
    }
  }
}

