/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import java.util.Objects;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.HostUDFWrapper;
import ai.rapids.cudf.NativeDepsLoader;

/**
 * HyperLogLogPlusPlus(HLLPP) host UDF aggregation utils
 */
public class HyperLogLogPlusPlusHostUDF extends HostUDFWrapper {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public HyperLogLogPlusPlusHostUDF(AggregationType type, int precision) {
    super(createHLLPPHostUDF(type, precision));
    this.type = type;
    this.precision = precision;
  }

  @Override
  public int hashCode() {
    return Objects.hash(this.getClass().getName(), type, precision);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    HyperLogLogPlusPlusHostUDF other = (HyperLogLogPlusPlusHostUDF) o;
    return type == other.type && precision == other.precision;
  }

  /**
   * HyperLogLogPlusPlus(HLLPP) aggregation types
   */
  public enum AggregationType {

    /**
     * Compute hash codes for the input, generate HyperLogLogPlusPlus(HLLPP)
     * sketches from hash codes, and merge all the sketches into one sketch, output
     * is a struct scalar with multiple long values.
     */
    Reduction(0),

    /**
     * Merge all HyperLogLogPlusPlus(HLLPP) sketches in the input column into one
     * sketch. Input is a struct column with multiple long columns which is
     * consistent with Spark. Output is a struct scalar with multiple long values.
     */
    ReductionMERGE(1),

    /**
     * Compute hash codes for the input, generate HyperLogLogPlusPlus(HLLPP)
     * sketches from hash codes, and merge the sketches in the same group. Output is
     * a struct column with multiple long columns which is consistent with Spark.
     */
    GroupBy(2),

    /**
     * Merge HyperLogLogPlusPlus(HLLPP) sketches in the same group.
     * Input is a struct column with multiple long columns which is consistent with
     * Spark.
     */
    GroupByMerge(3);

    final int nativeId;

    AggregationType(int nativeId) {
      this.nativeId = nativeId;
    }
  }

  /**
   * Create a HyperLogLogPlusPlus(HLLPP) host UDF
   */
  private static long createHLLPPHostUDF(AggregationType type, int precision) {
    return createHLLPPHostUDF(type.nativeId, precision);
  }

  /**
   * Compute the approximate count distinct value from sketch values.
   *
   * The input is sketch values, must be given in the format:
   * `Struct<INT64, INT64,...>`,
   * The value of num_registers_per_sketch = 2^precision
   * The children num of this Struct is: num_registers_per_sketch / 10 + 1,
   * Here 10 means a INT64 contains 10 register values,
   * each register value is 6 bits.
   * Register value is the number of leading zero bits in xxhash64 hash code.
   * xxhash64 hash code is 64 bits, Register value is 6 bits,
   * 6 bits is enough to hold the max value 64.
   *
   * @param input     The sketch column which constains Struct<INT64, INT64, ...>
   *                  values.
   * @param precision The num of bits for HLLPP register addressing.
   * @return A INT64 column with each value indicates the approximate count
   *         distinct value.
   */
  public static ColumnVector estimateDistinctValueFromSketches(ColumnView input, int precision) {
    return new ColumnVector(estimateDistinctValueFromSketches(input.getNativeView(), precision));
  }

  private static native long createHLLPPHostUDF(int type, int precision);

  private static native long estimateDistinctValueFromSketches(long inputHandle, int precision);

  private AggregationType type;
  private int precision;

  /**
   * TODO: move this to cuDF HostUDFWrapper
   */
  @Override
  public void close() throws Exception {
    close(udfNativeHandle);
  }

  /**
   * TODO: move this to cuDF HostUDFWrapper
   */
  static native void close(long ptr);
}