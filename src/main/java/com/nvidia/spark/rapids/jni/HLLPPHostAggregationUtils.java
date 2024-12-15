/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

 import ai.rapids.cudf.NativeDepsLoader;
 
 /**
  * HyperLogLogPlusPlus(HLLPP) host aggregation utils
  */
 public class HLLPPHostAggregationUtils {
   static {
     NativeDepsLoader.loadNativeDeps();
   }
 
   public enum AggregationType {
     Reduction(0),
     SegmentedReduction(1),
     GroupByAggregation(2);
 
     final int nativeId;
 
     AggregationType(int nativeId) {this.nativeId = nativeId;}
   }
 
   /**
    * Create a HLLPP host UDF
    */
   public static long createNativeHLLPPHostUDF(AggregationType type) {
     return createNativeHLLPPHostUDF(type.nativeId);
   }
 
   private static native long createNativeHLLPPHostUDF(int type);
 }
 