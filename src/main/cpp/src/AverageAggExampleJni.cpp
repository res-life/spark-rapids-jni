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

#include "average_agg_example.hpp"
#include "cudf_jni_apis.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_AverageAggExample_avg(JNIEnv* env,
                                                                               jclass,
                                                                               jlong keyOffsets,
                                                                               jlong groupedData)
{
  JNI_NULL_CHECK(env, keyOffsets, "keyOffsets is null", 0);
  JNI_NULL_CHECK(env, groupedData, "groupedData is null", 0);
  auto const& offsets = *reinterpret_cast<cudf::column_view const*>(keyOffsets);
  auto const& data    = *reinterpret_cast<cudf::column_view const*>(groupedData);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return cudf::jni::release_as_jlong(spark_rapids_jni::compute_intermediate_avg(offsets, data));
  }
  JNI_CATCH(env, 0);
}
}
