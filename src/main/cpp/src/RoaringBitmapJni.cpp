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

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"
#include "nvtx_ranges.hpp"
#include "roaring_bitmap.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

void close_host_buffer_noexcept(JNIEnv* env, jobject buffer) noexcept
{
  auto const buffer_class = env->GetObjectClass(buffer);
  if (buffer_class == nullptr) {
    env->ExceptionClear();
    return;
  }
  auto const close_method = env->GetMethodID(buffer_class, "close", "()V");
  if (close_method == nullptr) {
    env->ExceptionClear();
  } else {
    env->CallVoidMethod(buffer, close_method);
    if (env->ExceptionCheck()) { env->ExceptionClear(); }
  }
  env->DeleteLocalRef(buffer_class);
}

}  // namespace

extern "C" {

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_RoaringBitmap_buildAndSerialize64Native(
  JNIEnv* env,
  jclass,
  jlong positions_handle,
  jlongArray existing_addresses,
  jlongArray existing_lengths,
  jobject host_memory_allocator,
  jlong stream_handle,
  jlongArray output_cardinality)
{
  SRJ_FUNC_RANGE();
  JNI_NULL_CHECK(env, positions_handle, "positions is null", nullptr);
  JNI_NULL_CHECK(env, existing_addresses, "existing addresses are null", nullptr);
  JNI_NULL_CHECK(env, existing_lengths, "existing lengths are null", nullptr);
  JNI_NULL_CHECK(env, host_memory_allocator, "host memory allocator is null", nullptr);
  JNI_NULL_CHECK(env, output_cardinality, "output cardinality is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jlongArray addresses(env, existing_addresses);
    cudf::jni::native_jlongArray lengths(env, existing_lengths);
    cudf::jni::native_jlongArray cardinality(env, output_cardinality);
    CUDF_EXPECTS(addresses.size() == lengths.size(),
                 "existing bitmap address and length arrays must have the same size",
                 std::invalid_argument);
    CUDF_EXPECTS(cardinality.size() == 1,
                 "output cardinality array must have length 1",
                 std::invalid_argument);

    auto bitmaps = std::vector<std::span<std::byte const>>{};
    bitmaps.reserve(addresses.size());
    for (auto i = 0; i < addresses.size(); ++i) {
      CUDF_EXPECTS(addresses[i] != 0, "existing bitmap address must not be zero");
      CUDF_EXPECTS(lengths[i] > 0, "existing bitmap length must be positive");
      bitmaps.emplace_back(
        std::bit_cast<std::byte const*>(static_cast<std::uintptr_t>(addresses[i])),
        static_cast<std::size_t>(lengths[i]));
    }

    auto const positions = std::bit_cast<cudf::column_view const*>(positions_handle);
    auto const stream    = cuda::stream_ref{std::bit_cast<cudaStream_t>(stream_handle)};
    auto result = spark_rapids_jni::build_and_serialize_roaring64(*positions, bitmaps, stream);
    CUDF_EXPECTS(result.data.size() <= static_cast<std::size_t>(std::numeric_limits<jlong>::max()),
                 "serialized bitmap is too large");

    auto const output_size = static_cast<jlong>(result.data.size());
    auto output = cudf::jni::allocate_host_buffer(env, output_size, true, host_memory_allocator);
    auto const output_address =
      std::bit_cast<void*>(cudf::jni::get_host_buffer_address(env, output));
    try {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        output_address, result.data.data(), result.data.size(), cudaMemcpyDefault, stream.get()));
      stream.sync();
    } catch (...) {
      close_host_buffer_noexcept(env, output);
      throw;
    }
    cardinality[0] = static_cast<jlong>(result.cardinality);
    return output;
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"
