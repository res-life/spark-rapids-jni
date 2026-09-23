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

#include "roaring_bitmap.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <nvbench/nvbench.cuh>
#include <roaring/roaring64.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

using roaring64_ptr = std::unique_ptr<roaring64_bitmap_t, decltype(&roaring64_bitmap_free)>;

auto make_positions(std::int64_t num_rows, std::int64_t stride)
{
  auto values = std::vector<std::int64_t>(num_rows);
  for (auto index = std::int64_t{0}; index < num_rows; ++index) {
    values[index] = index * stride;
  }
  return cudf::test::fixed_width_column_wrapper<std::int64_t>(values.begin(), values.end())
    .release();
}

void gpu_build_and_serialize(nvbench::state& state)
{
  auto const num_rows = state.get_int64("num_rows");
  auto const stride   = state.get_int64("stride");
  auto positions      = make_positions(num_rows, stride);
  auto const stream   = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));

  state.exec(
    nvbench::exec_tag::timer | nvbench::exec_tag::sync, [&](nvbench::launch&, auto& timer) {
      timer.start();
      auto result     = spark_rapids_jni::build_and_serialize_roaring64(*positions, {}, stream);
      auto serialized = std::vector<std::byte>(result.data.size());
      CUDF_CUDA_TRY(cudaMemcpyAsync(serialized.data(),
                                    result.data.data(),
                                    result.data.size(),
                                    cudaMemcpyDefault,
                                    stream.get()));
      stream.sync();
      timer.stop();
    });
  state.add_element_count(static_cast<std::size_t>(num_rows));
}

void cpu_copy_add_and_serialize(nvbench::state& state)
{
  auto const num_rows = state.get_int64("num_rows");
  auto const stride   = state.get_int64("stride");
  auto positions      = make_positions(num_rows, stride);
  auto host_positions = std::vector<std::int64_t>(num_rows);
  auto const stream   = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));

  state.exec(
    nvbench::exec_tag::timer | nvbench::exec_tag::sync, [&](nvbench::launch&, auto& timer) {
      timer.start();
      CUDF_CUDA_TRY(cudaMemcpyAsync(host_positions.data(),
                                    positions->view().data<std::int64_t>(),
                                    num_rows * sizeof(std::int64_t),
                                    cudaMemcpyDefault,
                                    stream.get()));
      stream.sync();

      auto bitmap  = roaring64_ptr{roaring64_bitmap_create(), &roaring64_bitmap_free};
      auto context = roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};
      for (auto const position : host_positions) {
        roaring64_bitmap_add_bulk(bitmap.get(), &context, static_cast<std::uint64_t>(position));
      }
      roaring64_bitmap_run_optimize(bitmap.get());
      auto serialized = std::vector<char>(roaring64_bitmap_portable_size_in_bytes(bitmap.get()));
      roaring64_bitmap_portable_serialize(bitmap.get(), serialized.data());
      timer.stop();
    });
  state.add_element_count(static_cast<std::size_t>(num_rows));
}

}  // namespace

NVBENCH_BENCH(gpu_build_and_serialize)
  .set_name("Roaring64 GPU column build, serialize, and copy to host")
  .add_int64_axis("num_rows", {10'000, 1'000'000, 10'000'000})
  .add_int64_axis("stride", {1, 17});

NVBENCH_BENCH(cpu_copy_add_and_serialize)
  .set_name("Roaring64 CPU copy, per-row add, and serialize")
  .add_int64_axis("num_rows", {10'000, 1'000'000, 10'000'000})
  .add_int64_axis("stride", {1, 17});
