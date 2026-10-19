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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "range_sequence.hpp"

#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/filling.hpp>
#include <cudf/lists/filling.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <nvbench/nvbench.cuh>

#include <memory>
#include <numeric>
#include <vector>

namespace {
template <typename T>
std::unique_ptr<cudf::column> upload(std::vector<T> const& values,
                                     cudf::type_id type,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  auto output = cudf::make_fixed_width_column(cudf::data_type{type},
                                              static_cast<cudf::size_type>(values.size()),
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              mr);
  cudf::detail::cuda_memcpy(cudf::device_span<T>{output->mutable_view().data<T>(), values.size()},
                            cudf::host_span<T const>{values},
                            stream);
  return output;
}

std::unique_ptr<cudf::column> list_sequences(std::vector<int64_t> const& starts,
                                             std::vector<cudf::size_type> const& sizes,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  auto const start_column = upload(starts, cudf::type_id::INT64, stream, mr);
  auto const size_column  = upload(sizes, cudf::type_id::INT32, stream, mr);
  auto lists = cudf::lists::sequences(start_column->view(), size_column->view(), stream, mr);
  // Transfer the data child without copying so this baseline retains its single-output footprint.
  auto contents = lists->release();
  return std::move(contents.children[cudf::lists_column_view::child_column_index]);
}

std::unique_ptr<cudf::column> concatenate_sequences(std::vector<int64_t> const& starts,
                                                    std::vector<cudf::size_type> const& sizes,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> sequences;
  std::vector<cudf::column_view> views;
  for (std::size_t i = 0; i < starts.size(); ++i) {
    cudf::numeric_scalar<int64_t> start{starts[i], true, stream, mr};
    sequences.push_back(cudf::sequence(sizes[i], start, stream, mr));
    views.push_back(sequences.back()->view());
  }
  if (sequences.size() == 1) { return std::move(sequences.front()); }
  return cudf::concatenate(views, stream, mr);
}

std::unique_ptr<cudf::column> host_sequence(std::vector<int64_t> const& starts,
                                            std::vector<cudf::size_type> const& sizes,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  auto const num_rows = std::accumulate(sizes.begin(), sizes.end(), cudf::size_type{0});
  std::vector<int64_t> values(num_rows);
  cudf::size_type output_index = 0;
  for (std::size_t i = 0; i < starts.size(); ++i) {
    std::iota(values.begin() + output_index, values.begin() + output_index + sizes[i], starts[i]);
    output_index += sizes[i];
  }
  auto output = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  // The pageable host vector must remain alive until its upload completes, including on CUDA 13.
  cudf::detail::cuda_memcpy(
    cudf::device_span<int64_t>{output->mutable_view().data<int64_t>(), values.size()},
    cudf::host_span<int64_t const>{values},
    stream);
  return output;
}

void range_sequence(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("rows"));
  auto const num_ranges = static_cast<cudf::size_type>(state.get_int64("ranges"));
  auto const method     = state.get_string("method");
  std::vector<int64_t> starts(num_ranges);
  std::vector<cudf::size_type> sizes(num_ranges, num_rows / num_ranges);
  sizes.back() += num_rows % num_ranges;
  for (int i = 0; i < num_ranges; ++i) {
    starts[i] = int64_t{i} * (num_rows + 100L);
  }
  auto const stream = cudf::get_default_stream();
  // Use pooled allocations for every method, as in the Spark GPU execution path.
  rmm::mr::cuda_async_memory_resource pool{512 * 1'024 * 1'024};
  cudf::test::scoped_current_device_resource current_resource{pool};
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_element_count(num_rows);
  state.add_global_memory_writes<int64_t>(num_rows);
  // Host metadata preparation/upload is part of each measured invocation for every method.
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::unique_ptr<cudf::column> output;
    if (method == "kernel") {
      output = spark_rapids_jni::generate_range_sequence(starts, sizes, stream, pool);
    } else if (method == "host") {
      output = host_sequence(starts, sizes, stream, pool);
    } else if (method == "lists") {
      output = list_sequences(starts, sizes, stream, pool);
    } else {
      output = concatenate_sequences(starts, sizes, stream, pool);
    }
  });
}
}  // namespace

NVBENCH_BENCH(range_sequence)
  .set_name("Range sequence")
  .add_int64_axis("rows", {4'096, 65'536, 524'288, 2'097'152, 4'194'304})
  .add_int64_axis("ranges", {1, 2, 32, 128, 1'024})
  .add_string_axis("method", {"kernel", "host", "lists", "concatenate"});
