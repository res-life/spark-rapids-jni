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

#include "cast_string.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/functional>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <stdexcept>

namespace spark_rapids_jni {

namespace detail {
namespace {

// int64 = [0...0] 1011
// tid 0:  handle indices [0 , 1 ] 11
// tid 1:  handle indices [2 , 3 ] 01
// ......
// tid 30: handle indices [28, 29] 00
// tid 31: handle indices [30, 31] 00
CUDF_KERNEL void long_to_binary_string_kernel(cudf::column_device_view d_longs,
                                              char* d_chars,
                                              cudf::detail::input_offsetalator d_offsets,
                                              int num_threads_per_row)
{
  auto const tidx    = cudf::detail::grid_1d::global_thread_id();
  auto const row_idx = tidx / num_threads_per_row;
  if (row_idx >= d_longs.size()) { return; }
  auto const lane_idx            = tidx % num_threads_per_row;
  auto const num_bits_per_thread = 64 / num_threads_per_row;
  auto const string_offset       = d_offsets[row_idx];
  auto const str_len             = d_offsets[row_idx + 1] - string_offset;
  auto const value               = d_longs.element<int64_t>(row_idx);

  int first_bit_idx_in_lane = lane_idx * num_bits_per_thread;
  for (auto i = 0; i < num_bits_per_thread; ++i) {
    int curr_bit_idx = first_bit_idx_in_lane + i;
    if (curr_bit_idx < str_len) {
      d_chars[string_offset + (str_len - curr_bit_idx - 1)] =
        '0' + ((value & (1UL << curr_bit_idx)) >> curr_bit_idx);
    }
  }
}

template <typename SizeAndExecuteFunction>
CUDF_KERNEL void strings_children_kernel2(SizeAndExecuteFunction fn, cudf::size_type exec_size)
{
  auto tid = cudf::detail::grid_1d::global_thread_id();
  if (tid < exec_size) { fn(tid); }
}

template <typename SizeAndExecuteFunction>
auto make_strings_children2(SizeAndExecuteFunction size_and_exec_fn,
                              cudf::size_type strings_count,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  // This is called twice -- once for computing sizes and once for writing chars.
  // Reducing the number of places size_and_exec_fn is inlined speeds up compile time.
  auto for_each_fn = [strings_count, stream](SizeAndExecuteFunction& size_and_exec_fn) {
    auto constexpr block_size = 256;
    auto grid                 = cudf::detail::grid_1d{strings_count, block_size};
    strings_children_kernel2<<<grid.num_blocks, block_size, 0, stream.value()>>>(size_and_exec_fn,
                                                                                 strings_count);
  };

  // Compute the output sizes
  auto output_sizes        = rmm::device_uvector<cudf::size_type>(strings_count, stream);
  size_and_exec_fn.d_sizes = output_sizes.data();
  size_and_exec_fn.d_chars = nullptr;
  for_each_fn(size_and_exec_fn);

  // Convert the sizes to offsets
  auto [offsets_column, bytes] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);
  size_and_exec_fn.d_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());

  // Now build the chars column
  rmm::device_uvector<char> chars(bytes, stream, mr);
  cudf::experimental::prefetch::detail::prefetch("gather", chars, stream);
  size_and_exec_fn.d_chars = chars.data();

  // Execute the function fn again to fill in the chars data.
  if (bytes > 0) {
    constexpr int block_size          = 256;
    constexpr int num_threads_per_row = 32;
    auto grid =
      cudf::detail::grid_1d{static_cast<int64_t>(strings_count) * num_threads_per_row, block_size};
    long_to_binary_string_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
      size_and_exec_fn.d_longs,
      size_and_exec_fn.d_chars,
      size_and_exec_fn.d_offsets,
      num_threads_per_row);
  }

  return std::pair(std::move(offsets_column), std::move(chars));
}

struct long_to_binary_string_fn {
  cudf::column_device_view d_longs;
  cudf::size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  __device__ void long_to_binary_string(cudf::size_type idx)
  {
    auto const value = d_longs.element<int64_t>(idx);
    char* d_buffer   = d_chars + d_offsets[idx];
    for (auto i = d_sizes[idx] - 1; i >= 0; --i) {
      *d_buffer++ = '0' + ((value & (1UL << i)) >> i);
    }
  }

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_longs.is_null(idx)) {
      if (d_chars == nullptr) { d_sizes[idx] = 0; }
      return;
    }
    if (d_chars != nullptr) {
      long_to_binary_string(idx);
    } else {
      // If the value is 0, the size should be 1
      d_sizes[idx] = max(1, 64 - __clzll(d_longs.element<int64_t>(idx)));
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> long_to_binary_string2(cudf::column_view const& input,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return cudf::make_empty_column(cudf::type_id::STRING);

  CUDF_EXPECTS(input.type().id() == cudf::type_id::INT64, "Input column must be long type");

  auto const d_column = cudf::column_device_view::create(input, stream);

  auto [offsets, chars] =
    make_strings_children2(long_to_binary_string_fn{*d_column}, input.size(), stream, mr);

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input, stream, mr));
}

std::unique_ptr<cudf::column> long_to_binary_string(cudf::column_view const& input,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return cudf::make_empty_column(cudf::type_id::STRING);

  CUDF_EXPECTS(input.type().id() == cudf::type_id::INT64, "Input column must be long type");

  auto const d_column = cudf::column_device_view::create(input, stream);

  auto [offsets, chars] = cudf::strings::detail::make_strings_children(
    long_to_binary_string_fn{*d_column}, input.size(), stream, mr);

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input, stream, mr));
}

}  // namespace detail

// external API
std::unique_ptr<cudf::column> long_to_binary_string(cudf::column_view const& input,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::long_to_binary_string(input, stream, mr);
}

std::unique_ptr<cudf::column> long_to_binary_string2(cudf::column_view const& input,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::long_to_binary_string2(input, stream, mr);
}

}  // namespace spark_rapids_jni
