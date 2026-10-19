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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace spark_rapids_jni {
namespace {

struct range_descriptor {
  int64_t start;
  cudf::size_type end;
};

struct range_end {
  __device__ cudf::size_type operator()(range_descriptor range) const { return range.end; }
};

cudf::size_type validate_ranges(std::span<int64_t const> starts,
                                std::span<cudf::size_type const> sizes)
{
  CUDF_EXPECTS(starts.size() == sizes.size(),
               "starts and sizes must have the same length",
               std::invalid_argument);
  auto constexpr max_rows = std::numeric_limits<cudf::size_type>::max();
  CUDF_EXPECTS(starts.size() <= max_rows, "Too many ranges", std::invalid_argument);

  int64_t total_rows = 0;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    auto const size = sizes[i];
    CUDF_EXPECTS(size >= 0, "Range sizes must be nonnegative", std::invalid_argument);
    CUDF_EXPECTS(size == 0 || starts[i] <= std::numeric_limits<int64_t>::max() - (size - 1),
                 "Range sequence overflows INT64",
                 std::overflow_error);
    total_rows += size;
    CUDF_EXPECTS(
      total_rows <= max_rows, "Range sequence exceeds column size limit", std::overflow_error);
  }
  return static_cast<cudf::size_type>(total_rows);
}

template <bool single_range>
CUDF_KERNEL void generate_ranges(range_descriptor const* ranges,
                                 cudf::size_type num_ranges,
                                 int64_t single_start,
                                 cudf::size_type num_rows,
                                 int64_t* output)
{
  auto const index = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (index >= num_rows) { return; }
  auto const row = static_cast<cudf::size_type>(index);
  if constexpr (single_range) {
    output[row] = single_start + row;
  } else {
    auto const ends = thrust::make_transform_iterator(ranges, range_end{});
    // upper_bound skips repeated ends from empty ranges and preserves gaps in file positions.
    auto const range = thrust::upper_bound(thrust::seq, ends, ends + num_ranges, row) - ends;
    auto const begin = range == 0 ? 0 : ranges[range - 1].end;
    output[row]      = ranges[range].start + (row - begin);
  }
}

}  // namespace

std::unique_ptr<cudf::column> generate_range_sequence(std::span<int64_t const> starts,
                                                      std::span<cudf::size_type const> sizes,
                                                      cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  auto const num_rows = validate_ranges(starts, sizes);
  if (num_rows == 0) { return cudf::make_empty_column(cudf::type_id::INT64); }

  auto output = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const data           = output->mutable_view().data<int64_t>();
  auto constexpr block_size = 256;
  auto const num_blocks     = (int64_t{num_rows} + block_size - 1) / block_size;

  if (starts.size() == 1) {
    generate_ranges<true>
      <<<num_blocks, block_size, 0, stream.get()>>>(nullptr, 1, starts.front(), num_rows, data);
  } else {
    std::vector<range_descriptor> ranges(starts.size());
    cudf::size_type end = 0;
    for (std::size_t i = 0; i < starts.size(); ++i) {
      end += sizes[i];
      ranges[i] = range_descriptor{starts[i], end};
    }
    // Finish uploading the small host metadata before releasing it, including on CUDA 13.
    auto device_ranges = cudf::detail::make_device_uvector(ranges, stream, mr);
    generate_ranges<false><<<num_blocks, block_size, 0, stream.get()>>>(
      device_ranges.data(), static_cast<cudf::size_type>(starts.size()), 0, num_rows, data);
  }
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  return output;
}

}  // namespace spark_rapids_jni
