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
#include "range_sequence_detail.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

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

struct range_start {
  __device__ int64_t operator()(range_descriptor range) const { return range.start; }
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

CUDF_KERNEL void generate_single_range(int64_t start, cudf::size_type num_rows, int64_t* output)
{
  auto const row = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (row < num_rows) { output[row] = start + row; }
}

template <typename StartIterator, typename EndIterator>
CUDF_KERNEL void generate_ranges(StartIterator starts,
                                 EndIterator ends,
                                 cudf::size_type num_ranges,
                                 cudf::size_type num_rows,
                                 int64_t* output)
{
  auto const index = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (index >= num_rows) { return; }
  auto const row = static_cast<cudf::size_type>(index);
  // upper_bound skips repeated ends from empty ranges and preserves gaps in file positions.
  auto const range = thrust::upper_bound(thrust::seq, ends, ends + num_ranges, row) - ends;
  auto const begin = range == 0 ? 0 : ends[range - 1];
  output[row]      = starts[range] + (row - begin);
}

}  // namespace

std::unique_ptr<cudf::column> generate_range_sequence(std::span<int64_t const> starts,
                                                      std::span<cudf::size_type const> sizes,
                                                      cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  auto const num_rows = validate_ranges(starts, sizes);
  if (num_rows == 0) { return cudf::make_empty_column(cudf::type_id::INT64); }

  auto constexpr rows_per_range = detail::range_sequence_device_scan_rows_per_range;
  auto const min_ranges_for_rows =
    (static_cast<std::size_t>(num_rows) + rows_per_range - 1) / rows_per_range;
  auto const device_scan = starts.size() >= detail::range_sequence_device_scan_threshold &&
                           starts.size() >= min_ranges_for_rows;

  auto output = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const data           = output->mutable_view().data<int64_t>();
  auto constexpr block_size = 256;
  auto const num_blocks     = (int64_t{num_rows} + block_size - 1) / block_size;
  auto const num_ranges     = static_cast<cudf::size_type>(starts.size());

  if (starts.size() == 1) {
    generate_single_range<<<num_blocks, block_size, 0, stream.get()>>>(
      starts.front(), num_rows, data);
  } else if (device_scan) {
    // Allocate both buffers before queuing copies: allocation failure must not leave a pending
    // read of host inputs that the caller may immediately release.
    rmm::device_uvector<int64_t> device_starts(starts.size(), stream, mr);
    rmm::device_uvector<cudf::size_type> device_ends(sizes.size(), stream, mr);
    cudf::detail::cuda_memcpy_async(cudf::device_span<int64_t>{device_starts},
                                    cudf::host_span<int64_t const>{starts.data(), starts.size()},
                                    stream);
    cudf::detail::cuda_memcpy_async(
      cudf::device_span<cudf::size_type>{device_ends},
      cudf::host_span<cudf::size_type const>{sizes.data(), sizes.size()},
      stream);
    // The host arrays may be freed on return, including when the subsequent scan allocation fails.
    cudf::detail::sync_stream(stream);
    thrust::inclusive_scan(rmm::exec_policy_nosync(stream, mr),
                           device_ends.begin(),
                           device_ends.end(),
                           device_ends.begin());
    generate_ranges<<<num_blocks, block_size, 0, stream.get()>>>(
      device_starts.data(), device_ends.data(), num_ranges, num_rows, data);
  } else {
    std::vector<range_descriptor> ranges(starts.size());
    cudf::size_type end = 0;
    for (std::size_t i = 0; i < starts.size(); ++i) {
      end += sizes[i];
      ranges[i] = range_descriptor{starts[i], end};
    }
    // Finish uploading the small host metadata before releasing it, including on CUDA 13.
    auto device_ranges = cudf::detail::make_device_uvector(ranges, stream, mr);
    generate_ranges<<<num_blocks, block_size, 0, stream.get()>>>(
      thrust::make_transform_iterator(device_ranges.data(), range_start{}),
      thrust::make_transform_iterator(device_ranges.data(), range_end{}),
      num_ranges,
      num_rows,
      data);
  }
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  return output;
}

}  // namespace spark_rapids_jni
