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
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/filling.hpp>
#include <cudf/lists/filling.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <limits>
#include <stdexcept>
#include <utility>

namespace spark_rapids_jni {
namespace {

cudf::size_type validate_ranges(std::span<int64_t const> starts,
                                std::span<cudf::size_type const> sizes)
{
  CUDF_EXPECTS(starts.size() == sizes.size(),
               "starts and sizes must have the same length",
               std::invalid_argument);
  auto constexpr max_rows = std::numeric_limits<cudf::size_type>::max();
  // The LIST offsets column has one more row than the starts/sizes columns.
  CUDF_EXPECTS(starts.size() < max_rows, "Too many ranges", std::invalid_argument);

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

}  // namespace

std::unique_ptr<cudf::column> generate_range_sequence(std::span<int64_t const> starts,
                                                      std::span<cudf::size_type const> sizes,
                                                      cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  auto const num_rows = validate_ranges(starts, sizes);
  if (num_rows == 0) { return cudf::make_empty_column(cudf::type_id::INT64); }

  auto const num_ranges = static_cast<cudf::size_type>(starts.size());

  if (starts.size() == 1) {
    cudf::numeric_scalar<int64_t> start{starts.front(), true, stream, mr};
    return cudf::sequence(num_rows, start, stream, mr);
  }

  // Allocate both inputs before queuing copies, so allocation failures cannot leave reads of
  // host buffers pending after the caller releases them.
  auto start_column = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, num_ranges, cudf::mask_state::UNALLOCATED, stream, mr);
  auto size_column = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_ranges, cudf::mask_state::UNALLOCATED, stream, mr);
  cudf::detail::cuda_memcpy_async(
    cudf::device_span<int64_t>{start_column->mutable_view().data<int64_t>(), starts.size()},
    cudf::host_span<int64_t const>{starts.data(), starts.size()},
    stream);
  cudf::detail::cuda_memcpy_async(
    cudf::device_span<cudf::size_type>{size_column->mutable_view().data<cudf::size_type>(),
                                       sizes.size()},
    cudf::host_span<cudf::size_type const>{sizes.data(), sizes.size()},
    stream);
  // Complete host uploads before libcudf can throw, including on CUDA 13.
  cudf::detail::sync_stream(stream);

  auto lists    = cudf::lists::sequences(start_column->view(), size_column->view(), stream, mr);
  auto contents = lists->release();
  // Move the owning data child out of the LIST without allocating or copying another output.
  return std::move(contents.children[cudf::lists_column_view::child_column_index]);
}

}  // namespace spark_rapids_jni
