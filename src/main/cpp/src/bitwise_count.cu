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

#include <bitwise_count.hpp>
#include <cuda/functional>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/types.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/transform.h>

#include <memory>

namespace spark_rapids_jni {

namespace { // anonymous namespace

__device__ int64_t highestOneBit(int64_t i) {
  // HD, Figure 3-1
  i |= (i >> 1);
  i |= (i >> 2);
  i |= (i >> 4);
  i |= (i >> 8);
  i |= (i >> 16);
  i |= (i >> 32);
  return i - (i >> 1); // TODO Java here is >>>
}

std::unique_ptr<cudf::column>
highestOneBit_impl(cudf::column_view const &input, rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr) {
  auto result = cudf::make_fixed_width_column(
      cudf::data_type(cudf::type_id::INT64), input.size(),
      cudf::mask_state::UNINITIALIZED, stream, mr);
  auto const d_result =
      cudf::column_device_view::create(result->view(), stream);

  thrust::transform(rmm::exec_policy_nosync(stream), input.begin<int64_t>(),
                    input.end<int64_t>(),
                    result->mutable_view().begin<int64_t>(),
                    cuda::proclaim_return_type<int64_t>(
                        [] __device__(int64_t v) { return highestOneBit(v); }));
  result->set_null_mask(cudf::detail::copy_bitmask(input, stream, mr),
                        input.null_count(), stream);
  return result;
}

} // anonymous namespace

std::unique_ptr<cudf::column> highestOneBit(cudf::column_view const &input,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr) {
  return highestOneBit_impl(input, stream, mr);
}

} // namespace spark_rapids_jni
