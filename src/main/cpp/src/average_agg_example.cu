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

#include "average_agg_example.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>

namespace spark_rapids_jni {
namespace detail {
namespace {

struct avg_fn {
  // inputs
  int const* key_offsets_ptr;
  cudf::column_device_view d_data;

  // outputs
  int* sum_output;
  bool* sum_validity;
  int* count_output;

  __device__ void operator()(cudf::size_type group_idx) const
  {
    bool is_sum_valid = false;
    int sum           = 0;
    int count         = 0;
    for (auto i = key_offsets_ptr[group_idx]; i < key_offsets_ptr[group_idx + 1]; ++i) {
      if (d_data.is_valid(i)) {
        is_sum_valid = true;
        sum += d_data.element<int>(i);
        ++count;
      }
    }

    sum_output[group_idx]   = sum;
    sum_validity[group_idx] = is_sum_valid;
    count_output[group_idx] = count;
  }
};

}  // anonymous namespace

std::unique_ptr<cudf::column> compute_avg(cudf::column_view const& key_offsets,
                                          cudf::column_view const& grouped_data,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  auto num_groups = key_offsets.size() - 1;

  auto const d_data = cudf::column_device_view::create(grouped_data, stream);

  auto sum_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_groups, cudf::mask_state::UNALLOCATED, stream, mr);
  rmm::device_uvector<bool> sum_validity(num_groups, stream);
  auto count_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_groups, cudf::mask_state::UNALLOCATED, stream, mr);

  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     thrust::make_counting_iterator(0),
                     num_groups,
                     avg_fn{key_offsets.begin<int>(),
                            *d_data,
                            sum_col->mutable_view().data<int>(),
                            sum_validity.data(),
                            count_col->mutable_view().data<int>()});

  auto [sum_null_mask, sum_null_count] = cudf::detail::valid_if(
    sum_validity.begin(), sum_validity.end(), cuda::std::identity{}, stream, mr);
  if (sum_null_count > 0) { sum_col->set_null_mask(std::move(sum_null_mask), sum_null_count); }

  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::move(sum_col));
  children.push_back(std::move(count_col));
  return cudf::make_structs_column(num_groups,
                                   std::move(children),
                                   0,                     // null count
                                   rmm::device_buffer{},  // null mask
                                   stream);
}

}  // namespace detail

std::unique_ptr<cudf::column> compute_intermediate_avg(cudf::column_view const& key_offsets,
                                                       cudf::column_view const& grouped_data,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_avg(key_offsets, grouped_data, stream, mr);
}

}  // namespace spark_rapids_jni
