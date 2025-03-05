/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/filling.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/fill.h>
#include <thrust/tabulate.h>

#include <nvbench/nvbench.cuh>

void sequences(nvbench::state& state)
{
  auto const stream = cudf::get_default_stream();
  cudf::size_type const num_rows{(cudf::size_type)state.get_int64("num_rows")};
  auto starts = make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
  auto sizes  = make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
  thrust::fill(rmm::exec_policy(stream),
               starts->mutable_view().begin<int32_t>(),
               starts->mutable_view().end<int32_t>(),
               1);
  thrust::fill(rmm::exec_policy(stream),
               sizes->mutable_view().begin<int32_t>(),
               sizes->mutable_view().end<int32_t>(),
               5);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::lists::sequences(starts->view(), sizes->view());
  });
}

NVBENCH_BENCH(sequences).set_name("sequences").add_int64_axis("num_rows", {100 * 1024 * 1024});
