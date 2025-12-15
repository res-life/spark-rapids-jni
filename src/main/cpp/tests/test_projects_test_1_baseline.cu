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

#include "test_utilities.hpp"
#include "timer.hpp"

#include <cudf_test/base_fixture.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda/std/functional>
#include <thrust/tabulate.h>

#include <memory>
#include <vector>

struct ProjectsBaseline : public cudf::test::BaseFixture {};

std::unique_ptr<cudf::table> create_random_table(cudf::size_type row_count,
                                                 cudf::size_type col_count,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  using columns_vector = std::vector<std::unique_ptr<cudf::column>>;
  columns_vector cols_vec;

  for (cudf::size_type i = 0; i < col_count; i++) {
    // make a int column with 20% nulls
    rmm::device_uvector<bool> null_mask(row_count, stream, mr);
    thrust::tabulate(
      rmm::exec_policy(stream), null_mask.begin(), null_mask.end(), [] __device__(auto idx) {
        return idx % 5 != 0;
      });
    auto [bitmask, null_count] =
      cudf::detail::valid_if(null_mask.begin(), null_mask.end(), cuda::std::identity{}, stream, mr);
    auto col = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32}, row_count, std::move(bitmask), null_count, stream, mr);
    cols_vec.push_back(std::move(col));
  }

  return std::make_unique<cudf::table>(std::move(cols_vec));
}

/**
 * @brief Device function to process a single element.
 */
__device__ inline void process_element(cudf::column_device_view const& input,
                                       int row_idx,
                                       int* output)
{
  if (row_idx >= input.size()) { return; }
  if (input.is_null(row_idx)) {
    output[row_idx] = 0;
    return;
  }
  int32_t element = input.element<int>(row_idx);
  if (element % 2 == 0) {
    output[row_idx] = 123;
  } else {
    output[row_idx] = 456;
  }
}

/**
 * @brief Regular kernel to project one column.
 */
template <int block_size>
__launch_bounds__(block_size, 1) CUDF_KERNEL
  void project_baseline_kernel(cudf::column_device_view input, int* output)
{
  auto const row_idx = cudf::detail::grid_1d::global_thread_id();

  // process the element
  process_element(input, row_idx, output);
}

TEST_F(ProjectsBaseline, Ints)
{
  // prepare memory pool
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
    &cuda_mr,
    2ULL * 1024 * 1024 * 1024  // 2GB initial pool
  );
  rmm::mr::set_current_device_resource(&pool_mr);

  cudf::size_type const n_rows = 1'000'000;
  cudf::size_type const n_cols = 400;

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // create input/output table
  auto intput_int_table = create_random_table(n_rows, n_cols, stream, mr);
  auto output_int_table = create_random_table(n_rows, n_cols, stream, mr);

  stream.synchronize();

  timer t;

  // invoke column by column
  for (int i = 0; i < n_cols; i++) {
    auto& input_col        = intput_int_table->get_column(i);
    auto& output_col       = output_int_table->get_column(i);
    auto input_device_view = cudf::column_device_view::create(input_col);

    constexpr int block_size = 128;
    int num_blocks           = (n_rows + block_size - 1) / block_size;

    // call baseline kernel for each column
    project_baseline_kernel<block_size><<<num_blocks, block_size, 0, stream>>>(
      *input_device_view, output_col.mutable_view().begin<int>());
  }

  stream.synchronize();

  t.print_elapsed_micros();

  printf("ProjectsBaseline: Processed %d rows and %d columns\n", n_rows, n_cols);
}
