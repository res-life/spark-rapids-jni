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
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
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

// allocate 400 small columns
TEST_F(ProjectsBaseline, copy_if_else)
{
  // prepare memory pool
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
    &cuda_mr,
    2ULL * 1024 * 1024 * 1024  // 2GB initial pool
  );
  rmm::mr::set_current_device_resource(&pool_mr);

  cudf::size_type const n_rows = 1'000'000;
  // 400 columns
  cudf::size_type const n_cols = 400;

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  timer t;

  // create input table
  auto intput_int_table = create_random_table(n_rows, n_cols, stream, mr);

  stream.synchronize();

  printf("allocate 400 small columns time: \n");
  t.print_elapsed_micros();
  t.reset();

  // invoke column by column
  for (int i = 0; i < n_cols; i++) {
    auto& input_col = intput_int_table->get_column(i);

    // make bools from null mask
    auto bools  = cudf::is_valid(input_col.view(), stream, mr);
    auto scalar = cudf::make_fixed_width_scalar(0, stream, mr);

    // copy_if_else
    cudf::copy_if_else(input_col.view(), *scalar, bools->view(), stream, mr);
  }

  stream.synchronize();

  printf("compute time: \n");
  t.print_elapsed_micros();

  printf("ProjectsBaseline: Processed %d rows and %d columns\n", n_rows, n_cols);
}
