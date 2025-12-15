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
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda/std/functional>
#include <thrust/tabulate.h>

#include <memory>
#include <vector>

struct FusedBaseline : public cudf::test::BaseFixture {};

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
 * @brief Fused kernel to project multiple columns in a single kernel.
 * Logically treats all columns as a single column by mapping CUDA thread id to (col_idx, row_idx).
 */
template <int block_size>
__launch_bounds__(block_size, 1) CUDF_KERNEL
  void project_fused_kernel(cudf::device_span<cudf::column_device_view> input_cols,
                            cudf::device_span<int*> output_cols,
                            int num_rows)
{
  // calculate column and row index
  auto const tidx            = cudf::detail::grid_1d::global_thread_id();
  int num_threads_per_column = (num_rows + block_size - 1) / block_size * block_size;
  int col_idx                = tidx / num_threads_per_column;
  int row_idx                = tidx % num_threads_per_column;

  // bounds check
  if (col_idx >= input_cols.size()) { return; }

  cudf::column_device_view& input = input_cols[col_idx];
  int* output                     = output_cols[col_idx];

  // process the element
  process_element(input, row_idx, output);
}

TEST_F(FusedBaseline, Ints)
{
  // prepare memory pool
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
    &cuda_mr,
    2ULL * 1024 * 1024 * 1024  // 2GB initial pool
  );
  rmm::mr::set_current_device_resource(&pool_mr);

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  cudf::size_type const n_rows = 1'000'000;
  cudf::size_type const n_cols = 400;

  // create input/output table
  auto intput_int_table = create_random_table(n_rows, n_cols, stream, mr);
  auto output_int_table = create_random_table(n_rows, n_cols, stream, mr);

  stream.synchronize();

  timer t;

  std::vector<cudf::column_device_view> input_cols;
  std::vector<int*> output_cols;

  // collect input and output columns on host
  for (int i = 0; i < n_cols; i++) {
    auto& input_col        = intput_int_table->get_column(i);
    auto& output_col       = output_int_table->get_column(i);
    auto input_device_view = cudf::column_device_view::create(input_col);
    input_cols.push_back(*input_device_view);
    output_cols.push_back(output_col.mutable_view().begin<int>());
  }

  // copy input columns on host to device
  rmm::device_uvector<cudf::column_device_view> d_input_cols(input_cols.size(), stream, mr);
  cudaMemcpyAsync(d_input_cols.data(),
                  input_cols.data(),
                  input_cols.size() * sizeof(cudf::column_device_view),
                  cudaMemcpyHostToDevice,
                  cudf::get_default_stream());
  cudf::device_span<cudf::column_device_view> device_input_cols_span(d_input_cols.data(),
                                                                     d_input_cols.size());

  // copy output columns on host to device
  rmm::device_uvector<int*> d_output_cols(output_cols.size(), stream, mr);
  cudaMemcpyAsync(d_output_cols.data(),
                  output_cols.data(),
                  output_cols.size() * sizeof(int*),
                  cudaMemcpyHostToDevice,
                  cudf::get_default_stream());
  cudf::device_span<int*> device_output_cols_span(d_output_cols.data(), d_output_cols.size());

  constexpr int block_size   = 128;
  long num_blocks_per_column = (n_rows + block_size - 1) / block_size;
  long num_blocks            = num_blocks_per_column * n_cols;

  // call fused kernel
  project_fused_kernel<block_size><<<num_blocks, block_size, 0, stream>>>(
    device_input_cols_span, device_output_cols_span, n_rows);

  stream.synchronize();

  t.print_elapsed_micros();

  printf("FusedBaseline: Processed %d rows and %d columns\n", n_rows, n_cols);
}
