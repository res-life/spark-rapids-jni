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

#include <benchmarks/common/generate_input.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

/**
 * @brief Regular kernel to project one column
 */
template <int block_size>
__launch_bounds__(block_size, 1) CUDF_KERNEL
  void project_fused_kernel(cudf::device_span<cudf::column_device_view> input_cols,
                            cudf::device_span<int*> output_cols,
                            int num_rows)
{
  auto const tidx            = cudf::detail::grid_1d::global_thread_id();
  int num_threads_per_column = (num_rows + block_size - 1) / block_size * block_size;

  int col_idx = tidx / num_threads_per_column;
  int row_idx = tidx % num_threads_per_column;

  if (row_idx >= num_rows) { return; }

  cudf::column_device_view& input = input_cols[col_idx];
  int* output                     = output_cols[col_idx];

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

void projects_fused(nvbench::state& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.get_int64("num_rows")};
  cudf::size_type const n_cols{(cudf::size_type)state.get_int64("num_cols")};

  // column types
  std::vector<cudf::type_id> const col_types(n_cols, cudf::type_id::INT32);

  // input table
  auto const intput_int_table = create_random_table(col_types, row_count{n_rows});

  // output table
  auto output_int_table = create_random_table(col_types, row_count{n_rows});

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
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
    rmm::device_uvector<cudf::column_device_view> d_input_cols(input_cols.size(),
                                                               cudf::get_default_stream());
    cudaMemcpyAsync(d_input_cols.data(),
                    input_cols.data(),
                    input_cols.size() * sizeof(cudf::column_device_view),
                    cudaMemcpyHostToDevice,
                    cudf::get_default_stream());
    cudf::device_span<cudf::column_device_view> device_input_cols_span(d_input_cols.data(),
                                                                       d_input_cols.size());

    // copy output columns on host to device
    rmm::device_uvector<int*> d_output_cols(output_cols.size(), cudf::get_default_stream());
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
    project_fused_kernel<block_size><<<num_blocks, block_size, 0, cudf::get_default_stream()>>>(
      device_input_cols_span, device_output_cols_span, n_rows);
  });
}

NVBENCH_BENCH(projects_fused)
  .set_name("projects_fused")
  .add_int64_axis("num_rows", {1024 * 1024})
  .add_int64_axis("num_cols", {400});
