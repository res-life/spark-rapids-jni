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
 * @brief Fused kernel to project multiple columns in a single kernel.
 */
template <int block_size>
__launch_bounds__(block_size, 1) CUDF_KERNEL
  void project_baseline_kernel(cudf::column_device_view input, int* output)
{
  auto const row_idx = cudf::detail::grid_1d::global_thread_id();
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

void projects_baseline(nvbench::state& state)
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
    // invoke column by column
    for (int i = 0; i < n_cols; i++) {
      auto& input_col        = intput_int_table->get_column(i);
      auto& output_col       = output_int_table->get_column(i);
      auto input_device_view = cudf::column_device_view::create(input_col);

      constexpr int block_size = 128;
      int num_blocks           = (n_rows + block_size - 1) / block_size;

      // call baseline kernel for each column
      project_baseline_kernel<block_size>
        <<<num_blocks, block_size, 0, cudf::get_default_stream()>>>(
          *input_device_view, output_col.mutable_view().begin<int>());
    }
  });
}

NVBENCH_BENCH(projects_baseline)
  .set_name("projects_baseline")
  .add_int64_axis("num_rows", {1024L * 1024L})
  .add_int64_axis("num_cols", {400L});
