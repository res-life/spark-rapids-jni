/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/filling.hpp>

#include <nvbench/nvbench.cuh>
#include <cast_string.hpp>

static void bench_random_parse_uri(nvbench::state& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.get_int64("num_rows")};

  auto const table = create_random_table({cudf::type_id::INT64}, row_count{n_rows});

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto out =
      spark_rapids_jni::long_to_binary_string2(table->get_column(0).view());
  });

  state.add_buffer_size(n_rows, "trc", "Total Rows");
}

NVBENCH_BENCH(bench_random_parse_uri)
  .set_name("long-to-binary")
  .add_int64_axis("num_rows", {1000000, 10000000});
