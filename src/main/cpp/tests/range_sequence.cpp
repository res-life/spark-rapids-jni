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

#include <cudf_test/base_fixture.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
struct RangeSequenceTests : cudf::test::BaseFixture {};

auto constexpr min = std::numeric_limits<int64_t>::min();
auto constexpr max = std::numeric_limits<int64_t>::max();

void assert_sequence(std::vector<int64_t> const& starts,
                     std::vector<cudf::size_type> const& sizes,
                     cuda::stream_ref stream = cudf::get_default_stream())
{
  std::vector<int64_t> expected;
  for (std::size_t range = 0; range < starts.size(); ++range) {
    for (cudf::size_type row = 0; row < sizes[range]; ++row) {
      expected.push_back(starts[range] + row);
    }
  }
  auto const output = spark_rapids_jni::generate_range_sequence(starts, sizes, stream);
  EXPECT_EQ(output->type(), cudf::data_type{cudf::type_id::INT64});
  EXPECT_EQ(
    expected,
    cudf::detail::make_std_vector(
      cudf::device_span<int64_t const>(output->view().data<int64_t>(), output->size()), stream));
  EXPECT_FALSE(output->nullable());
}

TEST_F(RangeSequenceTests, EmptyAndSingleRange)
{
  assert_sequence({max - 2}, {3});
  assert_sequence({}, {});
  assert_sequence({0, max}, {0, 0});
}

TEST_F(RangeSequenceTests, ScanPaths)
{
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  for (auto const ranges : {std::size_t{3}, std::size_t{1'024}, std::size_t{1'025}}) {
    std::vector<int64_t> starts(ranges, -1);
    std::vector<cudf::size_type> sizes(ranges, 1);
    starts.front() = min;
    sizes.front()  = 2;
    sizes[1]       = 0;
    starts.back()  = max - 2;
    sizes.back()   = 3;
    assert_sequence(starts, sizes, stream);
  }
}

TEST_F(RangeSequenceTests, InvalidRanges)
{
  auto generate = [](std::vector<int64_t> const& starts,
                     std::vector<cudf::size_type> const& sizes) {
    static_cast<void>(spark_rapids_jni::generate_range_sequence(starts, sizes));
  };
  auto constexpr max_rows = std::numeric_limits<cudf::size_type>::max();
  EXPECT_THROW(generate({0}, {}), std::invalid_argument);
  EXPECT_THROW(generate({0}, {-1}), std::invalid_argument);
  EXPECT_THROW(generate({max}, {2}), std::overflow_error);
  EXPECT_THROW(generate({0, 0}, {max_rows, 1}), std::overflow_error);

  // Reserve uninitialized arrays: oversized input is rejected before any element is read.
  auto constexpr count = std::size_t{max_rows} + 1;
  auto const starts    = std::make_unique_for_overwrite<int64_t[]>(count);
  auto const sizes     = std::make_unique_for_overwrite<cudf::size_type[]>(count);
  EXPECT_THROW(static_cast<void>(spark_rapids_jni::generate_range_sequence(
                 std::span<int64_t const>{starts.get(), count},
                 std::span<cudf::size_type const>{sizes.get(), count})),
               std::invalid_argument);
}

TEST_F(RangeSequenceTests, AllocationFailures)
{
  auto expect_failure = [](std::size_t ranges, std::size_t limit, cudf::size_type size = 1) {
    rmm::mr::limiting_resource_adaptor limited{rmm::mr::cuda_async_memory_resource{}, limit};
    EXPECT_THROW(static_cast<void>(spark_rapids_jni::generate_range_sequence(
                   std::vector<int64_t>(ranges, 0),
                   std::vector<cudf::size_type>(ranges, size),
                   cudf::get_default_stream(),
                   limited)),
                 rmm::bad_alloc);
  };
  expect_failure(1, 0);
  expect_failure(2, 4'096 * sizeof(int64_t), 2'048);
  auto constexpr count = std::size_t{2'048};
  for (auto const metadata_bytes : {std::size_t{0},
                                    count * sizeof(int64_t),
                                    count * (sizeof(int64_t) + sizeof(cudf::size_type))}) {
    expect_failure(count, count * sizeof(int64_t) + metadata_bytes);
  }
}
}  // namespace
