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

#include "range_sequence_detail.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using long_column = cudf::test::fixed_width_column_wrapper<int64_t>;
struct RangeSequenceTests : cudf::test::BaseFixture {};

TEST_F(RangeSequenceTests, GapsAndInputOrder)
{
  std::vector<int64_t> const starts{100, 20, 500, 21};
  std::vector<cudf::size_type> const sizes{3, 2, 1, 3};
  auto const output   = spark_rapids_jni::generate_range_sequence(starts, sizes);
  auto const expected = long_column{100, 101, 102, 20, 21, 500, 21, 22, 23};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, expected);
  EXPECT_FALSE(output->nullable());
}

TEST_F(RangeSequenceTests, EmptyRanges)
{
  std::vector<int64_t> const starts{1, 100, 2, 3, 200, 4};
  std::vector<cudf::size_type> const sizes{0, 2, 0, 0, 1, 0};
  auto const output   = spark_rapids_jni::generate_range_sequence(starts, sizes);
  auto const expected = long_column{100, 101, 200};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, expected);
  for (auto const& lengths : std::vector<std::vector<cudf::size_type>>{{}, {0, 0}}) {
    auto const empty =
      spark_rapids_jni::generate_range_sequence(std::vector<int64_t>(lengths.size(), 0), lengths);
    EXPECT_EQ(empty->type(), cudf::data_type{cudf::type_id::INT64});
    EXPECT_EQ(empty->size(), 0);
    EXPECT_FALSE(empty->nullable());
  }
}

TEST_F(RangeSequenceTests, IntegerBoundaries)
{
  auto constexpr min = std::numeric_limits<int64_t>::min();
  auto constexpr max = std::numeric_limits<int64_t>::max();
  std::vector<int64_t> const starts{min, -1, max - 2, max};
  std::vector<cudf::size_type> const sizes{3, 3, 3, 0};
  auto const output = spark_rapids_jni::generate_range_sequence(starts, sizes);
  auto const expected =
    long_column{min, min + 1, min + 2, int64_t{-1}, int64_t{0}, int64_t{1}, max - 2, max - 1, max};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, expected);
  auto const last = spark_rapids_jni::generate_range_sequence(std::array<int64_t, 1>{max},
                                                              std::array<cudf::size_type, 1>{1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*last, (long_column{max}));
}

TEST_F(RangeSequenceTests, InvalidRanges)
{
  std::vector<int64_t> const starts{0, 100};
  EXPECT_THROW(static_cast<void>(spark_rapids_jni::generate_range_sequence(
                 starts, std::vector<cudf::size_type>{1})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(spark_rapids_jni::generate_range_sequence(
                 starts, std::vector<cudf::size_type>{0, -1})),
               std::invalid_argument);
  EXPECT_THROW(
    static_cast<void>(spark_rapids_jni::generate_range_sequence(
      starts, std::vector<cudf::size_type>{std::numeric_limits<cudf::size_type>::max(), 1})),
    std::overflow_error);
  EXPECT_THROW(static_cast<void>(spark_rapids_jni::generate_range_sequence(
                 std::vector<int64_t>{0, std::numeric_limits<int64_t>::max()},
                 std::vector<cudf::size_type>{0, 2})),
               std::overflow_error);
}

TEST_F(RangeSequenceTests, UnevenAndFragmentedRanges)
{
  std::mt19937 random{5309};
  for (int iteration = 0; iteration < 50; ++iteration) {
    auto const count = 1 + random() % 1'024;
    std::vector<int64_t> starts(count);
    std::vector<cudf::size_type> sizes(count);
    std::vector<int64_t> expected;
    for (std::size_t range = 0; range < count; ++range) {
      starts[range] = static_cast<int64_t>(random()) - (int64_t{1} << 31);
      sizes[range]  = random() % 257;
      for (int64_t i = 0; i < sizes[range]; ++i) {
        expected.push_back(starts[range] + i);
      }
    }
    auto const output          = spark_rapids_jni::generate_range_sequence(starts, sizes);
    auto const expected_column = long_column(expected.begin(), expected.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, expected_column);
  }
}

TEST_F(RangeSequenceTests, NonDefaultStreamAndBoundedWorkingSet)
{
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  cuda::stream_ref const stream_ref{stream};
  auto constexpr num_rows = 512 * 1'024;
  // Async frees ensure an implicit cudaFree synchronization cannot hide stream/lifetime errors.
  // Enough for one output plus range metadata, but not a second full-sized output.
  rmm::mr::limiting_resource_adaptor limited{rmm::mr::cuda_async_memory_resource{8 * 1'024 * 1'024},
                                             num_rows * sizeof(int64_t) + 4'096};
  std::vector<int64_t> starts(128);
  std::vector<cudf::size_type> sizes(128, num_rows / 128);
  for (std::size_t i = 0; i < starts.size(); ++i) {
    starts[i] = i * 10'000;
  }
  {
    auto const output =
      spark_rapids_jni::generate_range_sequence(starts, sizes, stream_ref, limited);
    auto const host = cudf::detail::make_std_vector(
      cudf::device_span<int64_t const>{output->view().data<int64_t>(), num_rows}, stream_ref);
    for (int i = 0; i < num_rows; ++i) {
      ASSERT_EQ(host[i], starts[i / sizes.front()] + i % sizes.front());
    }
  }
  stream.synchronize();
  EXPECT_EQ(limited.get_allocated_bytes(), std::size_t{0});
}

TEST_F(RangeSequenceTests, SingleRangeUsesOnlyOutputAllocation)
{
  auto constexpr num_rows = 4'096;
  rmm::mr::limiting_resource_adaptor limited{rmm::mr::cuda_memory_resource{},
                                             num_rows * sizeof(int64_t)};
  auto const output =
    spark_rapids_jni::generate_range_sequence(std::array<int64_t, 1>{-10},
                                              std::array<cudf::size_type, 1>{num_rows},
                                              cudf::get_default_stream(),
                                              limited);
  std::vector<int64_t> expected(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    expected[i] = -10 + i;
  }
  auto const expected_column = long_column(expected.begin(), expected.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, expected_column);
}

TEST_F(RangeSequenceTests, AllocationFailurePropagates)
{
  rmm::mr::limiting_resource_adaptor limited{rmm::mr::cuda_memory_resource{}, 1'024};
  EXPECT_THROW(static_cast<void>(
                 spark_rapids_jni::generate_range_sequence(std::array<int64_t, 1>{0},
                                                           std::array<cudf::size_type, 1>{4'096},
                                                           cudf::get_default_stream(),
                                                           limited)),
               rmm::bad_alloc);
  EXPECT_EQ(limited.get_allocated_bytes(), std::size_t{0});
}

TEST_F(RangeSequenceTests, MetadataAllocationFailureReleasesOutput)
{
  auto constexpr num_rows = 4'096;
  rmm::mr::limiting_resource_adaptor limited{rmm::mr::cuda_memory_resource{},
                                             num_rows * sizeof(int64_t)};
  EXPECT_THROW(static_cast<void>(spark_rapids_jni::generate_range_sequence(
                 std::array<int64_t, 2>{0, 10'000},
                 std::array<cudf::size_type, 2>{num_rows / 2, num_rows / 2},
                 cudf::get_default_stream(),
                 limited)),
               rmm::bad_alloc);
  EXPECT_EQ(limited.get_allocated_bytes(), std::size_t{0});
  auto const output =
    spark_rapids_jni::generate_range_sequence(std::array<int64_t, 1>{0},
                                              std::array<cudf::size_type, 1>{num_rows},
                                              cudf::get_default_stream(),
                                              limited);
  EXPECT_EQ(output->size(), num_rows);
}

TEST_F(RangeSequenceTests, ScanCrossoverAndSparseRanges)
{
  auto constexpr threshold = spark_rapids_jni::detail::range_sequence_device_scan_threshold;
  for (auto const count : {threshold - 1, threshold, threshold + 1, 4 * threshold}) {
    std::vector<int64_t> starts(count);
    std::vector<cudf::size_type> sizes(count);
    std::vector<int64_t> expected;
    for (std::size_t i = 0; i < count; ++i) {
      starts[i] = i % 2 == 0 ? -static_cast<int64_t>(i) * 100 : static_cast<int64_t>(i) * 100;
      // Empty ranges at both ends and throughout, with uneven nonempty ranges in input order.
      sizes[i] =
        i == 0 || i + 1 == count ? 0 : (i % 7 == 0 ? static_cast<cudf::size_type>(i % 5 + 1) : 0);
      for (cudf::size_type row = 0; row < sizes[i]; ++row) {
        expected.push_back(starts[i] + row);
      }
    }
    auto const output          = spark_rapids_jni::generate_range_sequence(starts, sizes);
    auto const expected_column = long_column(expected.begin(), expected.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, expected_column);
    EXPECT_FALSE(output->nullable());
  }
}

TEST_F(RangeSequenceTests, DeviceScanOnNonDefaultStream)
{
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  cuda::stream_ref const stream_ref{stream};
  rmm::mr::cuda_async_memory_resource mr{16 * 1'024 * 1'024};
  auto constexpr count = spark_rapids_jni::detail::range_sequence_device_scan_threshold;
  std::vector<int64_t> starts(count, std::numeric_limits<int64_t>::max());
  std::vector<cudf::size_type> sizes(count, 0);
  auto const first  = count / 3;
  auto const last   = 2 * count / 3;
  starts[first]     = std::numeric_limits<int64_t>::min();
  starts[last]      = std::numeric_limits<int64_t>::max() - 2;
  sizes[first]      = 3;
  sizes[last]       = 3;
  auto const output = spark_rapids_jni::generate_range_sequence(starts, sizes, stream_ref, mr);
  // Input buffers can be changed immediately after return, before output generation completes.
  std::fill(starts.begin(), starts.end(), 0);
  std::fill(sizes.begin(), sizes.end(), 0);
  auto const host = cudf::detail::make_std_vector(
    cudf::device_span<int64_t const>{output->view().data<int64_t>(), 6}, stream_ref);
  auto constexpr min = std::numeric_limits<int64_t>::min();
  auto constexpr max = std::numeric_limits<int64_t>::max();
  EXPECT_EQ(host, (std::vector<int64_t>{min, min + 1, min + 2, max - 2, max - 1, max}));

  auto const empty = spark_rapids_jni::generate_range_sequence(starts, sizes, stream_ref, mr);
  EXPECT_EQ(empty->size(), 0);
  EXPECT_FALSE(empty->nullable());
}

TEST_F(RangeSequenceTests, DeviceScanAllocationFailuresReleaseOutput)
{
  auto constexpr count = spark_rapids_jni::detail::range_sequence_device_scan_threshold;
  std::vector<int64_t> starts(count, 0);
  std::vector<cudf::size_type> sizes(count, 1);
  auto constexpr output_bytes = count * sizeof(int64_t);
  for (auto const limit : {output_bytes,
                           output_bytes + count * sizeof(int64_t),
                           output_bytes + count * (sizeof(int64_t) + sizeof(cudf::size_type))}) {
    rmm::mr::limiting_resource_adaptor limited{rmm::mr::cuda_memory_resource{}, limit};
    EXPECT_THROW(static_cast<void>(spark_rapids_jni::generate_range_sequence(
                   starts, sizes, cudf::get_default_stream(), limited)),
                 rmm::bad_alloc);
    EXPECT_EQ(limited.get_allocated_bytes(), std::size_t{0});
  }
}

TEST_F(RangeSequenceTests, DeviceScanBoundedWorkingSet)
{
  auto constexpr count        = spark_rapids_jni::detail::range_sequence_device_scan_threshold;
  auto constexpr num_rows     = 2 * count;
  auto constexpr output_bytes = num_rows * sizeof(int64_t);
  // One output, the 12-byte-per-range metadata, and scan scratch; less than two full outputs.
  auto constexpr limit =
    output_bytes + count * (sizeof(int64_t) + sizeof(cudf::size_type)) + 65'536;
  static_assert(limit < 2 * output_bytes);
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  cuda::stream_ref const stream_ref{stream};
  rmm::mr::limiting_resource_adaptor limited{
    rmm::mr::cuda_async_memory_resource{16 * 1'024 * 1'024}, limit};
  std::vector<int64_t> starts(count);
  std::vector<cudf::size_type> sizes(count, 2);
  for (std::size_t i = 0; i < count; ++i) {
    starts[i] = static_cast<int64_t>(i) * 100;
  }
  {
    auto const output =
      spark_rapids_jni::generate_range_sequence(starts, sizes, stream_ref, limited);
    auto const host = cudf::detail::make_std_vector(
      cudf::device_span<int64_t const>{output->view().data<int64_t>(), num_rows}, stream_ref);
    for (std::size_t row = 0; row < num_rows; ++row) {
      ASSERT_EQ(host[row], starts[row / 2] + static_cast<int64_t>(row % 2));
    }
  }
  stream.synchronize();
  EXPECT_EQ(limited.get_allocated_bytes(), std::size_t{0});
}

TEST_F(RangeSequenceTests, RowDensityCrossover)
{
  auto constexpr count = spark_rapids_jni::detail::range_sequence_device_scan_threshold;
  auto constexpr rows_per_range =
    spark_rapids_jni::detail::range_sequence_device_scan_rows_per_range;
  std::vector<int64_t> starts(count);
  for (std::size_t i = 0; i < count; ++i) {
    starts[i] = -static_cast<int64_t>(i) * 1'000;
  }
  for (auto const size : {rows_per_range - 1, rows_per_range, rows_per_range + 1}) {
    std::vector<cudf::size_type> sizes(count, static_cast<cudf::size_type>(size));
    auto const output = spark_rapids_jni::generate_range_sequence(starts, sizes);
    auto const host   = cudf::detail::make_std_vector(
      cudf::device_span<int64_t const>{output->view().data<int64_t>(),
                                         static_cast<std::size_t>(output->size())},
      cudf::get_default_stream());
    ASSERT_EQ(host.size(), count * size);
    for (std::size_t row = 0; row < host.size(); ++row) {
      ASSERT_EQ(host[row], starts[row / size] + static_cast<int64_t>(row % size));
    }
  }
}

TEST_F(RangeSequenceTests, EmptyOutputAndValidationBeforeDeviceAllocation)
{
  rmm::mr::limiting_resource_adaptor limited{rmm::mr::cuda_memory_resource{}, 0};
  cudf::test::scoped_current_device_resource current_resource{limited};
  auto const stream = cudf::get_default_stream();
  auto const empty  = spark_rapids_jni::generate_range_sequence(
    std::array<int64_t, 2>{0, std::numeric_limits<int64_t>::max()},
    std::array<cudf::size_type, 2>{0, 0},
    stream,
    limited);
  EXPECT_EQ(empty->type(), cudf::data_type{cudf::type_id::INT64});
  EXPECT_EQ(empty->size(), 0);
  EXPECT_FALSE(empty->nullable());
  EXPECT_THROW(static_cast<void>(spark_rapids_jni::generate_range_sequence(
                 std::array<int64_t, 1>{0}, std::array<cudf::size_type, 1>{-1}, stream, limited)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(spark_rapids_jni::generate_range_sequence(
                 std::array<int64_t, 1>{std::numeric_limits<int64_t>::max()},
                 std::array<cudf::size_type, 1>{2},
                 stream,
                 limited)),
               std::overflow_error);
  EXPECT_EQ(limited.get_allocated_bytes(), std::size_t{0});
}
}  // namespace
