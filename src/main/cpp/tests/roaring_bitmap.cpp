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
 * See the License for the specific language governing permissions and limitations under
 * the License.
 */

#include "roaring_bitmap.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/roaring_bitmap.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <roaring/roaring64.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

auto constexpr max_position = std::int64_t{9223372030412324864LL};

using roaring64_ptr = std::unique_ptr<roaring64_bitmap_t, decltype(&roaring64_bitmap_free)>;

std::vector<std::byte> serialize_with_croaring(std::vector<std::uint64_t> const& positions)
{
  auto bitmap  = roaring64_ptr{roaring64_bitmap_create(), &roaring64_bitmap_free};
  auto context = roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};
  for (auto const position : positions) {
    roaring64_bitmap_add_bulk(bitmap.get(), &context, position);
  }
  auto result = std::vector<std::byte>(roaring64_bitmap_portable_size_in_bytes(bitmap.get()));
  std::ignore =
    roaring64_bitmap_portable_serialize(bitmap.get(), std::bit_cast<char*>(result.data()));
  return result;
}

std::vector<std::byte> copy_to_host(spark_rapids_jni::serialized_roaring_bitmap const& bitmap,
                                    cuda::stream_ref stream)
{
  auto result = std::vector<std::byte>(bitmap.data.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    result.data(), bitmap.data.data(), bitmap.data.size(), cudaMemcpyDefault, stream.get()));
  stream.sync();
  return result;
}

void expect_contains(std::vector<std::byte> const& bytes,
                     std::vector<std::uint64_t> const& probes,
                     std::vector<bool> const& expected,
                     cuda::stream_ref stream)
{
  auto* roaring = roaring64_bitmap_portable_deserialize_safe(
    std::bit_cast<char const*>(bytes.data()), bytes.size());
  ASSERT_NE(roaring, nullptr);
  for (auto index = std::size_t{0}; index < probes.size(); ++index) {
    EXPECT_EQ(roaring64_bitmap_contains(roaring, probes[index]), expected[index]);
  }
  roaring64_bitmap_free(roaring);

  auto const serialized = std::span<cuda::std::byte const>{
    std::bit_cast<cuda::std::byte const*>(bytes.data()), bytes.size()};
  auto bitmap = cudf::roaring_bitmap(cudf::roaring_bitmap_type::BITS_64, serialized);
  cudf::test::fixed_width_column_wrapper<std::uint64_t> probe_column(probes.begin(), probes.end());
  cudf::test::fixed_width_column_wrapper<bool> expected_column(expected.begin(), expected.end());
  auto actual = bitmap.contains_async(probe_column, stream);
  stream.sync();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, *actual);
}

class RoaringBitmapTest : public cudf::test::BaseFixture {};

TEST_F(RoaringBitmapTest, Empty)
{
  auto const stream = cudf::get_default_stream();
  cudf::test::fixed_width_column_wrapper<std::int64_t> positions{};
  auto result = spark_rapids_jni::build_and_serialize_roaring64(positions, {}, stream);
  auto bytes  = copy_to_host(result, stream);

  EXPECT_EQ(result.cardinality, 0);
  EXPECT_EQ(bytes, std::vector<std::byte>(8));
  expect_contains(bytes, {}, {}, stream);
}

TEST_F(RoaringBitmapTest, UnsortedDuplicateAndBoundaryPositions)
{
  auto const stream = cudf::get_default_stream();
  auto const position_values =
    std::vector<std::int64_t>{max_position, 7, 0, 7, (std::int64_t{1} << 32) + 2, 65536, 3};
  cudf::test::fixed_width_column_wrapper<std::int64_t> positions(position_values.begin(),
                                                                 position_values.end());
  auto result = spark_rapids_jni::build_and_serialize_roaring64(positions, {}, stream);
  auto bytes  = copy_to_host(result, stream);

  EXPECT_EQ(result.cardinality, 6);
  expect_contains(bytes,
                  {0,
                   3,
                   7,
                   8,
                   65536,
                   (std::uint64_t{1} << 32) + 2,
                   static_cast<std::uint64_t>(max_position),
                   static_cast<std::uint64_t>(max_position) + 1},
                  {true, true, true, false, true, true, true, false},
                  stream);
}

TEST_F(RoaringBitmapTest, ExistingBitmapUnion)
{
  auto const stream         = cudf::get_default_stream();
  auto const existing_bytes = serialize_with_croaring(
    {1, 2, 3, (std::uint64_t{1} << 32) + 5, static_cast<std::uint64_t>(max_position)});
  auto const existing_span =
    std::span<std::byte const>{existing_bytes.data(), existing_bytes.size()};

  auto const new_position_values = std::vector<std::int64_t>{3, 4, (std::int64_t{1} << 32) + 6};
  cudf::test::fixed_width_column_wrapper<std::int64_t> new_positions(new_position_values.begin(),
                                                                     new_position_values.end());
  auto merged =
    spark_rapids_jni::build_and_serialize_roaring64(new_positions, {existing_span}, stream);
  auto merged_bytes = copy_to_host(merged, stream);

  EXPECT_EQ(merged.cardinality, 7);
  expect_contains(merged_bytes,
                  {0,
                   1,
                   2,
                   3,
                   4,
                   (std::uint64_t{1} << 32) + 5,
                   (std::uint64_t{1} << 32) + 6,
                   (std::uint64_t{1} << 32) + 7,
                   static_cast<std::uint64_t>(max_position)},
                  {false, true, true, true, true, true, true, false, true},
                  stream);
}

TEST_F(RoaringBitmapTest, ConsecutiveRangeUsesRunContainer)
{
  auto const stream = cudf::get_default_stream();
  auto positions    = std::vector<std::int64_t>(10'000);
  std::generate(
    positions.begin(), positions.end(), [value = std::int64_t{1234}]() mutable { return value++; });
  cudf::test::fixed_width_column_wrapper<std::int64_t> input(positions.begin(), positions.end());
  auto result = spark_rapids_jni::build_and_serialize_roaring64(input, {}, stream);
  auto bytes  = copy_to_host(result, stream);

  EXPECT_EQ(result.cardinality, positions.size());
  EXPECT_LT(bytes.size(), 128);
  expect_contains(bytes, {1233, 1234, 11233, 11234}, {false, true, true, false}, stream);
}

TEST_F(RoaringBitmapTest, RejectsInvalidInput)
{
  auto const stream = cudf::get_default_stream();
  cudf::test::fixed_width_column_wrapper<std::int32_t> wrong_type{1, 2};
  EXPECT_THROW(
    std::ignore = spark_rapids_jni::build_and_serialize_roaring64(wrong_type, {}, stream),
    std::invalid_argument);

  cudf::test::fixed_width_column_wrapper<std::int64_t> with_null{{1, 2}, {true, false}};
  EXPECT_THROW(std::ignore = spark_rapids_jni::build_and_serialize_roaring64(with_null, {}, stream),
               std::invalid_argument);

  cudf::test::fixed_width_column_wrapper<std::int64_t> negative{-1};
  EXPECT_THROW(std::ignore = spark_rapids_jni::build_and_serialize_roaring64(negative, {}, stream),
               std::invalid_argument);

  cudf::test::fixed_width_column_wrapper<std::int64_t> too_large{max_position + 1};
  EXPECT_THROW(std::ignore = spark_rapids_jni::build_and_serialize_roaring64(too_large, {}, stream),
               std::invalid_argument);

  auto malformed = std::vector<std::byte>(7);
  cudf::test::fixed_width_column_wrapper<std::int64_t> valid{1};
  EXPECT_THROW(std::ignore = spark_rapids_jni::build_and_serialize_roaring64(
                 valid, {std::span<std::byte const>{malformed.data(), malformed.size()}}, stream),
               std::invalid_argument);
}

TEST_F(RoaringBitmapTest, NonDefaultStream)
{
  cudf::test::fixed_width_column_wrapper<std::int64_t> positions{1, 2, 3, 65536};
  rmm::cuda_stream stream;
  auto result = spark_rapids_jni::build_and_serialize_roaring64(positions, {}, stream);
  auto bytes  = copy_to_host(result, stream);

  EXPECT_EQ(result.cardinality, 4);
  expect_contains(bytes, {0, 1, 2, 3, 65536}, {false, true, true, true, true}, stream);
}

}  // namespace
