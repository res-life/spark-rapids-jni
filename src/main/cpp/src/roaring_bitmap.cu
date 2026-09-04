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

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/cstdint>
#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace spark_rapids_jni {
namespace {

auto constexpr max_position              = std::uint64_t{9223372030412324864ULL};
auto constexpr serial_cookie_no_run      = std::uint32_t{12346};
auto constexpr serial_cookie             = std::uint32_t{12347};
auto constexpr no_offset_threshold       = std::uint32_t{4};
auto constexpr array_container_threshold = std::uint32_t{4096};
auto constexpr words_per_container       = std::uint32_t{1024};
auto constexpr threads_per_block         = std::uint32_t{256};

enum class source_kind : std::uint8_t { ARRAY, BITSET, RUN, NEW_VALUES };
enum class output_encoding : std::uint8_t { ARRAY, BITSET, RUN };

struct source_descriptor {
  std::uint64_t key48;
  std::uint64_t payload_offset;
  std::uint32_t cardinality;
  std::uint32_t payload_size;
  source_kind kind;
};

struct container_descriptor {
  std::uint64_t key48;
  std::uint64_t source_begin;
  std::uint32_t source_count;
  std::uint32_t cardinality;
  std::uint32_t run_count;
  std::uint32_t serialized_size;
  std::uint64_t output_offset;
  output_encoding encoding;
};

struct bucket_descriptor {
  std::uint64_t container_begin;
  std::uint32_t container_count;
  std::uint32_t high32;
  std::uint32_t header_size;
  bool has_run;
};

[[noreturn]] void invalid_bitmap(std::string const& reason)
{
  throw std::invalid_argument("Invalid portable Roaring64 bitmap: " + reason);
}

void require_bitmap(bool condition, std::string const& reason)
{
  if (not condition) { invalid_bitmap(reason); }
}

class portable_reader {
 public:
  explicit portable_reader(std::span<std::byte const> bytes) : _bytes{bytes} {}

  [[nodiscard]] std::size_t position() const { return _position; }
  [[nodiscard]] std::size_t size() const { return _bytes.size(); }

  std::uint8_t read_u8()
  {
    require(1);
    return std::to_integer<std::uint8_t>(_bytes[_position++]);
  }

  std::uint16_t read_u16()
  {
    require(2);
    auto const result = static_cast<std::uint16_t>(read_at(_position)) |
                        static_cast<std::uint16_t>(read_at(_position + 1) << 8);
    _position += 2;
    return result;
  }

  std::uint32_t read_u32()
  {
    require(4);
    auto result = std::uint32_t{0};
    for (auto i = std::size_t{0}; i < 4; ++i) {
      result |= static_cast<std::uint32_t>(read_at(_position + i)) << (8 * i);
    }
    _position += 4;
    return result;
  }

  std::uint64_t read_u64()
  {
    require(8);
    auto result = std::uint64_t{0};
    for (auto i = std::size_t{0}; i < 8; ++i) {
      result |= static_cast<std::uint64_t>(read_at(_position + i)) << (8 * i);
    }
    _position += 8;
    return result;
  }

 private:
  void require(std::size_t bytes) const
  {
    require_bitmap(bytes <= _bytes.size() - std::min(_position, _bytes.size()),
                   "truncated payload");
  }

  [[nodiscard]] std::uint8_t read_at(std::size_t position) const
  {
    return std::to_integer<std::uint8_t>(_bytes[position]);
  }

  std::span<std::byte const> _bytes;
  std::size_t _position{0};
};

void validate_position(std::uint32_t high32, std::uint16_t key16, std::uint16_t low16)
{
  auto const position =
    (static_cast<std::uint64_t>(high32) << 32) | (static_cast<std::uint64_t>(key16) << 16) | low16;
  require_bitmap(position <= max_position, "position exceeds the Iceberg maximum");
}

void validate_array_container(portable_reader& reader,
                              std::uint32_t high32,
                              std::uint16_t key16,
                              std::uint32_t cardinality)
{
  auto previous = std::uint16_t{0};
  for (auto i = std::uint32_t{0}; i < cardinality; ++i) {
    auto const value = reader.read_u16();
    require_bitmap(i == 0 or value > previous, "array container is not strictly sorted");
    validate_position(high32, key16, value);
    previous = value;
  }
}

void validate_bitset_container(portable_reader& reader,
                               std::uint32_t high32,
                               std::uint16_t key16,
                               std::uint32_t cardinality)
{
  auto actual_cardinality = std::uint32_t{0};
  auto last_value         = std::uint16_t{0};
  auto found_value        = false;
  for (auto word_index = std::uint32_t{0}; word_index < words_per_container; ++word_index) {
    auto word = reader.read_u64();
    actual_cardinality += std::popcount(word);
    if (word != 0) {
      last_value  = static_cast<std::uint16_t>(word_index * 64 + (63 - std::countl_zero(word)));
      found_value = true;
    }
  }
  require_bitmap(actual_cardinality == cardinality, "bitset cardinality does not match header");
  require_bitmap(found_value, "empty bitset container");
  validate_position(high32, key16, last_value);
}

void validate_run_container(portable_reader& reader,
                            std::uint32_t high32,
                            std::uint16_t key16,
                            std::uint32_t cardinality)
{
  auto const run_count = reader.read_u16();
  require_bitmap(run_count != 0, "empty run container");
  auto actual_cardinality = std::uint32_t{0};
  auto previous_end       = std::uint32_t{0};
  for (auto i = std::uint32_t{0}; i < run_count; ++i) {
    auto const start  = reader.read_u16();
    auto const length = reader.read_u16();
    auto const end    = static_cast<std::uint32_t>(start) + length;
    require_bitmap(end <= std::numeric_limits<std::uint16_t>::max(), "run exceeds uint16 range");
    require_bitmap(i == 0 or start > previous_end + 1, "runs overlap or are adjacent");
    actual_cardinality += static_cast<std::uint32_t>(length) + 1;
    require_bitmap(actual_cardinality <= 65536, "run cardinality overflow");
    previous_end = end;
  }
  require_bitmap(actual_cardinality == cardinality, "run cardinality does not match header");
  validate_position(high32, key16, static_cast<std::uint16_t>(previous_end));
}

void parse_roaring32(portable_reader& reader,
                     std::uint32_t high32,
                     std::size_t bitmap_start,
                     std::size_t global_base,
                     std::vector<source_descriptor>& descriptors)
{
  auto const cookie    = reader.read_u32();
  auto const has_run   = (cookie & 0xffffU) == serial_cookie;
  auto container_count = std::uint32_t{0};
  if (cookie == serial_cookie_no_run) {
    container_count = reader.read_u32();
  } else if (has_run) {
    container_count = (cookie >> 16) + 1;
  } else {
    invalid_bitmap("unknown Roaring32 cookie");
  }
  require_bitmap(container_count > 0 and container_count <= 65536,
                 "invalid Roaring32 container count");

  auto run_flags = std::vector<std::uint8_t>((container_count + 7) / 8, 0);
  if (has_run) {
    for (auto& flag : run_flags) {
      flag = reader.read_u8();
    }
  }

  auto keys          = std::vector<std::uint16_t>(container_count);
  auto cardinalities = std::vector<std::uint32_t>(container_count);
  for (auto i = std::uint32_t{0}; i < container_count; ++i) {
    keys[i]          = reader.read_u16();
    cardinalities[i] = static_cast<std::uint32_t>(reader.read_u16()) + 1;
    require_bitmap(i == 0 or keys[i] > keys[i - 1], "container keys are not strictly sorted");
  }

  auto const has_offsets = not has_run or container_count >= no_offset_threshold;
  auto offsets           = std::vector<std::uint32_t>(container_count, 0);
  if (has_offsets) {
    for (auto& offset : offsets) {
      offset = reader.read_u32();
    }
  }

  for (auto i = std::uint32_t{0}; i < container_count; ++i) {
    auto const relative_position = reader.position() - bitmap_start;
    if (has_offsets) {
      require_bitmap(offsets[i] == relative_position, "container offset does not match payload");
    }

    auto const is_run         = has_run and ((run_flags[i / 8] >> (i % 8)) & 1U) != 0;
    auto const payload_offset = reader.position();
    auto kind                 = source_kind::ARRAY;
    if (is_run) {
      kind = source_kind::RUN;
      validate_run_container(reader, high32, keys[i], cardinalities[i]);
    } else if (cardinalities[i] > array_container_threshold) {
      kind = source_kind::BITSET;
      validate_bitset_container(reader, high32, keys[i], cardinalities[i]);
    } else {
      validate_array_container(reader, high32, keys[i], cardinalities[i]);
    }
    auto const payload_size = reader.position() - payload_offset;
    require_bitmap(payload_size <= std::numeric_limits<std::uint32_t>::max(),
                   "container payload is too large");
    descriptors.push_back(source_descriptor{(static_cast<std::uint64_t>(high32) << 16) | keys[i],
                                            global_base + payload_offset,
                                            cardinalities[i],
                                            static_cast<std::uint32_t>(payload_size),
                                            kind});
  }
}

void parse_roaring64(std::span<std::byte const> bitmap,
                     std::size_t global_base,
                     std::vector<source_descriptor>& descriptors)
{
  portable_reader reader{bitmap};
  auto const bucket_count = reader.read_u64();
  require_bitmap(bucket_count <= std::numeric_limits<std::uint32_t>::max(),
                 "bucket count exceeds the portable format limit");
  require_bitmap(bucket_count <= (reader.size() - reader.position()) / 12,
                 "bucket count exceeds payload size");

  auto previous_high32 = std::uint32_t{0};
  for (auto bucket = std::uint64_t{0}; bucket < bucket_count; ++bucket) {
    auto const high32 = reader.read_u32();
    require_bitmap(bucket == 0 or high32 > previous_high32,
                   "Roaring64 high keys are not strictly sorted");
    auto const bitmap_start = reader.position();
    parse_roaring32(reader, high32, bitmap_start, global_base, descriptors);
    previous_high32 = high32;
  }
  require_bitmap(reader.position() == reader.size(), "trailing bytes after bitmap");
}

__device__ std::uint16_t device_read_u16(std::uint8_t const* data)
{
  return static_cast<std::uint16_t>(data[0]) |
         static_cast<std::uint16_t>(static_cast<std::uint16_t>(data[1]) << 8);
}

__device__ std::uint64_t device_read_u64(std::uint8_t const* data)
{
  auto result = std::uint64_t{0};
#pragma unroll
  for (auto i = 0; i < 8; ++i) {
    result |= static_cast<std::uint64_t>(data[i]) << (8 * i);
  }
  return result;
}

__device__ void device_write_u16(std::uint8_t* data, std::uint16_t value)
{
  data[0] = static_cast<std::uint8_t>(value);
  data[1] = static_cast<std::uint8_t>(value >> 8);
}

__device__ void device_write_u32(std::uint8_t* data, std::uint32_t value)
{
#pragma unroll
  for (auto i = 0; i < 4; ++i) {
    data[i] = static_cast<std::uint8_t>(value >> (8 * i));
  }
}

__device__ void device_write_u64(std::uint8_t* data, std::uint64_t value)
{
#pragma unroll
  for (auto i = 0; i < 8; ++i) {
    data[i] = static_cast<std::uint8_t>(value >> (8 * i));
  }
}

__device__ std::uint32_t lower_bound_array(std::uint8_t const* data,
                                           std::uint32_t size,
                                           std::uint16_t target)
{
  auto first = std::uint32_t{0};
  auto count = size;
  while (count != 0) {
    auto const step = count / 2;
    auto const it   = first + step;
    if (device_read_u16(data + 2 * it) < target) {
      first = it + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

__device__ std::uint32_t lower_bound_new_values(std::uint64_t const* data,
                                                std::uint32_t size,
                                                std::uint16_t target)
{
  auto first = std::uint32_t{0};
  auto count = size;
  while (count != 0) {
    auto const step = count / 2;
    auto const it   = first + step;
    if (static_cast<std::uint16_t>(data[it]) < target) {
      first = it + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

__device__ std::uint64_t range_mask(std::uint32_t begin, std::uint32_t end)
{
  auto const left  = begin == 0 ? ~std::uint64_t{0} : (~std::uint64_t{0} << begin);
  auto const right = end == 63 ? ~std::uint64_t{0} : ((std::uint64_t{1} << (end + 1)) - 1);
  return left & right;
}

__device__ std::uint64_t source_word(source_descriptor const& source,
                                     std::uint32_t word_index,
                                     std::uint8_t const* existing_data,
                                     std::uint64_t const* new_values)
{
  auto const word_begin = static_cast<std::uint16_t>(word_index * 64);
  auto const word_end   = static_cast<std::uint16_t>(word_begin + 63);
  if (source.kind == source_kind::BITSET) {
    return device_read_u64(existing_data + source.payload_offset + word_index * 8);
  }

  auto result = std::uint64_t{0};
  if (source.kind == source_kind::ARRAY) {
    auto const* data = existing_data + source.payload_offset;
    auto index       = lower_bound_array(data, source.cardinality, word_begin);
    while (index < source.cardinality) {
      auto const value = device_read_u16(data + 2 * index);
      if (value > word_end) { break; }
      result |= std::uint64_t{1} << (value - word_begin);
      ++index;
    }
  } else if (source.kind == source_kind::NEW_VALUES) {
    auto const* data = new_values + source.payload_offset;
    auto index       = lower_bound_new_values(data, source.cardinality, word_begin);
    while (index < source.cardinality) {
      auto const value = static_cast<std::uint16_t>(data[index]);
      if (value > word_end) { break; }
      result |= std::uint64_t{1} << (value - word_begin);
      ++index;
    }
  } else {
    auto const* data     = existing_data + source.payload_offset;
    auto const run_count = device_read_u16(data);
    auto first           = std::uint32_t{0};
    auto count           = static_cast<std::uint32_t>(run_count);
    while (count != 0) {
      auto const step  = count / 2;
      auto const run   = first + step;
      auto const start = static_cast<std::uint32_t>(device_read_u16(data + 2 + 4 * run));
      auto const end   = start + device_read_u16(data + 4 + 4 * run);
      if (end < word_begin) {
        first = run + 1;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    for (auto run = first; run < run_count; ++run) {
      auto const start = static_cast<std::uint32_t>(device_read_u16(data + 2 + 4 * run));
      auto const end   = start + device_read_u16(data + 4 + 4 * run);
      if (start > word_end) { break; }
      auto const clipped_begin = start > word_begin ? start : word_begin;
      auto const clipped_end   = end < word_end ? end : word_end;
      result |= range_mask(clipped_begin - word_begin, clipped_end - word_begin);
    }
  }
  return result;
}

__device__ void build_union_words(container_descriptor const& container,
                                  source_descriptor const* sources,
                                  std::uint8_t const* existing_data,
                                  std::uint64_t const* new_values,
                                  std::uint64_t* words)
{
  for (auto word_index = threadIdx.x; word_index < words_per_container; word_index += blockDim.x) {
    auto value = std::uint64_t{0};
    for (auto source_index = std::uint32_t{0}; source_index < container.source_count;
         ++source_index) {
      value |= source_word(
        sources[container.source_begin + source_index], word_index, existing_data, new_values);
    }
    words[word_index] = value;
  }
}

__global__ void validate_and_copy_positions(std::int64_t const* input,
                                            std::uint64_t* output,
                                            std::uint64_t size,
                                            int* invalid)
{
  for (auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < size;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    auto const value = input[index];
    if (value < 0 or static_cast<std::uint64_t>(value) > max_position) {
      atomicExch(invalid, 1);
    } else {
      output[index] = static_cast<std::uint64_t>(value);
    }
  }
}

struct different_key48 {
  std::uint64_t const* values;

  __device__ bool operator()(std::uint64_t index) const
  {
    return index == 0 or (values[index] >> 16) != (values[index - 1] >> 16);
  }
};

__global__ void make_new_source_descriptors(std::uint64_t const* values,
                                            std::uint64_t value_count,
                                            std::uint64_t const* starts,
                                            std::uint64_t descriptor_count,
                                            source_descriptor* output)
{
  for (auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < descriptor_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    auto const begin = starts[index];
    auto const end   = index + 1 < descriptor_count ? starts[index + 1] : value_count;
    output[index]    = source_descriptor{values[begin] >> 16,
                                      begin,
                                      static_cast<std::uint32_t>(end - begin),
                                      0,
                                      source_kind::NEW_VALUES};
  }
}

struct source_less {
  __device__ bool operator()(source_descriptor const& left, source_descriptor const& right) const
  {
    if (left.key48 != right.key48) { return left.key48 < right.key48; }
    if (left.kind != right.kind) {
      return static_cast<std::uint8_t>(left.kind) < static_cast<std::uint8_t>(right.kind);
    }
    return left.payload_offset < right.payload_offset;
  }
};

struct different_source_key {
  source_descriptor const* sources;

  __device__ bool operator()(std::uint64_t index) const
  {
    return index == 0 or sources[index].key48 != sources[index - 1].key48;
  }
};

__global__ void initialize_containers(source_descriptor const* sources,
                                      std::uint64_t source_count,
                                      std::uint64_t const* starts,
                                      std::uint64_t container_count,
                                      container_descriptor* containers)
{
  for (auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < container_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    auto const begin  = starts[index];
    auto const end    = index + 1 < container_count ? starts[index + 1] : source_count;
    containers[index] = container_descriptor{sources[begin].key48,
                                             begin,
                                             static_cast<std::uint32_t>(end - begin),
                                             0,
                                             0,
                                             0,
                                             0,
                                             output_encoding::ARRAY};
  }
}

__global__ void measure_containers(source_descriptor const* sources,
                                   std::uint8_t const* existing_data,
                                   std::uint64_t const* new_values,
                                   container_descriptor* containers,
                                   std::uint64_t container_count)
{
  auto const container_index = static_cast<std::uint64_t>(blockIdx.x);
  if (container_index >= container_count) { return; }
  __shared__ std::uint64_t words[words_per_container];
  __shared__ std::uint32_t cardinalities[threads_per_block];
  __shared__ std::uint32_t run_counts[threads_per_block];

  auto const container = containers[container_index];
  build_union_words(container, sources, existing_data, new_values, words);
  __syncthreads();

  auto cardinality = std::uint32_t{0};
  auto run_count   = std::uint32_t{0};
  for (auto word_index = threadIdx.x; word_index < words_per_container; word_index += blockDim.x) {
    auto const word = words[word_index];
    cardinality += __popcll(word);
    auto const previous_high_bit =
      word_index == 0 ? std::uint64_t{0} : ((words[word_index - 1] >> 63) & 1U);
    run_count += __popcll(word & ~((word << 1) | previous_high_bit));
  }
  cardinalities[threadIdx.x] = cardinality;
  run_counts[threadIdx.x]    = run_count;
  __syncthreads();

  for (auto stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      cardinalities[threadIdx.x] += cardinalities[threadIdx.x + stride];
      run_counts[threadIdx.x] += run_counts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    containers[container_index].cardinality = cardinalities[0];
    containers[container_index].run_count   = run_counts[0];
  }
}

struct different_high32 {
  container_descriptor const* containers;

  __device__ bool operator()(std::uint64_t index) const
  {
    return index == 0 or (containers[index].key48 >> 16) != (containers[index - 1].key48 >> 16);
  }
};

__global__ void initialize_buckets(container_descriptor const* containers,
                                   std::uint64_t container_count,
                                   std::uint64_t const* starts,
                                   std::uint64_t bucket_count,
                                   bucket_descriptor* buckets)
{
  for (auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < bucket_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    auto const begin = starts[index];
    auto const end   = index + 1 < bucket_count ? starts[index + 1] : container_count;
    buckets[index]   = bucket_descriptor{begin,
                                       static_cast<std::uint32_t>(end - begin),
                                       static_cast<std::uint32_t>(containers[begin].key48 >> 16),
                                       0,
                                       false};
  }
}

__device__ std::uint32_t base_container_size(container_descriptor const& container)
{
  return container.cardinality <= array_container_threshold ? 2 * container.cardinality : 8192;
}

__device__ std::uint32_t run_container_size(container_descriptor const& container)
{
  return 2 + 4 * container.run_count;
}

__global__ void choose_encodings(bucket_descriptor* buckets,
                                 container_descriptor* containers,
                                 std::uint64_t bucket_count)
{
  auto const bucket_index = static_cast<std::uint64_t>(blockIdx.x);
  if (bucket_index >= bucket_count) { return; }

  __shared__ bool use_run_header;
  __shared__ bool has_negative_delta;
  __shared__ std::uint32_t selected_container;
  if (threadIdx.x == 0) {
    auto const bucket   = buckets[bucket_index];
    auto negative_sum   = std::int64_t{0};
    auto minimum_delta  = std::int32_t{2147483647};
    auto minimum_index  = std::uint32_t{0};
    auto found_negative = false;
    for (auto local = std::uint32_t{0}; local < bucket.container_count; ++local) {
      auto const index = bucket.container_begin + local;
      auto const delta = static_cast<std::int32_t>(run_container_size(containers[index])) -
                         static_cast<std::int32_t>(base_container_size(containers[index]));
      if (delta < 0) {
        negative_sum += delta;
        found_negative = true;
      }
      if (delta < minimum_delta) {
        minimum_delta = delta;
        minimum_index = local;
      }
    }
    auto const count         = bucket.container_count;
    auto const run_header    = count < no_offset_threshold ? 4 + (count + 7) / 8 + 4 * count
                                                           : 4 + (count + 7) / 8 + 8 * count;
    auto const no_run_header = 8 + 8 * count;
    auto const payload_delta = found_negative ? negative_sum : minimum_delta;
    use_run_header     = static_cast<std::int64_t>(run_header) - no_run_header + payload_delta < 0;
    has_negative_delta = found_negative;
    selected_container = minimum_index;
    buckets[bucket_index].has_run     = use_run_header;
    buckets[bucket_index].header_size = use_run_header ? run_header : no_run_header;
  }
  __syncthreads();

  auto const bucket = buckets[bucket_index];
  for (auto local = threadIdx.x; local < bucket.container_count; local += blockDim.x) {
    auto const index = bucket.container_begin + local;
    auto const base  = base_container_size(containers[index]);
    auto const run   = run_container_size(containers[index]);
    auto const select_run =
      use_run_header and (has_negative_delta ? run < base : local == selected_container);
    if (select_run) {
      containers[index].encoding        = output_encoding::RUN;
      containers[index].serialized_size = run;
    } else if (containers[index].cardinality <= array_container_threshold) {
      containers[index].encoding        = output_encoding::ARRAY;
      containers[index].serialized_size = 2 * containers[index].cardinality;
    } else {
      containers[index].encoding        = output_encoding::BITSET;
      containers[index].serialized_size = 8192;
    }
  }
}

__global__ void assign_bucket_ids(bucket_descriptor const* buckets,
                                  std::uint64_t bucket_count,
                                  container_descriptor const* containers,
                                  std::uint32_t* bucket_ids,
                                  std::uint64_t* container_sizes)
{
  auto const bucket_index = static_cast<std::uint64_t>(blockIdx.x);
  if (bucket_index >= bucket_count) { return; }
  auto const bucket = buckets[bucket_index];
  for (auto local = threadIdx.x; local < bucket.container_count; local += blockDim.x) {
    auto const index       = bucket.container_begin + local;
    bucket_ids[index]      = static_cast<std::uint32_t>(bucket_index);
    container_sizes[index] = containers[index].serialized_size;
  }
}

__global__ void compute_bucket_sizes(bucket_descriptor const* buckets,
                                     container_descriptor const* containers,
                                     std::uint64_t const* relative_offsets,
                                     std::uint64_t bucket_count,
                                     std::uint64_t* bucket_sizes)
{
  for (auto bucket_index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       bucket_index < bucket_count;
       bucket_index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    auto const bucket          = buckets[bucket_index];
    auto const last_index      = bucket.container_begin + bucket.container_count - 1;
    bucket_sizes[bucket_index] = 4 + bucket.header_size + relative_offsets[last_index] +
                                 containers[last_index].serialized_size;
  }
}

__global__ void assign_output_offsets(bucket_descriptor const* buckets,
                                      std::uint64_t const* bucket_offsets,
                                      std::uint32_t const* bucket_ids,
                                      std::uint64_t const* relative_offsets,
                                      container_descriptor* containers,
                                      std::uint64_t container_count)
{
  for (auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < container_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    auto const bucket_index         = bucket_ids[index];
    containers[index].output_offset = bucket_offsets[bucket_index] + 4 +
                                      buckets[bucket_index].header_size + relative_offsets[index];
  }
}

__global__ void write_outer_header(std::uint8_t* output, std::uint64_t bucket_count)
{
  if (blockIdx.x == 0 and threadIdx.x == 0) { device_write_u64(output, bucket_count); }
}

__global__ void write_bucket_headers(std::uint8_t* output,
                                     bucket_descriptor const* buckets,
                                     std::uint64_t const* bucket_offsets,
                                     container_descriptor const* containers,
                                     std::uint64_t bucket_count)
{
  auto const bucket_index = static_cast<std::uint64_t>(blockIdx.x);
  if (bucket_index >= bucket_count) { return; }
  auto const bucket         = buckets[bucket_index];
  auto* high_key_output     = output + bucket_offsets[bucket_index];
  auto* bitmap_output       = high_key_output + 4;
  auto const run_bytes      = (bucket.container_count + 7) / 8;
  auto const keys_offset    = bucket.has_run ? 4 + run_bytes : 8;
  auto const offsets_offset = keys_offset + 4 * bucket.container_count;
  auto const write_offsets  = not bucket.has_run or bucket.container_count >= no_offset_threshold;

  if (threadIdx.x == 0) {
    device_write_u32(high_key_output, bucket.high32);
    if (bucket.has_run) {
      auto const cookie = serial_cookie | ((bucket.container_count - 1) << 16);
      device_write_u32(bitmap_output, cookie);
      for (auto local = std::uint32_t{0}; local < bucket.container_count; ++local) {
        auto const container = containers[bucket.container_begin + local];
        if (container.encoding == output_encoding::RUN) {
          bitmap_output[4 + local / 8] |= static_cast<std::uint8_t>(1U << (local % 8));
        }
      }
    } else {
      device_write_u32(bitmap_output, serial_cookie_no_run);
      device_write_u32(bitmap_output + 4, bucket.container_count);
    }
  }
  __syncthreads();

  for (auto local = threadIdx.x; local < bucket.container_count; local += blockDim.x) {
    auto const container = containers[bucket.container_begin + local];
    auto* key_output     = bitmap_output + keys_offset + 4 * local;
    device_write_u16(key_output, static_cast<std::uint16_t>(container.key48));
    device_write_u16(key_output + 2, static_cast<std::uint16_t>(container.cardinality - 1));
    if (write_offsets) {
      device_write_u32(
        bitmap_output + offsets_offset + 4 * local,
        static_cast<std::uint32_t>(container.output_offset - (bucket_offsets[bucket_index] + 4)));
    }
  }
}

__device__ std::uint32_t prefix_counts(std::uint32_t local_count,
                                       std::uint32_t* counts,
                                       std::uint32_t* total)
{
  counts[threadIdx.x + 1] = local_count;
  if (threadIdx.x == 0) { counts[0] = 0; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (auto i = std::uint32_t{1}; i <= blockDim.x; ++i) {
      counts[i] += counts[i - 1];
    }
    *total = counts[blockDim.x];
  }
  __syncthreads();
  return counts[threadIdx.x];
}

__global__ void serialize_containers(std::uint8_t* output,
                                     source_descriptor const* sources,
                                     std::uint8_t const* existing_data,
                                     std::uint64_t const* new_values,
                                     container_descriptor const* containers,
                                     std::uint64_t container_count)
{
  auto const container_index = static_cast<std::uint64_t>(blockIdx.x);
  if (container_index >= container_count) { return; }
  __shared__ std::uint64_t words[words_per_container];
  __shared__ std::uint32_t first_prefix[threads_per_block + 1];
  __shared__ std::uint32_t second_prefix[threads_per_block + 1];
  __shared__ std::uint32_t first_total;
  __shared__ std::uint32_t second_total;

  auto const container = containers[container_index];
  auto* payload        = output + container.output_offset;
  build_union_words(container, sources, existing_data, new_values, words);
  __syncthreads();

  if (container.encoding == output_encoding::BITSET) {
    for (auto word_index = threadIdx.x; word_index < words_per_container;
         word_index += blockDim.x) {
      device_write_u64(payload + 8 * word_index, words[word_index]);
    }
    return;
  }

  auto const first_word = static_cast<std::uint32_t>(threadIdx.x) * 4;
  if (container.encoding == output_encoding::ARRAY) {
    auto local_count = std::uint32_t{0};
#pragma unroll
    for (auto word = std::uint32_t{0}; word < 4; ++word) {
      local_count += __popcll(words[first_word + word]);
    }
    auto output_index = prefix_counts(local_count, first_prefix, &first_total);
#pragma unroll
    for (auto local_word = std::uint32_t{0}; local_word < 4; ++local_word) {
      auto value = words[first_word + local_word];
      while (value != 0) {
        auto const bit = static_cast<std::uint32_t>(__ffsll(value) - 1);
        device_write_u16(payload + 2 * output_index,
                         static_cast<std::uint16_t>((first_word + local_word) * 64 + bit));
        ++output_index;
        value &= value - 1;
      }
    }
    return;
  }

  auto local_starts = std::uint32_t{0};
  auto local_ends   = std::uint32_t{0};
#pragma unroll
  for (auto local_word = std::uint32_t{0}; local_word < 4; ++local_word) {
    auto const word_index = first_word + local_word;
    auto const word       = words[word_index];
    auto const previous   = word_index == 0 ? std::uint64_t{0} : (words[word_index - 1] >> 63);
    auto const next       = word_index + 1 == words_per_container ? std::uint64_t{0}
                                                                  : ((words[word_index + 1] & 1U) << 63);
    local_starts += __popcll(word & ~((word << 1) | previous));
    local_ends += __popcll(word & ~((word >> 1) | next));
  }
  auto start_index = prefix_counts(local_starts, first_prefix, &first_total);
  auto end_index   = prefix_counts(local_ends, second_prefix, &second_total);
  if (threadIdx.x == 0) { device_write_u16(payload, static_cast<std::uint16_t>(first_total)); }

#pragma unroll
  for (auto local_word = std::uint32_t{0}; local_word < 4; ++local_word) {
    auto const word_index = first_word + local_word;
    auto const word       = words[word_index];
    auto const previous   = word_index == 0 ? std::uint64_t{0} : (words[word_index - 1] >> 63);
    auto starts           = word & ~((word << 1) | previous);
    while (starts != 0) {
      auto const bit = static_cast<std::uint32_t>(__ffsll(starts) - 1);
      device_write_u16(payload + 2 + 4 * start_index,
                       static_cast<std::uint16_t>(word_index * 64 + bit));
      ++start_index;
      starts &= starts - 1;
    }
  }
  __syncthreads();

#pragma unroll
  for (auto local_word = std::uint32_t{0}; local_word < 4; ++local_word) {
    auto const word_index = first_word + local_word;
    auto const word       = words[word_index];
    auto const next       = word_index + 1 == words_per_container ? std::uint64_t{0}
                                                                  : ((words[word_index + 1] & 1U) << 63);
    auto ends             = word & ~((word >> 1) | next);
    while (ends != 0) {
      auto const bit   = static_cast<std::uint32_t>(__ffsll(ends) - 1);
      auto const end   = static_cast<std::uint16_t>(word_index * 64 + bit);
      auto* run        = payload + 2 + 4 * end_index;
      auto const start = device_read_u16(run);
      device_write_u16(run + 2, static_cast<std::uint16_t>(end - start));
      ++end_index;
      ends &= ends - 1;
    }
  }
}

std::uint32_t grid_size(std::uint64_t size)
{
  auto const blocks = (size + threads_per_block - 1) / threads_per_block;
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(blocks, 65535));
}

template <typename Predicate>
std::uint64_t select_starts(std::uint64_t count,
                            rmm::device_uvector<std::uint64_t>& starts,
                            Predicate predicate,
                            cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr)
{
  if (count == 0) { return 0; }
  auto const begin = thrust::make_counting_iterator<std::uint64_t>(0);
  auto const end   = thrust::copy_if(
    rmm::exec_policy_nosync(stream, mr), begin, begin + count, starts.begin(), predicate);
  return static_cast<std::uint64_t>(end - starts.begin());
}

}  // namespace

serialized_roaring_bitmap build_and_serialize_roaring64(
  cudf::column_view const& positions,
  std::vector<std::span<std::byte const>> const& existing_bitmaps,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(positions.type().id() == cudf::type_id::INT64,
               "positions must have INT64 type",
               std::invalid_argument);
  CUDF_EXPECTS(
    not positions.has_nulls(), "positions must not contain nulls", std::invalid_argument);

  auto existing_descriptors = std::vector<source_descriptor>{};
  auto raw_bytes            = std::size_t{0};
  for (auto const bitmap : existing_bitmaps) {
    require_bitmap(not bitmap.empty(), "empty input payload");
    require_bitmap(bitmap.size() <= std::numeric_limits<std::size_t>::max() - raw_bytes,
                   "combined input size overflow");
    parse_roaring64(bitmap, raw_bytes, existing_descriptors);
    raw_bytes += bitmap.size();
  }

  auto existing_data   = rmm::device_buffer(raw_bytes, stream, mr);
  auto existing_offset = std::size_t{0};
  for (auto const bitmap : existing_bitmaps) {
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(static_cast<std::uint8_t*>(existing_data.data()) + existing_offset,
                      bitmap.data(),
                      bitmap.size(),
                      cudaMemcpyDefault,
                      stream.get()));
    existing_offset += bitmap.size();
  }

  auto new_values = rmm::device_uvector<std::uint64_t>(positions.size(), stream, mr);
  if (positions.size() != 0) {
    auto invalid = rmm::device_uvector<int>(1, stream, mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(invalid.data(), 0, sizeof(int), stream.get()));
    validate_and_copy_positions<<<grid_size(positions.size()),
                                  threads_per_block,
                                  0,
                                  stream.get()>>>(
      positions.begin<std::int64_t>(), new_values.data(), positions.size(), invalid.data());
    CUDF_CUDA_TRY(cudaPeekAtLastError());
    auto host_invalid = 0;
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(&host_invalid, invalid.data(), sizeof(int), cudaMemcpyDefault, stream.get()));
    stream.sync();
    CUDF_EXPECTS(host_invalid == 0,
                 "positions must be between 0 and the Iceberg maximum position",
                 std::invalid_argument);
  }

  auto policy = rmm::exec_policy_nosync(stream, mr);
  thrust::sort(policy, new_values.begin(), new_values.end());
  auto const unique_end      = thrust::unique(policy, new_values.begin(), new_values.end());
  auto const new_value_count = static_cast<std::uint64_t>(unique_end - new_values.begin());

  auto new_starts = rmm::device_uvector<std::uint64_t>(new_value_count, stream, mr);
  auto const new_descriptor_count =
    select_starts(new_value_count, new_starts, different_key48{new_values.data()}, stream, mr);
  auto const source_count = existing_descriptors.size() + new_descriptor_count;
  if (source_count == 0) {
    auto output = rmm::device_buffer(sizeof(std::uint64_t), stream, mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(output.data(), 0, output.size(), stream.get()));
    return serialized_roaring_bitmap{std::move(output), 0};
  }

  auto sources = rmm::device_uvector<source_descriptor>(source_count, stream, mr);
  if (not existing_descriptors.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(sources.data(),
                                  existing_descriptors.data(),
                                  existing_descriptors.size() * sizeof(source_descriptor),
                                  cudaMemcpyDefault,
                                  stream.get()));
  }
  if (new_descriptor_count != 0) {
    make_new_source_descriptors<<<grid_size(new_descriptor_count),
                                  threads_per_block,
                                  0,
                                  stream.get()>>>(new_values.data(),
                                                  new_value_count,
                                                  new_starts.data(),
                                                  new_descriptor_count,
                                                  sources.data() + existing_descriptors.size());
    CUDF_CUDA_TRY(cudaPeekAtLastError());
  }

  thrust::sort(policy, sources.begin(), sources.end(), source_less{});
  auto source_starts = rmm::device_uvector<std::uint64_t>(source_count, stream, mr);
  auto const container_count =
    select_starts(source_count, source_starts, different_source_key{sources.data()}, stream, mr);
  CUDF_EXPECTS(container_count <= std::numeric_limits<std::uint32_t>::max(),
               "too many Roaring containers");

  auto containers = rmm::device_uvector<container_descriptor>(container_count, stream, mr);
  initialize_containers<<<grid_size(container_count), threads_per_block, 0, stream.get()>>>(
    sources.data(), source_count, source_starts.data(), container_count, containers.data());
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  measure_containers<<<static_cast<std::uint32_t>(container_count),
                       threads_per_block,
                       0,
                       stream.get()>>>(sources.data(),
                                       static_cast<std::uint8_t const*>(existing_data.data()),
                                       new_values.data(),
                                       containers.data(),
                                       container_count);
  CUDF_CUDA_TRY(cudaPeekAtLastError());

  auto bucket_starts = rmm::device_uvector<std::uint64_t>(container_count, stream, mr);
  auto const bucket_count =
    select_starts(container_count, bucket_starts, different_high32{containers.data()}, stream, mr);
  CUDF_EXPECTS(bucket_count <= std::numeric_limits<std::uint32_t>::max(),
               "too many Roaring64 buckets");
  auto buckets = rmm::device_uvector<bucket_descriptor>(bucket_count, stream, mr);
  initialize_buckets<<<grid_size(bucket_count), threads_per_block, 0, stream.get()>>>(
    containers.data(), container_count, bucket_starts.data(), bucket_count, buckets.data());
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  choose_encodings<<<static_cast<std::uint32_t>(bucket_count),
                     threads_per_block,
                     0,
                     stream.get()>>>(buckets.data(), containers.data(), bucket_count);
  CUDF_CUDA_TRY(cudaPeekAtLastError());

  auto bucket_ids      = rmm::device_uvector<std::uint32_t>(container_count, stream, mr);
  auto container_sizes = rmm::device_uvector<std::uint64_t>(container_count, stream, mr);
  assign_bucket_ids<<<static_cast<std::uint32_t>(bucket_count),
                      threads_per_block,
                      0,
                      stream.get()>>>(
    buckets.data(), bucket_count, containers.data(), bucket_ids.data(), container_sizes.data());
  CUDF_CUDA_TRY(cudaPeekAtLastError());

  auto relative_offsets = rmm::device_uvector<std::uint64_t>(container_count, stream, mr);
  thrust::exclusive_scan_by_key(policy,
                                bucket_ids.begin(),
                                bucket_ids.end(),
                                container_sizes.begin(),
                                relative_offsets.begin());
  auto bucket_sizes   = rmm::device_uvector<std::uint64_t>(bucket_count, stream, mr);
  auto bucket_offsets = rmm::device_uvector<std::uint64_t>(bucket_count, stream, mr);
  compute_bucket_sizes<<<grid_size(bucket_count), threads_per_block, 0, stream.get()>>>(
    buckets.data(), containers.data(), relative_offsets.data(), bucket_count, bucket_sizes.data());
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  thrust::exclusive_scan(
    policy, bucket_sizes.begin(), bucket_sizes.end(), bucket_offsets.begin(), std::uint64_t{8});

  auto total_size  = std::uint64_t{0};
  auto last_offset = std::uint64_t{0};
  auto last_size   = std::uint64_t{0};
  CUDF_CUDA_TRY(cudaMemcpyAsync(&last_offset,
                                bucket_offsets.data() + bucket_count - 1,
                                sizeof(last_offset),
                                cudaMemcpyDefault,
                                stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(&last_size,
                                bucket_sizes.data() + bucket_count - 1,
                                sizeof(last_size),
                                cudaMemcpyDefault,
                                stream.get()));
  stream.sync();
  CUDF_EXPECTS(last_offset <= std::numeric_limits<std::uint64_t>::max() - last_size,
               "serialized bitmap size overflow");
  total_size = last_offset + last_size;
  CUDF_EXPECTS(total_size <= std::numeric_limits<std::size_t>::max(),
               "serialized bitmap is too large");

  assign_output_offsets<<<grid_size(container_count), threads_per_block, 0, stream.get()>>>(
    buckets.data(),
    bucket_offsets.data(),
    bucket_ids.data(),
    relative_offsets.data(),
    containers.data(),
    container_count);
  CUDF_CUDA_TRY(cudaPeekAtLastError());

  auto output = rmm::device_buffer(static_cast<std::size_t>(total_size), stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(output.data(), 0, output.size(), stream.get()));
  write_outer_header<<<1, 1, 0, stream.get()>>>(static_cast<std::uint8_t*>(output.data()),
                                                bucket_count);
  write_bucket_headers<<<static_cast<std::uint32_t>(bucket_count),
                         threads_per_block,
                         0,
                         stream.get()>>>(static_cast<std::uint8_t*>(output.data()),
                                         buckets.data(),
                                         bucket_offsets.data(),
                                         containers.data(),
                                         bucket_count);
  serialize_containers<<<static_cast<std::uint32_t>(container_count),
                         threads_per_block,
                         0,
                         stream.get()>>>(static_cast<std::uint8_t*>(output.data()),
                                         sources.data(),
                                         static_cast<std::uint8_t const*>(existing_data.data()),
                                         new_values.data(),
                                         containers.data(),
                                         container_count);
  CUDF_CUDA_TRY(cudaPeekAtLastError());

  auto const cardinality = thrust::transform_reduce(
    policy,
    containers.begin(),
    containers.end(),
    [] __device__(container_descriptor const& container) -> std::uint64_t {
      return static_cast<std::uint64_t>(container.cardinality);
    },
    std::uint64_t{0},
    thrust::plus<std::uint64_t>{});
  return serialized_roaring_bitmap{std::move(output), cardinality};
}

}  // namespace spark_rapids_jni
