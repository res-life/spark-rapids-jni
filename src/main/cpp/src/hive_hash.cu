/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "hash.cuh"
#include "hash.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>

namespace spark_rapids_jni {

namespace {

using hive_hash_value_t = int32_t;

constexpr hive_hash_value_t HIVE_HASH_FACTOR = 31;
constexpr hive_hash_value_t HIVE_INIT_HASH   = 0;

struct col_info {
  cudf::type_id type_id;
  cudf::size_type
    nested_num_children_or_basic_col_idx;  // Number of children for nested types, or column index
                                           // in `basic_cdvs` for basic types
};

hive_hash_value_t __device__ inline compute_int(int32_t key) { return key; }

hive_hash_value_t __device__ inline compute_long(int64_t key)
{
  return (static_cast<uint64_t>(key) >> 32) ^ key;
}

hive_hash_value_t __device__ inline compute_bytes(int8_t const* data, cudf::size_type const len)
{
  hive_hash_value_t ret = HIVE_INIT_HASH;
  for (auto i = 0; i < len; i++) {
    ret = ret * HIVE_HASH_FACTOR + static_cast<int32_t>(data[i]);
  }
  return ret;
}

template <typename Key>
struct hive_hash_function {
  // 'seed' is not used in 'hive_hash_function', but required by 'element_hasher'.
  constexpr hive_hash_function(uint32_t) {}

  [[nodiscard]] hive_hash_value_t __device__ inline operator()(Key const& key) const
  {
    CUDF_UNREACHABLE("Unsupported type for hive hash");
  }
};  // struct hive_hash_function

template <>
hive_hash_value_t __device__ inline hive_hash_function<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  auto const data = reinterpret_cast<int8_t const*>(key.data());
  auto const len  = key.size_bytes();
  return compute_bytes(data, len);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<bool>::operator()(bool const& key) const
{
  return compute_int(static_cast<int32_t>(key));
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<int8_t>::operator()(int8_t const& key) const
{
  return compute_int(static_cast<int32_t>(key));
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<int16_t>::operator()(
  int16_t const& key) const
{
  return compute_int(static_cast<int32_t>(key));
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<int32_t>::operator()(
  int32_t const& key) const
{
  return compute_int(key);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<int64_t>::operator()(
  int64_t const& key) const
{
  return compute_long(key);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<float>::operator()(float const& key) const
{
  auto normalized = spark_rapids_jni::normalize_nans(key);
  auto* p_int     = reinterpret_cast<int32_t const*>(&normalized);
  return compute_int(*p_int);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<double>::operator()(double const& key) const
{
  auto normalized = spark_rapids_jni::normalize_nans(key);
  auto* p_long    = reinterpret_cast<int64_t const*>(&normalized);
  return compute_long(*p_long);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<cudf::timestamp_D>::operator()(
  cudf::timestamp_D const& key) const
{
  auto* p_int = reinterpret_cast<int32_t const*>(&key);
  return compute_int(*p_int);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<cudf::timestamp_us>::operator()(
  cudf::timestamp_us const& key) const
{
  auto time_as_long            = *reinterpret_cast<int64_t const*>(&key);
  constexpr int MICRO_PER_SEC  = 1000000;
  constexpr int NANO_PER_MICRO = 1000;

  int64_t ts  = time_as_long / MICRO_PER_SEC;
  int64_t tns = (time_as_long % MICRO_PER_SEC) * NANO_PER_MICRO;

  int64_t result = ts;
  result <<= 30;
  result |= tns;

  result = (static_cast<uint64_t>(result) >> 32) ^ result;
  return static_cast<hive_hash_value_t>(result);
}

using hash_functor_t = cudf::experimental::row::hash::element_hasher<hive_hash_function, bool>;

class primitive_col_hash_functor {
 private:
  cudf::column_device_view _cdv;
  hash_functor_t _hash_functor;

 public:
  primitive_col_hash_functor(cudf::column_device_view cdv) : _cdv{cdv}, _hash_functor{true, HIVE_INIT_HASH, HIVE_INIT_HASH} {}

  /**
   * @brief Return the hash value of a row in the given table.
   *
   * @param row_index The row index to compute the hash value of
   * @return The hash value of the row
   */
  __device__ hive_hash_value_t operator()(cudf::size_type row_index) const noexcept
  {
    return cudf::type_dispatcher<cudf::experimental::dispatch_void_if_nested>(
      _cdv.type(), _hash_functor, _cdv, row_index);
  }
};

std::unique_ptr<cudf::column> calc_primitive_col(cudf::column_view const& input,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto const d_input = cudf::column_device_view::create(input);
  auto output = cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<hive_hash_value_t>()),
                                          input.size(),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  auto output_view = output->mutable_view();
  thrust::tabulate(rmm::exec_policy(stream),
                   output_view.begin<hive_hash_value_t>(),
                   output_view.end<hive_hash_value_t>(),
                   primitive_col_hash_functor(*d_input));
  return output;
}

class list_acc_functor {
 private:
  cudf::column_device_view _d_offsets;
  cudf::column_device_view _d_child_hash;

 public:
  list_acc_functor(cudf::column_device_view d_offsets, cudf::column_device_view d_child_hash)
    : _d_offsets{d_offsets}, _d_child_hash(d_child_hash)
  {
  }

  /**
   * @brief Return the hash value of a row in the given table.
   *
   * @param row_index The row index to compute the hash value of
   * @return The hash value of the row
   */
  __device__ hive_hash_value_t operator()(cudf::size_type row_index) const noexcept
  {
    int hash                       = HIVE_INIT_HASH;
    auto const child_row_idx_begin = _d_offsets.element<cudf::size_type>(row_index);
    auto const child_row_idx_end   = _d_offsets.element<cudf::size_type>(row_index + 1);
    for (size_t i = child_row_idx_begin; i < child_row_idx_end; i++) {
      auto new_hash = _d_child_hash.element<int32_t>(i);
      hash          = hash * HIVE_HASH_FACTOR + new_hash;
    }
    return hash;
  }
};

class acc_functor {
 private:
  cudf::column_device_view _d_hash;
  cudf::column_device_view _d_hash_new;

 public:
  acc_functor(cudf::column_device_view d_hash, cudf::column_device_view d_hash_new)
    : _d_hash{d_hash}, _d_hash_new(d_hash_new)
  {
  }

  /**
   * @brief Return the hash value of a row in the given table.
   *
   * @param row_index The row index to compute the hash value of
   * @return The hash value of the row
   */
  __device__ hive_hash_value_t operator()(cudf::size_type row_index) const noexcept
  {
    int hash     = _d_hash.element<hive_hash_value_t>(row_index);
    int hash_new = _d_hash_new.element<hive_hash_value_t>(row_index);
    return hash * HIVE_HASH_FACTOR + hash_new;
  }
};

// forward declaration
std::unique_ptr<cudf::column> calc_list_col(cudf::column_view const& input,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr);
std::unique_ptr<cudf::column> calc_struct_col(cudf::column_view const& input,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> calc_col(cudf::column_view const& input,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  if (input.type().id() == cudf::type_id::LIST) {
    return calc_list_col(input, stream, mr);
  } else if (input.type().id() == cudf::type_id::STRUCT) {
    return calc_struct_col(input, stream, mr);
  } else {
    return calc_primitive_col(input, stream, mr);
  }
}

std::unique_ptr<cudf::column> calc_list_col(cudf::column_view const& input,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto output = cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<hive_hash_value_t>()),
                                          input.size(),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  auto output_view = output->mutable_view();

  auto const list_col       = cudf::lists_column_view(input);
  auto const offsets        = list_col.offsets();
  auto const child          = list_col.get_sliced_child(stream);
  auto const d_offsets      = cudf::column_device_view::create(offsets);
  auto const child_output   = calc_col(child, stream, mr);
  auto const d_child_output = cudf::column_device_view::create(child_output->view());

  thrust::tabulate(rmm::exec_policy(stream),
                   output_view.begin<hive_hash_value_t>(),
                   output_view.end<hive_hash_value_t>(),
                   list_acc_functor{*d_offsets, *d_child_output});

  return output;
}

std::unique_ptr<cudf::column> calc_struct_col(cudf::column_view const& input,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{

  // std::vector<std::unique_ptr<cudf::column>> ret;
  // ret.push_back();
  
  // auto output = cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<hive_hash_value_t>()),
  //                                         input.size(),
  //                                         cudf::mask_state::UNALLOCATED,
  //                                         stream,
  //                                         mr);
  // auto output_view = output->mutable_view();


  auto const struct_col       = cudf::structs_column_view(input);
  cudf::column_view const& c0 = struct_col.get_sliced_child(0, stream);
  auto output                 = calc_col(c0, stream, mr);
  auto output_view            = output->mutable_view();
  auto d_output               = cudf::column_device_view::create(output_view);

  for (auto child_idx = 1; child_idx < input.num_children(); child_idx++) {
    cudf::column_view const& col = struct_col.get_sliced_child(child_idx, stream);
    auto hash_new                = calc_col(col, stream, mr);
    auto d_hash_new              = cudf::column_device_view::create(hash_new->view());

    thrust::tabulate(rmm::exec_policy(stream),
                     output_view.begin<hive_hash_value_t>(),
                     output_view.end<hive_hash_value_t>(),
                     acc_functor{*d_output, *d_hash_new});
  }
  return output;
}

std::unique_ptr<cudf::column> calc_table(cudf::table_view const& input,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  auto output      = calc_col(input.column(0), stream, mr);
  auto output_view = output->mutable_view();
  auto d_output    = cudf::column_device_view::create(output_view);

  for (auto i = 1; i < input.num_columns(); i++) {
    cudf::column_view const& col = input.column(i);
    auto hash_new                = calc_col(col, stream, mr);
    auto d_hash_new              = cudf::column_device_view::create(hash_new->view());
    thrust::tabulate(rmm::exec_policy(stream),
                     output_view.begin<hive_hash_value_t>(),
                     output_view.end<hive_hash_value_t>(),
                     acc_functor{*d_output, *d_hash_new});
  }
  return output;
}

void check_nested_depth(cudf::table_view const& input)
{
  using column_checker_fn_t = std::function<int(cudf::column_view const&)>;

  column_checker_fn_t get_nested_depth = [&](cudf::column_view const& col) {
    if (col.type().id() == cudf::type_id::LIST) {
      auto const child_col = cudf::lists_column_view(col).child();
      return 1 + get_nested_depth(child_col);
    }
    if (col.type().id() == cudf::type_id::STRUCT) {
      int max_child_depth = 0;
      for (auto child = col.child_begin(); child != col.child_end(); ++child) {
        max_child_depth = std::max(max_child_depth, get_nested_depth(*child));
      }
      return 1 + max_child_depth;
    }
    // Primitive type
    return 0;
  };

  for (auto i = 0; i < input.num_columns(); i++) {
    cudf::column_view const& col = input.column(i);
    CUDF_EXPECTS(get_nested_depth(col) <= MAX_STACK_DEPTH,
                 "The " + std::to_string(i) +
                   "-th column exceeds the maximum allowed nested depth. " +
                   "Current depth: " + std::to_string(get_nested_depth(col)) + ", " +
                   "Maximum allowed depth: " + std::to_string(MAX_STACK_DEPTH));
  }
}

}  // namespace

std::unique_ptr<cudf::column> hive_hash(cudf::table_view const& input,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  check_nested_depth(input);
  return calc_table(input, stream, mr);
}

}  // namespace spark_rapids_jni
