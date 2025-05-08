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

#include "cast_date.hpp"
#include "utils.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/offsets_iterator.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/error.hpp>

namespace spark_rapids_jni {

namespace {

/**
 * @brief Functor for converting a date to a string
 *
 * This is designed to be used with make_strings_children
 */
struct date_to_staring_fn {
  cudf::column_device_view d_dates;
  cudf::size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ int32_t get_str_len_for_year(int32_t year) const
  {
    int32_t len = 0;
    if (year < 0) {
      // Need '-' char for negative years
      ++len;
      year = -year;
    }

    // Count digits in year
    int digits = int_str_converter::get_digits(year);
    if (digits > 4) {
      len += digits;
    } else {
      // year 0-999 will pad with zeros in the leading, e.g.: 0001, 0099
      len += 4;
    }

    return len;
  }

  __device__ int32_t get_days(cudf::timestamp_D const ts_days) const
  {
    return *reinterpret_cast<int32_t const*>(&ts_days);
  }

  __device__ cudf::size_type compute_output_size(cudf::timestamp_D const ts_days) const
  {
    auto const days = get_days(ts_days);
    auto const date = date_time_utils::to_date(days);

    // Has the following format:
    //   -123456-01-01
    //   -1900-01-01
    //   0000-01-01
    //   0000-01-01

    // 6 is the length of the string "-MM-DD"
    return get_str_len_for_year(date.year) + 6;
  }

  __device__ void date_to_string(cudf::timestamp_D const ts_days, char* ptr) const
  {
    auto const days = get_days(ts_days);
    auto const date = date_time_utils::to_date(days);

    // write year
    int year = date.year;
    if (year < 0) {
      *ptr++ = '-';
      year   = -year;
    }
    int positive_year_len = get_str_len_for_year(year);
    ptr                   = int_str_converter::int2str(ptr, positive_year_len, year);

    // write month
    *ptr++ = '-';
    ptr    = int_str_converter::int2str(ptr, 2, date.month);

    // write day
    *ptr++ = '-';
    int_str_converter::int2str(ptr, 2, date.day);
  }

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_dates.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const ts_days = d_dates.element<cudf::timestamp_D>(idx);
    if (d_chars) {
      date_to_string(ts_days, d_chars + d_offsets[idx]);
    } else {
      d_sizes[idx] = compute_output_size(ts_days);
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> cast_date_to_string(cudf::column_view const& input,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::TIMESTAMP_DAYS, "Input must be date type.");
  if (input.is_empty()) return cudf::make_empty_column(cudf::type_id::STRING);
  auto d_dates                 = cudf::column_device_view::create(input, stream);
  auto [offsets_column, chars] = cudf::strings::detail::make_strings_children(
    date_to_staring_fn{*d_dates}, input.size(), stream, mr);

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets_column),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::copy_bitmask(input, stream, mr));
}

}  // namespace spark_rapids_jni
