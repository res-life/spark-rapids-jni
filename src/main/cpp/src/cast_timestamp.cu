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

#include "cast_timestamp.hpp"
#include "utils.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/offsets_iterator.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/error.hpp>

#include <thrust/pair.h>

namespace spark_rapids_jni {

namespace {

/**
 * @brief Functor for converting a timestamp to a string
 *
 * This is designed to be used with make_strings_children
 */
struct timestamp_to_staring_fn {
  cudf::column_device_view d_timestamps;
  cudf::size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ time_segments get_time_segments(int64_t us_in_day) const
  {
    time_segments result;
    constexpr int64_t const NUM_US_PER_SECOND = 1'000'000L;
    result.microseconds                       = us_in_day % NUM_US_PER_SECOND;
    us_in_day /= NUM_US_PER_SECOND;
    result.second = us_in_day % 60;
    us_in_day /= 60;
    result.minute = us_in_day % 60;
    result.hour   = us_in_day /= 60;
    return result;
  }

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

  /**
   * @brief Get the len of string for microseconds, note omit the tailing zeros
   * e.g.:
   *   0 -> empty string
   *   1 -> .000001
   *   1000 -> .001
   * @param us The input value, MUST be in range of [0, 999999]
   */
  __device__ int32_t get_str_len_for_microseconds(int32_t us) const
  {
    if (us == 0) { return 0; }

    // 7 is the max length of microseconds, e.g.: ".123456", ".000001"
    int len = 7;
    while (us % 10 == 0) {
      --len;
      us /= 10;
    }
    return len;
  }

  __device__ void write_microseconds(char* str, int32_t us) const
  {
    if (us == 0) { return; }

    int len = get_str_len_for_microseconds(us);
    *str++  = '.';
    --len;

    int base = 100'000;
    for (int i = 0; i < len; i++) {
      int digit = us / base;
      us -= digit * base;
      str[i] = '0' + digit;
      base /= 10;
    }
  }

  __device__ thrust::pair<int32_t, int64_t> get_days_time(cudf::timestamp_us const ts_us) const
  {
    constexpr int64_t const NUM_OF_US_PER_DAY = 24L * 60L * 60L * 1'000'000L;
    int64_t const v                           = *reinterpret_cast<int64_t const*>(&ts_us);
    int64_t days;
    int64_t time;
    if (v < 0) {
      days = v / NUM_OF_US_PER_DAY;
      time = v % NUM_OF_US_PER_DAY;
      if (time != 0) {
        days -= 1;
        time += NUM_OF_US_PER_DAY;
      }
    } else {
      days = v / NUM_OF_US_PER_DAY;
      time = v % NUM_OF_US_PER_DAY;
    }
    return thrust::make_pair(static_cast<int32_t>(days), time);
  }

  __device__ cudf::size_type compute_output_size(cudf::timestamp_us const ts_us) const
  {
    auto const [days, time] = get_days_time(ts_us);
    auto const date         = date_time_utils::to_date(days);
    auto const time_parts   = get_time_segments(time);

    // Has the following format:
    //   -123456-01-01 01:02:03.123456
    //   -1900-01-01 01:02:03.123456
    //   -1900-01-01 01:02:03.0003
    //   0001-01-01 01:02:03
    //   0000-01-01 00:00:00

    // 15 is the length of the string "-MM-DD HH:MM:SS"
    return get_str_len_for_year(date.year) + get_str_len_for_microseconds(time_parts.microseconds) +
           15;
  }

  __device__ void timestamp_to_string(cudf::timestamp_us const ts_us, char* ptr) const
  {
    auto const [days, time] = get_days_time(ts_us);
    auto const date         = date_time_utils::to_date(days);
    auto const time_parts   = get_time_segments(time);

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
    ptr    = int_str_converter::int2str(ptr, 2, date.day);

    // write hour
    *ptr++ = ' ';
    ptr    = int_str_converter::int2str(ptr, 2, time_parts.hour);

    // write minute
    *ptr++ = ':';
    ptr    = int_str_converter::int2str(ptr, 2, time_parts.minute);

    // write second
    *ptr++ = ':';
    ptr    = int_str_converter::int2str(ptr, 2, time_parts.second);

    // write microseconds
    write_microseconds(ptr, time_parts.microseconds);
  }

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_timestamps.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const tstamp = d_timestamps.element<cudf::timestamp_us>(idx);
    if (d_chars) {
      timestamp_to_string(tstamp, d_chars + d_offsets[idx]);
    } else {
      d_sizes[idx] = compute_output_size(tstamp);
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> cast_timestamp_to_string(cudf::column_view const& input,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS,
               "Input must be timestamp type.");

  if (input.is_empty()) return cudf::make_empty_column(cudf::type_id::STRING);
  auto d_timestamps            = cudf::column_device_view::create(input, stream);
  auto [offsets_column, chars] = cudf::strings::detail::make_strings_children(
    timestamp_to_staring_fn{*d_timestamps}, input.size(), stream, mr);

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets_column),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::copy_bitmask(input, stream, mr));
}

}  // namespace spark_rapids_jni
