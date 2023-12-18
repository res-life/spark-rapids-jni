/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "datetime_parser.hpp"

#include <vector>

#include <iostream>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>

#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

using column                   = cudf::column;
using column_device_view       = cudf::column_device_view;
using column_view              = cudf::column_view;
using lists_column_device_view = cudf::detail::lists_column_device_view;
using size_type                = cudf::size_type;
using string_view              = cudf::string_view;
using struct_view              = cudf::struct_view;
using table_view               = cudf::table_view;

namespace {

/**
 * Represents local date time in a time zone.
 */
struct timestamp_components {
  int32_t year;  // max 6 digits
  int8_t month;
  int8_t day;
  int8_t hour;
  int8_t minute;
  int8_t second;
  int32_t microseconds;
};

/**
 * Convert a local time in a time zone to a UTC timestamp
 */
__device__ __host__ thrust::tuple<cudf::timestamp_us, bool> to_utc_timestamp(
  timestamp_components const& components, cudf::string_view const& time_zone)
{
  // TODO replace the following fake implementation
  long seconds = components.year * 365L * 86400L + components.month * 30L * 86400L +
                 components.day * 86400L + components.hour * 3600L + components.minute * 60L +
                 components.second;
  long us = seconds * 1000000L + components.microseconds;
  return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{us}}, true);
}

/**
 * Convert a local time in a time zone to a UTC timestamp
 */
__device__ __host__ thrust::tuple<cudf::timestamp_us, bool> to_utc_timestamp(
  timestamp_components const& components)
{
  return to_utc_timestamp(components, cudf::string_view("UTC", 3));
}

/**
 * Is white space
 */
__device__ __host__ inline bool is_whitespace(const char chr)
{
  switch (chr) {
    case ' ':
    case '\r':
    case '\t':
    case '\n': return true;
    default: return false;
  }
}

/**
 * Whether the given two strings are equal,
 * used to compare special timestamp strings ignoring case:
 *   "epoch", "now", "today", "yesterday", "tomorrow"
 * the expect string should be lower-case a-z chars
 */
__device__ inline bool equals_ascii_ignore_case(char const *actual_begin,
                                                char const *actual_end,
                                                char const *expect_begin,
                                                char const *expect_end) {
  if (actual_end - actual_begin != expect_end - expect_begin) { return false; }

  while (expect_begin < expect_end) {
    // the diff between upper case and lower case for a same char is 32
    if (*actual_begin != *expect_begin && *actual_begin != (*expect_begin - 32)) { return false; }
    actual_begin++;
    expect_begin++;
  }
  return true;
}

/**
 * Ported from Spark
 */
__device__ __host__ bool is_valid_digits(int segment, int digits)
{
  // A Long is able to represent a timestamp within [+-]200 thousand years
  const int constexpr maxDigitsYear = 6;
  // For the nanosecond part, more than 6 digits is allowed, but will be truncated.
  return segment == 6 || (segment == 0 && digits >= 4 && digits <= maxDigitsYear) ||
     // For the zoneId segment(7), it's could be zero digits when it's a region-based zone ID
     (segment == 7 && digits <= 2) ||
     (segment != 0 && segment != 6 && segment != 7 && digits > 0 && digits <= 2);
}


struct parse_timestamp_string_fn {
  column_device_view const d_strings;
  // The list column of transitions to figure out the correct offset
  // to adjust the timestamp. The type of the values in this column is
  // LIST<STRUCT<utcInstant: int64, tzInstant: int64, utcOffset: int32, looseTzInstant: int64>>.
  lists_column_device_view const transitions;
  column_device_view const tz_indices;
  column_device_view const special_datetime_names;
  size_type default_tz_index;

  __device__ thrust::tuple<cudf::timestamp_us, bool> operator()(const cudf::size_type& idx) const
  {
    if (!d_strings.is_valid(idx)) {
      return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}}, false);
    }

    auto const d_str = d_strings.element<cudf::string_view>(idx);

    timestamp_components ts_comp{};
    char const * tz_lit_ptr = nullptr;
    size_type tz_lit_len = 0;
    if (!parse_string_to_timestamp_us(&ts_comp, &tz_lit_ptr, &tz_lit_len, d_str)) {
      return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}}, false);
    }

    int tz_index;
    if (tz_lit_ptr == nullptr) {
      tz_index = default_tz_index;
    }
    else {
      tz_index = parse_time_zone(string_view(tz_lit_ptr, tz_lit_len));
      if (tz_index == tz_indices.size()) {
        return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}}, false);
      }
    }
    auto const loose_epoch = compute_loose_epoch_s(ts_comp);
    auto const tz_offset = extract_timezone_offset(loose_epoch, tz_index);
    auto const ts_unaligned = compute_epoch_us(ts_comp);

    return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{ts_unaligned - tz_offset}}, true);
  }

  __device__ inline int parse_time_zone(string_view const &tz_lit) const
  {
    // TODO: replace with more efficient approach (such as binary search or prefix tree)
    auto predicate = [&tz = tz_indices, &tz_lit] __device__(auto const i) {
      return tz.element<string_view>(i) == tz_lit;
    };
    auto ret = thrust::find_if(thrust::seq,
                               thrust::make_counting_iterator(0),
                               thrust::make_counting_iterator(tz_indices.size()),
                               predicate);
    return *ret;
  }

  __device__ inline int64_t extract_timezone_offset(int64_t loose_epoch_second, size_type tz_index) const
  {
    auto const utc_offsets  = transitions.child().child(2);
    auto const loose_instants = transitions.child().child(3);

    auto const tz_transitions = cudf::list_device_view{transitions, tz_index};
    auto const list_size      = tz_transitions.size();

    auto const transition_times = cudf::device_span<int64_t const>(
        loose_instants.data<int64_t>() + tz_transitions.element_offset(0),
        static_cast<size_t>(list_size));

    auto const it = thrust::upper_bound(
        thrust::seq, transition_times.begin(), transition_times.end(), loose_epoch_second);
    auto const idx         = static_cast<size_type>(thrust::distance(transition_times.begin(), it));
    auto const list_offset = tz_transitions.element_offset(idx - 1);

    return static_cast<int64_t>(utc_offsets.element<int32_t>(list_offset));
  }

  __device__ inline int64_t compute_loose_epoch_s(timestamp_components const& ts) const
  {
    return (ts.year * 400 + (ts.month - 1) * 31 + ts.day - 1) * 86400L + ts.hour * 3600L + ts.minute * 60L + ts.second;
  }

  __device__ inline int64_t compute_epoch_us(timestamp_components const& ts) const
  {
    auto const ymd =  // chrono class handles the leap year calculations for us
        cuda::std::chrono::year_month_day(
            cuda::std::chrono::year{ts.year},
            cuda::std::chrono::month{static_cast<uint32_t>(ts.month)},
            cuda::std::chrono::day{static_cast<uint32_t>(ts.day)});
    auto days = cuda::std::chrono::sys_days(ymd).time_since_epoch().count();

    int64_t timestamp_s = (days * 24L * 3600L) + (ts.hour * 3600L) + (ts.minute * 60L) + ts.second;

    return timestamp_s * 1000000L + ts.microseconds;
  }

  /**
   * Ported from Spark:
   *   https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
   *   org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L394
   *
   * Parse a string with time zone to a timestamp.
   * The bool in the returned tuple is false if the parse failed.
   */
  __device__ inline bool parse_string_to_timestamp_us(
      timestamp_components *ts_comp,
      char const **parsed_tz_ptr,
      size_type *parsed_tz_length,
      cudf::string_view const &timestamp_str) const {

    if (timestamp_str.empty()) { return false; }

    const char *curr_ptr = timestamp_str.data();
    const char *end_ptr = curr_ptr + timestamp_str.size_bytes();

    // trim left
    while (curr_ptr < end_ptr && is_whitespace(*curr_ptr)) {
      ++curr_ptr;
    }
    // trim right
    while (curr_ptr < end_ptr - 1 && is_whitespace(*(end_ptr - 1))) {
      --end_ptr;
    }

    // special strings: epoch, now, today, yesterday, tomorrow
    for (size_type i = 0; i < special_datetime_names.size(); i++) {
      auto const& ref = special_datetime_names.element<string_view>(i);
      if (equals_ascii_ignore_case(curr_ptr, end_ptr, ref.data(), ref.data() + ref.size_bytes())) {
        *parsed_tz_ptr = ref.data();
        *parsed_tz_length = ref.size_bytes();
        return true;
      }
    }

    if (curr_ptr == end_ptr) { return false; }

    const char *const bytes = curr_ptr;
    const size_type bytes_length = end_ptr - curr_ptr;

    int segments[] = {1, 1, 1, 0, 0, 0, 0, 0, 0};
    int segments_len = 9;
    int i = 0;
    int current_segment_value = 0;
    int current_segment_digits = 0;
    size_t j = 0;
    int digits_milli = 0;
    bool just_time = false;
    thrust::optional<int> year_sign;
    if ('-' == bytes[j] || '+' == bytes[j]) {
      if ('-' == bytes[j]) {
        year_sign = -1;
      } else {
        year_sign = 1;
      }
      j += 1;
    }

    while (j < bytes_length) {
      char b = bytes[j];
      int parsed_value = static_cast<int32_t>(b - '0');
      if (parsed_value < 0 || parsed_value > 9) {
        if (0 == j && 'T' == b) {
          just_time = true;
          i += 3;
        } else if (i < 2) {
          if (b == '-') {
            if (!is_valid_digits(i, current_segment_digits)) {
              return false;
            }
            segments[i] = current_segment_value;
            current_segment_value = 0;
            current_segment_digits = 0;
            i += 1;
          } else if (0 == i && ':' == b && !year_sign.has_value()) {
            just_time = true;
            if (!is_valid_digits(3, current_segment_digits)) {
              return false;
            }
            segments[3] = current_segment_value;
            current_segment_value = 0;
            current_segment_digits = 0;
            i = 4;
          } else {
            return false;
          }
        } else if (2 == i) {
          if (' ' == b || 'T' == b) {
            if (!is_valid_digits(i, current_segment_digits)) {
              return false;
            }
            segments[i] = current_segment_value;
            current_segment_value = 0;
            current_segment_digits = 0;
            i += 1;
          } else {
            return false;
          }
        } else if (3 == i || 4 == i) {
          if (':' == b) {
            if (!is_valid_digits(i, current_segment_digits)) {
              return false;
            }
            segments[i] = current_segment_value;
            current_segment_value = 0;
            current_segment_digits = 0;
            i += 1;
          } else {
            return false;
          }
        } else if (5 == i || 6 == i) {
          if ('.' == b && 5 == i) {
            if (!is_valid_digits(i, current_segment_digits)) {
              return false;
            }
            segments[i] = current_segment_value;
            current_segment_value = 0;
            current_segment_digits = 0;
            i += 1;
          } else {
            if (!is_valid_digits(i, current_segment_digits)) {
              return false;
            }
            segments[i] = current_segment_value;
            current_segment_value = 0;
            current_segment_digits = 0;
            i += 1;
            *parsed_tz_ptr = bytes + j;
            *parsed_tz_length = bytes_length - j;
            j = bytes_length - 1;
          }
          if (i == 6 && '.' != b) { i += 1; }
        } else {
          if (i < segments_len && (':' == b || ' ' == b)) {
            if (!is_valid_digits(i, current_segment_digits)) {
              return false;
            }
            segments[i] = current_segment_value;
            current_segment_value = 0;
            current_segment_digits = 0;
            i += 1;
          } else {
            return false;
          }
        }
      } else {
        if (6 == i) { digits_milli += 1; }
        // We will truncate the nanosecond part if there are more than 6 digits, which results
        // in loss of precision
        if (6 != i || current_segment_digits < 6) {
          current_segment_value = current_segment_value * 10 + parsed_value;
        }
        current_segment_digits += 1;
      }
      j += 1;
    }

    if (!is_valid_digits(i, current_segment_digits)) { return false; }
    segments[i] = current_segment_value;

    while (digits_milli < 6) {
      segments[6] *= 10;
      digits_milli += 1;
    }

    segments[0] *= year_sign.value_or(1);
    // above is ported from Spark.

    // set components
    ts_comp->year = segments[0];
    ts_comp->month = static_cast<int8_t>(segments[1]);
    ts_comp->day = static_cast<int8_t>(segments[2]);
    ts_comp->hour = static_cast<int8_t>(segments[3]);
    ts_comp->minute = static_cast<int8_t>(segments[4]);
    ts_comp->second = static_cast<int8_t>(segments[5]);
    ts_comp->microseconds = segments[6];

    return true;
  }
};

/**
 *
 * Trims and parses timestamp string column to a timestamp column and a is valid column
 *
 */
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> to_timestamp(
  cudf::strings_column_view const& input,
  table_view const& transitions,
  cudf::strings_column_view tz_indices,
  cudf::strings_column_view special_datetime_lit,
  size_type default_tz_index,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto d_strings = cudf::column_device_view::create(input.parent(), stream);

  // get the fixed transitions
  auto const ft_cdv_ptr        = column_device_view::create(transitions.column(0), stream);
  auto const fixed_transitions = lists_column_device_view{*ft_cdv_ptr};

  auto d_tz_indices = cudf::column_device_view::create(tz_indices.parent(), stream);
  auto d_special_datetime_lit = cudf::column_device_view::create(special_datetime_lit.parent(), stream);

  auto output_timestamp =
    cudf::make_timestamp_column(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS},
                                input.size(),
                                cudf::mask_state::UNALLOCATED,
                                stream,
                                mr);
  // record which string is failed to parse.
  auto output_bool =
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                  input.size(),
                                  cudf::mask_state::UNALLOCATED,
                                  stream,
                                  mr);

  thrust::transform(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(input.size()),
      thrust::make_zip_iterator(
          thrust::make_tuple(output_timestamp->mutable_view().begin<cudf::timestamp_us>(),
                             output_bool->mutable_view().begin<bool>())),
      parse_timestamp_string_fn{*d_strings,
                                fixed_transitions,
                                *d_tz_indices,
                                *d_special_datetime_lit,
                                default_tz_index});

  return std::make_pair(std::move(output_timestamp), std::move(output_bool));
}

/**
 * Parse string column with time zone to timestamp column,
 * Returns a pair of timestamp column and a bool indicates whether successed.
 */
std::pair<std::unique_ptr<cudf::column>, bool> parse_string_to_timestamp(
  cudf::strings_column_view const& input,
  table_view const& transitions,
  cudf::strings_column_view tz_indices,
  cudf::strings_column_view special_datetime_lit,
  size_type default_tz_index,
  bool ansi_mode)
{
  auto timestamp_type = cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
  if (input.size() == 0) {
    return std::make_pair(cudf::make_empty_column(timestamp_type.id()), true);
  }

  auto const stream = cudf::get_default_stream();
  auto const mr     = rmm::mr::get_current_device_resource();
  auto [timestamp_column, validity_column] =
    to_timestamp(input, transitions, tz_indices, special_datetime_lit, default_tz_index, stream, mr);

  // generate bitmask from `validity_column`
  auto validity_view = validity_column->mutable_view();
  auto [valid_bitmask, valid_null_count] = cudf::detail::valid_if(
      validity_view.begin<bool>(), validity_view.end<bool>(), thrust::identity<bool>{}, stream, mr);

  if (ansi_mode && input.null_count() < valid_null_count) {
    // has invalid value in validity column under ansi mode
    return std::make_pair(nullptr, false);
  }
  timestamp_column->set_null_mask(valid_bitmask, valid_null_count, stream);

  return std::make_pair(std::move(timestamp_column), true);
}

}  // namespace

namespace spark_rapids_jni {

/**
 * Parse string column with time zone to timestamp column,
 * Returns a pair of timestamp column and a bool indicates whether successed.
 * If does not have time zone in string, use the default time zone.
 */
std::pair<std::unique_ptr<cudf::column>, bool> string_to_timestamp(
  cudf::strings_column_view const& input,
  std::string_view const& default_time_zone,
  bool allow_special_expressions,
  bool ansi_mode)
{
  CUDF_EXPECTS(default_time_zone.size() > 0, "should specify default time zone");
  return parse_string_to_timestamp(
    input, default_time_zone, true, allow_special_expressions, ansi_mode);
}

/**
 * Parse string column with time zone to timestamp column,
 * Returns a pair of timestamp column and a bool indicates whether successed.
 * Do not use the time zone in string.
 * If allow_time_zone is false and string contains time zone, then the string is invalid.
 */
std::pair<std::unique_ptr<cudf::column>, bool> string_to_timestamp_without_time_zone(
  cudf::strings_column_view const& input,
  bool allow_time_zone,
  bool allow_special_expressions,
  bool ansi_mode)
{
  return parse_string_to_timestamp(input,
                                   std::string_view(""),  // specify empty time zone
                                   allow_time_zone,
                                   allow_special_expressions,
                                   ansi_mode);
}

}  // namespace spark_rapids_jni
