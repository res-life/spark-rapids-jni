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
#include <cudf/search.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
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
  const char* default_time_zone;
  lists_column_device_view const transitions;
  column_device_view const tz_indices;
  size_type default_time_zone_char_len;
  bool allow_time_zone;
  bool allow_special_expressions;

  static constexpr int const num_special_datetime = 5;
  static constexpr char const* special_dt_expr[5]{"epoch", "now", "today", "tomorrow", "yesterday"};
  static constexpr int const special_dt_len[5]{5, 3, 5, 8, 9};

  __device__ thrust::tuple<cudf::timestamp_us, bool> operator()(const cudf::size_type& idx) const
  {
    auto const d_str = d_strings.element<cudf::string_view>(idx);

    timestamp_components ts_comp{};
    auto tz_string = cudf::string_view{default_time_zone, default_time_zone_char_len};

    if (!parse_string_to_timestamp_us(&ts_comp, tz_string, d_str)) {
      return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}}, false);
    }

    /**
    if (default_time_zone_char_len == 0) {
      // invoke from `string_to_timestamp_without_time_zone`
      if (just_time || !allow_time_zone && tz.has_value()) {
        return thrust::make_tuple(error_us, false);
      } else {
        return to_utc_timestamp(components);
      }
    } else {
      // invoke from `string_to_timestamp`
      if (just_time) {
        // Update here to support the following format:
        //   `[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
        //   `T[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
        // by set local date in a time zone: year-month-day.
        // Above 2 formats are time zone related, Spark uses LocalDate.now(zoneId)

        // do not support currently
        return thrust::make_tuple(error_us, false);
      } else {
        return to_utc_timestamp(components, time_zone);
      }
    }
    */
  }

  __device__ inline auto parse_time_zone(cudf::string_view const &tz_string) const
  {
    // TODO: replace with more efficient approach (such as binary search or prefix tree)
    auto predicate = [&tz = tz_indices, &tz_string] __device__(auto const i) {
      return tz.element<string_view>(i) == tz_string;
    };
    return thrust::find_if(thrust::seq,
                           thrust::make_counting_iterator(0),
                           thrust::make_counting_iterator(tz_indices.size()),
                           predicate);
  }

  __device__ inline size_type extract_timezone_offset(int64_t loose_epoch_second, size_type tz_index) const
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

    return utc_offsets.element<int32_t>(list_offset);
  }

  __device__ inline int64_t compute_loose_epoch_sec(timestamp_components const& ts)
  {

  }

  __device__ inline int64_t compute_epoch_us(timestamp_components const& ts)
  {
    auto const ymd =  // chrono class handles the leap year calculations for us
        cuda::std::chrono::year_month_day(
            cuda::std::chrono::year{ts.year},
            cuda::std::chrono::month{static_cast<uint32_t>(ts.month)},
            cuda::std::chrono::day{static_cast<uint32_t>(ts.day)});
    auto days = cuda::std::chrono::sys_days(ymd).time_since_epoch().count();

    int64_t timestamp = (days * 24L * 3600L) + (ts.hour * 3600L) + (ts.minute * 60L) + ts.second;


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
      cudf::string_view &parsed_tz,
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
    if (allow_special_expressions) {
      for (size_type i = 0; i < num_special_datetime; i++) {
        auto const *pos = special_dt_expr[i];
        auto const *end = special_dt_expr[i] + special_dt_len[i];
        if (equals_ascii_ignore_case(curr_ptr, end_ptr, pos, end)) {
          parsed_tz = cudf::string_view(special_dt_expr[i], special_dt_len[i]);
          return true;
        }
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
            parsed_tz = cudf::string_view(bytes + j, bytes_length - j);
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
  std::string_view const& default_time_zone,
  bool allow_time_zone,
  bool allow_special_expressions,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto output_timestamp =
    cudf::make_timestamp_column(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS},
                                input.size(),
                                cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                input.null_count(),
                                stream,
                                mr);
  // record which string is failed to parse.
  auto output_bool =
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                  input.size(),
                                  cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                  input.null_count(),
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
                              default_time_zone.data(),
                              static_cast<cudf::size_type>(default_time_zone.size()),
                              allow_time_zone,
                              allow_special_expressions});

  return std::make_pair(std::move(output_timestamp), std::move(output_bool));
}

/**
 * Set the null mask of timestamp column according to the validity column.
 */
void update_bitmask(cudf::column& timestamp_column,
                    cudf::column const& validity_column,
                    rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource* mr)
{
  auto const& ts_view    = timestamp_column.view();
  auto const& valid_view = validity_column.view();
  std::vector<cudf::bitmask_type const*> masks;
  std::vector<cudf::size_type> offsets;
  if (timestamp_column.nullable()) {
    masks.push_back(ts_view.null_mask());
    offsets.push_back(ts_view.offset());
  }

  // generate bitmask from `validity_column`
  auto [valid_bitmask, valid_null_count] = cudf::detail::valid_if(
    valid_view.begin<bool>(), valid_view.end<bool>(), thrust::identity<bool>{}, stream, mr);

  masks.push_back(static_cast<cudf::bitmask_type*>(valid_bitmask.data()));
  offsets.push_back(0);

  // merge 2 bitmasks
  auto [null_mask, null_count] =
    cudf::detail::bitmask_and(masks, offsets, timestamp_column.size(), stream, mr);

  timestamp_column.set_null_mask(null_mask, null_count);
}

/**
 * Parse string column with time zone to timestamp column,
 * Returns a pair of timestamp column and a bool indicates whether successed.
 */
std::pair<std::unique_ptr<cudf::column>, bool> parse_string_to_timestamp(
  cudf::strings_column_view const& input,
  std::string_view const& default_time_zone,
  bool allow_time_zone,
  bool allow_special_expressions,
  bool ansi_mode)
{
  auto timestamp_type = cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
  if (input.size() == 0) {
    return std::make_pair(cudf::make_empty_column(timestamp_type.id()), true);
  }

  auto const stream = cudf::get_default_stream();
  auto const mr     = rmm::mr::get_current_device_resource();
  auto [timestamp_column, validity_column] =
    to_timestamp(input, default_time_zone, allow_time_zone, allow_special_expressions, stream, mr);

  if (ansi_mode) {
    // create scalar, value is false, is_valid is true
    cudf::numeric_scalar<bool> false_scalar{false, true, stream, mr};
    if (cudf::contains(*validity_column, false_scalar, stream)) {
      // has invalid value in validity column under ansi mode
      return std::make_pair(nullptr, false);
    } else {
      update_bitmask(*timestamp_column, *validity_column, stream, mr);
      return std::make_pair(std::move(timestamp_column), true);
    }
  } else {
    update_bitmask(*timestamp_column, *validity_column, stream, mr);
    return std::make_pair(std::move(timestamp_column), true);
  }
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
