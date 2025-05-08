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

#pragma once

#include <cudf/types.hpp>

#include <cuda/std/limits>

namespace spark_rapids_jni {

struct date_segments {
  __device__ date_segments() : year(1970), month(1), day(1) {}

  __device__ date_segments(int32_t y, uint8_t m, uint8_t d) : year(y), month(m), day(d) {}

  // max 7 digits
  int32_t year;

  // 1-12
  uint8_t month;

  // 1-31; it is 29 for leap February, or 28 for regular February
  uint8_t day;
};

struct time_segments {
  __device__ time_segments() : hour(0), minute(0), second(0), microseconds(0) {}

  __device__ time_segments(uint8_t h, uint8_t m, uint8_t s, int32_t us)
    : hour(h), minute(m), second(s), microseconds(us)
  {
  }

  // 0-23
  uint8_t hour;

  // 0-59
  uint8_t minute;

  // 0-59
  uint8_t second;

  // 0-999,999, only parse 6 digits, ignore/truncate the rest digits
  int32_t microseconds;
};

struct date_time_utils {
  __device__ static bool is_leap_year(int year)
  {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
  }

  __device__ static int days_in_month(int year, int month)
  {
    if (month == 2) { return is_leap_year(year) ? 29 : 28; }
    return (month == 4 || month == 6 || month == 9 || month == 11) ? 30 : 31;
  }

  /**
   * Returns the number of days since 1970-01-01.
   * Refer to https://howardhinnant.github.io/date_algorithms.html#days_from_civil
   */
  __device__ static int64_t to_epoch_day(int year, int month, int day)
  {
    int32_t y          = year - (month <= 2);
    const int32_t era  = (y >= 0 ? y : y - 399) / 400;
    const uint32_t yoe = static_cast<uint32_t>(y - era * 400);                           // [0, 399]
    const uint32_t doy = (153 * (month > 2 ? month - 3 : month + 9) + 2) / 5 + day - 1;  // [0, 365]
    const uint32_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;  // [0, 146096]
    return era * 146097L + doe - 719468L;
  }

  /**
   * Returns year/month/day triple from epoch day.
   * Preconditions:  epoch_day is number of days since 1970-01-01 and is in the range:
   *   [numeric_limits<int>::min(), numeric_limits<int>::max()].
   * Refer to https://howardhinnant.github.io/date_algorithms.html#civil_from_days
   */
  __device__ static date_segments to_date(int32_t epoch_day) noexcept
  {
    int64_t z = static_cast<int64_t>(epoch_day);
    z += 719468;
    const int32_t era  = static_cast<int32_t>((z >= 0 ? z : z - 146096) / 146097);
    const uint32_t doe = static_cast<uint32_t>(z - era * 146097);                // [0, 146096]
    const uint32_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;  // [0, 399]
    const int32_t y    = static_cast<uint32_t>(yoe) + era * 400;
    const uint32_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);                // [0, 365]
    const uint32_t mp  = (5 * doy + 2) / 153;                                    // [0, 11]
    const uint32_t d   = doy - (153 * mp + 2) / 5 + 1;                           // [1, 31]
    const uint32_t m   = mp < 10 ? mp + 3 : mp - 9;                              // [1, 12]
    return date_segments(y + (m <= 2), static_cast<uint8_t>(m), static_cast<uint8_t>(d));
  }
};

struct size_type_iterator {
  cudf::size_type const* values;
  __device__ cudf::size_type operator()(cudf::size_type const& idx) const { return values[idx]; }
};

struct const_size_type_iterator {
  cudf::size_type const value;
  __device__ cudf::size_type operator()(cudf::size_type const&) const { return value; }
};

/**
 * Represents timestamp with microsecond accuracy.
 * Max year is six-digits, approximately [-300000, 300000]
 */
struct ts_segments {
  __device__ ts_segments()
    : year(1970), month(1), day(1), hour(0), minute(0), second(0), microseconds(0)
  {
  }

  /**
   * Is the segments of the timestamp valid?
   */
  __device__ bool is_valid_ts() const { return is_valid_date() && is_valid_time(); }

  /**
   * Get epoch day.
   * Note can not use cuda::std::chrono::year_month_day, because year is short
   * type in cuda::std::chrono, year range in cuda::std::chrono is [-32,767 ,
   * 32,767] which is too small. Spark supports 6 digits year in timestamp and 7
   * digits year in date.
   */
  __device__ int64_t to_epoch_day() const
  {
    return date_time_utils::to_epoch_day(year, month, day);
  }

  // 4-6 digits
  int32_t year;

  // 1-12
  int32_t month;

  // 1-31; it is 29 for leap February, or 28 for regular February
  int32_t day;

  // 0-23
  int32_t hour;

  // 0-59
  int32_t minute;

  // 0-59
  int32_t second;

  // 0-999999, only parse 6 digits, ignore/truncate the rest digits
  int32_t microseconds;

  __device__ bool is_valid_date() const
  {
    // Java year range: [-999,999,999, 999,999,999]
    if (year < -999999999 || year > 999999999) { return false; }

    if (month < 1 || month > 12 || day < 1) {
      return false;  // Invalid month or day
    }

    // Check for leap year, February has 29 days in leap year
    if (month == 2 && date_time_utils::is_leap_year(year)) { return (day <= 29); }

    // Check against the standard days
    return (day <= date_time_utils::days_in_month(year, month));
  }

  __device__ bool is_valid_time() const
  {
    return (hour >= 0 && hour < 24) && (minute >= 0 && minute < 60) &&
           (second >= 0 && second < 60) && (microseconds >= 0 && microseconds < 1000000);
  }
};

struct overflow_checker {
  /**
   * Check overflow for int, long addition
   */
  template <typename T>
  __device__ static bool check_signed_add_overflow(T a, T b, T& result)
  {
    if (b > 0 && a > cuda::std::numeric_limits<T>::max() - b) {
      return true;  // Overflow occurred
    }
    if (b < 0 && a < cuda::std::numeric_limits<T>::min() - b) {
      return true;  // Underflow occurred
    }
    result = a + b;
    return false;  // No overflow
  }

  /**
   * Calculate the timestamp from epoch seconds and microseconds with checking
   * overflow
   * @param seconds seconds from epoch
   * @param microseconds MUST be in range [0, 999999]
   * @param[out] result timestamp in microseconds
   * @return true if overflow occurred, flase otherwise
   */
  __device__ static bool get_timestamp_with_check(int64_t seconds,
                                                  int32_t microseconds,
                                                  int64_t& result)
  {
    constexpr int64_t micros_per_sec       = 1000000;
    constexpr int64_t max_v                = cuda::std::numeric_limits<int64_t>::max();
    constexpr int64_t min_v                = cuda::std::numeric_limits<int64_t>::min();
    constexpr int64_t max_positive_seconds = max_v / micros_per_sec;
    constexpr int64_t min_negative_seconds = min_v / micros_per_sec - 1;
    result                                 = seconds * micros_per_sec + microseconds;
    if (seconds > max_positive_seconds || seconds < min_negative_seconds) {
      return true;  // Overflow occurred
    }

    if (seconds > 0) { return microseconds > max_v - seconds * micros_per_sec; }

    if (seconds == min_negative_seconds) {
      // 224192L is calculated from 9999999999999999 / 1000000
      // BigDecimal(min_negative_seconds) * micros_per_sec - BigDecimal(min_v)
      return microseconds >= 224192L;
    }

    return false;
  }
};

struct int_str_converter {
  /**
   * @brief Get the number of digits for int value
   * @param v The input value, MUST be positive
   */
  __device__ static int32_t get_digits(int32_t v)
  {
    int digits = 0;
    while (v > 0) {
      ++digits;
      v /= 10;
    }
    return digits;
  }

  /**
   * Converts an integer to a string with zero padding (up to 9 chars).
   */
  __device__ static char* int2str(char* str, int bytes, int val)
  {
    char tmpl[9] = {'0', '0', '0', '0', '0', '0', '0', '0', '0'};
    char* ptr    = tmpl;
    while (val > 0) {
      int digit = val % 10;
      *ptr++    = '0' + digit;
      val       = val / 10;
    }
    ptr = tmpl + bytes - 1;
    while (bytes-- > 0) {
      *str++ = *ptr--;
    }
    return str;
  }
};

}  // namespace spark_rapids_jni
