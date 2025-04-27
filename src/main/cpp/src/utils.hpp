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

namespace spark_rapids_jni {

/**
 * Represents timestamp with microsecond accuracy.
 */
struct ts_segments {
  __device__ ts_segments()
    : year(1970), month(1), day(1), hour(0), minute(0), second(0), microseconds(0)
  {
  }

  __device__ bool is_valid_ts() { return is_valid_date() && is_valid_time(); }

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

  __device__ bool is_leap_year() { return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0); }

  __device__ int days_in_month()
  {
    if (month == 2) { return is_leap_year() ? 29 : 28; }
    return (month == 4 || month == 6 || month == 9 || month == 11) ? 30 : 31;
  }

  __device__ bool is_valid_date()
  {
    if (month < 1 || month > 12 || day < 1) {
      return false;  // Invalid month or day
    }

    // Check for leap year, February has 29 days in leap year
    if (month == 2 && is_leap_year()) { return (day <= 29); }

    // Check against the standard days
    return (day <= days_in_month());
  }

  __device__ bool is_valid_time()
  {
    return (hour >= 0 && hour < 24) && (minute >= 0 && minute < 60) &&
           (second >= 0 && second < 60) && (microseconds >= 0 && microseconds < 1000000);
  }
};

}  // namespace spark_rapids_jni
