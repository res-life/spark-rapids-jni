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
#include "timezones.cpp"

#include <cassert>
#include <cstring>

#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

using timestamp_col =
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep>;
using micros_col =
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep>;
struct DateTimeParserTest : public cudf::test::BaseFixture {
protected:
  void SetUp() override {
    transitions = make_transitions_table();
  }
  std::unique_ptr<cudf::table> transitions;
  std::unique_ptr<cudf::column> tz_indices;
  std::unique_ptr<cudf::column> special_tz;

private:
  std::unique_ptr<cudf::table> make_transitions_table()
  {
    auto instants_from_utc_col = int64_col({int64_min,
                                            int64_min,
                                            -1585904400L,
                                            -933667200L,
                                            -922093200L,
                                            -908870400L,
                                            -888829200L,
                                            -650019600L,
                                            515527200L,
                                            558464400L,
                                            684867600L});
    auto instants_to_utc_col   = int64_col({int64_min,
                                            int64_min,
                                            -1585904400L,
                                            -933634800L,
                                            -922064400L,
                                            -908838000L,
                                            -888796800L,
                                            -649990800L,
                                            515559600L,
                                            558493200L,
                                            684896400L});
    auto utc_offsets_col =
        int32_col({18000, 29143, 28800, 32400, 28800, 32400, 28800, 28800, 32400, 28800, 28800});
    auto struct_column = cudf::test::structs_column_wrapper{
        {instants_from_utc_col, instants_to_utc_col, utc_offsets_col},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
    auto offsets       = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 11};
    auto list_nullmask = std::vector<bool>(1, 1);
    auto [null_mask, null_count] =
        cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
    auto list_column = cudf::make_lists_column(
        2, offsets.release(), struct_column.release(), null_count, std::move(null_mask));
    auto columns = std::vector<std::unique_ptr<cudf::column>>{};
    columns.push_back(std::move(list_column));
    return std::make_unique<cudf::table>(std::move(columns));
  }
};

TEST_F(DateTimeParserTest, ParseTimestamp)
{
  auto ts_strings = cudf::test::strings_column_wrapper(
    {
      "2023",
      " 2023 ",
      " 2023-11 ",
      " 2023-11-5 ",
      " 2023-11-05 3:04:55   ",
      " 2023-11-05T03:4:55   ",
      " 2023-11-05T3:4:55   ",
      "  2023-11-5T3:4:55.",
      "  2023-11-5T3:4:55.Iran",
      "  2023-11-5T3:4:55.1 ",
      "  2023-11-5T3:4:55.1Iran",
      "  2023-11-05T03:04:55.123456  ",
      "  2023-11-05T03:04:55.123456Iran  ",
      " 222222 ",
      " ",  // invalid
      "",   // invalid
      "1-"  // invalid

    },
    {

      0,  // null bit
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1

    });
  auto d_2023_1_1           = (2023L * 365L * 86400L + 1 * 30L * 86400L + 1 * 86400L) * 1000000L;
  auto d_2023_11_1          = (2023L * 365L * 86400L + 11 * 30L * 86400L + 1 * 86400L) * 1000000L;
  auto d_2023_11_5          = (2023L * 365L * 86400L + 11L * 30L * 86400L + 5L * 86400L) * 1000000L;
  auto t_3_4_55             = (3L * 3600L + 4L * 60L + 55L) * 1000000L;
  auto d_2023_11_5_t_3_4_55 = d_2023_11_5 + t_3_4_55;
  auto ts_col               = timestamp_col(
    {

      0L,
      d_2023_1_1,
      d_2023_11_1,
      d_2023_11_5,
      d_2023_11_5_t_3_4_55,
      d_2023_11_5_t_3_4_55,
      d_2023_11_5_t_3_4_55,
      d_2023_11_5_t_3_4_55,
      d_2023_11_5_t_3_4_55,
      d_2023_11_5_t_3_4_55 + 100000,
      d_2023_11_5_t_3_4_55 + 100000,
      d_2023_11_5_t_3_4_55 + 123456,
      d_2023_11_5_t_3_4_55 + 123456,
      (222222L * 365L * 86400L + 1 * 30L * 86400L + 1 * 86400L) * 1000000L,
      0L,
      0L,
      0L

    },
    {
      0,  // null bit
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,  // null bit
      0,  // null bit
      0   // null bit
    });
  auto ret =
    spark_rapids_jni::string_to_timestamp(cudf::strings_column_view(ts_strings), "Z", true, false);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(ts_col, *(ret.first));
  assert(ret.second == true);

  ts_strings = cudf::test::strings_column_wrapper(
    {

      "invalid"

    },
    {

      1

    });
  ts_col = timestamp_col(
    {

      0L

    },
    {0

    });
  ret =
    spark_rapids_jni::string_to_timestamp(cudf::strings_column_view(ts_strings), "Z", true, true);
  assert(ret.first == nullptr);
  assert(ret.second == false);

  ts_strings = cudf::test::strings_column_wrapper(
    {

      " Epoch  ", " NOW ", "  today  ", "  tomoRRow  ", "  yesTERday  "

    },
    {

      1, 1, 1, 1, 1

    });
  ts_col = timestamp_col(
    {// Temp implement: epoch -> 111, now -> 222, ... , yesterday -> 555
     111L,
     222L,
     333L,
     444L,
     555L

    },
    {1, 1, 1, 1, 1

    });
  ret =
    spark_rapids_jni::string_to_timestamp(cudf::strings_column_view(ts_strings), "Z", true, true);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(ts_col, *(ret.first));
  assert(ret.second == true);
}
