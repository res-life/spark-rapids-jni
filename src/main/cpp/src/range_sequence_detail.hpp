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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>

namespace spark_rapids_jni::detail {

// End-to-end L4 measurements include host validation/upload and output generation. Long ranges
// favor the packed host-scan metadata during generation, so range count alone is insufficient.
inline constexpr std::size_t range_sequence_device_scan_threshold      = 65'536;
inline constexpr std::size_t range_sequence_device_scan_rows_per_range = 64;

}  // namespace spark_rapids_jni::detail
