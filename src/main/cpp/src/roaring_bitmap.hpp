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

#pragma once

#include <cudf/column/column_view.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace spark_rapids_jni {

struct serialized_roaring_bitmap {
  rmm::device_buffer data;
  std::uint64_t cardinality;
};

/**
 * @brief Builds a portable Roaring64 bitmap from device positions and existing host payloads.
 */
serialized_roaring_bitmap build_and_serialize_roaring64(
  cudf::column_view const& positions,
  std::vector<std::span<std::byte const>> const& existing_bitmaps,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
