/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <vector>

namespace xllm::kernel::npu::megakernel {

struct SingleActiveExpert {
  int64_t expert_id = -1;
  uint32_t start_offset = 0;
  uint32_t token_count = 0;
};

bool group_list_to_token_counts(const std::vector<int64_t>& group_list,
                                int64_t expanded_token_count,
                                int64_t num_experts,
                                std::vector<uint32_t>* expert_token_counts);

bool find_single_active_expert(const std::vector<uint32_t>& expert_token_counts,
                               SingleActiveExpert* active_expert);

}  // namespace xllm::kernel::npu::megakernel
