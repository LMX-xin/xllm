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

#include "kernels/npu/megakernel/moe_single_expert_adapter.h"

#include <cstddef>
#include <limits>

namespace xllm::kernel::npu::megakernel {

bool group_list_to_token_counts(const std::vector<int64_t>& group_list,
                                int64_t expanded_token_count,
                                int64_t num_experts,
                                std::vector<uint32_t>* expert_token_counts) {
  if (expert_token_counts == nullptr || expanded_token_count < 0 ||
      num_experts < 0 ||
      group_list.size() != static_cast<size_t>(num_experts) ||
      expanded_token_count > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  std::vector<uint32_t> counts;
  counts.reserve(group_list.size());

  int64_t total = 0;
  for (const int64_t count : group_list) {
    if (count < 0 || count > std::numeric_limits<uint32_t>::max() ||
        total + count > expanded_token_count) {
      return false;
    }
    counts.push_back(static_cast<uint32_t>(count));
    total += count;
  }

  if (total != expanded_token_count) {
    return false;
  }

  *expert_token_counts = std::move(counts);
  return true;
}

bool find_single_active_expert(const std::vector<uint32_t>& expert_token_counts,
                               SingleActiveExpert* active_expert) {
  if (active_expert == nullptr) {
    return false;
  }

  SingleActiveExpert candidate;
  uint32_t offset = 0;
  int64_t active_count = 0;
  for (size_t expert_id = 0; expert_id < expert_token_counts.size();
       ++expert_id) {
    const uint32_t token_count = expert_token_counts[expert_id];
    if (token_count > 0) {
      ++active_count;
      candidate.expert_id = static_cast<int64_t>(expert_id);
      candidate.start_offset = offset;
      candidate.token_count = token_count;
    }
    offset += token_count;
  }

  if (active_count != 1 || candidate.start_offset != 0) {
    return false;
  }

  *active_expert = candidate;
  return true;
}

}  // namespace xllm::kernel::npu::megakernel
