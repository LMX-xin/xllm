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

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace xllm::kernel::npu::megakernel {
namespace {

TEST(MoeSingleExpertAdapterTest, ConvertsPerExpertGroupListToCounts) {
  std::vector<uint32_t> counts;

  const bool ok =
      group_list_to_token_counts(std::vector<int64_t>{2, 0, 3}, 5, 3, &counts);

  EXPECT_TRUE(ok);
  EXPECT_EQ(counts, (std::vector<uint32_t>{2, 0, 3}));
}

TEST(MoeSingleExpertAdapterTest, RejectsInvalidPerExpertGroupList) {
  std::vector<uint32_t> counts;

  const bool ok =
      group_list_to_token_counts(std::vector<int64_t>{2, -1, 4}, 5, 3, &counts);

  EXPECT_FALSE(ok);
  EXPECT_TRUE(counts.empty());
}

TEST(MoeSingleExpertAdapterTest, FindsOnlySingleActiveExpertAtZeroStart) {
  SingleActiveExpert active;

  const bool ok =
      find_single_active_expert(std::vector<uint32_t>{4, 0, 0}, &active);

  EXPECT_TRUE(ok);
  EXPECT_EQ(active.expert_id, 0);
  EXPECT_EQ(active.start_offset, 0);
  EXPECT_EQ(active.token_count, 4);
}

TEST(MoeSingleExpertAdapterTest, FindsSingleActiveExpertAfterEmptyPrefix) {
  SingleActiveExpert active;

  const bool ok =
      find_single_active_expert(std::vector<uint32_t>{0, 4, 0}, &active);

  EXPECT_TRUE(ok);
  EXPECT_EQ(active.expert_id, 1);
  EXPECT_EQ(active.start_offset, 0);
  EXPECT_EQ(active.token_count, 4);
}

TEST(MoeSingleExpertAdapterTest, RejectsMultipleActiveExperts) {
  SingleActiveExpert active;

  const bool ok =
      find_single_active_expert(std::vector<uint32_t>{2, 0, 3}, &active);

  EXPECT_FALSE(ok);
}

}  // namespace
}  // namespace xllm::kernel::npu::megakernel
