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

#include "kernels/npu/megakernel/dense_ffn_runner.h"

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/torch_npu.h>

#include "kernels/npu/megakernel/moe_single_expert_adapter.h"

namespace xllm::kernel::npu::megakernel {
namespace {

class DenseFfnRunnerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

torch::Tensor swiglu(const torch::Tensor& input) {
  return at_npu::native::custom_ops::npu_swiglu(input);
}

void expect_allclose(const torch::Tensor& actual,
                     const torch::Tensor& expected) {
  const torch::Tensor diff =
      (actual.to(torch::kFloat32) - expected.to(torch::kFloat32)).abs();
  const float max_diff = diff.max().item<float>();
  const int64_t max_index = diff.argmax().item<int64_t>();

  EXPECT_TRUE(torch::allclose(actual, expected, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max_diff=" << max_diff << ", max_index=" << max_index;
}

TEST_F(DenseFfnRunnerTest, MatchesTorchNpuDenseFfnReference) {
  const torch::Device device("npu:0");
  const torch::TensorOptions fp16_options =
      torch::TensorOptions().dtype(torch::kHalf).device(device);
  const int64_t token_count = 2;
  const int64_t hidden_size = 64;
  const int64_t intermediate_size = 128;

  torch::manual_seed(20260529);
  const torch::Tensor x =
      torch::randn({token_count, hidden_size}, fp16_options);
  const torch::Tensor w1 =
      torch::randn({hidden_size, intermediate_size * 2}, fp16_options);
  const torch::Tensor w2 =
      torch::randn({intermediate_size, hidden_size}, fp16_options);

  DenseFfnRunParams params;
  params.x = x;
  params.w1 = w1;
  params.w2 = w2;
  params.hidden_size = hidden_size;
  params.intermediate_size = intermediate_size;

  torch::Tensor actual;
  ASSERT_TRUE(try_run_dense_ffn_fp16_no_quant(params, &actual));

  const torch::Tensor expected =
      torch::matmul(swiglu(torch::matmul(x, w1)), w2);
  expect_allclose(actual, expected);
}

TEST_F(DenseFfnRunnerTest, MatchesSingleActiveExpertAdapterReference) {
  const torch::Device device("npu:0");
  const torch::TensorOptions fp16_options =
      torch::TensorOptions().dtype(torch::kHalf).device(device);
  const int64_t token_count = 2;
  const int64_t hidden_size = 64;
  const int64_t intermediate_size = 128;
  const int64_t num_experts = 3;

  std::vector<uint32_t> expert_token_counts;
  ASSERT_TRUE(
      group_list_to_token_counts(std::vector<int64_t>{0, token_count, 0},
                                 token_count,
                                 num_experts,
                                 &expert_token_counts));

  SingleActiveExpert active_expert;
  ASSERT_TRUE(find_single_active_expert(expert_token_counts, &active_expert));
  ASSERT_EQ(active_expert.expert_id, 1);

  torch::manual_seed(20260530);
  const torch::Tensor expand_hidden_states =
      torch::randn({token_count, hidden_size}, fp16_options);
  const torch::Tensor w13 = torch::randn(
      {num_experts, hidden_size, intermediate_size * 2}, fp16_options);
  const torch::Tensor w2 =
      torch::randn({num_experts, intermediate_size, hidden_size}, fp16_options);

  DenseFfnRunParams params;
  params.x = expand_hidden_states.slice(
      0,
      active_expert.start_offset,
      active_expert.start_offset + active_expert.token_count);
  params.w1 = w13.select(0, active_expert.expert_id);
  params.w2 = w2.select(0, active_expert.expert_id);
  params.hidden_size = hidden_size;
  params.intermediate_size = intermediate_size;

  torch::Tensor actual;
  ASSERT_TRUE(try_run_dense_ffn_fp16_no_quant(params, &actual));

  const torch::Tensor expected =
      torch::matmul(swiglu(torch::matmul(params.x, params.w1)), params.w2);
  expect_allclose(actual, expected);
}

}  // namespace
}  // namespace xllm::kernel::npu::megakernel
