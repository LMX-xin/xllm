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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "kernels/npu/megakernel/dense_ffn_runner.h"
#include "layers/common/fused_moe_base.h"
#include "layers/npu_torch/fused_moe.h"
#include "platform/device.h"
#include "tests_utils.h"

namespace xllm {
namespace layer {
namespace {

constexpr const char* kMegakernelMoeEnv =
    "XLLM_ENABLE_MEGAKERNEL_SINGLE_EXPERT_MOE";

class EnvGuard final {
 public:
  explicit EnvGuard(std::string name) : name_(std::move(name)) {
    const char* value = std::getenv(name_.c_str());
    if (value != nullptr) {
      old_value_ = value;
    }
  }

  ~EnvGuard() {
    if (old_value_.has_value()) {
      setenv(name_.c_str(), old_value_.value().c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  void set(const std::string& value) {
    setenv(name_.c_str(), value.c_str(), 1);
  }

  void unset() { unsetenv(name_.c_str()); }

 private:
  std::string name_;
  std::optional<std::string> old_value_;
};

class FusedMoEMegakernelTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }

  void SetUp() override {
    options_ = torch::TensorOptions()
                   .dtype(torch::kHalf)
                   .device(torch::Device("npu:0"))
                   .requires_grad(false);
    parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  }

  ModelArgs make_model_args(int64_t num_experts,
                            int64_t hidden_size,
                            int64_t intermediate_size) const {
    ModelArgs model_args;
    model_args.hidden_size() = hidden_size;
    model_args.hidden_act() = "silu";
    model_args.n_routed_experts() = static_cast<int32_t>(num_experts);
    model_args.num_experts_per_tok() = 1;
    model_args.n_group() = static_cast<int32_t>(num_experts);
    model_args.topk_group() = static_cast<int32_t>(num_experts);
    model_args.routed_scaling_factor() = 1.0F;
    model_args.moe_intermediate_size() =
        static_cast<int32_t>(intermediate_size);
    model_args.n_shared_experts() = 0;
    model_args.norm_topk_prob() = false;
    model_args.scoring_func() = "softmax";
    model_args.topk_method() = "greedy";
    return model_args;
  }

  FusedMoE make_fused_moe(int64_t num_experts,
                          int64_t hidden_size,
                          int64_t intermediate_size) const {
    const ModelArgs model_args =
        make_model_args(num_experts, hidden_size, intermediate_size);
    const FusedMoEArgs moe_args{
        .is_gated = true,
        .enable_result_reduction = true,
        .skip_gate_load = true,
    };
    QuantArgs quant_args;
    return FusedMoE(FusedMoEImpl(
        model_args, moe_args, quant_args, parallel_args_, options_));
  }

  torch::Tensor make_small_tensor(const std::string& key,
                                  const std::vector<int64_t>& shape,
                                  float scale) const {
    torch::Tensor tensor =
        test::seeded_tensor(key, shape, torch::kFloat32, options_.device());
    tensor = (tensor - 0.5F) * scale;
    return tensor.to(options_);
  }

  std::unordered_map<std::string, torch::Tensor> make_expert_weights(
      int64_t num_experts,
      int64_t hidden_size,
      int64_t intermediate_size) const {
    std::unordered_map<std::string, torch::Tensor> weight_dict;
    for (int64_t expert_id = 0; expert_id < num_experts; ++expert_id) {
      const std::string expert_prefix =
          "experts." + std::to_string(expert_id) + ".";
      const std::string seed_prefix =
          "npu.fused_moe.megakernel.expert_" + std::to_string(expert_id);
      weight_dict[expert_prefix + "gate_proj.weight"] = make_small_tensor(
          seed_prefix + ".gate", {intermediate_size, hidden_size}, 0.04F);
      weight_dict[expert_prefix + "up_proj.weight"] = make_small_tensor(
          seed_prefix + ".up", {intermediate_size, hidden_size}, 0.04F);
      weight_dict[expert_prefix + "down_proj.weight"] = make_small_tensor(
          seed_prefix + ".down", {hidden_size, intermediate_size}, 0.04F);
    }
    return weight_dict;
  }

  void expect_dense_runner_available(
      const std::unordered_map<std::string, torch::Tensor>& weight_dict,
      const torch::Tensor& hidden_states,
      int64_t active_expert_id,
      int64_t hidden_size,
      int64_t intermediate_size) const {
    const std::string expert_prefix =
        "experts." + std::to_string(active_expert_id) + ".";
    torch::Tensor w1 =
        torch::cat({weight_dict.at(expert_prefix + "gate_proj.weight"),
                    weight_dict.at(expert_prefix + "up_proj.weight")},
                   0)
            .transpose(0, 1)
            .contiguous();
    torch::Tensor w2 = weight_dict.at(expert_prefix + "down_proj.weight")
                           .transpose(0, 1)
                           .contiguous();

    xllm::kernel::npu::megakernel::DenseFfnRunParams params;
    params.x = hidden_states;
    params.w1 = w1;
    params.w2 = w2;
    params.hidden_size = hidden_size;
    params.intermediate_size = intermediate_size;

    torch::Tensor output;
    ASSERT_TRUE(xllm::kernel::npu::megakernel::try_run_dense_ffn_fp16_no_quant(
        params, &output));
    ASSERT_TRUE(output.defined());
  }

  void expect_output_close(const torch::Tensor& actual,
                           const torch::Tensor& expected) const {
    torch::Tensor actual_fp32 = actual.to(torch::kFloat32).cpu();
    torch::Tensor expected_fp32 = expected.to(torch::kFloat32).cpu();
    torch::Tensor diff = (actual_fp32 - expected_fp32).abs();
    const float max_diff = diff.max().item<float>();
    const int64_t max_index = diff.argmax().item<int64_t>();
    EXPECT_TRUE(torch::allclose(
        actual_fp32, expected_fp32, /*rtol=*/2e-2, /*atol=*/2e-2))
        << "max_diff=" << max_diff << ", max_index=" << max_index;
    EXPECT_TRUE(torch::isfinite(actual_fp32).all().item<bool>());
  }

  torch::TensorOptions options_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
};

TEST_F(FusedMoEMegakernelTest,
       SingleActiveExpertEnvOnMatchesEnvOffAfterMoeCombine) {
  const int64_t token_count = 4;
  const int64_t hidden_size = 64;
  const int64_t intermediate_size = 128;
  const int64_t num_experts = 3;
  const int64_t active_expert_id = 1;

  const std::unordered_map<std::string, torch::Tensor> weight_dict =
      make_expert_weights(num_experts, hidden_size, intermediate_size);
  const torch::Tensor hidden_states = make_small_tensor(
      "npu.fused_moe.megakernel.hidden", {token_count, hidden_size}, 0.2F);
  const torch::Tensor topk_weights =
      torch::linspace(0.5,
                      1.0,
                      token_count,
                      torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .device(options_.device()))
          .reshape({token_count, 1})
          .to(options_);
  const torch::Tensor topk_ids = torch::full(
      {token_count, 1},
      active_expert_id,
      torch::TensorOptions().dtype(torch::kLong).device(options_.device()));

  expect_dense_runner_available(weight_dict,
                                hidden_states,
                                active_expert_id,
                                hidden_size,
                                intermediate_size);

  FusedMoE baseline_moe =
      make_fused_moe(num_experts, hidden_size, intermediate_size);
  FusedMoE megakernel_moe =
      make_fused_moe(num_experts, hidden_size, intermediate_size);
  StateDict state_dict(weight_dict);
  baseline_moe->load_state_dict(state_dict);
  megakernel_moe->load_state_dict(state_dict);

  ModelInputParams input_params;
  EnvGuard env_guard(kMegakernelMoeEnv);
  env_guard.unset();
  torch::Tensor expected = baseline_moe->forward_with_selected_experts(
      hidden_states, topk_weights, topk_ids, input_params);
  Device(options_.device()).synchronize_default_stream();

  env_guard.set("1");
  torch::Tensor actual = megakernel_moe->forward_with_selected_experts(
      hidden_states, topk_weights, topk_ids, input_params);
  Device(options_.device()).synchronize_default_stream();

  expect_output_close(actual, expected);
}

}  // namespace
}  // namespace layer
}  // namespace xllm
