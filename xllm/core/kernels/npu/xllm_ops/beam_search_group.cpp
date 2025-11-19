/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <nlohmann/json.hpp>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include "acl/acl.h"
#include "aclnn_beam_search_group.h"
#include "beam_search_group.h"

#define CHECK_ACL_SUCCESS(expr, msg) \
  do {                               \
    auto _ret = (expr);              \
    if (_ret != ACL_SUCCESS) {       \
      LOG(ERROR) << msg;             \
      throw std::runtime_error(msg); \
    }                                \
  } while (0)
namespace xllm_ops {

void beam_search_group(const torch::Tensor& log_probs,
                       const torch::Tensor& top_tokens,
                       const torch::Tensor& top_probs,
                       torch::Tensor& sequence_group,
                       int64_t current_step,
                       torch::Tensor& out_token_ids,
                       torch::Tensor& out_token_index,
                       torch::Tensor& out_log_probs,
                       torch::Tensor& out_beam_count_prefix_sums) {
  xllm_ops_utils::check_tensor(log_probs, "log_probs", "beam_search");
  xllm_ops_utils::check_tensor(top_tokens, "top_tokens", "beam_search");
  xllm_ops_utils::check_tensor(top_probs, "top_probs", "beam_search");
  xllm_ops_utils::check_tensor(sequence_group, "sequence_group", "beam_search");
  aclTensor* log_probs_ids = nullptr;
  aclTensor* top_tokens_ids = nullptr;
  aclTensor* top_logprobs_ids = nullptr;
  aclTensor* sequence_group_ids = nullptr;
  aclTensor* out_token_npu_ids = nullptr;
  aclTensor* out_token_npu_index = nullptr;
  aclTensor* out_log_probs_npu = nullptr;
  aclTensor* out_beam_count_prefix_sums_npu = nullptr;
  aclTensor* out_sequence_npu = nullptr;
  int32_t device_id = log_probs.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  xllm_ops_utils::create_acltensor(&log_probs_ids, log_probs);
  xllm_ops_utils::create_acltensor(&top_tokens_ids, top_tokens);
  xllm_ops_utils::create_acltensor(&top_logprobs_ids, top_probs);
  xllm_ops_utils::create_acltensor(&sequence_group_ids, sequence_group);

  xllm_ops_utils::create_acltensor(&out_token_npu_ids, out_token_ids);
  xllm_ops_utils::create_acltensor(&out_token_npu_index, out_token_index);
  xllm_ops_utils::create_acltensor(&out_log_probs_npu, out_log_probs);
  xllm_ops_utils::create_acltensor(&out_beam_count_prefix_sums_npu,
                                   out_beam_count_prefix_sums);
  xllm_ops_utils::create_acltensor(&out_sequence_npu, sequence_group);
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  LOG(INFO) << "beam_search_group shape: log_probs: " << log_probs.sizes();
  LOG(INFO) << "beam_search_group shape: top_tokens: " << top_tokens.sizes();
  LOG(INFO) << "beam_search_group shape: top_probs: " << top_probs.sizes();
  LOG(INFO) << "beam_search_group shape: sequence_group: "
            << sequence_group.sizes();
  LOG(INFO) << "beam_search_group shape: current_step: " << current_step;
  // LOG(INFO) << "beam_search_ops logprobs: " << logprobs;
  // LOG(INFO) << "beam_search_ops top_tokens: " << top_tokens;
  // LOG(INFO) << "beam_search_ops top_logprobs: " << top_logprobs;

  CHECK_ACL_SUCCESS(
      aclnnBeamSearchGroupGetWorkspaceSize(log_probs_ids,
                                           top_tokens_ids,
                                           top_logprobs_ids,
                                           sequence_group_ids,
                                           current_step,
                                           out_token_npu_ids,
                                           out_token_npu_index,
                                           out_log_probs_npu,
                                           out_beam_count_prefix_sums_npu,
                                           out_sequence_npu,
                                           &workspace_size,
                                           &executor),
      "beam_search_group: failed to get workspace size");
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(
        aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
        "beam_search_group: failed to allocate workspace");
  }
  CHECK_ACL_SUCCESS(
      aclnnBeamSearchGroup(workspace_addr, workspace_size, executor, stream),
      "beam_search_group: failed to perform beam search");
  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                    "beam_search_group: failed to synchronize stream");
  aclDestroyTensor(log_probs_ids);
  aclDestroyTensor(top_tokens_ids);
  aclDestroyTensor(top_logprobs_ids);
  aclDestroyTensor(sequence_group_ids);
  aclDestroyTensor(out_token_npu_ids);
  aclDestroyTensor(out_token_npu_index);
  aclDestroyTensor(out_log_probs_npu);
  aclDestroyTensor(out_beam_count_prefix_sums_npu);
  aclDestroyTensor(out_sequence_npu);
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(aclrtFree(workspace_addr),
                      "beam_search_group: failed to free workspace");
  }
}
}  // namespace xllm_ops