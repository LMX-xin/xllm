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

#include "beam_search.h"
namespace xllm_ops {

void beam_search(const torch::Tensor& logprobs,
                 const torch::Tensor& top_tokens,
                 const torch::Tensor& top_logprobs,
                 torch::Tensor& src_seq_idxes,
                 torch::Tensor& out_logprobs,
                 torch::Tensor& out_tokens,
                 torch::Tensor& group_offset) {
  xllm_ops_utils::check_tensor(logprobs, "logprobs", "beam_search");
  xllm_ops_utils::check_tensor(top_tokens, "top_tokens", "beam_search");
  xllm_ops_utils::check_tensor(top_logprobs, "top_logprobs", "beam_search");
  auto num_seq = logprobs.size(0);
  torch::Tensor chosen_tokens = top_tokens.dim() > 1 ? top_tokens.select(1, 0)
                                                     : top_tokens.reshape({num_seq});
  torch::Tensor chosen_logprobs =
      top_logprobs.dim() > 1 ? top_logprobs.select(1, 0)
                             : top_logprobs.reshape({num_seq});
  out_tokens.copy_(chosen_tokens);
  out_logprobs.copy_(chosen_logprobs);
  src_seq_idxes.copy_(torch::arange(num_seq, src_seq_idxes.options()));
  group_offset.copy_(torch::zeros_like(src_seq_idxes));
}
}  // namespace xllm_ops
