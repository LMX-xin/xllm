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

#pragma once

#include "attention.h"

namespace xllm {
namespace layer {
class XAttentionImpl : public AttentionImpl {
 public:
  using AttentionImpl::AttentionImpl;

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      KVCache& kv_cache,
      std::optional<torch::Tensor> output = std::nullopt) override;

  // 两段式解码：先计算共享 KV cache 的 attention，再计算非共享 KV cache 的
  // attention 公开以便单元测试使用
  void TwoStageDecode(const AttentionMetadata& attn_metadata,
                      torch::Tensor& query,
                      torch::Tensor& output_tensor,
                      uint32_t batch_size,
                      uint32_t beam_size,
                      uint32_t total_beam);

  // 一段式解码：直接使用完整的 KV cache 计算 attention
  // 公开以便单元测试使用
  void SingleStageDecode(const AttentionMetadata& attn_metadata,
                         torch::Tensor& query,
                         torch::Tensor& output_tensor,
                         torch::Tensor& key,
                         uint32_t batch_size);
};
TORCH_MODULE(XAttention);

}  // namespace layer
}  // namespace xllm