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

#include <torch/torch.h>

namespace xllm::kernel::npu::megakernel {

struct DenseFfnRunParams {
  torch::Tensor x;
  torch::Tensor w1;
  torch::Tensor w2;
  int64_t hidden_size = 0;
  int64_t intermediate_size = 0;
};

bool is_single_expert_moe_enabled();

bool try_run_dense_ffn_fp16_no_quant(const DenseFfnRunParams& params,
                                     torch::Tensor* output);

}  // namespace xllm::kernel::npu::megakernel
