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

#include "beam_searcher.h"

namespace xllm {
BeamSearchOutput BeamSearcher::forward(const torch::Tensor& logprobs,
                                       const torch::Tensor& top_tokens,
                                       const torch::Tensor& top_logprobs,
                                       torch::Tensor& sequence_group,
                                       int64_t current_step) const {
#if defined(USE_NPU)
  BeamSearchOutput output;

  LOG(INFO) << "[BeamSearcher] cur_step=" << current_step
            << ", seq_group_in=" << sequence_group.sizes()
            << ", logprobs=" << logprobs.sizes()
            << ", top_tokens=" << top_tokens.sizes()
            << ", top_logprobs=" << top_logprobs.sizes();

  int64_t num_seq = logprobs.numel();
  output.out_tokens =
      torch::empty({num_seq, 1}, logprobs.options().dtype(torch::kInt32));
  output.out_logprobs =
      torch::empty({num_seq, 1}, logprobs.options().dtype(torch::kFloat32));
  output.src_seq_idxes =
      torch::empty({num_seq, 1}, logprobs.options().dtype(torch::kInt32));
  output.out_beam_count_prefix_sums =
      torch::empty({num_seq, 1}, logprobs.options().dtype(torch::kInt32));
  output.out_sequence = sequence_group.clone();
  if (current_step == 0) {
    xllm_ops::beam_search(logprobs.reshape({-1, 1}),
                          top_tokens.reshape({-1, 1}).to(torch::kInt32),
                          top_logprobs.reshape({-1, 1}),
                          sequence_group,
                          current_step,
                          output.out_tokens,
                          output.src_seq_idxes,
                          output.out_logprobs,
                          output.out_beam_count_prefix_sums,
                          output.out_sequence);
  } else {
    xllm_ops::beam_search(logprobs.reshape({-1, 1}),
                          top_tokens.reshape({num_seq, -1}).to(torch::kInt32),
                          top_logprobs.reshape({num_seq, -1}),
                          sequence_group,
                          current_step,
                          output.out_tokens,
                          output.src_seq_idxes,
                          output.out_logprobs,
                          output.out_beam_count_prefix_sums,
                          output.out_sequence);
  }
  output.src_seq_idxes = output.src_seq_idxes.reshape({-1});
  output.out_logprobs = output.out_logprobs.reshape({-1});
  output.out_tokens = output.out_tokens.reshape({-1});
  output.out_beam_count_prefix_sums =
      output.out_beam_count_prefix_sums.reshape({-1});
  sequence_group.copy_(output.out_sequence);
  LOG(INFO) << "[BeamSearcher] out_tokens=" << output.out_tokens.sizes()
            << ", out_seq=" << output.out_sequence.sizes()
            << ", src_idxes=" << output.src_seq_idxes.sizes();
  return output;
#else
  LOG(FATAL) << "BeamSearcher is only implemented for NPU backend.";
#endif
}
}  // namespace xllm