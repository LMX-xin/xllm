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

#include "llm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "kernels/npu/xllm_ops/beam_search_group.h"
#include "kernels/npu/xllm_ops/cache_select.h"
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

LLMWorkerImpl::LLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {}

bool LLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";
  device_.set_device();

  // Try to create a causal LM model
  model_ = create_llm_model(context);

  // Dont find model in causal models
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);

  if (FLAGS_enable_eplb) {
    eplb_executor_ = std::make_unique<EplbExecutor>(model_.get(), device_);
  }

  // if (FLAGS_enable_beam_search_kernel) {
  //   beam_searcher_ = std::make_unique<BeamSearcher>();
  // }
  return true;
}

std::optional<ForwardOutput> LLMWorkerImpl::step(
    const BatchedForwardInputs& inputs) {
  device_.set_device();
  Timer timer;
  std::vector<torch::Tensor> flatten_tokens_micro_batches;
  std::vector<torch::Tensor> flatten_positions_micro_batches;
  std::vector<ModelInputParams> input_params_micro_batches;
  auto& concated_sampling_params = inputs.concated_sampling_params;

  std::vector<folly::SemiFuture<bool>> futures;

  for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
    flatten_tokens_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].token_ids));
    flatten_positions_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].positions));
    input_params_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].input_params));

    if (options_.instance_role() == InstanceRole::PREFILL &&
        options_.kv_cache_transfer_mode() == "PUSH" &&
        !inputs.micro_inputs[i].transfer_kv_infos.empty()) {
#if defined(USE_NPU)
      std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
          std::make_shared<NPULayerSynchronizerImpl>(
              context_.get_model_args().n_layers());
      const_cast<ModelInputParams*>(&(input_params_micro_batches[i]))
          ->layer_synchronizer = layer_synchronizer;

      futures.emplace_back(kv_cache_transfer_->push_kv_blocks_async(
          inputs.micro_inputs[i].transfer_kv_infos,
          context_.get_parallel_args(),
          layer_synchronizer,
          is_spec_draft_));
#endif
    }
  }
  if (FLAGS_enable_eplb) {
    eplb_executor_->eplb_execute(inputs.micro_inputs[0].eplb_info);
  }
  // prefer params-provided beam width; default to 1 when absent
  int32_t beam_width = input_params_micro_batches[0].beam_width > 0
                           ? input_params_micro_batches[0].beam_width
                           : 512;
  int32_t batch = input_params_micro_batches[0].num_sequences;
  int32_t max_decode_step = 3;
  auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(device());
  torch::Tensor sequence_group =
      torch::zeros({batch, beam_width, max_decode_step}, int_opts);
  auto args = context_.get_model_args();
  int32_t head_size = static_cast<int32_t>(args.head_dim());
  int32_t head_num =
      static_cast<int32_t>(args.n_kv_heads().value_or(args.n_heads()));
  int32_t layer_num = static_cast<int32_t>(args.n_layers());
  auto dtype = util::parse_dtype(args.dtype(), device());
  auto fp_opts = torch::TensorOptions().dtype(dtype).device(device());
  std::vector<torch::Tensor> shared_k_cache(
      layer_num,
      torch::zeros({batch, beam_width, head_num, max_decode_step, head_size},
                   fp_opts));
  std::vector<torch::Tensor> shared_v_cache(
      layer_num,
      torch::zeros({batch, beam_width, head_num, max_decode_step, head_size},
                   fp_opts));
  torch::Tensor beam_width_tensor = torch::full({1}, beam_width, int_opts);

  input_params_micro_batches[0].shared_k_cache = shared_k_cache;
  input_params_micro_batches[0].shared_v_cache = shared_v_cache;
  input_params_micro_batches[0].beam_width_tensor = beam_width_tensor;
  input_params_micro_batches[0].beam_width = beam_width;
  ForwardOutput output;
  BeamSearchOutput beam_search_output;
  torch::Tensor last_hidden_states;
  SampleOutput sample_output;
  // TODO: need to check
  torch::Tensor block_tables =
      torch::arange(batch, int_opts).reshape({batch, 1});
  // block tables value is 0-batch-1
  input_params_micro_batches[0].block_tables = block_tables;
  torch::Tensor acc_logprob;
  for (auto current_step = 0; current_step < max_decode_step; ++current_step) {
    torch::Tensor current_step_tensor =
        torch::full({1}, current_step, int_opts);
    input_params_micro_batches[0].current_step_tensor = current_step_tensor;
    input_params_micro_batches[0].current_step = current_step;
    // temporarily use [0], will be adapted in next pr
    // call model executor forward to get hidden states
    auto hidden_states =
        model_executor_->forward(flatten_tokens_micro_batches,
                                 flatten_positions_micro_batches,
                                 kv_caches_,
                                 input_params_micro_batches);
    last_hidden_states = hidden_states;

    if (!hidden_states.defined()) {
      return std::nullopt;
    }

    torch::Tensor logits;
    if (concated_sampling_params.selected_token_idxes.defined()) {
      logits = model_->logits(hidden_states,
                              concated_sampling_params.selected_token_idxes);
    }

    if (FLAGS_enable_eplb) {
      output.expert_load_data = expert_load_data_;
      output.prepared_layer_id = eplb_executor_->get_ready_layer_id();
      if (output.prepared_layer_id != -1) {
        eplb_executor_->reset_ready_layer_id();
      }
    }

    if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
        !options_.enable_speculative_decode()) {
      auto ret = device_.synchronize_default_stream();
      // in p-d disaggregation scene, all micro batches should be in same
      // prefill/decode stage, so, to judge transfer_kv_infos.empty,
      // just use micro inputs.micro_inputs[0] here
      if (options_.instance_role() == InstanceRole::PREFILL &&
          options_.kv_cache_transfer_mode() == "PUSH" &&
          !inputs.micro_inputs[0].transfer_kv_infos.empty()) {
        auto results =
            folly::collectAll(futures).within(std::chrono::seconds(60)).get();
        for (const auto& result : results) {
          if (!result.value()) {
            LOG(ERROR) << "kv_cache_transfer_ failed";
            return std::nullopt;
          }
        }
      }
      if (FLAGS_enable_eplb) {
        return output;
      }
      return std::nullopt;
    }

    // driver prepare model output

    // if (concated_sampling_params.selected_token_idxes.defined()) {
    sample_output = sampler_->forward(logits, concated_sampling_params);
    output.logits = logits;

    // beam search kernel

    // if (concated_sampling_params.use_beam_search &&
    //     inputs.acc_logprob.numel() > 0) {
    {
      // BeamSearchGroup 要求 logProbs 为 FP32
      if (!inputs.acc_logprob.defined() || inputs.acc_logprob.numel() == 0) {
        auto fp32_opts =
            torch::TensorOptions().dtype(torch::kFloat32).device(device());
        acc_logprob = torch::zeros({batch * beam_width, 1}, fp32_opts);
      } else {
        acc_logprob = inputs.acc_logprob;
        if (acc_logprob.scalar_type() != torch::kFloat32) {
          acc_logprob = acc_logprob.to(torch::kFloat32);
        }
      }
      // reshape top_tokens and top_logprobs ,第一维为batch * beam_width
      torch::Tensor top_tokens;
      torch::Tensor top_logprobs;

      if (current_step == 0) {
        top_tokens = sample_output.top_tokens.to(torch::kInt32)
                         .reshape({batch * beam_width, 1});
        top_logprobs =
            sample_output.top_logprobs.reshape({batch * beam_width, 1});
      } else {
        top_tokens = sample_output.top_tokens.to(torch::kInt32)
                         .reshape({batch * beam_width, beam_width});
        top_logprobs = sample_output.top_logprobs.reshape(
            {batch * beam_width, beam_width});
      }
      auto bs_tuple = xllm_ops::beam_search_group(
          acc_logprob, top_tokens, top_logprobs, sequence_group, current_step);
      beam_search_output.out_tokens = std::get<0>(bs_tuple);
      beam_search_output.src_seq_idxes = std::get<1>(bs_tuple);
      beam_search_output.out_logprobs = std::get<2>(bs_tuple);
      beam_search_output.group_offsets = std::get<3>(bs_tuple);
    }
    // beam_search_output = beam_searcher_->forward(inputs.acc_logprob,
    //                                                sample_output.top_tokens,
    //                                                sample_output.top_logprobs,
    //                                                sequence_group);

    if (current_step > 0) {
      // xllm_ops::cache_select(beam_search_output.out_tokens,
      //   beam_search_output.group_offset,
      //   decode_k_cache,
      //   decode_v_cache,
      //   beam_width,
      //   current_round);

      std::vector<torch::Tensor> unshared_k_cache;
      std::vector<torch::Tensor> unshared_v_cache;
      for (auto i = 0; i < layer_num; ++i) {
        unshared_k_cache.push_back(kv_caches_[i].get_k_cache());
        unshared_v_cache.push_back(kv_caches_[i].get_v_cache());
      }

      xllm_ops::cache_select(beam_search_output.out_tokens,
                             unshared_k_cache,
                             unshared_v_cache,
                             input_params_micro_batches[0].block_tables,
                             beam_search_output.group_offsets,
                             current_step,
                             beam_width,
                             layer_num);
    }
    // }
    flatten_tokens_micro_batches[0] = beam_search_output.out_tokens;
  }
  // set sample output to output
  output.sample_output = sample_output;
  // carry over the sampling params
  output.do_sample = concated_sampling_params.do_sample;
  output.logprobs = concated_sampling_params.logprobs;
  output.max_top_logprobs = concated_sampling_params.max_top_logprobs;
  // set beam search output to output
  output.beam_search_output = beam_search_output;
  // }

  // if running in multi_stream_parallel step, all micro batches
  // should be in same prefill stage, so, to judge empty_kv_cache,
  // just use micro batch 0 here
  if (options_.enable_speculative_decode() && !is_spec_draft_) {
    if (input_params_micro_batches[0].q_seq_lens_vec[0] > 1) {
      output.sample_output.embeddings = last_hidden_states;
    } else if (concated_sampling_params.sample_idxes.defined()) {
      // auto sample_idxes =
      //     concated_sampling_params.selected_token_idxes.index_select(
      //         /*dim=*/0, concated_sampling_params.sample_idxes);
      auto embeddings = last_hidden_states.index_select(
          /*dim=*/0, concated_sampling_params.sample_idxes);
      output.sample_output.embeddings = embeddings;
    }
  }

  auto ret = device_.synchronize_default_stream();

  if (options_.instance_role() == InstanceRole::PREFILL &&
      options_.kv_cache_transfer_mode() == "PUSH" &&
      !inputs.micro_inputs[0].transfer_kv_infos.empty()) {
    auto results =
        folly::collectAll(futures).within(std::chrono::seconds(60)).get();
    for (const auto& result : results) {
      if (!result.value()) {
        LOG(ERROR) << "kv_cache_transfer_ failed";
        return std::nullopt;
      }
    }
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      device_.index());

  // Expose the per-sequence generated token ids for this step so that
  // upper layers can directly decode without re-materializing sequences.
  // sequence_group has shape [batch, beam_width, max_decode_step].
  // Select the first beam by default; if beam search is enabled and a different
  // beam is desired, upper layers can use beam_search_output to choose.
  {
    const int64_t chosen_beam = 0;
    if (sequence_group.defined()) {
      auto final_tokens = sequence_group.index(
          {torch::indexing::Slice(), chosen_beam, torch::indexing::Slice()});
      if (final_tokens.defined()) {
        if (final_tokens.dtype() != torch::kLong) {
          final_tokens = final_tokens.to(torch::kLong);
        }
        // Shape: [batch, max_decode_step]
        output.sample_output.next_tokens = final_tokens.contiguous();
      }
    }
  }

  return output;
}

}  // namespace xllm
