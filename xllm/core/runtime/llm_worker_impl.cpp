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
#include <sstream>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"
#if defined(USE_NPU)
#include "kernels/npu/xllm_ops/cache_select.h"
#endif

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

  if (FLAGS_enable_beam_search_kernel) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }
  return true;
}

std::optional<ForwardOutput> LLMWorkerImpl::step(
    const BatchedForwardInputs& inputs) {
  device_.set_device();
  Timer timer;
  if (!inputs.micro_inputs.empty() && inputs.micro_inputs[0].total_round > 0) {
    return step_rec(inputs);
  }
  if (!inputs.micro_inputs.empty() && inputs.micro_inputs[0].total_round > 0) {
    std::vector<torch::Tensor> flatten_tokens_micro_batches;
    std::vector<torch::Tensor> flatten_positions_micro_batches;
    std::vector<ModelInputParams> input_params_micro_batches;
    const auto& concated_sampling_params =
        inputs.concated_decoder_sampling_params.selected_token_idxes.defined()
            ? inputs.concated_decoder_sampling_params
            : inputs.concated_sampling_params;

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

    int32_t total_rounds = inputs.micro_inputs[0].total_round;
    ForwardOutput output;
    if (FLAGS_enable_eplb) {
      output.expert_load_data = expert_load_data_;
      output.prepared_layer_id = eplb_executor_->get_ready_layer_id();
      if (output.prepared_layer_id != -1) {
        eplb_executor_->reset_ready_layer_id();
      }
    }

    for (int32_t round = 0; round <= total_rounds; ++round) {
      for (auto i = 0; i < input_params_micro_batches.size(); ++i) {
        auto& mip = input_params_micro_batches[i];
        if (!mip.current_round_tensor_list.empty() && round >= 0 &&
            round <
                static_cast<int32_t>(mip.current_round_tensor_list.size())) {
          mip.current_round_tensor = mip.current_round_tensor_list[round];
        }
      }
      auto hidden_states =
          model_executor_->forward(flatten_tokens_micro_batches,
                                   flatten_positions_micro_batches,
                                   kv_caches_,
                                   input_params_micro_batches);
      if (!hidden_states.defined()) {
        return std::nullopt;
      }

      torch::Tensor logits;
      if (concated_sampling_params.selected_token_idxes.defined()) {
        logits = model_->logits(hidden_states,
                                concated_sampling_params.selected_token_idxes);
      }
      if (round == total_rounds) {
        SampleOutput sample_output;
        if (concated_sampling_params.selected_token_idxes.defined()) {
          sample_output = sampler_->forward(logits, concated_sampling_params);
          output.logits = logits;

          BeamSearchOutput beam_search_output;
          if (concated_sampling_params.use_beam_search &&
              inputs.acc_logprob.numel() > 0) {
            beam_search_output =
                beam_searcher_->forward(inputs.acc_logprob,
                                        sample_output.top_tokens,
                                        sample_output.top_logprobs);
          }
          output.sample_output = sample_output;
          output.do_sample = concated_sampling_params.do_sample;
          output.logprobs = concated_sampling_params.logprobs;
          output.max_top_logprobs = concated_sampling_params.max_top_logprobs;
          output.beam_search_output = beam_search_output;

#if defined(USE_NPU)
          int32_t beam_width = inputs.micro_inputs[0].beam_width;
          if (beam_width > 1 && round > 0) {
            xllm_ops::cache_select(beam_search_output.out_tokens,
                                   beam_search_output.group_offset,
                                   input_params_micro_batches[0].decode_k_cache,
                                   input_params_micro_batches[0].decode_v_cache,
                                   beam_width,
                                   round);
          }
#endif
        }
      } else {
#if defined(USE_NPU)
        int32_t beam_width = inputs.micro_inputs[0].beam_width;
        if (concated_sampling_params.selected_token_idxes.defined() &&
            beam_width > 1 && round > 0) {
          auto sample_output =
              sampler_->forward(logits, concated_sampling_params);
          auto beam_search_output =
              beam_searcher_->forward(inputs.acc_logprob,
                                      sample_output.top_tokens,
                                      sample_output.top_logprobs);
          xllm_ops::cache_select(beam_search_output.out_tokens,
                                 beam_search_output.group_offset,
                                 input_params_micro_batches[0].decode_k_cache,
                                 input_params_micro_batches[0].decode_v_cache,
                                 beam_width,
                                 round);
        }
#endif
      }
    }

    if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
        !options_.enable_speculative_decode()) {
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
      if (FLAGS_enable_eplb) {
        return output;
      }
      return std::nullopt;
    }

    auto ret = device_.synchronize_default_stream();
    COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
    DeviceMonitor::get_instance().update_active_activation_memory(
        device_.index());
    return output;
  }
  std::vector<torch::Tensor> flatten_tokens_micro_batches;
  std::vector<torch::Tensor> flatten_positions_micro_batches;
  std::vector<ModelInputParams> input_params_micro_batches;
  const auto& concated_sampling_params =
      inputs.concated_decoder_sampling_params.selected_token_idxes.defined()
          ? inputs.concated_decoder_sampling_params
          : inputs.concated_sampling_params;

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

  // Debug: print global KV cache and new_cache_slots shapes before forward
  {
    auto format_sizes = [](const c10::IntArrayRef& sizes) {
      std::stringstream ss;
      ss << "[";
      for (size_t i = 0; i < sizes.size(); ++i) {
        ss << sizes[i];
        if (i + 1 < sizes.size()) ss << ", ";
      }
      ss << "]";
      return ss.str();
    };

    bool has_inputs = !inputs.micro_inputs.empty();
    bool is_decode = false;
    if (!input_params_micro_batches.empty()) {
      const auto& mip0 = input_params_micro_batches[0];
      is_decode = (mip0.decode_seq_range.first != -1);
    }
    int cur_round = has_inputs ? inputs.micro_inputs[0].current_round : -1;
    int beam_w = has_inputs ? inputs.micro_inputs[0].beam_width : -1;
    VLOG(1) << "[KV_DEBUG] phase=" << (is_decode ? "DECODE" : "PREFILL")
            << " current_round=" << cur_round << " beam_width=" << beam_w;

    if (!kv_caches_.empty()) {
      auto k = kv_caches_[0].get_k_cache();
      auto v = kv_caches_[0].get_v_cache();
      if (k.defined()) {
        VLOG(1) << "[KV_DEBUG] global_k_cache shape: "
                << format_sizes(k.sizes());
      } else {
        LOG(INFO) << "[KV_DEBUG] global_k_cache is undefined";
      }
      if (v.defined()) {
        VLOG(1) << "[KV_DEBUG] global_v_cache shape: "
                << format_sizes(v.sizes());
      } else {
        LOG(INFO) << "[KV_DEBUG] global_v_cache is undefined";
      }
    }

    if (!input_params_micro_batches.empty()) {
      const auto& mip0 = input_params_micro_batches[0];
      if (mip0.new_cache_slots.defined()) {
        VLOG(1) << "[KV_DEBUG] new_cache_slots shape: "
                << format_sizes(mip0.new_cache_slots.sizes())
                << " numel=" << mip0.new_cache_slots.numel();
      } else {
        LOG(INFO) << "[KV_DEBUG] new_cache_slots is undefined";
      }

      if (mip0.block_tables.defined()) {
        VLOG(1) << "[KV_DEBUG] block_tables shape: "
                << format_sizes(mip0.block_tables.sizes())
                << " numel=" << mip0.block_tables.numel();
      } else {
        LOG(INFO) << "[KV_DEBUG] block_tables is undefined";
      }
    }
  }

  // temporarily use [0], will be adapted in next pr
  // call model executor forward to get hidden states
  VLOG(1) << "model_executor_->forward";
  {
    int32_t current_round = inputs.micro_inputs[0].current_round;
    for (auto i = 0; i < input_params_micro_batches.size(); ++i) {
      auto& mip = input_params_micro_batches[i];
      if (!mip.current_round_tensor_list.empty() && current_round >= 0 &&
          current_round <
              static_cast<int32_t>(mip.current_round_tensor_list.size())) {
        mip.current_round_tensor = mip.current_round_tensor_list[current_round];
      }
    }
  }
  auto hidden_states = model_executor_->forward(flatten_tokens_micro_batches,
                                                flatten_positions_micro_batches,
                                                kv_caches_,
                                                input_params_micro_batches);
  VLOG(1) << "model_executor_->forward done";

  // Debug: print global KV cache and new_cache_slots shapes before forward
  {
    auto format_sizes = [](const c10::IntArrayRef& sizes) {
      std::stringstream ss;
      ss << "[";
      for (size_t i = 0; i < sizes.size(); ++i) {
        ss << sizes[i];
        if (i + 1 < sizes.size()) ss << ", ";
      }
      ss << "]";
      return ss.str();
    };

    bool has_inputs = !inputs.micro_inputs.empty();
    bool is_decode = false;
    if (!input_params_micro_batches.empty()) {
      const auto& mip0 = input_params_micro_batches[0];
      is_decode = (mip0.decode_seq_range.first != -1);
    }
    int cur_round = has_inputs ? inputs.micro_inputs[0].current_round : -1;
    int beam_w = has_inputs ? inputs.micro_inputs[0].beam_width : -1;
    VLOG(1) << "[KV_DEBUG] phase=" << (is_decode ? "DECODE" : "PREFILL")
            << " current_round=" << cur_round << " beam_width=" << beam_w;

    if (!kv_caches_.empty()) {
      auto k = kv_caches_[0].get_k_cache();
      auto v = kv_caches_[0].get_v_cache();
      if (k.defined()) {
        VLOG(1) << "[KV_DEBUG] global_k_cache shape: "
                << format_sizes(k.sizes());
      } else {
        LOG(INFO) << "[KV_DEBUG] global_k_cache is undefined";
      }
      if (v.defined()) {
        VLOG(1) << "[KV_DEBUG] global_v_cache shape: "
                << format_sizes(v.sizes());
      } else {
        LOG(INFO) << "[KV_DEBUG] global_v_cache is undefined";
      }
    }

    if (!input_params_micro_batches.empty()) {
      const auto& mip0 = input_params_micro_batches[0];
      if (mip0.new_cache_slots.defined()) {
        VLOG(1) << "[KV_DEBUG] new_cache_slots shape: "
                << format_sizes(mip0.new_cache_slots.sizes())
                << " numel=" << mip0.new_cache_slots.numel();
      } else {
        LOG(INFO) << "[KV_DEBUG] new_cache_slots is undefined";
      }
      if (mip0.block_tables.defined()) {
        VLOG(1) << "[KV_DEBUG] block_tables shape: "
                << format_sizes(mip0.block_tables.sizes())
                << " numel=" << mip0.block_tables.numel();
      } else {
        LOG(INFO) << "[KV_DEBUG] block_tables is undefined";
      }
    }
  }
  if (!hidden_states.defined()) {
    return std::nullopt;
  }

  torch::Tensor logits;
  if (concated_sampling_params.selected_token_idxes.defined()) {
    logits = model_->logits(hidden_states,
                            concated_sampling_params.selected_token_idxes);
  }

  ForwardOutput output;
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
  SampleOutput sample_output;
  VLOG(1) << "inputs acc logprob:" << inputs.acc_logprob.numel();
  VLOG(1) << "selected token is defined:"
          << concated_sampling_params.selected_token_idxes.defined();
  if (concated_sampling_params.selected_token_idxes.defined()) {
    VLOG(1) << "[SEL/WORKER] sel.defined=1 numel="
            << concated_sampling_params.selected_token_idxes.size(0);
    sample_output = sampler_->forward(logits, concated_sampling_params);
    output.logits = logits;

    // beam search kernel
    BeamSearchOutput beam_search_output;
    VLOG(1) << "inputs acc logprob:" << inputs.acc_logprob.numel();
    if (concated_sampling_params.use_beam_search &&
        inputs.acc_logprob.numel() > 0) {
      VLOG(1) << "beam search kernel";
      beam_search_output = beam_searcher_->forward(inputs.acc_logprob,
                                                   sample_output.top_tokens,
                                                   sample_output.top_logprobs);
    }

    // set sample output to output
    output.sample_output = sample_output;
    // carry over the sampling params
    output.do_sample = concated_sampling_params.do_sample;
    output.logprobs = concated_sampling_params.logprobs;
    output.max_top_logprobs = concated_sampling_params.max_top_logprobs;
    // set beam search output to output
    output.beam_search_output = beam_search_output;
    // cache select kernel
    // 只有在 decode stage 才需要 cache select

    int32_t current_round = inputs.micro_inputs[0].current_round;
    int32_t beam_width = inputs.micro_inputs[0].beam_width;
    if (beam_width > 1 && current_round > 0) {
      VLOG(1) << "cache select" << beam_width << " " << current_round;
      xllm_ops::cache_select(beam_search_output.out_tokens,
                             beam_search_output.group_offset,
                             input_params_micro_batches[0].decode_k_cache,
                             input_params_micro_batches[0].decode_v_cache,
                             beam_width,
                             current_round);
      VLOG(1) << "cache_select done";
    }
  }

  // if running in multi_stream_parallel step, all micro batches
  // should be in same prefill stage, so, to judge empty_kv_cache,
  // just use micro batch 0 here
  if (options_.enable_speculative_decode() && !is_spec_draft_) {
    if (input_params_micro_batches[0].q_seq_lens_vec[0] > 1) {
      output.sample_output.embeddings = hidden_states;
    } else if (concated_sampling_params.sample_idxes.defined()) {
      // auto sample_idxes =
      //     concated_sampling_params.selected_token_idxes.index_select(
      //         /*dim=*/0, concated_sampling_params.sample_idxes);
      auto embeddings = hidden_states.index_select(
          /*dim=*/0, concated_sampling_params.sample_idxes);
      output.sample_output.embeddings = embeddings;
    }
  }

  // if running in multi_stream_parallel step, all micro batches
  // should be in same prefill stage, so, to judge empty_kv_cache,
  // just use micro batch 0 here
  if (options_.enable_speculative_decode() && !is_spec_draft_) {
    if (input_params_micro_batches[0].q_seq_lens_vec[0] > 1) {
      output.sample_output.embeddings = hidden_states;
    } else if (concated_sampling_params.sample_idxes.defined()) {
      // auto sample_idxes =
      //     concated_sampling_params.selected_token_idxes.index_select(
      //         /*dim=*/0, concated_sampling_params.sample_idxes);
      auto embeddings = hidden_states.index_select(
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

  return output;
}

std::optional<ForwardOutput> LLMWorkerImpl::step_rec(
    const BatchedForwardInputs& inputs) {
  device_.set_device();
  Timer timer;
  std::vector<torch::Tensor> flatten_tokens_micro_batches;
  std::vector<torch::Tensor> flatten_positions_micro_batches;
  std::vector<ModelInputParams> input_params_micro_batches;
  const auto& concated_sampling_params =
      inputs.concated_decoder_sampling_params.selected_token_idxes.defined()
          ? inputs.concated_decoder_sampling_params
          : inputs.concated_sampling_params;

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

  int32_t total_rounds = inputs.micro_inputs[0].total_round;
  ForwardOutput output;
  if (FLAGS_enable_eplb) {
    output.expert_load_data = expert_load_data_;
    output.prepared_layer_id = eplb_executor_->get_ready_layer_id();
    if (output.prepared_layer_id != -1) {
      eplb_executor_->reset_ready_layer_id();
    }
  }

  for (int32_t round = 0; round <= total_rounds; ++round) {
    for (auto i = 0; i < input_params_micro_batches.size(); ++i) {
      auto& mip = input_params_micro_batches[i];
      if (!mip.current_round_tensor_list.empty() && round >= 0 &&
          round < static_cast<int32_t>(mip.current_round_tensor_list.size())) {
        mip.current_round_tensor = mip.current_round_tensor_list[round];
      }
    }
    auto hidden_states =
        model_executor_->forward(flatten_tokens_micro_batches,
                                 flatten_positions_micro_batches,
                                 kv_caches_,
                                 input_params_micro_batches);
    if (!hidden_states.defined()) {
      return std::nullopt;
    }

    torch::Tensor logits;
    if (concated_sampling_params.selected_token_idxes.defined()) {
      logits = model_->logits(hidden_states,
                              concated_sampling_params.selected_token_idxes);
    }
    if (round == total_rounds) {
      SampleOutput sample_output;
      if (concated_sampling_params.selected_token_idxes.defined()) {
        sample_output = sampler_->forward(logits, concated_sampling_params);
        output.logits = logits;

        BeamSearchOutput beam_search_output;
        if (concated_sampling_params.use_beam_search &&
            inputs.acc_logprob.numel() > 0) {
          beam_search_output =
              beam_searcher_->forward(inputs.acc_logprob,
                                      sample_output.top_tokens,
                                      sample_output.top_logprobs);
        }
        output.sample_output = sample_output;
        output.do_sample = concated_sampling_params.do_sample;
        output.logprobs = concated_sampling_params.logprobs;
        output.max_top_logprobs = concated_sampling_params.max_top_logprobs;
        output.beam_search_output = beam_search_output;

#if defined(USE_NPU)
        int32_t beam_width = inputs.micro_inputs[0].beam_width;
        if (beam_width > 1 && round > 0) {
          xllm_ops::cache_select(beam_search_output.out_tokens,
                                 beam_search_output.group_offset,
                                 input_params_micro_batches[0].decode_k_cache,
                                 input_params_micro_batches[0].decode_v_cache,
                                 beam_width,
                                 round);
        }
#endif
      }
    } else {
#if defined(USE_NPU)
      int32_t beam_width = inputs.micro_inputs[0].beam_width;
      if (concated_sampling_params.selected_token_idxes.defined() &&
          beam_width > 1 && round > 0) {
        auto sample_output =
            sampler_->forward(logits, concated_sampling_params);
        auto beam_search_output =
            beam_searcher_->forward(inputs.acc_logprob,
                                    sample_output.top_tokens,
                                    sample_output.top_logprobs);
        xllm_ops::cache_select(beam_search_output.out_tokens,
                               beam_search_output.group_offset,
                               input_params_micro_batches[0].decode_k_cache,
                               input_params_micro_batches[0].decode_v_cache,
                               beam_width,
                               round);
      }
#endif
    }
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    auto ret = device_.synchronize_default_stream();
    if (options_.instance_role() == InstanceRole::PREFILL &&
        options_.kv_cache_transfer_mode() == "PUSH" &&
        !inputs.micro_inputs[0].transfer_kv_infos.empty()) {
      auto results =
          folly::collectAll(futures).within(std::chrono::seconds(60)).get();
      for (const auto& result : results) {
        if (!result.value()) {
          return std::nullopt;
        }
      }
    }
    if (FLAGS_enable_eplb) {
      return output;
    }
    return std::nullopt;
  }

  auto ret = device_.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      device_.index());
  return output;
}

}  // namespace xllm
