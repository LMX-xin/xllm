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

#include "flashinfer_planinfo.h"

#include "core/util/utils.h"
#include "flashinfer_workspace.h"
#include "kernels/cuda/function_factory.h"
#include "kernels/cuda/utils.h"

namespace xllm {
namespace layer {
namespace flashinfer {

void update_plan_info(std::shared_ptr<PlanInfo> plan_info,
                      const std::string& backend,
                      const AttentionMetadata& attn_meta,
                      c10::ScalarType query_dtype,
                      c10::ScalarType key_dtype,
                      c10::ScalarType output_dtype,
                      int32_t head_dim_qk,
                      int32_t head_dim_vo,
                      int32_t num_qo_heads,
                      int32_t num_kv_heads,
                      int32_t block_size,
                      int32_t window_size_left,
                      bool enable_cuda_graph,
                      bool causal,
                      bool use_tensor_core) {
  CHECK(plan_info->layer_id != -1) << "Need to set layer_id to PlanInfo.";
  if (plan_info->layer_id != 0) return;

  // 1. prefill plan info
  if (causal) {
    plan_info->uri = kernel::cuda::get_batch_prefill_uri(
        backend,
        query_dtype,
        key_dtype,
        output_dtype,
        attn_meta.q_cu_seq_lens.scalar_type(),
        head_dim_qk,
        head_dim_vo,
        /*pos_encoding_mode=*/0,
        /*use_sliding_window=*/false,
        /*use_logits_soft_cap=*/false,
        /*use_fp16_qk_reduction=*/false);

    torch::Tensor qo_indptr_host = attn_meta.q_cu_seq_lens.to(torch::kCPU);
    torch::Tensor kv_cu_seq_lens_host =
        attn_meta.kv_cu_seq_lens.to(torch::kCPU);
    torch::Tensor kv_len_arr_host =
        kv_cu_seq_lens_host.slice(0, 1) - kv_cu_seq_lens_host.slice(0, 0, -1);
    const int64_t total_num_rows = qo_indptr_host[-1].item<int64_t>();
    const int64_t batch_size = qo_indptr_host.size(0) - 1;
    auto call_plan_func = [&](auto&& func) {
      return func.call(
          FlashinferWorkspace::get_instance().get_float_workspace_buffer(),
          FlashinferWorkspace::get_instance().get_int_workspace_buffer(),
          FlashinferWorkspace::get_instance()
              .get_page_locked_int_workspace_buffer(),
          qo_indptr_host,
          kv_cu_seq_lens_host,
          kv_len_arr_host,
          total_num_rows,
          batch_size,
          num_qo_heads,
          num_kv_heads,
          /*page_size=*/1,
          enable_cuda_graph,
          head_dim_qk,
          head_dim_vo,
          causal);
    };
    if (backend == "fa2") {
      plan_info->plan_info = call_plan_func(
          kernel::cuda::FunctionFactory::get_instance().fa2_prefill_plan_func(
              plan_info->uri));
    } else {
      plan_info->plan_info = call_plan_func(
          kernel::cuda::FunctionFactory::get_instance().fa3_prefill_plan_func(
              plan_info->uri));
    }
  } else {
    // 2. decode plan info
    if (use_tensor_core) {
      plan_info->uri = kernel::cuda::get_batch_prefill_uri(
          /*backend=*/"fa2",
          query_dtype,
          key_dtype,
          output_dtype,
          attn_meta.paged_kv_indptr.scalar_type(),
          head_dim_qk,
          head_dim_vo,
          /*pos_encoding_mode=*/0,
          /*use_sliding_window=*/false,
          /*use_logits_soft_cap=*/false,
          /*use_fp16_qk_reduction=*/false);
      const int64_t batch_size = attn_meta.paged_kv_last_page_len.size(0);
      torch::Tensor qo_indptr_host =
          kernel::cuda::get_cache_buffer(batch_size + 1, torch::kCPU);
      torch::Tensor qo_indptr = qo_indptr_host.to(torch::kCUDA);
      torch::Tensor paged_kv_indptr_host =
          attn_meta.paged_kv_indptr.to(torch::kCPU);
      torch::Tensor kv_len_arr_host = attn_meta.kv_seq_lens.to(torch::kCPU);
      plan_info->plan_info =
          kernel::cuda::FunctionFactory::get_instance()
              .fa2_prefill_plan_func(plan_info->uri)
              .call(FlashinferWorkspace::get_instance()
                        .get_float_workspace_buffer(),
                    FlashinferWorkspace::get_instance()
                        .get_int_workspace_buffer(),
                    FlashinferWorkspace::get_instance()
                        .get_page_locked_int_workspace_buffer(),
                    qo_indptr_host,
                    paged_kv_indptr_host,
                    kv_len_arr_host,
                    batch_size,  // total_num_rows
                    batch_size,
                    num_qo_heads,  // num_qo_heads
                    num_kv_heads,  // num_kv_heads
                    block_size,    // block_size
                    enable_cuda_graph,
                    head_dim_qk,  // head_dim_qk
                    head_dim_vo,  // head_dim_vo
                    /*causal=*/false);
    } else {
      plan_info->uri = kernel::cuda::get_batch_decode_uri(
          query_dtype,
          key_dtype,
          output_dtype,
          attn_meta.paged_kv_indptr.scalar_type(),
          head_dim_qk,
          head_dim_vo,
          /*pos_encoding_mode=*/0,
          /*use_sliding_window=*/false,
          /*use_logits_soft_cap=*/false);
      torch::Tensor paged_kv_indptr_host =
          attn_meta.paged_kv_indptr.to(torch::kCPU);
      const int64_t batch_size = attn_meta.paged_kv_last_page_len.size(0);
      torch::Tensor empty_q_data =
          torch::empty({0}, torch::TensorOptions().dtype(query_dtype));
      torch::Tensor empty_kv_data =
          torch::empty({0}, torch::TensorOptions().dtype(key_dtype));
      plan_info->plan_info =
          kernel::cuda::FunctionFactory::get_instance()
              .decode_plan_func(plan_info->uri)
              .call(FlashinferWorkspace::get_instance()
                        .get_float_workspace_buffer(),
                    FlashinferWorkspace::get_instance()
                        .get_int_workspace_buffer(),
                    FlashinferWorkspace::get_instance()
                        .get_page_locked_int_workspace_buffer(),
                    paged_kv_indptr_host,
                    batch_size,
                    num_qo_heads,
                    num_kv_heads,
                    block_size,
                    enable_cuda_graph,
                    window_size_left,
                    /*logits_soft_cap=*/0.0,
                    head_dim_qk,
                    head_dim_vo,
                    empty_q_data,
                    empty_kv_data);
    }
  }
}

void initialize_two_stage_decode_cache(AttentionMetadata& attn_metadata,
                                       const torch::Tensor& query,
                                       uint32_t batch_size,
                                       uint32_t beam_size,
                                       uint32_t total_beam,
                                       int32_t num_heads,
                                       int32_t head_size) {
  bool is_layer_0 = (attn_metadata.plan_info->layer_id == 0);

  if (is_layer_0) {
    bool need_init =
        !attn_metadata.two_stage_decode_cache.has_value() ||
        attn_metadata.two_stage_decode_cache->cached_batch_size !=
            static_cast<int32_t>(batch_size) ||
        attn_metadata.two_stage_decode_cache->cached_beam_size !=
            static_cast<int32_t>(beam_size) ||
        attn_metadata.two_stage_decode_cache->cached_num_heads != num_heads ||
        attn_metadata.two_stage_decode_cache->cached_head_size != head_size;

    if (need_init) {
      TwoStageDecodeCache cache;

      auto fp32_options =
          torch::TensorOptions().dtype(torch::kFloat32).device(query.device());

      cache.shared_lse =
          torch::zeros({batch_size * beam_size, num_heads, 1}, fp32_options);
      cache.shared_o = torch::zeros(
          {batch_size * beam_size, num_heads, head_size}, query.options());
      cache.unshared_lse =
          torch::zeros({total_beam, num_heads, 1}, fp32_options);
      cache.unshared_o =
          torch::zeros({total_beam, num_heads, head_size}, query.options());

      cache.q_cu_seq_lens_shared = torch::arange(
          0,
          (batch_size + 1) * beam_size,
          beam_size,
          torch::TensorOptions().dtype(torch::kInt32).device(query.device()));

      cache.paged_kv_indptr_expanded = torch::arange(
          batch_size * beam_size + 1, attn_metadata.paged_kv_indptr.options());

      cache.paged_kv_last_page_len_expanded =
          torch::full({batch_size * beam_size},
                      0,
                      attn_metadata.paged_kv_last_page_len.options());
      cache.paged_kv_last_page_len_expanded.fill_(attn_metadata.step + 1);

      // paged_kv_indices 的计算
      auto batch_offsets =
          torch::zeros({batch_size}, attn_metadata.paged_kv_indices.options());
      batch_offsets = batch_offsets.unsqueeze(1).expand({-1, beam_size});
      auto beam_offsets =
          torch::arange(beam_size, attn_metadata.paged_kv_indices.options());
      auto batch_beam_offsets = batch_offsets * beam_size + beam_offsets;
      cache.paged_kv_indices_expanded = batch_beam_offsets.flatten();

      cache.cached_batch_size = static_cast<int32_t>(batch_size);
      cache.cached_beam_size = static_cast<int32_t>(beam_size);
      cache.cached_num_heads = num_heads;
      cache.cached_head_size = head_size;
      auto max_val = attn_metadata.kv_cu_seq_lens.max();
      int32_t shared_kv_len = max_val.item().toInt();
      int32_t real_shared_kv_len = shared_kv_len * batch_size;
      cache.real_shared_kv_len = real_shared_kv_len;
      attn_metadata.two_stage_decode_cache = cache;
    }
  }
}

}  // namespace flashinfer
}  // namespace layer
}  // namespace xllm
