
/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#include "core/layers/cuda/xattention.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

#include "core/common/global_flags.h"
#include "core/layers/common/attention_metadata.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

class XAttentionDecodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 检查是否有 CUDA 设备
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test";
    }

    device_ = torch::Device(torch::kCUDA, 0);
    options_ = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
    fp32_options_ =
        torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    int32_options_ =
        torch::TensorOptions().dtype(torch::kInt32).device(device_);

    // 测试参数
    batch_size_ = 1;
    beam_size_ = 4;
    total_beam_ = batch_size_ * beam_size_;
    num_heads_ = 8;
    num_kv_heads_ = 8;
    head_size_ = 128;
    shared_kv_len_ = 1024;
    real_shared_kv_len_ = 13;
    unshared_kv_len_ = 2;
    max_decode_step_ = unshared_kv_len_;
    cache_step_ = 0;

    // 初始化全局标志
    FLAGS_max_seqs_per_batch = 4;
    FLAGS_max_token_per_req = 1024;

    // 创建 XAttention 实例
    xattention_ = XAttention(num_heads_,
                             head_size_,
                             1.0f / std::sqrt(head_size_),
                             num_kv_heads_,
                             -1);
  }

  // 创建 AttentionMetadata 用于两段式解码测试
  AttentionMetadata CreateTwoStageDecodeMetadata() {
    AttentionMetadata metadata;
    metadata.is_prefill = false;
    metadata.is_chunked_prefill = false;
    metadata.is_dummy = false;
    metadata.max_seq_len = shared_kv_len_ + unshared_kv_len_;
    metadata.max_query_len = 1;
    metadata.enable_cuda_graph = false;
    metadata.use_tensor_core = true;

    // kv_cu_seq_lens: [batch_size + 1] for shared KV cache
    // 例如 batch_size=2, shared_kv_len=1024: [0, 1024, 2048]
    metadata.kv_cu_seq_lens =
        torch::arange(0,
                      (batch_size_ + 1) * real_shared_kv_len_,
                      real_shared_kv_len_,
                      int32_options_)
            .contiguous();

    // q_cu_seq_lens: [total_beam + 1] for decode mode
    // 每个 beam 的 query 长度为 1
    metadata.q_cu_seq_lens =
        torch::arange(0, total_beam_ + 1, 1, int32_options_).contiguous();

    // kv_seq_lens: [batch_size] for decode mode
    metadata.kv_seq_lens =
        torch::full({batch_size_}, shared_kv_len_, int32_options_).contiguous();

    metadata.compute_dtype = "float";

    LOG(INFO) << "CreateTwoStageDecodeMetadata:";
    LOG(INFO) << "  batch_size: " << batch_size_
              << ", beam_size: " << beam_size_
              << ", total_beam: " << total_beam_;
    LOG(INFO) << "  real_shared_kv_len: " << real_shared_kv_len_
              << ", unshared_kv_len: " << unshared_kv_len_;
    LOG(INFO) << "  kv_cu_seq_lens shape: " << metadata.kv_cu_seq_lens.sizes()
              << ", values: " << metadata.kv_cu_seq_lens;
    LOG(INFO) << "  q_cu_seq_lens shape: " << metadata.q_cu_seq_lens.sizes()
              << ", values: " << metadata.q_cu_seq_lens;

    // 创建 PlanInfo
    metadata.plan_info = std::make_shared<PlanInfo>();
    metadata.plan_info->layer_id = 0;
    metadata.plan_info->uri = "";

    metadata.unshared_plan_info = std::make_shared<PlanInfo>();
    metadata.unshared_plan_info->layer_id = 0;
    metadata.unshared_plan_info->uri = "";

    // 按照一阶段的逻辑计算 paged_kv 相关参数
    // 1. 计算 shared_kv_lens_each_batch: [batch_size]
    auto shared_kv_lens_each_batch = torch::diff(metadata.kv_cu_seq_lens);

    // 2. 计算 unshared_kv_len（从 step 获取）
    int32_t step_value = 1;  // 默认 step = 1
    int32_t unshared_kv_len =
        step_value + 1;  // unshared_kv_len = current_step + 1

    // 3. 计算 batch_beam_shared_kv_lens: [batch_size * beam_size]
    // (shared_kv_lens_each_batch.unsqueeze(1).expand({-1, beam_size}) +
    // unshared_kv_len).flatten()
    auto batch_beam_shared_kv_lens =
        (shared_kv_lens_each_batch.unsqueeze(1).expand({-1, beam_size_}) +
         unshared_kv_len)
            .flatten();

    // 4. 计算 paged_kv_indptr: [batch_size * beam_size + 1]
    auto cumsum_result = torch::cumsum(batch_beam_shared_kv_lens, 0);
    metadata.paged_kv_indptr = torch::cat(
        {torch::zeros({1}, int32_options_), cumsum_result.to(int32_options_)},
        0);
    LOG(INFO) << "  paged_kv_indptr shape: " << metadata.paged_kv_indptr.sizes()
              << ", values: " << metadata.paged_kv_indptr;

    // 5. 构造 paged_kv_indices
    // paged_kv_indices 是给一段式使用的，shape 为 [batch*beam_size * (step +
    // real_shared_kv_len_)] 对于每个 (batch, beam)，索引为: 0, 1, 2, ...,
    // real_shared_kv_len_-1, shared_kv_len+0, shared_kv_len+1, ...,
    // shared_kv_len+step 其中 step < max_decode_step_，为 0 或 1
    // 注意：shared_kv_len 在这里指的是 unshared 区域的起始偏移

    uint32_t unshared_kv_begin_offset =
        FLAGS_max_token_per_req * FLAGS_max_seqs_per_batch;

    // 每个 (batch, beam) 的有效长度为: real_shared_kv_len_ + (step + 1)
    int32_t total_len_per_beam = real_shared_kv_len_ + unshared_kv_len;
    int64_t total_elements = batch_size_ * beam_size_ * total_len_per_beam;

    std::vector<int32_t> paged_kv_indices_vec;
    paged_kv_indices_vec.reserve(total_elements);

    for (int64_t b = 0; b < batch_size_; ++b) {
      int32_t batch_offset = metadata.kv_cu_seq_lens[b].item<int32_t>();
      for (int64_t beam = 0; beam < beam_size_; ++beam) {
        // shared 部分: 0, 1, 2, ..., real_shared_kv_len_-1 (从 batch
        // 的起始偏移开始)
        for (int32_t i = 0; i < real_shared_kv_len_; ++i) {
          paged_kv_indices_vec.push_back(batch_offset + i);
        }

        // unshared 部分: shared_kv_len + 0, shared_kv_len + 1, ...,
        // shared_kv_len + step 其中 shared_kv_len = unshared_kv_begin_offset +
        // (b * beam_size + beam) * max_decode_step
        int64_t beam_idx = b * beam_size_ + beam;
        int64_t unshared_base =
            unshared_kv_begin_offset + beam_idx * unshared_kv_len;
        for (int32_t i = 0; i < unshared_kv_len; ++i) {
          paged_kv_indices_vec.push_back(
              static_cast<int32_t>(unshared_base + i));
        }
      }
    }

    // 转换为 tensor - 先创建在 CPU 上，然后移动到 CUDA
    auto cpu_options =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    metadata.paged_kv_indices =
        torch::from_blob(paged_kv_indices_vec.data(),
                         {static_cast<int64_t>(paged_kv_indices_vec.size())},
                         cpu_options)
            .clone()
            .to(device_);

    LOG(INFO) << "  paged_kv_indices construction:";
    LOG(INFO) << "    total_len_per_beam = " << total_len_per_beam
              << " (real_shared_kv_len=" << real_shared_kv_len_
              << " + unshared_kv_len=" << unshared_kv_len << ")";
    LOG(INFO) << "    unshared_kv_begin_offset = " << unshared_kv_begin_offset;
    if (batch_size_ == 1 && beam_size_ == 4) {
      // 打印第一个 beam 的索引作为示例
      LOG(INFO) << "    First beam indices (first " << total_len_per_beam
                << " values):";
      for (int32_t i = 0; i < total_len_per_beam && i < 20; ++i) {
        LOG(INFO) << "      [" << i << "] = " << paged_kv_indices_vec[i];
      }
    }
    LOG(INFO) << "  paged_kv_indices shape: "
              << metadata.paged_kv_indices.sizes()
              << ", values: " << metadata.paged_kv_indices;

    // 8. 计算 paged_kv_last_page_len: [batch_size * beam_size]，值都是 1
    metadata.paged_kv_last_page_len =
        torch::ones({batch_size_ * beam_size_}, int32_options_);
    LOG(INFO) << "  paged_kv_last_page_len shape: "
              << metadata.paged_kv_last_page_len.sizes()
              << ", values: " << metadata.paged_kv_last_page_len;

    // 创建 KV cache tensors
    // full_k_cache: [max_seqs_per_batch * max_token_per_req, num_kv_heads,
    // head_size]
    int64_t full_cache_size =
        FLAGS_max_seqs_per_batch * FLAGS_max_token_per_req +
        unshared_kv_len * total_beam_;
    metadata.full_k_cache =
        torch::randn({full_cache_size, num_kv_heads_, head_size_}, options_);
    metadata.full_v_cache =
        torch::randn({full_cache_size, num_kv_heads_, head_size_}, options_);

    // unshared_k_cache: [batch_size * beam_size, max_decode_step, num_kv_heads,
    // head_size] 先 slice 得到 3D tensor，再 view 成正确的 4D shape
    auto unshared_k_3d = metadata.full_k_cache.slice(
        0, unshared_kv_begin_offset, full_cache_size);
    auto unshared_v_3d = metadata.full_v_cache.slice(
        0, unshared_kv_begin_offset, full_cache_size);

    // view 成 4D: [batch_size * beam_size, max_decode_step, num_kv_heads,
    // head_size]
    metadata.unshared_k_cache =
        unshared_k_3d.view({static_cast<int64_t>(total_beam_),
                            max_decode_step_,
                            num_kv_heads_,
                            head_size_});
    metadata.unshared_v_cache =
        unshared_v_3d.view({static_cast<int64_t>(total_beam_),
                            max_decode_step_,
                            num_kv_heads_,
                            head_size_});

    LOG(INFO) << "  unshared_k_cache shape: "
              << metadata.unshared_k_cache.sizes() << " (expected: ["
              << total_beam_ << ", " << max_decode_step_ << ", "
              << num_kv_heads_ << ", " << head_size_ << "])";
    LOG(INFO) << "  unshared_v_cache shape: "
              << metadata.unshared_v_cache.sizes();
    LOG(INFO) << "  full_k_cache shape: " << metadata.full_k_cache.sizes();
    LOG(INFO) << "  full_v_cache shape: " << metadata.full_v_cache.sizes();

    // step tensor
    metadata.step = torch::tensor({step_value}, int32_options_);

    // 创建工作空间 buffers (按照 FlashinferWorkspace 的规范)
    // float_workspace_buffer: CUDA 上的 UInt8
    metadata.float_workspace_buffer = torch::empty(
        {128 * 1024 * 1024}, torch::dtype(torch::kUInt8).device(device_));
    // int_workspace_buffer: CUDA 上的 UInt8
    metadata.int_workspace_buffer = torch::empty(
        {8 * 1024 * 1024}, torch::dtype(torch::kUInt8).device(device_));
    // page_locked_int_workspace_buffer: CPU pinned memory 上的 UInt8
    metadata.page_locked_int_workspace_buffer = torch::empty(
        {8 * 1024 * 1024},
        torch::dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));

    return metadata;
  }

  // 创建 AttentionMetadata 用于一段式解码测试
  AttentionMetadata CreateSingleStageDecodeMetadata() {
    AttentionMetadata metadata = CreateTwoStageDecodeMetadata();
    // 一段式解码不需要 two_stage_decode_cache
    metadata.two_stage_decode_cache = std::nullopt;
    return metadata;
  }

  // 创建 query tensor
  torch::Tensor CreateQuery() {
    // query: [total_beam, num_heads, head_size]
    return torch::randn({total_beam_, num_heads_, head_size_}, options_);
  }

  // 创建 key tensor（用于一段式解码）
  torch::Tensor CreateKey() {
    // key: [batch_size, beam_size, num_kv_heads, head_size]
    return torch::randn({batch_size_, beam_size_, num_kv_heads_, head_size_},
                        options_);
  }

  // 创建 output tensor
  torch::Tensor CreateOutput() {
    // output: [total_beam, num_heads, head_size]
    return torch::zeros({total_beam_, num_heads_, head_size_}, options_);
  }

  torch::Device device_ = torch::Device(torch::kCPU);
  torch::TensorOptions options_;
  torch::TensorOptions fp32_options_;
  torch::TensorOptions int32_options_;

  uint32_t batch_size_ = 0;
  uint32_t beam_size_ = 0;
  uint32_t total_beam_ = 0;
  int64_t num_heads_ = 0;
  int64_t num_kv_heads_ = 0;
  int64_t head_size_ = 0;
  int64_t shared_kv_len_ = 0;
  int64_t real_shared_kv_len_ = 0;
  int64_t unshared_kv_len_ = 0;
  int64_t max_decode_step_ = 0;
  int64_t cache_step_ = 0;

  XAttention xattention_;
};

// 测试两段式解码函数
TEST_F(XAttentionDecodeTest, TwoStageDecodeBasic) {
  LOG(INFO) << "=== TwoStageDecodeBasic Test Start ===";

  auto metadata = CreateTwoStageDecodeMetadata();
  auto query = CreateQuery();
  auto output = CreateOutput();

  LOG(INFO) << "Test setup:";
  LOG(INFO) << "  query shape: " << query.sizes()
            << ", dtype: " << query.scalar_type();
  LOG(INFO) << "  output shape: " << output.sizes()
            << ", dtype: " << output.scalar_type();
  LOG(INFO) << "  full_k_cache shape: " << metadata.full_k_cache.sizes();
  LOG(INFO) << "  full_v_cache shape: " << metadata.full_v_cache.sizes();
  LOG(INFO) << "  unshared_k_cache shape: "
            << metadata.unshared_k_cache.sizes();
  LOG(INFO) << "  unshared_v_cache shape: "
            << metadata.unshared_v_cache.sizes();

  // 确保是第一层（用于初始化 cache）
  metadata.plan_info->layer_id = 0;

  // 执行两段式解码
  try {
    LOG(INFO) << "Calling TwoStageDecode...";
    xattention_->TwoStageDecode(
        metadata, query, output, batch_size_, beam_size_, total_beam_);
    LOG(INFO) << "TwoStageDecode completed successfully";
  } catch (const std::exception& e) {
    std::string error_msg = e.what();
    LOG(ERROR) << "Exception in TwoStageDecode: " << error_msg;
    if (error_msg.find("PyModule_Create2") != std::string::npos ||
        error_msg.find("DynamicLibrary") != std::string::npos) {
      GTEST_SKIP() << "FlashInfer AOT library requires Python symbols. "
                      "Please ensure Python libraries are properly linked. "
                      "Error: "
                   << error_msg;
    }
    throw;  // Re-throw if it's a different error
  }

  // 验证 output 是否被正确修改（不应该全是零）
  auto output_abs_sum = output.abs().sum().item<float>();
  EXPECT_GT(output_abs_sum, 0.0f) << "Output should not be all zeros";

  // 验证 two_stage_decode_cache 是否被初始化
  ASSERT_TRUE(metadata.two_stage_decode_cache.has_value())
      << "two_stage_decode_cache should be initialized";
}

// 测试两段式解码在非第一层的调用（复用 cache）
TEST_F(XAttentionDecodeTest, TwoStageDecodeReuseCache) {
  auto metadata = CreateTwoStageDecodeMetadata();
  auto query = CreateQuery();
  auto output = CreateOutput();

  // 第一层：初始化 cache
  metadata.plan_info->layer_id = 0;
  try {
    xattention_->TwoStageDecode(
        metadata, query, output, batch_size_, beam_size_, total_beam_);
  } catch (const std::exception& e) {
    std::string error_msg = e.what();
    if (error_msg.find("PyModule_Create2") != std::string::npos ||
        error_msg.find("DynamicLibrary") != std::string::npos) {
      GTEST_SKIP() << "FlashInfer AOT library requires Python symbols. "
                      "Error: "
                   << error_msg;
    }
    throw;
  }

  ASSERT_TRUE(metadata.two_stage_decode_cache.has_value());

  // 非第一层：应该复用 cache
  metadata.plan_info->layer_id = 1;
  auto query2 = CreateQuery();
  auto output2 = CreateOutput();

  try {
    xattention_->TwoStageDecode(
        metadata, query2, output2, batch_size_, beam_size_, total_beam_);
  } catch (const std::exception& e) {
    std::string error_msg = e.what();
    if (error_msg.find("PyModule_Create2") != std::string::npos ||
        error_msg.find("DynamicLibrary") != std::string::npos) {
      GTEST_SKIP() << "FlashInfer AOT library requires Python symbols. "
                      "Error: "
                   << error_msg;
    }
    throw;
  }

  // 验证 cache 仍然存在
  ASSERT_TRUE(metadata.two_stage_decode_cache.has_value())
      << "two_stage_decode_cache should still exist";
}

// 测试一段式解码函数
TEST_F(XAttentionDecodeTest, SingleStageDecodeBasic) {
  auto metadata = CreateSingleStageDecodeMetadata();
  auto query = CreateQuery();
  auto key = CreateKey();
  auto output = CreateOutput();

  // 执行一段式解码
  try {
    xattention_->SingleStageDecode(metadata, query, output, key, batch_size_);
  } catch (const std::exception& e) {
    std::string error_msg = e.what();
    if (error_msg.find("PyModule_Create2") != std::string::npos ||
        error_msg.find("DynamicLibrary") != std::string::npos) {
      GTEST_SKIP() << "FlashInfer AOT library requires Python symbols. "
                      "Error: "
                   << error_msg;
    }
    throw;
  }

  // 验证 output 是否被正确修改（不应该全是零）
  auto output_abs_sum = output.abs().sum().item<float>();
  EXPECT_GT(output_abs_sum, 0.0f) << "Output should not be all zeros";
}

// 测试不同 batch_size 的情况
TEST_F(XAttentionDecodeTest, TwoStageDecodeDifferentBatchSize) {
  // 测试较小的 batch_size
  batch_size_ = 1;
  beam_size_ = 2;
  total_beam_ = batch_size_ * beam_size_;

  auto metadata = CreateTwoStageDecodeMetadata();
  metadata.plan_info->layer_id = 0;

  auto query = CreateQuery();
  auto output = CreateOutput();

  try {
    xattention_->TwoStageDecode(
        metadata, query, output, batch_size_, beam_size_, total_beam_);
  } catch (const std::exception& e) {
    std::string error_msg = e.what();
    if (error_msg.find("PyModule_Create2") != std::string::npos ||
        error_msg.find("DynamicLibrary") != std::string::npos) {
      GTEST_SKIP() << "FlashInfer AOT library requires Python symbols. "
                      "Error: "
                   << error_msg;
    }
    throw;
  }
}

// 测试不同 beam_size 的情况
TEST_F(XAttentionDecodeTest, TwoStageDecodeDifferentBeamSize) {
  batch_size_ = 2;
  beam_size_ = 1;
  total_beam_ = batch_size_ * beam_size_;

  auto metadata = CreateTwoStageDecodeMetadata();
  metadata.plan_info->layer_id = 0;

  auto query = CreateQuery();
  auto output = CreateOutput();

  try {
    xattention_->TwoStageDecode(
        metadata, query, output, batch_size_, beam_size_, total_beam_);
  } catch (const std::exception& e) {
    std::string error_msg = e.what();
    if (error_msg.find("PyModule_Create2") != std::string::npos ||
        error_msg.find("DynamicLibrary") != std::string::npos) {
      GTEST_SKIP() << "FlashInfer AOT library requires Python symbols. "
                      "Error: "
                   << error_msg;
    }
    throw;
  }
}

// // 测试一段式解码在不同 batch_size 的情况
// TEST_F(XAttentionDecodeTest, SingleStageDecodeDifferentBatchSize) {
//   batch_size_ = 1;
//   beam_size_ = 2;
//   total_beam_ = batch_size_ * beam_size_;

//   auto metadata = CreateSingleStageDecodeMetadata();
//   auto query = CreateQuery();
//   auto key = CreateKey();
//   auto output = CreateOutput();

//   try {
//     xattention_->SingleStageDecode(metadata, query, output, key,
//     batch_size_);
//   } catch (const std::exception& e) {
//     std::string error_msg = e.what();
//     if (error_msg.find("PyModule_Create2") != std::string::npos ||
//         error_msg.find("DynamicLibrary") != std::string::npos) {
//       GTEST_SKIP() << "FlashInfer AOT library requires Python symbols. "
//                       "Error: " << error_msg;
//     }
//     throw;
//   }
// }

}  // namespace layer
}  // namespace xllm
