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

#include "kernels/npu/megakernel/dense_ffn_runner.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(XLLM_ENABLE_MEGAKERNEL_COMPILE)
#include <acl/acl.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/core/npu/register/OptionRegister.h>

#include "control/binding/common/control_tensor_serializer.h"
#include "control/frontend/graph_builder.h"
#include "control/frontend/plan.h"
#include "kernels/npu/aclnn/pytorch_npu_helper.hpp"
#endif

namespace xllm::kernel::npu::megakernel {
namespace {

constexpr const char* kEnableEnv = "XLLM_ENABLE_MEGAKERNEL_SINGLE_EXPERT_MOE";

bool parse_bool_env(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  std::string normalized(value);
  std::transform(
      normalized.begin(),
      normalized.end(),
      normalized.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return normalized == "1" || normalized == "true" || normalized == "on" ||
         normalized == "yes";
}

#if defined(XLLM_ENABLE_MEGAKERNEL_COMPILE)
constexpr uint32_t kDefaultCubeTileMHint = 128;
constexpr size_t kMaxControlCacheEntries = 128;
constexpr size_t kMaxWeightCacheEntries = 256;

struct ControlCacheEntry {
  torch::Tensor worker_queue_plan_cpu;
  torch::Tensor task_desc_table_cpu;
  torch::Tensor sync_resource_table_cpu;
  std::vector<std::vector<int64_t>> data_output_shapes;
  std::vector<torch::ScalarType> data_output_dtypes;
  size_t user_output_slot = 0;
};

bool is_megakernel_op_available() {
  static const bool is_available =
      aclnn::detail::get_op_api_func_addr("aclnnMegakernelGetWorkspaceSize") !=
          nullptr &&
      aclnn::detail::get_op_api_func_addr("aclnnMegakernel") != nullptr;
  return is_available;
}

megakernel_control::ModuleTensorDType module_dtype_from_tensor(
    const torch::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case torch::kChar:
      return megakernel_control::ModuleTensorDType::kInt8;
    case torch::kInt:
      return megakernel_control::ModuleTensorDType::kInt32;
    case torch::kFloat:
      return megakernel_control::ModuleTensorDType::kFloat32;
    case torch::kHalf:
      return megakernel_control::ModuleTensorDType::kFloat16;
    case torch::kBFloat16:
      return megakernel_control::ModuleTensorDType::kBFloat16;
    default:
      TORCH_CHECK(false, "unsupported megakernel dense FFN tensor dtype");
  }
}

torch::ScalarType scalar_type_from_module_dtype(
    megakernel_control::ModuleTensorDType dtype) {
  switch (dtype) {
    case megakernel_control::ModuleTensorDType::kInt8:
      return torch::kChar;
    case megakernel_control::ModuleTensorDType::kInt32:
      return torch::kInt;
    case megakernel_control::ModuleTensorDType::kFloat32:
      return torch::kFloat;
    case megakernel_control::ModuleTensorDType::kFloat16:
      return torch::kHalf;
    case megakernel_control::ModuleTensorDType::kBFloat16:
      return torch::kBFloat16;
  }
  TORCH_CHECK(false, "unsupported megakernel dense FFN output dtype");
}

megakernel_control::ModuleTensorMeta module_meta_from_tensor(
    const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.defined(), "megakernel dense FFN expects defined tensors");
  megakernel_control::ModuleTensorMeta meta;
  meta.dtype = module_dtype_from_tensor(tensor);
  for (const int64_t dim : tensor.sizes()) {
    TORCH_CHECK(dim >= 0, "megakernel tensor shape dim must be non-negative");
    meta.shape.push_back(static_cast<uint32_t>(dim));
  }
  meta.layout = tensor.is_contiguous() ? "contiguous" : "non_contiguous";
  meta.device = tensor.device().str();
  return meta;
}

const megakernel_control::ModuleTensor& module_tensor_by_id(
    const megakernel_control::ModuleGraphPayload& payload,
    uint32_t tensor_id) {
  for (const megakernel_control::ModuleTensor& tensor : payload.tensors) {
    if (tensor.tensor_id == tensor_id) {
      return tensor;
    }
  }
  TORCH_CHECK(false, "unknown megakernel plan tensor id");
}

std::vector<int64_t> shape_from_module_tensor(
    const megakernel_control::ModuleTensor& tensor) {
  std::vector<int64_t> shape;
  shape.reserve(tensor.meta.shape.size());
  for (const uint32_t dim : tensor.meta.shape) {
    shape.push_back(static_cast<int64_t>(dim));
  }
  return shape;
}

torch::Tensor as_byte_view(const torch::Tensor& tensor) {
  if (tensor.scalar_type() == torch::kChar) {
    return tensor;
  }
  return tensor.view(torch::kChar);
}

torch::Tensor control_to_device_byte_tensor(const torch::Tensor& tensor,
                                            const torch::Tensor& anchor) {
  return tensor.contiguous().to(anchor.options().dtype(torch::kChar));
}

std::string make_control_cache_key(int64_t token_count,
                                   int64_t hidden_size,
                                   int64_t intermediate_size) {
  std::ostringstream stream;
  stream << "dense_ffn_fp16_no_quant"
         << ";t=" << token_count << ";h=" << hidden_size
         << ";i=" << intermediate_size << ";target=ascend910c";
  return stream.str();
}

uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
  return divisor == 0U ? 0U : (value + divisor - 1U) / divisor;
}

uint32_t pick_cube_tile_m_hint(int64_t token_count,
                               uint32_t cube_worker_count) {
  if (token_count <= 0 || cube_worker_count == 0U) {
    return kDefaultCubeTileMHint;
  }

  const uint32_t m = static_cast<uint32_t>(token_count);
  const uint32_t default_row_workers =
      std::min(cube_worker_count, ceil_div_u32(m, kDefaultCubeTileMHint));
  if (default_row_workers != 0U && (m % default_row_workers) == 0U) {
    return kDefaultCubeTileMHint;
  }

  // The current megakernel compiler requires m_size % row_workers == 0.
  const uint32_t max_row_workers = std::min(cube_worker_count, m);
  for (uint32_t row_workers = max_row_workers; row_workers > 0U;
       --row_workers) {
    if ((m % row_workers) != 0U) {
      continue;
    }
    const uint32_t tile_m = m / row_workers;
    if (tile_m <= kDefaultCubeTileMHint) {
      return tile_m;
    }
  }
  return kDefaultCubeTileMHint;
}

std::unordered_map<std::string, ControlCacheEntry>& control_cache() {
  static std::unordered_map<std::string, ControlCacheEntry> cache;
  return cache;
}

std::mutex& control_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

bool compile_or_get_controls(const torch::Tensor& x,
                             const torch::Tensor& w1,
                             const torch::Tensor& w2,
                             int64_t hidden_size,
                             int64_t intermediate_size,
                             ControlCacheEntry* entry) {
  if (entry == nullptr) {
    return false;
  }

  const std::string cache_key =
      make_control_cache_key(x.size(0), hidden_size, intermediate_size);
  {
    std::lock_guard<std::mutex> lock(control_cache_mutex());
    auto it = control_cache().find(cache_key);
    if (it != control_cache().end()) {
      *entry = it->second;
      return true;
    }
    if (control_cache().size() >= kMaxControlCacheEntries) {
      return false;
    }
  }

  megakernel_control::GraphBuilder builder;
  const megakernel_control::GraphTensorHandle x_handle =
      builder.Input("x", module_meta_from_tensor(x));
  const megakernel_control::GraphTensorHandle w1_handle =
      builder.Parameter("w1", module_meta_from_tensor(w1));
  const megakernel_control::GraphTensorHandle w2_handle =
      builder.Parameter("w2", module_meta_from_tensor(w2));

  megakernel_control::GemmAttrs gemm1_attrs;
  gemm1_attrs.out_dtype = megakernel_control::ModuleTensorDType::kFloat16;
  const megakernel_control::GraphTensorHandle h =
      builder.Gemm(x_handle, w1_handle, gemm1_attrs);

  megakernel_control::SwigluAttrs swiglu_attrs;
  swiglu_attrs.beta = 1.0F;
  const megakernel_control::GraphTensorHandle y =
      builder.Swiglu(h, swiglu_attrs);

  megakernel_control::GemmAttrs gemm2_attrs;
  gemm2_attrs.out_dtype = megakernel_control::ModuleTensorDType::kFloat16;
  const megakernel_control::GraphTensorHandle out =
      builder.Gemm(y, w2_handle, gemm2_attrs);
  builder.Output(out);

  const auto target_model = megakernel_control::MakeAscendNpu910CTargetModel();
  megakernel_control::CompileOptions options;
  options.cube_tile_m_hint =
      pick_cube_tile_m_hint(x.size(0), target_model.workers.cube_worker_count);
  options.pad_worker_queues_to_target = false;
  const megakernel_control::MegakernelPlan plan =
      megakernel_control::CompileModuleGraphPayloadToPlan(
          builder.Finish(), options, target_model);
  if (!plan.ok()) {
    for (const auto& diagnostic : plan.diagnostics) {
      LOG(WARNING) << "megakernel dense FFN compile diagnostic: "
                   << diagnostic.message;
    }
    return false;
  }
  if (plan.runtime_metadata.user_output_tensor_ids.size() != 1) {
    LOG(WARNING) << "megakernel dense FFN expected one user output, got "
                 << plan.runtime_metadata.user_output_tensor_ids.size();
    return false;
  }

  auto controls = megakernel_control::binding::SerializeControlsToCpuTensors(
      plan.control_tables, "xllm dense ffn");
  ControlCacheEntry compiled_entry;
  compiled_entry.worker_queue_plan_cpu = std::get<0>(controls);
  compiled_entry.task_desc_table_cpu = std::get<1>(controls);
  compiled_entry.sync_resource_table_cpu = std::get<2>(controls);
  std::unordered_map<uint32_t, size_t> output_slot_by_tensor_id;
  for (const uint32_t tensor_id :
       plan.runtime_metadata.data_output_tensor_ids) {
    const megakernel_control::ModuleTensor& tensor =
        module_tensor_by_id(plan.payload, tensor_id);
    if (tensor.has_view) {
      LOG(WARNING) << "megakernel dense FFN data output view is unsupported";
      return false;
    }
    output_slot_by_tensor_id[tensor.tensor_id] =
        compiled_entry.data_output_shapes.size();
    compiled_entry.data_output_shapes.push_back(
        shape_from_module_tensor(tensor));
    compiled_entry.data_output_dtypes.push_back(
        scalar_type_from_module_dtype(tensor.meta.dtype));
  }
  const auto user_output_iter = output_slot_by_tensor_id.find(
      plan.runtime_metadata.user_output_tensor_ids.front());
  if (user_output_iter == output_slot_by_tensor_id.end()) {
    LOG(WARNING) << "megakernel dense FFN user output has no data output slot";
    return false;
  }
  compiled_entry.user_output_slot = user_output_iter->second;

  {
    std::lock_guard<std::mutex> lock(control_cache_mutex());
    control_cache().emplace(cache_key, compiled_entry);
  }
  *entry = std::move(compiled_entry);
  return true;
}

class InternalFormatGuard {
 public:
  InternalFormatGuard()
      : old_value_(c10_npu::option::GetOption("ALLOW_INTERNAL_FORMAT")) {
    c10_npu::option::SetOption("ALLOW_INTERNAL_FORMAT", "enable");
  }

  ~InternalFormatGuard() {
    c10_npu::option::SetOption(
        "ALLOW_INTERNAL_FORMAT",
        old_value_.has_value() ? old_value_.value() : "disable");
  }

 private:
  c10::optional<std::string> old_value_;
};

int64_t get_tensor_npu_format(const torch::Tensor& tensor) {
#ifdef TORCH_HIGHER_THAN_PTA6
  return at_npu::native::custom_ops::get_npu_format(tensor);
#else
  return at_npu::native::NPUNativeFunctions::get_npu_format(tensor);
#endif
}

torch::Tensor npu_format_cast(const torch::Tensor& tensor, int64_t format) {
#ifdef TORCH_HIGHER_THAN_PTA6
  return at_npu::native::custom_ops::npu_format_cast(tensor, format);
#else
  return at_npu::native::NPUNativeFunctions::npu_format_cast(tensor, format);
#endif
}

std::string make_weight_cache_key(const torch::Tensor& weight,
                                  bool swap_swiglu_halves) {
  std::ostringstream stream;
  stream << reinterpret_cast<uintptr_t>(weight.data_ptr())
         << ";dtype=" << c10::toString(weight.scalar_type())
         << ";offset=" << weight.storage_offset()
         << ";format=" << get_tensor_npu_format(weight)
         << ";swap_swiglu_halves=" << (swap_swiglu_halves ? 1 : 0) << ";sizes=";
  for (const int64_t size : weight.sizes()) {
    stream << size << ",";
  }
  stream << ";strides=";
  for (const int64_t stride : weight.strides()) {
    stream << stride << ",";
  }
  return stream.str();
}

std::unordered_map<std::string, torch::Tensor>& weight_cache() {
  static std::unordered_map<std::string, torch::Tensor> cache;
  return cache;
}

std::mutex& weight_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

torch::Tensor swap_swiglu_halves_for_value_gate(const torch::Tensor& weight) {
  const int64_t half_size = weight.size(1) / 2;
  std::vector<torch::Tensor> halves = {
      weight.slice(1, half_size, half_size * 2),
      weight.slice(1, 0, half_size),
  };
  return torch::cat(halves, 1);
}

bool get_or_pack_weight_nz(const torch::Tensor& weight,
                           bool swap_swiglu_halves,
                           torch::Tensor* packed_weight) {
  if (packed_weight == nullptr || !weight.defined() ||
      weight.scalar_type() != torch::kHalf || weight.device().is_cpu()) {
    return false;
  }

  if (swap_swiglu_halves && (weight.dim() != 2 || (weight.size(1) % 2) != 0)) {
    return false;
  }
  if (swap_swiglu_halves &&
      get_tensor_npu_format(weight) == ACL_FORMAT_FRACTAL_NZ) {
    return false;
  }

  if (!swap_swiglu_halves &&
      get_tensor_npu_format(weight) == ACL_FORMAT_FRACTAL_NZ) {
    *packed_weight = weight;
    return true;
  }

  const std::string cache_key =
      make_weight_cache_key(weight, swap_swiglu_halves);
  {
    std::lock_guard<std::mutex> lock(weight_cache_mutex());
    auto it = weight_cache().find(cache_key);
    if (it != weight_cache().end()) {
      *packed_weight = it->second;
      return true;
    }
    if (weight_cache().size() >= kMaxWeightCacheEntries) {
      return false;
    }
  }

  InternalFormatGuard internal_format_guard;
  torch::Tensor weight_for_pack =
      swap_swiglu_halves ? swap_swiglu_halves_for_value_gate(weight) : weight;
  torch::Tensor packed =
      npu_format_cast(weight_for_pack.contiguous(), ACL_FORMAT_FRACTAL_NZ);
  {
    std::lock_guard<std::mutex> lock(weight_cache_mutex());
    weight_cache().emplace(cache_key, packed);
  }
  *packed_weight = std::move(packed);
  return true;
}

bool validate_params(const DenseFfnRunParams& params) {
  if (!params.x.defined() || !params.w1.defined() || !params.w2.defined() ||
      params.x.device().is_cpu() || params.w1.device().is_cpu() ||
      params.w2.device().is_cpu() || params.x.scalar_type() != torch::kHalf ||
      params.w1.scalar_type() != torch::kHalf ||
      params.w2.scalar_type() != torch::kHalf || params.x.dim() != 2 ||
      params.w1.dim() != 2 || params.w2.dim() != 2 || params.hidden_size <= 0 ||
      params.intermediate_size <= 0) {
    return false;
  }

  return params.x.size(1) == params.hidden_size &&
         params.w1.size(0) == params.hidden_size &&
         params.w1.size(1) == params.intermediate_size * 2 &&
         params.w2.size(0) == params.intermediate_size &&
         params.w2.size(1) == params.hidden_size;
}

#endif

}  // namespace

bool is_single_expert_moe_enabled() { return parse_bool_env(kEnableEnv); }

bool try_run_dense_ffn_fp16_no_quant(const DenseFfnRunParams& params,
                                     torch::Tensor* output) {
#if !defined(XLLM_ENABLE_MEGAKERNEL_COMPILE)
  (void)params;
  (void)output;
  return false;
#else
  if (output == nullptr || !validate_params(params) ||
      !is_megakernel_op_available()) {
    return false;
  }

  try {
    torch::Tensor w1_nz;
    torch::Tensor w2_nz;
    if (!get_or_pack_weight_nz(params.w1, true, &w1_nz) ||
        !get_or_pack_weight_nz(params.w2, false, &w2_nz)) {
      return false;
    }

    ControlCacheEntry controls;
    if (!compile_or_get_controls(params.x,
                                 w1_nz,
                                 w2_nz,
                                 params.hidden_size,
                                 params.intermediate_size,
                                 &controls)) {
      return false;
    }

    std::vector<torch::Tensor> data_inputs = {
        as_byte_view(params.x),
        as_byte_view(w1_nz),
        as_byte_view(w2_nz),
    };
    std::vector<torch::Tensor> data_outputs;
    std::vector<torch::Tensor> output_byte_views;
    data_outputs.reserve(controls.data_output_shapes.size());
    output_byte_views.reserve(controls.data_output_shapes.size());
    for (size_t i = 0; i < controls.data_output_shapes.size(); ++i) {
      data_outputs.push_back(torch::empty(
          controls.data_output_shapes[i],
          params.x.options().dtype(controls.data_output_dtypes[i])));
      output_byte_views.push_back(as_byte_view(data_outputs.back()));
    }
    torch::TensorList data_inputs_tensor_list(data_inputs);
    torch::TensorList data_outputs_tensor_list(output_byte_views);
    torch::Tensor worker_queue_plan =
        control_to_device_byte_tensor(controls.worker_queue_plan_cpu, params.x);
    torch::Tensor task_desc_table =
        control_to_device_byte_tensor(controls.task_desc_table_cpu, params.x);
    torch::Tensor sync_resource_table = control_to_device_byte_tensor(
        controls.sync_resource_table_cpu, params.x);
    int64_t active_cube_worker_count = 0;
    int64_t active_vector_worker_count = 0;

    EXEC_NPU_CMD(aclnnMegakernel,
                 data_inputs_tensor_list,
                 data_outputs_tensor_list,
                 worker_queue_plan,
                 task_desc_table,
                 sync_resource_table,
                 active_cube_worker_count,
                 active_vector_worker_count);
    if (controls.user_output_slot >= data_outputs.size()) {
      return false;
    }
    *output = data_outputs[controls.user_output_slot];
    return true;
  } catch (const std::exception& error) {
    LOG(WARNING) << "megakernel dense FFN fallback: " << error.what();
    return false;
  }
#endif
}

}  // namespace xllm::kernel::npu::megakernel
