include(cc_library)

add_subdirectory(tilelang)

add_subdirectory(xllm_ops)

set(XLLM_MEGAKERNEL_ROOT "$ENV{MEGAKERNEL_ROOT}" CACHE PATH
    "Path to a megakernel source tree for optional xLLM integration")
if(NOT XLLM_MEGAKERNEL_ROOT)
  get_filename_component(_xllm_sibling_megakernel
    "${PROJECT_SOURCE_DIR}/../megakernel" ABSOLUTE)
  if(EXISTS "${_xllm_sibling_megakernel}/control/frontend/graph_builder.h")
    set(XLLM_MEGAKERNEL_ROOT "${_xllm_sibling_megakernel}")
  endif()
endif()

set(XLLM_MEGAKERNEL_DEFINES "")
set(XLLM_MEGAKERNEL_INCLUDES "")
if(XLLM_MEGAKERNEL_ROOT AND
   EXISTS "${XLLM_MEGAKERNEL_ROOT}/control/frontend/graph_builder.h")
  list(APPEND XLLM_MEGAKERNEL_DEFINES XLLM_ENABLE_MEGAKERNEL_COMPILE)
  list(APPEND XLLM_MEGAKERNEL_INCLUDES "${XLLM_MEGAKERNEL_ROOT}")
endif()

cc_library(
  NAME
    npu_kernels
  HDRS
    npu_ops_api.h
    aclnn/pytorch_npu_helper.hpp
  SRCS
    active.cpp
    attention.cpp
    fused_layernorm.cpp
    matmul.cpp
    megakernel/dense_ffn_runner.cpp
    megakernel/moe_single_expert_adapter.cpp
    npu_causal_conv1d.cpp
    npu_fused_infer_attention.cpp
    npu_gemma_rms_norm.cpp
    npu_grouped_matmul.cpp
    npu_moe_gating_topk_softmax.cpp
    npu_dispatch_ffn_combine.cpp
    npu_dispatch_gmm_combine_decode.cpp
    npu_moe_distribute_combine_v2.cpp
    npu_moe_distribute_dispatch_v2.cpp
    npu_moe_init_routing_v2.cpp
    npu_moe_token_unpermute.cpp
    rope.cpp
    npu_recurrent_gated_delta_rule.cpp
    utils.cpp
    w4a8_dynamic_moe_preprocess.cpp
    rec_constrained_topk.cpp
  DEPS
    :torch_npu_kernels
    :tilelang_kernels
    xllm_ops
    opapi
  DEFINES
    ${XLLM_MEGAKERNEL_DEFINES}
  INCLUDES
    ${XLLM_MEGAKERNEL_INCLUDES}
)
