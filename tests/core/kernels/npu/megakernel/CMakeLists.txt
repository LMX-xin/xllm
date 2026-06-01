include(cc_test)

cc_test(
  NAME
    moe_single_expert_adapter_test
  SRCS
    moe_single_expert_adapter_test.cpp
    ${PROJECT_SOURCE_DIR}/xllm/core/kernels/npu/megakernel/moe_single_expert_adapter.cpp
  DEPS
    GTest::gtest_main
  INCLUDES
    ${PROJECT_SOURCE_DIR}/xllm/core
)

cc_test(
  NAME
    dense_ffn_runner_test
  SRCS
    dense_ffn_runner_test.cpp
  DEPS
    :npu_kernels
    torch
    GTest::gtest_main
    glog::glog
)
