include(cc_library)
include(cc_test)

cc_library(
  NAME
    cuda_layers
  HDRS
    attention.h
    xattention.h
    flashinfer_workspace.h
    flashinfer_planinfo.h
  SRCS
    attention.cpp
    xattention.cpp
    flashinfer_workspace.cpp
    flashinfer_planinfo.cpp
  DEPS
    :common_layers
)

cc_test(
  NAME
    xattention_decode_test
  SRCS
    xattention_test.cpp
  DEPS
    :cuda_layers
    :common_layers
    :kernels
    :flags
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(xattention_decode_test PRIVATE brpc Python::Python torch_python)
