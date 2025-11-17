cd /export/home/limenxin1/xllm_lmx/xllm

# 1) 检查并清理 Git 代理（若有）
git config --global --get http.proxy || true
git config --global --get https.proxy || true
git config --global --unset http.proxy || true
git config --global --unset https.proxy || true
export http_proxy=
export https_proxy=

# 2) 同步并不做浅克隆，完整拉取所有子模块
git submodule sync --recursive
GIT_TRACE=1 GIT_CURL_VERBOSE=1 git submodule update --init --recursive --jobs 4

# 3) 针对嵌套子模块再执行一遍（xllm_ops 自带 catlass 子模块）
cd third_party/xllm_ops
git submodule sync --recursive
GIT_TRACE=1 GIT_CURL_VERBOSE=1 git submodule update --init --recursive --jobs 4
