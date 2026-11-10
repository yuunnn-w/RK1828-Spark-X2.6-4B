#!/usr/bin/env bash
# 取 Rockchip RKNN3 SDK（rknn3_api 头文件与板端运行时库）。
# 注意：这部分受 Rockchip 自己的 "RKNN SDK License" 约束，不能进本仓库，只能在你自己机器上克隆。
# 用法：
#   scripts/get_sdk.sh                # 克隆 airockchip/rknn3-toolkit 到仓库根 .sdk/（tag V1.2.0）
#   RKNN3_SDK_ROOT=/path scripts/...  # 已有 SDK 就不必克隆
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${RKNN3_SDK_ROOT:-$ROOT/.sdk}"
TAG="${RKNN3_TAG:-V1.2.0}"

if [ -d "$SDK/rknn3-runtime/rknn3-api/include" ]; then
  echo "[get_sdk] 已存在：$SDK"
  exit 0
fi

echo "[get_sdk] 克隆 airockchip/rknn3-toolkit ($TAG) -> $SDK"
# 国内网络若直连 GitHub 慢，可自行设置：git -c http.proxy=http://127.0.0.1:7890 ...
git clone --depth 1 --branch "$TAG" https://github.com/airockchip/rknn3-toolkit.git "$SDK"

echo "[get_sdk] 完成。头文件：$SDK/rknn3-runtime/rknn3-api/include/rknn3_api.h"
ls -1 "$SDK/rknn3-runtime/rknn3-api/lib/Linux/aarch64" 2>/dev/null | sed 's/^/  /'
