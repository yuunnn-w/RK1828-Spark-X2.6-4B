#!/usr/bin/env bash
# 编译自研推理服务 rknn_spark_server（默认在 RK1828 开发板上本机编译，aarch64）。
# 交叉编译：先 export CC=<aarch64-linux-gnu-gcc> CXX=<aarch64-linux-gnu-g++>，再跑本脚本。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${RKNN3_SDK_ROOT:-$ROOT/.sdk}"
BUILD="${BUILD_DIR:-$ROOT/server/build}"

if [ ! -f "$SDK/rknn3-runtime/rknn3-api/include/rknn3_api.h" ]; then
  echo "[build] 找不到 RKNN3 SDK，先执行 scripts/get_sdk.sh（或设置 RKNN3_SDK_ROOT）" >&2
  exit 1
fi

cmake -S "$ROOT/server" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DRKNN3_SDK_ROOT="$SDK"
cmake --build "$BUILD" -j"$(nproc)"

# 把运行时 .so 收进 build/lib（可执行文件的 RPATH 是 $ORIGIN/lib，所以整个目录一起拷走就能跑）
mkdir -p "$BUILD/lib"
found=0
for l in $(find "$SDK" /usr/lib /usr/local/lib -maxdepth 6 -name 'librknn3_api*.so*' 2>/dev/null | sort -u); do
  cp -nL "$l" "$BUILD/lib/" && found=1
done
if [ "$found" -eq 0 ]; then
  echo "[build] 警告：没在 SDK 目录里找到 librknn3_api*.so，运行前请自行确保 LD_LIBRARY_PATH 或 ./lib 里有它们" >&2
fi

echo
echo "[build] OK: $BUILD/rknn_spark_server"
"$BUILD/rknn_spark_server" --help 2>&1 | head -6 || true
