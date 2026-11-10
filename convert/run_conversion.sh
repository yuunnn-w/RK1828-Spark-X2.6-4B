#!/usr/bin/env bash
# Spark-X2.6-4B 的一键转换流水线（在 PC 上跑，不是在板子上跑）：
#   1) 下载模型（ModelScope 优先）
#   2) GRQ 量化导出 ONNX（w4a16 / group32，需要 CUDA）+ tokenizer.gguf + embed.bin
#   3) 转成 RKNN（指定上下文长度与 KV 精度）
# 要求：x86_64 Linux（或 macOS arm64）、Python 3.12、NVIDIA GPU（GRQ 需要 CUDA；无 GPU 会静默跳过量化，精度档位会变）
#
# 用法：
#   convert/run_conversion.sh --model-dir model/Spark-X2.6-4B --ctx 73728 --out model/out
#   convert/run_conversion.sh --ctx 65536 --kv int4        # int4 | fp16 | int8
#   convert/run_conversion.sh --no-quant                  # 跳过 GRQ（无 CUDA 时用于打通流程，精度会差）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT/model/Spark-X2.6-4B}"
OUT="${OUT:-$ROOT/model/out}"
CTX="${CTX:-73728}"
KV="${KV:-int4}"
CORES="${CORES:-8}"
SEQ="${SEQ:-128}"
PLATFORM="${PLATFORM:-rk1828}"
CALI="${CALI:-$ROOT/datasets/calib_real.json}"
MZ="${MODEL_ZOO:-$ROOT/.model-zoo}"
DO_QUANT=1
SIZE="4B"

while [ $# -gt 0 ]; do
  case "$1" in
    --model-dir) MODEL_DIR=$2; shift 2 ;;
    --out)       OUT=$2; shift 2 ;;
    --ctx)       CTX=$2; shift 2 ;;
    --kv)        KV=$2; shift 2 ;;
    --cores)     CORES=$2; shift 2 ;;
    --seq)       SEQ=$2; shift 2 ;;
    --platform)  PLATFORM=$2; shift 2 ;;
    --cali)      CALI=$2; shift 2 ;;
    --model-size) SIZE=$2; shift 2 ;;
    --no-quant)  DO_QUANT=0; shift ;;
    -h|--help)   sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数 $1" >&2; exit 1 ;;
  esac
done

log() { echo "[conv $(date '+%T')] $*"; }
mkdir -p "$OUT"

# ---- 0) 前置检查 ----
if [ "$(uname -s)" != "Linux" ] && [ "$(uname -s)" != "Darwin" ]; then
  echo "[conv] RKNN3 Toolkit 只发 x86_64 Linux / macOS arm64 的 wheel" >&2; exit 1
fi
python3 - <<'PY' || { echo "[conv] 先安装 rknn3-toolkit 1.2.0 的 wheel（见 docs/build.md）与 model-zoo 依赖"; exit 1; }
from rknn.api import RKNN, DEFAULT_RKNN_LLM_CONFIG  # noqa
PY
if [ ! -d "$MZ/examples/Spark_X2_5/python" ]; then
  log "克隆 rknn3-model-zoo (V1.2.0) -> $MZ"
  git clone --depth 1 --branch V1.2.0 https://github.com/airockchip/rknn3-model-zoo.git "$MZ"
fi
if [ ! -d "$MODEL_DIR" ]; then
  log "模型目录不存在，先下载"
  "$ROOT/model/download_model.sh"
fi

export PYTHONPATH="$MZ"

# ---- 1) GRQ 量化 + ONNX + tokenizer.gguf + embed.bin ----
cd "$MZ/examples/Spark_X2_5/python"
QUANT_ARGS=()
if [ "$DO_QUANT" = "1" ]; then
  if ! python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[conv] 警告：本机没有可用 CUDA —— GRQ 会被官方脚本静默跳过。要真量化请在有 GPU 的机器上跑，或加 --no-quant 明确接受 w4a16/normal 路径。" >&2
  fi
  QUANT_ARGS=(--quant --cali_dataset "$CALI")
  log "GRQ 量化导出 ONNX（w4a16 / group32，校准集 $CALI）"
else
  log "导出 ONNX（不做 GRQ）"
fi
python3 export_llm.py \
  --model_path "$MODEL_DIR" \
  --model_size "$SIZE" \
  --export_llm_path "$OUT/Spark-X2.6-4B.onnx" \
  "${QUANT_ARGS[@]}"

# ---- 2) 转 RKNN（我们的 export_rknn.py，参数用环境变量给）----
case "$KV" in
  int4) KVD=Int4_to_F16; KVM=GroupQuant; KVG=16; KVR=64 ;;
  int8) KVD=Int8_to_F16; KVM=GroupQuant; KVG=16; KVR=64 ;;
  fp16) KVD=Float16;     KVM=Normal;     KVG=32; KVR=32 ;;
  *) echo "[conv] --kv 只支持 int4 | int8 | fp16" >&2; exit 1 ;;
esac
SUF="kvi4"; [ "$KV" = "fp16" ] && SUF="kvfp16"; [ "$KV" = "int8" ] && SUF="kvi8"
NAME="Spark-X2.6-4B-${CTX}-${SUF}"
log "转换 RKNN：name=$NAME ctx=$CTX kv=$KVD/$KVM/g$KVG/r$KVR cores=$CORES platform=$PLATFORM"

# 每个 build 用独立工作目录（toolkit 的缓存目录是 CWD 相对的 ./tmp）
BUILDDIR="$OUT/build_$NAME"
mkdir -p "$BUILDDIR" && cd "$BUILDDIR"
SPARK_MODEL_SIZE="$SIZE" \
SPARK_ONNX="$OUT/Spark-X2.6-4B.onnx" \
SPARK_CONFIG="$OUT/Spark-X2.6-4B.config.pkl" \
SPARK_RKNN="$OUT/$NAME.rknn" \
SPARK_PLATFORM="$PLATFORM" \
SPARK_CTX="$CTX" SPARK_SEQ="$SEQ" SPARK_CORES="$CORES" \
SPARK_KV_DTYPE="$KVD" SPARK_KV_METHOD="$KVM" SPARK_KV_GROUP="$KVG" SPARK_KV_RESID="$KVR" \
SPARK_HOST_ROPE=1 SPARK_DO_QUANT="$DO_QUANT" \
python3 "$ROOT/convert/export_rknn.py"

cp -f "$OUT/Spark-X2.6-4B.tokenizer.gguf" "$OUT/Spark-X2.6-4B.tokenizer.gguf" 2>/dev/null || true
cp -f "$OUT/Spark-X2.6-4B.embed.bin" "$OUT/Spark-X2.6-4B.embed.bin" 2>/dev/null || true

log "产物："
ls -lh "$OUT" | grep -E "$NAME|tokenizer.gguf|embed.bin" || ls -lh "$OUT"
cat <<EOF

下一步（板端）：
  1) 把 $OUT/$NAME.rknn / .weight / .safetensors 与 Spark-X2.6-4B.tokenizer.gguf / .embed.bin
     拷到板上的 /opt/rk1828-spark/model/
  2) 先做内存判定（关键！Per-Core 合计必须 <= 5051.80 MB，否则 session_init 会失败）：
       rknn3_model_info /opt/rk1828-spark/model/$NAME.rknn 0xff
  3) 改 /etc/rk1828-spark.env 的 MODEL_NAME=$NAME 与 CTX=$CTX，然后：
       systemctl restart rk1828-spark.service && curl -s localhost:8080/health
EOF
