#!/usr/bin/env bash
# 下载模型源文件（XHToken/Spark-X2.6-4B，Apache-2.0）。
# 默认走 ModelScope（国内直连可用、省流量）；--hf 改走 HuggingFace。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${DEST:-$ROOT/model/Spark-X2.6-4B}"
REPO="XHToken/Spark-X2.6-4B"

if [ "${1:-}" = "--hf" ]; then
  echo "[download] HuggingFace: $REPO -> $DEST"
  echo "  需要 huggingface_hub；国内可先 export HF_ENDPOINT=https://hf-mirror.com"
  python3 - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
p = snapshot_download(repo, allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt", "spark2_5/*"],
                      local_dir=dest)
print("done:", p)
PY
  exit 0
fi

echo "[download] ModelScope: $REPO -> $DEST"
if ! python3 -c "import modelscope" >/dev/null 2>&1; then
  echo "[download] 需要 modelscope：pip install modelscope" >&2
  exit 1
fi
python3 - "$REPO" "$DEST" <<'PY'
import sys
from modelscope import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
p = snapshot_download(repo, local_dir=dest,
                      allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt", "*.model", "spark2_5/*"])
print("done:", p)
PY

echo
echo "[download] 完成。下一步（PC 侧转换，需要 x86_64 Linux + CUDA）："
echo "  convert/run_conversion.sh --model-dir $DEST --ctx 73728"
