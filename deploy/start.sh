#!/bin/sh
# 启动包装：读环境变量配置文件（默认 /etc/rk1828-spark.env）里可选的覆盖项，然后 exec 服务进程。
# 这样改参数不必动 systemd 单元；systemd 只负责常驻与重启。
PREFIX=${RK1828_SPARK_PREFIX:-${PREFIX:-/opt/rk1828-spark}}
CONF=${RK1828_SPARK_CONF:-/etc/rk1828-spark.env}
if [ -f "$CONF" ]; then
    # shellcheck disable=SC1090
    . "$CONF"
fi

BIN=${BIN:-$PREFIX/bin/rknn_spark_server}
MODEL_DIR=${MODEL_DIR:-$PREFIX/model}
MODEL_NAME=${MODEL_NAME:-Spark-X2.6-4B-73728-kvi4}
BASE_NAME=${BASE_NAME:-Spark-X2.6-4B}
CORE_MASK=${CORE_MASK:-0xff}
CTX=${CTX:-73728}

# 若把 librknn3_api.so / librknn3_api_rkcp.so 放在 $PREFIX/lib 而不是系统目录，这里会生效
if [ -d "$PREFIX/lib" ]; then
    LD_LIBRARY_PATH="$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH
fi

exec env TMPDIR="${TMPDIR:-/run/rknn3}" "$BIN" \
    "$MODEL_DIR/$MODEL_NAME.rknn" \
    "$MODEL_DIR/$MODEL_NAME.weight" \
    "$CORE_MASK" \
    "$MODEL_DIR/$BASE_NAME.tokenizer.gguf" \
    "$MODEL_DIR/$BASE_NAME.embed.bin" \
    "$CTX" \
    "$MODEL_DIR/$MODEL_NAME.safetensors" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8080}" \
    --model-alias "${ALIAS:-Spark-X2.6-4B}" \
    --thinking "${THINKING:-on}" \
    --temp "${TEMP:-1.0}" \
    --top-k "${TOP_K:-0}" \
    --top-p "${TOP_P:-0.95}" \
    --repeat-penalty "${REPEAT_PENALTY:-1.0}" \
    --frequency-penalty "${FREQ_PENALTY:-0.0}" \
    --presence-penalty "${PRESENCE_PENALTY:-0.0}" \
    --max-new-tokens "${MAX_NEW:-0}" \
    --max-new-tokens-cap "${MAX_NEW_CAP:-0}" \
    --request-timeout-sec "${REQ_TIMEOUT:-1200}" \
    --max-connections "${MAX_CONNS:-128}" \
    --max-queue "${MAX_QUEUE:-64}" \
    --thinking-effort-inject "${EFFORT_INJECT:-on}" \
    ${API_KEY:+--api-key "$API_KEY"} \
    ${EXTRA_ARGS:-}
