#!/bin/sh
# 进程守护：连续 4 次 /health 失败（约 80 s）才重启，避免把"模型正在加载"误判成故障。
# 卡片状态异常时只记录，绝不做 reset —— 这张卡的复位手段是整机断电，必须人工执行。
CONF=${RK1828_SPARK_CONF:-/etc/rk1828-spark.env}
if [ -f "$CONF" ]; then
    # shellcheck disable=SC1090
    . "$CONF"
fi
PORT=${PORT:-8080}
FAILS=/run/rk1828-spark-health.fails
UNIT=rk1828-spark
n=$(cat "$FAILS" 2>/dev/null || echo 0)
state=$(systemctl is-active "$UNIT".service 2>/dev/null)

if curl -sf -m 8 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo 0 > "$FAILS"
    exit 0
fi

# 单元还在 activating/starting/reloading（4B 级模型加载 45~90 s 属正常）时不累计失败，
# 否则会出现"重启 -> 重新加载 -> 又超时 -> 又重启"的自激循环
if [ "$state" = "activating" ] || [ "$state" = "starting" ] || [ "$state" = "reloading" ]; then
    logger -t rk1828-spark-health "health not ready but unit=${state}, skip (no restart loop)"
    echo 0 > "$FAILS"
    exit 0
fi

pid=$(systemctl show -p MainPID --value "$UNIT".service 2>/dev/null)
if [ -n "$pid" ] && [ "$pid" != "0" ] && [ -r "/proc/$pid/stat" ] && [ -r /proc/uptime ]; then
    line=$(cat "/proc/$pid/stat")
    rest=${line#*) }
    st=$(echo "$rest" | awk '{print $20}')
    up=$(awk '{print $1}' /proc/uptime)
    elapsed=$(awk -v u="$up" -v s="$st" 'BEGIN{printf "%d", u - s/100}')
    if [ "$elapsed" -ge 0 ] && [ "$elapsed" -lt 240 ]; then
        logger -t rk1828-spark-health "health not ready but main pid $pid only ${elapsed}s old (loading) -> no restart"
        echo 0 > "$FAILS"
        exit 0
    fi
fi

n=$((n + 1))
echo "$n" > "$FAILS"

card=$(TMPDIR=/run/rknn3 timeout 15 rknn-smi info 2>/dev/null | sed -n '7,9p' | tr -s ' ' | tr '\n' ' ')

if [ "$n" -ge 4 ]; then
    logger -t rk1828-spark-health "health failed ${n} times (unit=${state}) card=${card} -> restart ${UNIT}.service"
    systemctl restart "$UNIT".service
    echo 0 > "$FAILS"
else
    logger -t rk1828-spark-health "health probe failed (${n}/4, unit=${state})"
fi
exit 0
