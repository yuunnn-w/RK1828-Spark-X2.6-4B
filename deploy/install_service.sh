#!/usr/bin/env bash
# 把推理服务装成 systemd 常驻服务（开机自启 + 健康守护）。需要 root。
# ⚠️ 这是系统级改动：会写 /etc/systemd/system、/etc/rk1828-spark.env，并 enable 两个 unit。
# 用法：
#   PREFIX=/opt/rk1828-spark BIN=server/build/rknn_spark_server deploy/install_service.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${PREFIX:-/opt/rk1828-spark}"
BIN="${BIN:-$ROOT/server/build/rknn_spark_server}"
UNIT=rk1828-spark

if [ "$(id -u)" != "0" ]; then
  echo "[install] 需要 root（systemd 单元与 /etc 配置）" >&2
  exit 1
fi
if [ ! -x "$BIN" ]; then
  echo "[install] 找不到可执行文件 $BIN —— 先跑 scripts/build_server.sh" >&2
  exit 1
fi

mkdir -p "$PREFIX/bin" "$PREFIX/lib" "$PREFIX/model"
cp -f "$BIN" "$PREFIX/bin/rknn_spark_server"
if [ -d "$(dirname "$BIN")/lib" ]; then
  cp -fL "$(dirname "$BIN")"/lib/librknn3_api*.so* "$PREFIX/lib/" 2>/dev/null || true
fi
cp -f "$ROOT/deploy/start.sh" "$PREFIX/start.sh"
cp -f "$ROOT/deploy/rk1828-spark-health.sh" "$PREFIX/rk1828-spark-health.sh"
chmod +x "$PREFIX/start.sh" "$PREFIX/rk1828-spark-health.sh"

if [ ! -f /etc/rk1828-spark.env ]; then
  cp "$ROOT/deploy/rk1828-spark.env.example" /etc/rk1828-spark.env
  echo "[install] 已生成 /etc/rk1828-spark.env（改模型档位/端口/采样都改它）"
fi

for u in "$UNIT".service "$UNIT"-health.service; do
  sed "s|__PREFIX__|$PREFIX|g" "$ROOT/deploy/systemd/$u.in" > "/etc/systemd/system/$u"
done
cp "$ROOT/deploy/systemd/$UNIT-health.timer" /etc/systemd/system/

systemctl daemon-reload
systemctl enable --now "$UNIT".service
systemctl enable --now "$UNIT"-health.timer

echo "[install] 完成。模型文件请放到 $PREFIX/model/（见 model/download_model.sh 与 convert/）"
echo "[install] 状态：systemctl status $UNIT.service   日志：journalctl -u $UNIT -f"
sleep 2
systemctl --no-pager status "$UNIT".service 2>/dev/null | head -8 || true
