#!/usr/bin/env bash
# 一键全套测试：复读专项 + 十一阶段压力/兼容性 + 服务自指标 + 前后卡状态 + 服务日志摘录。
# 从仓库根运行：bash tools/run_stress_suite.sh
# 产物：results/repeat_probe.json、results/stress_report.json、results/stress_suite.log（本脚本自身的输出请自行重定向）
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
export TMPDIR=${TMPDIR:-/run/rknn3}

DIR="$(cd "$(dirname "$0")" && pwd)"
UNIT=${UNIT:-rk1828-spark}
mkdir -p results

log() { echo "[$(date '+%T')] $*"; }

card() {
  echo "--- 卡状态 $(date '+%T') ---"
  timeout 15 rknn-smi info 2>&1 | sed -n '1,12p'
  timeout 15 rknn-smi -v 2>&1 | head -8
  if command -v lspci >/dev/null; then
    lspci -vv -s 01:00.0 2>/dev/null | grep -aE 'LnkSta|UErr|CErr|MalformedTLP|Frequency' | head -6
  fi
}

unit() {
  systemctl is-active "$UNIT".service 2>/dev/null
  systemctl show -p MainPID,ExecMainStartTimestamp,NRestarts --value "$UNIT".service 2>/dev/null | head -3
}

log "==== 测试前状态 ===="
card
unit
curl -s -m 5 http://127.0.0.1:8080/health; echo

log "==== 1) 复读专项（24 组对抗性 prompt） ===="
python3 -u "$DIR/repeat_probe.py" --out results/repeat_probe.json > results/_rp.log 2>&1
RPRC=$?
tail -60 results/_rp.log

log "==== 2) 十一阶段压力/兼容性 ===="
python3 -u "$DIR/stress_server.py" --threads 24 --soak 120 --out results/stress_report.json > results/_ss.log 2>&1
STRC=$?
tail -60 results/_ss.log
log "压力脚本真实退出码 stress_rc=$STRC（注意：把命令接进管道会让 \$? 变成管道尾端的状态）"

log "==== 3) 服务自指标 ===="
curl -s -m 10 http://127.0.0.1:8080/stats

log "==== 4) 测试后卡状态 ===="
card
unit

log "==== 5) 最近服务日志 ===="
journalctl -u "$UNIT".service -n 40 --no-pager 2>/dev/null | grep -aiE 'error|fail|E RKNN|REQ ' | tail -25

log "STRESS_SUITE_DONE repeat_rc=$RPRC stress_rc=$STRC"
