#!/bin/bash
# 轻量温度取样器：每 60 s 记录一次 SoC 热区与 RK1828 卡的温度/利用率/显存，
# 用于评测与压测期间的峰值统计。只追加到 results/temp_watch.log，采到 SAMPLES 条后自动退出。
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
L=${OUT:-results/temp_watch.log}
SAMPLES=${1:-150}
for i in $(seq 1 "$SAMPLES"); do
  soc=$(for z in /sys/class/thermal/thermal_zone*; do awk '{printf "%.0f ", $1/1000}' "$z/temp" 2>/dev/null; done)
  smi=$(timeout 10 rknn-smi info 2>/dev/null)
  ctemp=$(echo "$smi" | awk -F'|' '/RK1828/{gsub(/[[:space:]]/,"",$4); print $4; exit}')
  cmem=$(echo "$smi" | awk -F'|' '/RK1828/{gsub(/[[:space:]]/,"",$5); print $5; exit}')
  util=$(echo "$smi" | awk -F'|' '/Online|Offline/{gsub(/[[:space:]]/,"",$5); print $5; exit}')
  echo "$(date '+%F %T') soc=[$soc] card_temp=${ctemp:-NA} util=${util:-NA}% mem=${cmem:-NA}" >> "$L"
  sleep 60
done
