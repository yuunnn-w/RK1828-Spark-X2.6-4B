#!/bin/bash
# KV cache 策略扫描：同一模型/同一 ctx 下，比较 runtime 默认策略与显式 RECURRENT/NORMAL 的
# prefill / decode 速率。每档都完整重启服务（顺带验证优雅收尾与重启可靠性），最后恢复生产配置。
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
export TMPDIR=/run/rknn3
L=results/kv_sweep.log
log() { echo "[$(date '+%F %T')] $*" >> "$L"; }
health() { curl -sf -m 8 http://127.0.0.1:8080/health >/dev/null 2>&1; }

cp -a /etc/rk1828-spark.env /etc/rk1828-spark.env.prod

set_args() {
  sed -i "s|^EXTRA_ARGS=.*|EXTRA_ARGS=\"--reasoning-format deepseek$1\"|" /etc/rk1828-spark.env
}

wait_stop() {
  for i in $(seq 1 30); do pgrep -f 'rknn_spark_serv[e]r' >/dev/null || break; sleep 2; done
}

run_case() {
  local name="$1"
  local a="$2"
  log "CASE $name args='$a'"
  systemctl stop rk1828-spark.service
  wait_stop
  set_args "$a"
  sh -c '. /etc/rk1828-spark.env; echo "env now: [$EXTRA_ARGS]"' >> "$L" 2>&1
  systemctl start rk1828-spark.service
  local ok=0
  for i in $(seq 1 60); do sleep 2; if health; then ok=1; break; fi; done
  if [ "$ok" -ne 1 ]; then
    log "CASE $name HEALTH_FAIL (模型可能装不下这种策略) -> stop"
    systemctl stop rk1828-spark.service
    wait_stop
    sleep 3
    return 1
  fi
  python3 -u tools/perf_probe.py >> "$L" 2>&1
  log "CASE $name perf rc=$?"
  export PATH=/usr/bin:/bin:/usr/sbin:/sbin
  timeout 15 rknn-smi info 2>&1 | sed -n '7,9p' >> "$L"
  return 0
}

log KV_SWEEP_START
run_case "default(不设policy)" ""
run_case "RECURRENT显式" " --kvcache-policy 0"
run_case "RECURRENT n_keep=2048" " --kvcache-policy 0 --kv-n-keep 2048"
run_case "NORMAL" " --kvcache-policy 1"

cp -a /etc/rk1828-spark.env.prod /etc/rk1828-spark.env
systemctl stop rk1828-spark.service
wait_stop
systemctl start rk1828-spark.service
for i in $(seq 1 60); do sleep 2; health && break; done
if health; then log "RESTORED_OK"; else log "RESTORED_FAIL"; fi
log KV_SWEEP_DONE
