#!/usr/bin/env bash
# 冒烟测试：一条命令确认服务活着、模型在、能生成。
# 用法：BASE=http://<板子IP>:8080 examples/curl.sh
BASE=${BASE:-http://127.0.0.1:8080}

echo "== /health"
curl -s -m 10 "$BASE/health"; echo

echo "== /v1/models"
curl -s -m 10 "$BASE/v1/models" | head -c 600; echo

echo "== /v1/chat/completions（非流式）"
curl -s -m 300 "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Spark-X2.6-4B","messages":[{"role":"user","content":"1+1 等于几？只回答数字。"}],"temperature":0,"max_tokens":128}' \
  | head -c 1200; echo

echo "== /v1/chat/completions（流式）"
curl -s -N -m 300 "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Spark-X2.6-4B","messages":[{"role":"user","content":"说一句中文问候。"}],"stream":true,"max_tokens":64}' \
  | head -12

echo "== /stats"
curl -s -m 10 "$BASE/stats"; echo
