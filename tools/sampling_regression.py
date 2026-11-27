#!/usr/bin/env python3
"""采样路径的对拍：同一 seed 的同一条请求，改前/改后输出是否逐字节一致；顺带测 top_p=1.0 与默认 top_p 的解码速度。

用法（板上）：python3 sampling_regression.py /tmp/reg.json
不带参数就写到 ./results/sampling_regression.json。

前三条走服务端默认采样（top_p=0.95 / top_k 关闭），后两条同一个 prompt 分别用默认 top_p 和
OpenAI 默认的 top_p=1.0，比较 decode_ms。改前 t_top_p1 约比 t_def 慢一倍，改后两者相同。

注意：把 top_k 关掉时的候选上限从"只在 top_p<1 时生效"改成"与 top_p 无关"之后，
同一份二进制在服务重启前后的 sampled 输出不保证逐字节一致（温度 1.0 的采样对运行时状态敏感）。
要做有意义的对拍，就把同一次对比的所有跑法都放在同一个服务实例上，重启后再各跑一遍对照。
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8080")
MODEL = os.environ.get("MODEL", "Spark-X2.6-4B")
WORDS = "请连续写出 40 个中文词语，用逗号分隔，不要解释。"

CASES = [
    ("d1", {"max_tokens": 96, "seed": 42}, "用两句话说明为什么天空是蓝色的。"),
    ("d2", {"max_tokens": 96, "seed": 42}, "列三步说明怎么做番茄炒蛋。"),
    ("d3", {"max_tokens": 96, "seed": 7}, "用一段话介绍什么是差分隐私。"),
    ("t_def", {"max_tokens": 64, "seed": 5}, WORDS),
    ("t_top_p1", {"max_tokens": 64, "seed": 5, "temperature": 1.0, "top_p": 1.0, "top_k": 0}, WORDS),
]


def post(extra, prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False}
    body.update(extra)
    t0 = time.time()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=400) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    el = time.time() - t0
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    u = d.get("usage") or {}
    return {
        "text": (msg.get("reasoning_content") or "") + "#" + (msg.get("content") or ""),
        "finish": ch.get("finish_reason"),
        "ct": u.get("completion_tokens"),
        "decode_ms": u.get("decode_ms"),
        "tps": u.get("tokens_per_second"),
        "wall": round(el, 2),
    }


out_path = sys.argv[1] if len(sys.argv) > 1 else "results/sampling_regression.json"
out = {}
for name, extra, prompt in CASES:
    out[name] = post(extra, prompt)
    v = out[name]
    ms = (v["decode_ms"] or 0) / max(1, v["ct"] or 1)
    print("%-9s finish=%-6s ct=%-4s decode_ms=%-7s ms/token=%-6.1f tps=%-8s" %
          (name, v["finish"], v["ct"], v["decode_ms"], ms, v["tps"]), flush=True)

d = os.path.dirname(out_path)
if d:
    os.makedirs(d, exist_ok=True)
json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1)
print("saved", out_path)
