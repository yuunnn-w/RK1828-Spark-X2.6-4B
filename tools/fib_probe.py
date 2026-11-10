#!/usr/bin/env python3
# 结案 SUITE 里唯一遗留的 P3 fib 用例：同一个"只输出代码"的 prompt，
# 在 thinking 开关 / reasoning_effort / 生成预算 三种组合下，看正式回答通道里
# 到底有没有出现 `def fib`。用来决定"长上下文编码任务"的推荐配置。
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8080"
MODEL = "Spark-X2.6-4B"
Q = "用 Python 写一个函数 fib(n) 返回第 n 个斐波那契数，只输出代码。"
GOLD = "def fib"

CASES = [
    ("A 基线 temp0 + 900 预算", {"temperature": 0, "max_tokens": 900}),
    ("B temp0 + 预算不限制", {"temperature": 0}),
    ("C temp0 + 2500 预算", {"temperature": 0, "max_tokens": 2500}),
    ("D reasoning_effort=none", {"temperature": 0, "max_tokens": 900, "reasoning_effort": "none"}),
    ("E chat_template thinking off", {"temperature": 0, "max_tokens": 900,
                                      "chat_template_kwargs": {"enable_thinking": False}}),
    ("F 厂商默认采样(temp1) 不限制", {"temperature": 1.0, "top_p": 0.95, "top_k": 0}),
]


def post(extra, timeout=1300):
    body = {"model": MODEL, "messages": [{"role": "user", "content": Q}], "stream": False}
    body.update(extra)
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace")), time.time() - t0


rows = []
for name, extra in CASES:
    try:
        d, sec = post(extra)
    except Exception as e:
        rows.append({"case": name, "hit": False, "err": repr(e)})
        print("[ERR ] %-32s %s" % (name, repr(e)), flush=True)
        continue
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    u = d.get("usage") or {}
    hit = GOLD in content
    rows.append({"case": name, "hit": hit, "finish": ch.get("finish_reason"),
                 "ct": u.get("completion_tokens"), "content_len": len(content),
                 "reasoning_len": len(reasoning), "sec": round(sec, 1)})
    print("[%s] %-32s finish=%-7s ct=%-5s content=%5d reasoning=%5d %.1fs"
          % ("HIT " if hit else "MISS", name, ch.get("finish_reason"), u.get("completion_tokens"),
             len(content), len(reasoning), sec), flush=True)
    print("      content 头部: %s" % content[:150].replace("\n", "\\n"), flush=True)

json.dump(rows, open("results/fib_probe.json", "w"), ensure_ascii=False, indent=1)
print("[fib_probe] 命中 %d/%d -> results/fib_probe.json"
      % (sum(1 for r in rows if r.get("hit")), len(rows)), flush=True)
