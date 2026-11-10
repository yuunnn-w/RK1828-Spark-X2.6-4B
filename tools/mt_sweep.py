#!/usr/bin/env python3
# 目的：验证"decode_ms 里那段与生成长度无关的时间"是不是正比于**请求要求的 max_tokens**
# 而不是实际生成的 token 数。做法：用同一个会在十几个 token 内自然结束的问题，
# 扫 max_tokens = 16/64/256/1024/4096，看 decode_ms 是否随 max_tokens 线性增长。
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8080"
MODEL = "Spark-X2.6-4B"
Q = "1+1 等于几？只回答数字。"

for mt in (16, 64, 256, 1024, 4096):
    body = {"model": MODEL, "messages": [{"role": "user", "content": Q}],
            "max_tokens": mt, "stream": False}
    t0 = time.time()
    try:
        req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1200) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print("mt=%-5d ERROR %r" % (mt, e), flush=True)
        continue
    u = d.get("usage", {})
    ch = (d.get("choices") or [{}])[0]
    print("mt=%-5d ct=%-4s prefill=%-8s decode=%-9s e2e_server=%-9s wall(incl queue)=%-7.1fs finish=%s"
          % (mt, u.get("completion_tokens"), u.get("prefill_ms"), u.get("decode_ms"),
             u.get("e2e_ms"), time.time() - t0, ch.get("finish_reason")), flush=True)
