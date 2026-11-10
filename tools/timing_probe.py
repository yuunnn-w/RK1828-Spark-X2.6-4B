#!/usr/bin/env python3
# 单次请求的固定开销测定：短 prompt 下扫 max_tokens，看 decode_ms 是否含一个与 token 数无关的固定段。
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8080"
MODEL = "Spark-X2.6-4B"

SHORT = "1+1="
MED = "请用一句话解释什么是牛顿第三定律。"
LONG = ("下面是一段资料，读完后只回答“收到”。\n" +
        ("岩石循环包括岩浆、沉积、变质三类岩石的相互转化，涉及熔化、结晶、侵蚀、沉积、埋藏、高温高压变质等过程。" * 40))


def one(prompt, max_tokens, timeout=600):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    e2e = (time.time() - t0) * 1000.0
    u = d.get("usage", {})
    ch = d.get("choices", [{}])[0]
    return u, ch.get("finish_reason"), e2e, (ch.get("message", {}) or {}).get("content", "")


def main():
    # 先确认服务就绪
    for _ in range(60):
        try:
            urllib.request.urlopen(BASE + "/health", timeout=8).read()
            break
        except Exception:
            time.sleep(3)
    print(f"[timing] base={BASE} model={MODEL}", flush=True)
    rows = []
    for name, prompt in (("short", SHORT), ("med", MED), ("long", LONG)):
        for mt in (1, 2, 4, 8, 16, 32, 64, 128):
            try:
                u, fin, e2e, txt = one(prompt, mt)
            except Exception as e:
                print(f"  {name:5s} mt={mt:3d} ERROR {type(e).__name__}: {e}", flush=True)
                continue
            rows.append((name, mt, u.get("prompt_tokens"), u.get("completion_tokens"),
                         u.get("prefill_ms"), u.get("decode_ms"), e2e, fin, len(txt)))
            print(f"  {name:5s} mt={mt:3d} pt={u.get('prompt_tokens'):5} ct={u.get('completion_tokens'):4} "
                  f"prefill={u.get('prefill_ms'):8.1f} decode={u.get('decode_ms'):9.1f} e2e={e2e:9.1f} "
                  f"finish={fin} chars={len(txt)}", flush=True)
    # 线性拟合 decode_ms = a + b * completion_tokens（按每个 prompt 分组）
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        if r[3]:
            g[r[0]].append((r[3], r[5]))
    fit = {}
    for k, pts in g.items():
        pts.sort()
        if len(pts) >= 2:
            (n0, d0), (n1, d1) = pts[0], pts[-1]
            if n1 != n0:
                b = (d1 - d0) / (n1 - n0)
                a = d0 - b * n0
                fit[k] = (a, b, n0, n1, d0, d1)
                print(f"[fit {k}] decode_ms ≈ {a:.0f} + {b:.2f}·tokens  ({n0}→{n1}: {d0:.0f}→{d1:.0f} ms)", flush=True)
    json.dump({"rows": rows, "fit": fit}, open("results/timing_probe.json", "w"), ensure_ascii=False)
    print("[timing] done -> results/timing_probe.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
