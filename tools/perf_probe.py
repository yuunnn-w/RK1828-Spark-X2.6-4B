#!/usr/bin/env python3
# 性能探针：同一份内容在 短/中/长 三档 prompt 下的 prefill 与 decode 速率，用于对比
# kvcache 策略等配置差异。decode_ms/prefill_ms 取自服务端自测（不含排队等待）。
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8080"
MODEL = "Spark-X2.6-4B"
FILLER = "岩石循环描述岩浆岩、沉积岩与变质岩在三类地质作用下的相互转化，涉及熔化、结晶、侵蚀、搬运、沉积、埋藏与变质。"
FILLER2 = "The rock cycle describes transformation among igneous, sedimentary and metamorphic rocks via melting, crystallization, erosion, transport, deposition and metamorphism. "

cases = [("short", "1+1 等于几？只回答数字。", 32),
         ("mid", FILLER * 120 + "\n上面这段文字里提到了几类岩石？只回答数字。", 32),
         ("long", FILLER * 420 + "\n上面这段文字里提到了几类岩石？只回答数字。", 32)]

out = []
for name, prompt, mt in cases:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mt, "temperature": 0, "stream": False}
    try:
        req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1200) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        print("  %-6s ERROR %r" % (name, e), flush=True)
        out.append({"case": name, "error": repr(e)})
        continue
    u = d.get("usage", {})
    pt = u.get("prompt_tokens") or 0
    ct = u.get("completion_tokens") or 0
    pf = u.get("prefill_ms") or 0.0
    dc = u.get("decode_ms") or 0.0
    row = {"case": name, "prompt_tokens": pt, "completion_tokens": ct, "prefill_ms": pf, "decode_ms": dc,
           "prefill_tps": round(pt / (pf / 1000.0), 1) if pf > 0 else 0,
           "decode_tps": round(ct / (dc / 1000.0), 1) if dc > 0 and ct else 0}
    out.append(row)
    print("  %-6s pt=%-6d ct=%-4d prefill=%8.1fms (%6.1f tok/s)  decode=%8.1fms (%5.1f tok/s)"
          % (name, pt, ct, pf, row["prefill_tps"], dc, row["decode_tps"]), flush=True)

json.dump(out, open("results/perf_probe.json", "w"), ensure_ascii=False, indent=1)
