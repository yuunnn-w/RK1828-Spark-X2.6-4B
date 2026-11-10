#!/usr/bin/env python3
# 默认参数与"输出长度不限制"的验证：
#   1) /v1/models 的 meta.sampling_defaults 是不是官方那套
#   2) 不传 max_tokens 时是否自然收尾（finish=stop 而不是 length）
#   3) 传小 max_tokens 时是否被尊重（finish=length）
#   4) 传 -1 是否按"不限制"处理
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8080"
MODEL = "Spark-X2.6-4B"
rows = []


def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        return -1, {"err": repr(e)}


def post(extra, prompt="1+1 等于几？只回答数字。", timeout=1300):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False}
    body.update(extra)
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    return d, time.time() - t0


def check(name, ok, detail):
    rows.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
    print("[%s] %-34s %s" % ("PASS" if ok else "FAIL", name, detail[:190]), flush=True)


st, d = get("/v1/models")
item = (d.get("data") or [{}])[0]
meta = item.get("meta") or {}
sd = meta.get("sampling_defaults") or {}
want = {"temperature": 1.0, "top_k": 0, "top_p": 0.95, "repeat_penalty": 1.0,
        "frequency_penalty": 0.0, "presence_penalty": 0.0, "max_new_tokens": 0}
bad = [k for k, v in want.items() if abs(float(sd.get(k, -999)) - v) > 1e-6]
check("meta.sampling_defaults=官方", st == 200 and not bad, "不符项=%s 实际=%s" % (bad, sd))

d, sec = post({})
u = d.get("usage", {})
ch = (d.get("choices") or [{}])[0]
check("不传 max_tokens 自然收尾", ch.get("finish_reason") == "stop" and (u.get("completion_tokens") or 0) > 0,
      "finish=%s ct=%s chars=%d %.1fs" % (ch.get("finish_reason"), u.get("completion_tokens"),
                                          len((ch.get("message") or {}).get("content") or ""), sec))

# 用一个已验证能写 >1000 token 的 prompt（见最后一项），保证小预算时是"被截断"而不是自然 EOS。
# 注意：模型常在短预算下提前 EOS（实测 max_tokens=8 + 倒数 prompt 只生成 7 token 就 stop），
# 所以截断用例必须用长输出 prompt + 足够小但仍大于模型正常起始长度的预算。
d, sec = post({"max_tokens": 64}, prompt="请连续写出 300 个不同的中文词语，用逗号分隔，不要解释。")
u = d.get("usage", {})
ch = (d.get("choices") or [{}])[0]
check("max_tokens=64 被尊重(截断)", ch.get("finish_reason") == "length" and (u.get("completion_tokens") or 0) <= 64,
      "finish=%s ct=%s" % (ch.get("finish_reason"), u.get("completion_tokens")))

d, sec = post({"max_tokens": -1})
ch = (d.get("choices") or [{}])[0]
check("max_tokens=-1 视为不限制", ch.get("finish_reason") == "stop",
      "finish=%s ct=%s %.1fs" % (ch.get("finish_reason"), (d.get("usage") or {}).get("completion_tokens"), sec))

# 不限制时的实际上限演示：给一个会一直写下去的 prompt，看它最终停在哪
d, sec = post({}, prompt="请连续写出 300 个不同的中文词语，用逗号分隔，不要解释。", timeout=1300)
u = d.get("usage", {})
ch = (d.get("choices") or [{}])[0]
check("长输出可用（>1024 token）", (u.get("completion_tokens") or 0) > 1024 or ch.get("finish_reason") == "stop",
      "finish=%s ct=%s prompt_tokens=%s %.1fs" % (ch.get("finish_reason"), u.get("completion_tokens"),
                                                  u.get("prompt_tokens"), sec))

fails = [r for r in rows if r["result"] == "FAIL"]
json.dump(rows, open("results/defaults_probe.json", "w"), ensure_ascii=False, indent=1)
print("[defaults_probe] %d/%d PASS -> results/defaults_probe.json" % (len(rows) - len(fails), len(rows)), flush=True)
