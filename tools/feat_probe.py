#!/usr/bin/env python3
# 新加的 OpenAI 特性专项验证：logit_bias、top_logprobs、reasoning_effort。
# logit_bias 的测法是自指的：先问一次拿到模型自然输出的第一个 token id，
# 再用 logit_bias 把这个 id 屏蔽掉，断言输出必须改变。
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8080"
MODEL = "Spark-X2.6-4B"
rows = []


def post(body, timeout=600):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        return -1, {"error": repr(e)}


def check(name, ok, detail):
    rows.append((name, "PASS" if ok else "FAIL", detail[:200]))
    print("[%s] %-28s %s" % ("PASS" if ok else "FAIL", name, detail[:180]), flush=True)


def ask(prompt, extra=None, max_tokens=48):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": False}
    if extra:
        body.update(extra)
    return post(body)


# 1) top_logprobs
st, d = ask("用单个汉字回答：天是什么感觉？", {"logprobs": True, "top_logprobs": 5})
lp = ((d.get("choices") or [{}])[0].get("logprobs") or {})
cont = lp.get("content") or []
top = (cont[0].get("top_logprobs") or []) if cont else []
check("top_logprobs=5", st == 200 and len(top) >= 2, "st=%s entries=%d ids=%s" % (st, len(top),
      [t.get("token_id") for t in top[:5]]))

# 2) logit_bias 屏蔽自然首选 token
st, d = ask("只回答一个阿拉伯数字：1+1=?", {"logprobs": True})
cont = ((d.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
first_id = cont[0]["token_id"] if cont else None
base_text = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
if first_id is None:
    check("logit_bias 屏蔽", False, "拿不到 baseline token id (st=%s)" % st)
else:
    st2, d2 = ask("只回答一个阿拉伯数字：1+1=?", {"logit_bias": {str(first_id): -100}})
    new_text = ((d2.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    check("logit_bias 屏蔽", st2 == 200 and new_text.strip() != base_text.strip() and len(new_text) > 0,
          "blocked_id=%s base=%r biased=%r" % (first_id, base_text[:40], new_text[:40]))
    # 正向偏置也应改变选择
    st3, d3 = ask("只回答一个阿拉伯数字：1+1=?", {"logit_bias": {str(first_id): 5.0, "2": 5.0}})
    t3 = ((d3.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    check("logit_bias 正偏置", st3 == 200 and len(t3) > 0, "st=%s out=%r" % (st3, t3[:40]))

# 3) reasoning_effort 两档
st, d = ask("21 乘以 13 等于多少？只回答数字。", {"reasoning_effort": "none"}, max_tokens=64)
m = (d.get("choices") or [{}])[0].get("message") or {}
off_ok = st == 200 and not m.get("reasoning_content") and bool((m.get("content") or "").strip())
check("reasoning_effort=none", off_ok,
      "st=%s reasoning=%s content=%r" % (st, bool(m.get("reasoning_content")), (m.get("content") or "")[:40]))
st, d = ask("21 乘以 13 等于多少？只回答数字。", {"reasoning_effort": "high"}, max_tokens=256)
m2 = (d.get("choices") or [{}])[0].get("message") or {}
on_ok = st == 200 and bool(m2.get("reasoning_content"))
check("reasoning_effort=high", on_ok,
      "st=%s reasoning_chars=%d content=%r" % (st, len(m2.get("reasoning_content") or ""), (m2.get("content") or "")[:40]))

fails = [r for r in rows if r[1] == "FAIL"]
print("[feat_probe] %d/%d PASS" % (len(rows) - len(fails), len(rows)), flush=True)
json.dump([{"check": a, "result": b, "detail": c} for a, b, c in rows],
          open("results/feat_probe.json", "w"), ensure_ascii=False, indent=1)
sys.exit(1 if fails else 0)
