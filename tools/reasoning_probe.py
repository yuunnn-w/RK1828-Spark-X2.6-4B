#!/usr/bin/env python3
# 思考通道专项验证：thinking on 时思考内容必须进 reasoning_content，
# 正文进 content；thinking off 时不得出现 reasoning_content。
# 覆盖非流式/流式、完整收尾/max_tokens 截断，并检查 SSE 每行必须是合法 UTF-8。
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8080"
MODEL = "Spark-X2.6-4B"
SKY = "用一句话说明为什么天空是蓝色的。"
MATH = "只回答一个阿拉伯数字：1+1=?"
rows = []


def check(name, ok, detail):
    rows.append((name, "PASS" if ok else "FAIL", detail[:240]))
    print("[%s] %-30s %s" % ("PASS" if ok else "FAIL", name, detail[:200]), flush=True)


def body_of(prompt, stream, max_tokens, thinking=None):
    b = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
         "temperature": 0, "stream": stream, "max_tokens": max_tokens}
    if thinking is not None:
        b["thinking"] = thinking
    return b


def post(body, timeout=600):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return -1, {"error": repr(e)}


def stream(body, timeout=600):
    """返回 (status, reasoning, content, finish, bad_utf8_lines, bad_json_lines)。"""
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    reasoning = []
    content = []
    finish = None
    bad_utf8 = 0
    bad_json = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.rstrip(b"\r\n")
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]":
                    break
                try:
                    text = payload.decode("utf-8", "strict")
                except UnicodeDecodeError:
                    bad_utf8 += 1
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    bad_json += 1
                    continue
                ch = (obj.get("choices") or [{}])[0]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                delta = ch.get("delta") or {}
                if isinstance(delta.get("reasoning_content"), str):
                    reasoning.append(delta["reasoning_content"])
                if isinstance(delta.get("content"), str):
                    content.append(delta["content"])
    except Exception as e:
        return -1, "", "", None, bad_utf8, bad_json, repr(e)
    return 200, "".join(reasoning), "".join(content), finish, bad_utf8, bad_json, ""


# 1) 非流式 + 截断：整段都应进 reasoning_content，content 为空
st, d = post(body_of(SKY, False, 64))
m = ((d.get("choices") or [{}])[0].get("message") or {})
fin = ((d.get("choices") or [{}])[0].get("finish_reason") or "")
rc = m.get("reasoning_content") or ""
ct = m.get("content") or ""
check("nonstream 截断", st == 200 and bool(rc.strip()) and not ct.strip() and fin == "length",
      "st=%s finish=%s reasoning_chars=%d content=%r" % (st, fin, len(rc), ct[:40]))

# 2) 非流式 + 完整收尾：reasoning/content 分离
st, d = post(body_of(SKY, False, 1500))
m = ((d.get("choices") or [{}])[0].get("message") or {})
rc = m.get("reasoning_content") or ""
ct = m.get("content") or ""
check("nonstream 完整", st == 200 and bool(rc.strip()) and bool(ct.strip())
      and "</think>" not in rc and "</think>" not in ct,
      "st=%s reasoning_chars=%d content_chars=%d content=%r" % (st, len(rc), len(ct), ct[:40]))

# 3) 流式 + 截断：只在 reasoning_content 里出现
st, rc, ct, fin, bad_utf8, bad_json, err = stream(body_of(SKY, True, 64))
check("stream 截断", st == 200 and bool(rc.strip()) and not ct.strip()
      and fin == "length" and bad_utf8 == 0 and bad_json == 0,
      "st=%s finish=%s reasoning_chars=%d content=%r bad_utf8=%d bad_json=%d %s"
      % (st, fin, len(rc), ct[:40], bad_utf8, bad_json, err))

# 4) 流式 + 完整：先 reasoning 后 content，SSE 每行 UTF-8 合法
st, rc, ct, fin, bad_utf8, bad_json, err = stream(body_of(SKY, True, 1500))
check("stream 完整", st == 200 and bool(rc.strip()) and bool(ct.strip())
      and "</think>" not in rc and "</think>" not in ct
      and bad_utf8 == 0 and bad_json == 0,
      "st=%s finish=%s reasoning_chars=%d content_chars=%d bad_utf8=%d bad_json=%d %s"
      % (st, fin, len(rc), len(ct), bad_utf8, bad_json, err))

# 5) 流式 + thinking off：正文必须在 content，不得出现 reasoning_content
st, rc, ct, fin, bad_utf8, bad_json, err = stream(body_of(MATH, True, 32, thinking=False))
check("stream thinking off", st == 200 and bool(ct.strip()) and not rc and bad_utf8 == 0 and bad_json == 0,
      "st=%s finish=%s reasoning=%r content=%r bad_utf8=%d %s"
      % (st, fin, rc[:20], ct[:40], bad_utf8, err))

# 6) 非流式 + thinking off：同样不得出现 reasoning_content
st, d = post(body_of(MATH, False, 32, thinking=False))
m = ((d.get("choices") or [{}])[0].get("message") or {})
rc = m.get("reasoning_content") or ""
ct = m.get("content") or ""
check("nonstream thinking off", st == 200 and bool(ct.strip()) and not rc,
      "st=%s reasoning=%r content=%r" % (st, rc[:20], ct[:40]))

fails = [r for r in rows if r[1] == "FAIL"]
print("[reasoning_probe] %d/%d PASS" % (len(rows) - len(fails), len(rows)), flush=True)
json.dump([{"check": a, "result": b, "detail": c} for a, b, c in rows],
          open("reasoning_probe.json", "w"), ensure_ascii=False, indent=1)
sys.exit(1 if fails else 0)
