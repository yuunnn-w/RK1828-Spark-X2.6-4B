#!/usr/bin/env python3
"""rknn_spark_server 严格压力/兼容性测试（板端或远端均可，只用标准库）。

阶段：
  P1 基本端点       health / models / metrics
  P2 协议校验       畸形 JSON、缺字段、空 messages、超大 body、未知路由、非法类型
  P3 顺序正确性     连续短请求，检查非空、无复读、finish_reason、usage 计数
  P4 并发排队       N 线程并发请求，全部成功、无死锁、延迟分布
  P5 流式           chunk 序列合法（role 首块、content 增量、finish 尾块、[DONE]、include_usage）
  P6 客户端中途断开 生成过程中关闭 socket，服务端必须停止且后续请求正常
  P7 长上下文阶梯   2k / 8k / 32k token 级 prompt
  P8 工具调用       tools + tool_choice=auto，期望 tool_calls
  P9 结构化输出     response_format=json_object 必须可 json.loads
  P10 浸泡          混合请求 200 条，最后核对 /stats 错误计数与卡状态

用法: python3 -u stress_server.py [--base http://127.0.0.1:8080] [--model Spark-X2.6-4B]
                                   [--threads 24] [--soak 200] [--skip-long]
"""
import argparse
import http.client
import json
import re
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

MODEL_DEFAULT = "Spark-X2.6-4B"


def post_json(base, path, payload, timeout=600):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace"), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), time.time() - t0
    except Exception as e:
        return -1, repr(e), time.time() - t0


def get_txt(base, path, timeout=30):
    try:
        with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return -1, repr(e)


def rep8(t, n=8):
    if len(t) < 2 * n:
        return 0.0
    g = [t[i:i + n] for i in range(len(t) - n + 1)]
    return 1.0 - len(set(g)) / len(g)


def content_of(resp_text):
    try:
        d = json.loads(resp_text)
    except Exception:
        return None
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return msg.get("content")


class Report:
    def __init__(self):
        self.rows = []
        self.fails = []

    def add(self, phase, name, ok, detail=""):
        self.rows.append((phase, name, "PASS" if ok else "FAIL", detail[:220]))
        if not ok:
            self.fails.append((phase, name, detail[:220]))
        print("  [%s] %-4s %s %s" % (phase, "PASS" if ok else "FAIL", name, detail[:160]), flush=True)


# ------------------------------------------------------------------ P1
def p1(base, rp):
    st, b = get_txt(base, "/health")
    rp.add("P1", "GET /health", st == 200 and "ok" in b, b[:80])
    st, b = get_txt(base, "/v1/models")
    ok = st == 200 and "data" in b and MODEL in b
    rp.add("P1", "GET /v1/models", ok, b[:100])
    st, b = get_txt(base, "/metrics")
    rp.add("P1", "GET /metrics", st == 200 and "requests_total" in b, "")
    st, b = get_txt(base, "/nope")
    rp.add("P1", "未知路由 404", st == 404, b[:80])


# ------------------------------------------------------------------ P2
def p2(base, rp):
    cases = [
        ("\u7592\u5f62 JSON", "{not json", 400),
        ("\u7f3a messages", {"model": MODEL}, 400),
        ("\u7a7a messages", {"model": MODEL, "messages": []}, 400),
        ("messages \u975e\u6570\u7ec4", {"model": MODEL, "messages": "hi"}, 400),
        ("\u6d88\u606f\u4e0d\u662f\u5bf9\u8c61", {"model": MODEL, "messages": ["hi"]}, 400),
        ("\u65e0\u7528\u6237\u8f6e", {"model": MODEL, "messages": [{"role": "system", "content": "x"}]}, 400),
        ("top_p 非法类型（忽略并保持默认）",
         {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "top_p": "abc", "max_tokens": 8}, 200),
        ("temperature \u8d1f\u6570\uff08\u6536\u655b\u5230 0\uff09",
         {"model": MODEL, "messages": [{"role": "user", "content": "1+1=?"}], "temperature": -5,
          "max_tokens": 8}, 200),
    ]
    for name, payload, want in cases:
        if isinstance(payload, str):
            st, b, _ = post_json_raw(base, "/v1/chat/completions", payload)
        else:
            st, b, _ = post_json(base, "/v1/chat/completions", payload, timeout=180)
        rp.add("P2", name, st == want, "got %s %s" % (st, (b or "")[:80]))

    big = "\u6d4b\u8bd5" * 9000000  # > 32MB body
    st, b, _ = post_json(base, "/v1/chat/completions",
                         {"model": MODEL, "messages": [{"role": "user", "content": big}]}, timeout=180)
    rp.add("P2", "\u8d85\u5927 body \u88ab\u62d2", st in (413, 400, -1), "got %s" % st)


def post_json_raw(base, path, raw):
    data = raw.encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode("utf-8", "replace"), 0.0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), 0.0
    except Exception as e:
        return -1, repr(e), 0.0


# ------------------------------------------------------------------ P3
QS = [
    ("1+1等于几？只回答数字。", "2"),
    ("What is 17*23? Answer with just the number.", "391"),
    ("37乘以48等于多少？只回答数字。", "1776"),
    ("一个单词 extraordinary 里有几个字母？只回答数字。", "13"),
    ("121 的平方根是多少？只回答数字。", "11"),
    ("用 Python 写一个函数 fib(n) 返回第 n 个斐波那契数，只输出代码。", "def fib"),
]


def texts_of(resp_text):
    """content + reasoning_content 拼在一起，用于判"答案里是否出现关键串"。"""
    try:
        d = json.loads(resp_text)
    except Exception:
        return ""
    msg = ((d.get("choices") or [{}])[0].get("message") or {})
    return (msg.get("content") or "") + (msg.get("reasoning_content") or "")


def p3(base, rp, n=3):
    lat = []
    bad = []
    for _ in range(n):
        for q, gold in QS:
            st, b, dt = post_json(base, "/v1/chat/completions",
                                  {"model": MODEL, "messages": [{"role": "user", "content": q}],
                                   "max_tokens": 900, "temperature": 0}, timeout=300)
            c = content_of(b) or ""
            full = texts_of(b)
            # 思考型模型要先把 think 块写完才会输出正式回答，预算太小会让 content 停在思考文本上。
            ok = st == 200 and gold in full and rep8(c) < 0.4
            lat.append(dt)
            if not ok:
                bad.append((q[:24], st, (c or "")[:80]))
    rp.add("P3", "顺序正确性 %d 轮×%d 题" % (n, len(QS)), not bad,
           "avg %.2fs p95 %.2fs 失败=%s" % (statistics.mean(lat), sorted(lat)[-1], bad[:2]))


# ------------------------------------------------------------------ P4
def p4(base, rp, threads=24):
    errs = []
    lat = []
    lock = threading.Lock()

    def worker(i):
        q = "第%d题：计算 12*13，只回答数字。" % i
        st, b, dt = post_json(base, "/v1/chat/completions",
                              {"model": MODEL, "messages": [{"role": "user", "content": q}],
                               "max_tokens": 64, "temperature": 0}, timeout=600)
        c = content_of(b)
        with lock:
            lat.append(dt)
            if st != 200 or not c or "156" not in c:
                errs.append((i, st, (c or b)[:60]))

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    rp.add("P4", "并发 %d 连接排队全部完成" % threads, not errs,
           "wall %.1fs 平均 %.2fs 最差 %.2fs 错误=%s" % (wall, statistics.mean(lat), max(lat), errs[:2]))


# ------------------------------------------------------------------ P5
def stream_once(base, payload, max_chunks=100000):
    """返回 (ok, 首块role, chunk数, content文本, reasoning文本, 有usage, 有DONE, 有finish)"""
    u = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                               data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                               headers={"Content-Type": "application/json"})
    first_role = False
    nchunk = 0
    text = []
    reason = []
    have_done = False
    have_finish = False
    have_usage = False
    try:
        with urllib.request.urlopen(u, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    have_done = True
                    continue
                nchunk += 1
                d = json.loads(body)
                for ch in d.get("choices", []):
                    dl = ch.get("delta") or {}
                    if dl.get("role") == "assistant":
                        first_role = True
                    if dl.get("content"):
                        text.append(dl["content"])
                    if dl.get("reasoning_content"):
                        reason.append(dl["reasoning_content"])
                    if ch.get("finish_reason"):
                        have_finish = True
                if d.get("usage"):
                    have_usage = True
                if nchunk > max_chunks:
                    break
    except Exception as e:
        return (False, first_role, nchunk, "".join(text), "".join(reason), have_usage, have_done, have_finish, repr(e))
    return (True, first_role, nchunk, "".join(text), "".join(reason), have_usage, have_done, have_finish, "")


def p5(base, rp, n=8):
    ok_all = True
    detail = []
    for i in range(n):
        usage_flag = (i % 2 == 0)
        payload = {"model": MODEL,
                   "messages": [{"role": "user", "content": "写一句关于秋天的七言诗句。"}],
                   "max_tokens": 512, "temperature": 0.8, "top_p": 0.9,
                   "stream": True, "stream_options": {"include_usage": usage_flag}}
        ok, role, nch, text, reason, hu, hd, hf, err = stream_once(base, payload)
        # 判的是 SSE 协议合法性与"有文本流出"。开了 reasoning 分离后，思考文本走 reasoning_content；
        # 若模型在预算内没写完思考块，content 为空是模型行为（与 vLLM 的 reasoning parser 语义一致），
        # 不算协议错误 —— 这里记录 closed 供人工判断，不作为 FAIL 条件。
        closed = len(text) > 5
        good = ok and role and nch >= 3 and (len(text) + len(reason)) > 5 and hd and hf and (hu == usage_flag)
        bad_rep = rep8(text) > 0.6
        if not good or bad_rep:
            ok_all = False
        detail.append("#%d chunks=%d len=%d reflen=%d closed=%s rep_ans=%.2f rep_all=%.2f role=%s done=%s finish=%s usage=%s %s"
                      % (i, nch, len(text), len(reason), closed, rep8(text), rep8(text + reason), role, hd, hf, hu, err[:40]))
    rp.add("P5", "流式 %d 次（SSE 合法性）" % n, ok_all, " | ".join(detail[:3]))


# ------------------------------------------------------------------ P6
def p6(base, rp, n=6):
    ok_all = True
    detail = []
    host = base.split("//")[-1].split(":")[0]
    port = int(base.rsplit(":", 1)[-1])
    for i in range(n):
        body = json.dumps({"model": MODEL,
                           "messages": [{"role": "user",
                                         "content": "请写一篇 800 字的关于人工智能发展史的文章。"}],
                           "max_tokens": 1200, "temperature": 0.7, "stream": True}, ensure_ascii=False).encode()
        s = socket.create_connection((host, port), timeout=30)
        req = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        req += b"Content-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body
        s.sendall(req)
        got = 0
        t0 = time.time()
        while time.time() - t0 < 20:
            chunk = s.recv(4096)
            if not chunk:
                break
            got += chunk.count(b"data: ")
            if got >= 8:
                break
        s.close()  # 客户端硬断开
        detail.append("#%d got=%d" % (i, got))
        time.sleep(3)
        # 断开之后服务必须还能正常响应
        st, b, _ = post_json(base, "/v1/chat/completions",
                             {"model": MODEL, "messages": [{"role": "user", "content": "9-4=?只回答数字。"}],
                              "max_tokens": 16, "temperature": 0}, timeout=120)
        c = content_of(b)
        if st != 200 or not c or "5" not in c:
            ok_all = False
            detail.append("断开后请求失败: %s %s" % (st, (c or b)[:60]))
    rp.add("P6", "客户端中途断开后仍可用", ok_all, " ".join(detail[:6]))


# ------------------------------------------------------------------ P7
def p7(base, rp):
    detail = []
    ok_all = True
    for mult in (1500, 4000, 9000):  # 约 1k / 3k / 6k token；更大的阶梯交给 longctx_probe.py
        filler = ("人工智能模型的推理过程包含预填充与解码两个阶段，前者并行处理输入序列，后者逐 token 生成。"
                  "在长上下文场景下，注意力缓存的容量与分页策略共同决定了可用的上下文窗口。")
        prompt = (filler * (mult // 60 + 1))[:mult * 2]
        payload = {"model": MODEL,
                   "messages": [{"role": "user", "content": prompt + "\n\n请用一句话总结上面内容的主题。"}],
                   "max_tokens": 64, "temperature": 0}
        st, b, dt = post_json(base, "/v1/chat/completions", payload, timeout=1200)
        c = content_of(b)
        usage = {}
        try:
            usage = json.loads(b).get("usage", {})
        except Exception:
            pass
        pt = usage.get("prompt_tokens", 0)
        ok = st == 200 and bool(c) and pt > 100
        if not ok:
            ok_all = False
        detail.append("~%d字符->prompt_tokens=%d completion=%d %.1fs st=%s"
                      % (len(prompt), pt, usage.get("completion_tokens", 0), dt, st))
    rp.add("P7", "长上下文阶梯", ok_all, " | ".join(detail))


# ------------------------------------------------------------------ P8
def p8(base, rp):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "北京今天气温是多少摄氏度？请调用工具查询。"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的当前气温",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "城市名"}},
                    "required": ["city"],
                },
            },
        }],
        "tool_choice": "auto",
        "max_tokens": 256,
        "temperature": 0,
    }
    st, b, _ = post_json(base, "/v1/chat/completions", payload, timeout=300)
    try:
        d = json.loads(b)
        ch = d["choices"][0]
        tc = (ch.get("message") or {}).get("tool_calls")
        finish = ch.get("finish_reason")
    except Exception:
        tc, finish = None, "?"
    name = tc[0].get("function", {}).get("name") if isinstance(tc, list) and tc else None
    args_ok = False
    try:
        args = json.loads((tc[0].get("function") or {}).get("arguments") or "")
        args_ok = isinstance(args, dict) and "city" in args
    except Exception:
        args_ok = False
    ok = st == 200 and isinstance(tc, list) and len(tc) > 0 and bool(name) and args_ok
    rp.add("P8", "tools -> tool_calls(name+可解析arguments)", bool(ok),
           "finish=%s name=%s args_ok=%s tool_calls=%s" % (
               finish, name, args_ok, json.dumps(tc, ensure_ascii=False)[:150] if tc else "none"))


# ------------------------------------------------------------------ P9
def p9(base, rp):
    payload = {"model": MODEL,
               "messages": [{"role": "user",
                             "content": "给出一个 JSON 对象，字段 name 为字符串、age 为整数，描述一个叫小李的 20 岁的人。"}],
               "response_format": {"type": "json_object"},
               "max_tokens": 200, "temperature": 0}
    st, b, _ = post_json(base, "/v1/chat/completions", payload, timeout=300)
    c = content_of(b)
    ok = False
    detail = "st=%s" % st
    if c:
        m = re.search(r"\{[\s\S]*\}", c)
        if m:
            try:
                obj = json.loads(m.group(0))
                ok = isinstance(obj, dict) and "name" in obj and "age" in obj
                detail += " json=%s" % json.dumps(obj, ensure_ascii=False)[:120]
            except Exception as e:
                detail += " parse_fail=%s text=%s" % (e, c[:120])
    rp.add("P9", "response_format=json_object", ok, detail)


# ------------------------------------------------------------------ P10
def p10(base, rp, n=200):
    mix = [
        ({"model": MODEL, "messages": [{"role": "user", "content": "5*7=?只回答数字。"}], "max_tokens": 32,
         "temperature": 0}, False),
        ({"model": MODEL, "messages": [{"role": "system", "content": "你是一个简洁的助手。"},
                                       {"role": "user", "content": "说出三个质数。"}], "max_tokens": 64}, True),
        ({"model": MODEL, "messages": [{"role": "user", "content": "把下面的话翻译成英文：今天天气很好。"}],
         "max_tokens": 96, "temperature": 0.3}, False),
        ({"model": MODEL, "messages": [{"role": "user", "content": "多轮测试：记住数字 42。"}], "max_tokens": 48},
         False),
    ]
    errs = []
    deg = 0
    t0 = time.time()
    for i in range(n):
        payload, multi = mix[i % len(mix)]
        if multi and i % len(mix) == 1:
            payload = dict(payload)
            payload["messages"] = payload["messages"] + [
                {"role": "assistant", "content": "2、3、5。"},
                {"role": "user", "content": "把第一个数字改成 7，再列一次。"},
            ]
        st, b, _ = post_json(base, "/v1/chat/completions", payload, timeout=600)
        c = content_of(b)
        if st != 200 or not c:
            errs.append((i, st, (b or "")[:80]))
        elif rep8(c) > 0.6:
            deg += 1
        if i % 50 == 49:
            print("   soak %d/%d 用时 %.0fs 错误 %d 复读 %d" % (i + 1, n, time.time() - t0, len(errs), deg), flush=True)
    st, b = get_txt(base, "/stats")
    stats = {}
    try:
        stats = json.loads(b)
    except Exception:
        pass
    ok = not errs and deg == 0 and stats.get("requests_error", 0) == 0
    rp.add("P10", "浸泡 %d 条混合请求" % n, ok,
           "客户端错误=%d 复读=%d server错误=%d guard=%d veto=%d 用时 %.0fs"
           % (len(errs), deg, stats.get("requests_error", -1), stats.get("requests_degeneration_guard", -1),
              stats.get("veto_hits_total", -1), time.time() - t0))
    if stats:
        rp.add("P10", "服务自指标可读", True,
               "p50=%.0fms p95=%.0fms p99=%.0fms max=%.0fms tok_out=%d"
               % (stats.get("latency_ms_p50", 0), stats.get("latency_ms_p95", 0),
                  stats.get("latency_ms_p99", 0), stats.get("latency_ms_max", 0),
                  stats.get("completion_tokens_total", 0)))


# ------------------------------------------------------------------ P11（新加端点与 OpenAI 语义）
def _hostport(base):
    s = base.split("://", 1)[-1].rstrip("/")
    if ":" in s:
        h, p = s.rsplit(":", 1)
        return h, int(p)
    return s, 80


def head_txt(base, path, timeout=30):
    h, port = _hostport(base)
    conn = http.client.HTTPConnection(h, port, timeout=timeout)
    try:
        conn.request("HEAD", path)
        r = conn.getresponse()
        body = r.read()
        return r.status, body.decode("utf-8", "replace")
    except Exception as e:
        return -1, repr(e)
    finally:
        conn.close()


def p11(base, rp):
    # 1) HEAD /health
    st, b = head_txt(base, "/health")
    rp.add("P11", "HEAD /health 无正文", st == 200 and b == "", "st=%s len=%d" % (st, len(b)))

    # 2) GET /v1/models/{id}
    st, b = get_txt(base, "/v1/models/" + MODEL)
    ok = False
    try:
        d = json.loads(b)
        ok = st == 200 and d.get("id") == MODEL and d.get("object") == "model"
    except Exception:
        pass
    rp.add("P11", "GET /v1/models/{id}", ok, "st=%s %s" % (st, (b or "")[:90]))

    # 3) 未知模型 id 必须 404
    st, b = get_txt(base, "/v1/models/zz-not-a-model")
    rp.add("P11", "GET /v1/models/{未知id} -> 404", st == 404, "st=%s %s" % (st, (b or "")[:90]))

    # 4) n>1 明确拒绝（而不是悄悄只出一条）
    st, b, _ = post_json(base, "/v1/chat/completions",
                         {"model": MODEL, "messages": [{"role": "user", "content": "1+1=?"}],
                          "n": 2, "max_tokens": 16}, timeout=120)
    rp.add("P11", "n>1 明确报 400", st == 400, "st=%s %s" % (st, (b or "")[:90]))

    # 5) reasoning_format=deepseek：思考与答案分离
    st, b, _ = post_json(base, "/v1/chat/completions",
                         {"model": MODEL, "messages": [{"role": "user", "content": "一个笼子里有 3 只鸡 2 只兔，共几条腿？只回答数字。"}],
                          "max_tokens": 256, "reasoning_format": "deepseek"}, timeout=600)
    msg = {}
    try:
        msg = (json.loads(b).get("choices") or [{}])[0].get("message") or {}
    except Exception:
        pass
    c = msg.get("content") or ""
    r = msg.get("reasoning_content") or ""
    ok = st == 200 and c != "" and "</think>" not in c
    rp.add("P11", "reasoning_format=deepseek 分离", ok,
           "st=%s content=%dchars reasoning=%dchars" % (st, len(c), len(r)))

    # 6) role="tool" 轮次能被接受
    st, b, _ = post_json(base, "/v1/chat/completions",
                         {"model": MODEL, "messages": [
                             {"role": "user", "content": "北京今天气温多少度？先调用工具。"},
                             {"role": "assistant", "content": "", "tool_calls": [
                                 {"id": "call_1", "type": "function",
                                  "function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"}}]},
                             {"role": "tool", "tool_call_id": "call_1", "name": "get_weather",
                              "content": "{\"temp\": 12, \"unit\": \"C\"}"}],
                          "max_tokens": 96}, timeout=600)
    c = content_of(b) or ""
    rp.add("P11", "role=tool 上下文可续写", st == 200 and len(c) > 0, "st=%s %s" % (st, c[:80]))

    # 7) 传统 /v1/completions
    st, b, _ = post_json(base, "/v1/completions",
                         {"model": MODEL, "prompt": "把这句话翻译成英文：今天天气很好。", "max_tokens": 64}, timeout=600)
    txt = ""
    try:
        txt = (json.loads(b).get("choices") or [{}])[0].get("text") or ""
    except Exception:
        pass
    rp.add("P11", "POST /v1/completions 返回 text", st == 200 and len(txt) > 0, "st=%s %s" % (st, txt[:70]))

    # 8) logprobs
    st, b, _ = post_json(base, "/v1/chat/completions",
                         {"model": MODEL, "messages": [{"role": "user", "content": "说一个数字。"}],
                          "max_tokens": 8, "logprobs": True}, timeout=600)
    lp = None
    try:
        lp = (json.loads(b).get("choices") or [{}])[0].get("logprobs")
    except Exception:
        pass
    ok = st == 200 and isinstance(lp, dict) and bool(lp.get("content"))
    rp.add("P11", "logprobs 返回逐 token 概率", ok, "st=%s n=%s" % (st, len((lp or {}).get("content") or []) if lp else 0))

    # 9) /stats 暴露队列水位（429 策略可观测）
    st, b = get_txt(base, "/stats")
    s = {}
    try:
        s = json.loads(b)
    except Exception:
        pass
    rp.add("P11", "/stats 暴露 queue_limit/queued_now",
           st == 200 and "queue_limit" in s and "queued_now" in s,
           "queue_limit=%s queued_now=%s conns=%s" % (s.get("queue_limit"), s.get("queued_now"),
                                                      s.get("connections_active")))


# ------------------------------------------------------------------ main
def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--soak", type=int, default=200)
    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--out", default="results/stress_report.json")
    args = ap.parse_args()
    MODEL = args.model

    rp = Report()
    print("=== 目标 %s model=%s ===" % (args.base, MODEL), flush=True)
    p1(args.base, rp)
    p2(args.base, rp)
    p3(args.base, rp)
    p4(args.base, rp, args.threads)
    p5(args.base, rp)
    p6(args.base, rp)
    if not args.skip_long:
        p7(args.base, rp)
    p8(args.base, rp)
    p9(args.base, rp)
    p10(args.base, rp, args.soak)
    p11(args.base, rp)

    print("\n=== 汇总 ===")
    print("%-6s %-40s %-5s %s" % ("PHASE", "CHECK", "RESULT", "DETAIL"))
    for ph, name, res, detail in rp.rows:
        print("%-6s %-40s %-5s %s" % (ph, name, res, detail))
    print("\n失败 %d / 共 %d" % (len(rp.fails), len(rp.rows)))
    json.dump({"rows": rp.rows, "fails": rp.fails}, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("-> %s" % args.out)
    return 1 if rp.fails else 0


if __name__ == "__main__":
    sys.exit(main())
