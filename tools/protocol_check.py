#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 yuunnn-w
# SPDX-License-Identifier: GPL-3.0-only
"""rknn_spark_server OpenAI 协议验收（标准库实现，可直接在板端跑）。

用法：
  python3 protocol_check.py --base http://127.0.0.1:8080 --slots off --save base.json
  python3 protocol_check.py --base http://127.0.0.1:8080 --slots on  --save slot.json
  python3 protocol_check.py --compare base.json slot.json

检查项：models 元数据、CORS 预检、chunked 请求体、流式 tool_calls 增量协议、
content=null、tool_choice=required、stop 句中截断、未知 response_format=400、
usage 尾块 choices=[]、/v1/responses（流式+非流式）、/v1/embeddings、多轮 KV 槽位续接。
"""
import argparse
import http.client
import json
import sys
import urllib.error
import urllib.request

TOOL_OPEN = "<tool" + "_call>"
TOOL_CLOSE = "</tool" + "_call>"
TAG_ARGK = "<arg" + "_key>"
TAG_ARGV = "<arg" + "_value>"
THINK_CLOSE = "</think" + ">"

PASS = []
FAIL = []


def ok(name, detail=""):
    PASS.append(name)
    print("PASS  %-38s %s" % (name, detail))


def bad(name, detail=""):
    FAIL.append(name)
    print("FAIL  %-38s %s" % (name, detail))


class Client(object):
    def __init__(self, base, key=None):
        if base.endswith("/"):
            base = base[:-1]
        self.base = base
        self.key = key
        if base.startswith("http://"):
            self.scheme, host = "http", base[len("http://"):]
        else:
            self.scheme, host = "https", base[len("https://"):]
        if "/" in host:
            host = host.split("/", 1)[0]
        self.host, self.port = (host.split(":", 1) + ["80"])[:2] if ":" in host else (host, "80" if self.scheme == "http" else "443")
        self.port = int(self.port)

    def _headers(self, extra=None):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        if extra:
            h.update(extra)
        return h

    def _conn(self, timeout):
        if self.scheme == "http":
            return http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        return http.client.HTTPSConnection(self.host, self.port, timeout=timeout)

    def post(self, path, body, timeout=600):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method="POST")
        for k, v in self._headers().items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")

    def post_stream(self, path, body, timeout=900):
        """返回 (status, chunks)；chunks 是 data: 行解析后的 JSON（[DONE] -> None）。"""
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method="POST")
        for k, v in self._headers().items():
            req.add_header(k, v)
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            return e.code, []
        out = []
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                out.append(None)
                continue
            try:
                out.append(json.loads(payload))
            except ValueError:
                out.append({"_raw": payload})
        r.close()
        return 200, out

    def post_chunked(self, path, body, timeout=600):
        data = json.dumps(body).encode("utf-8")
        conn = self._conn(timeout)
        conn.putrequest("POST", path, skip_accept_encoding=True)
        for k, v in self._headers({"Transfer-Encoding": "chunked"}).items():
            conn.putheader(k, v)
        conn.endheaders()
        for i in range(0, len(data), 307):
            piece = data[i:i + 307]
            conn.send(("%x\r\n" % len(piece)).encode() + piece + b"\r\n")
        conn.send(b"0\r\n\r\n")
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", "replace")
        conn.close()
        return resp.status, raw

    def options(self, path, timeout=30):
        req = urllib.request.Request(self.base + path, method="OPTIONS")
        req.add_header("Origin", "http://localhost")
        req.add_header("Access-Control-Request-Method", "POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers)

    def get(self, path, timeout=60):
        req = urllib.request.Request(self.base + path, method="GET")
        for k, v in self._headers().items():
            if k != "Content-Type":
                req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")


def delta_texts(chunks, field):
    out = []
    for c in chunks:
        if not isinstance(c, dict):
            continue
        for ch in c.get("choices") or []:
            d = ch.get("delta") or {}
            if field in d and isinstance(d[field], str):
                out.append(d[field])
    return "".join(out)


def accumulate_tool_calls(chunks):
    calls = {}
    order = []
    bad_reason = None
    for c in chunks:
        if not isinstance(c, dict):
            continue
        for ch in c.get("choices") or []:
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                if "index" not in tc:
                    bad_reason = "tool_calls delta without index"
                    continue
                idx = tc["index"]
                if idx not in calls:
                    calls[idx] = {"id": None, "type": None, "name": None, "arguments": ""}
                    order.append(idx)
                cur = calls[idx]
                if tc.get("id"):
                    cur["id"] = tc["id"]
                if tc.get("type"):
                    cur["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    cur["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    cur["arguments"] += fn["arguments"]
    return [calls[i] for i in order], bad_reason


def finish_reason(chunks):
    reason = None
    for c in chunks:
        if not isinstance(c, dict):
            continue
        for ch in c.get("choices") or []:
            if ch.get("finish_reason"):
                reason = ch["finish_reason"]
    return reason


def choices_empty_usage(chunks):
    for c in chunks:
        if isinstance(c, dict) and c.get("usage") and c.get("choices") == []:
            return c["usage"]
    return None


LONG_SYS = ("You are a careful assistant running on an embedded NPU. "
            "Follow the user's instructions exactly and keep answers short. ") * 48


def base_chat(thinking=False):
    return {
        "model": "Spark-X2.6-4B",
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"thinking": thinking},
        "messages": [{"role": "system", "content": LONG_SYS}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default="")
    ap.add_argument("--slots", choices=["on", "off"], default="off")
    ap.add_argument("--save", default="")
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()

    if args.compare:
        a = json.load(open(args.compare[0]))
        b = json.load(open(args.compare[1]))
        same = a.get("turns") == b.get("turns")
        print("turns A: %r" % (a.get("turns"),))
        print("turns B: %r" % (b.get("turns"),))
        if same:
            ok("multi-turn byte-identical", "slot on/off outputs match")
        else:
            bad("multi-turn byte-identical", "slot on/off outputs differ")
        print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
        return 0 if not FAIL else 1

    c = Client(args.base, args.key or None)
    expect_slots = args.slots == "on"
    record = {"turns": [], "prefix": []}

    # 1) /v1/models 元数据
    st, body = c.get("/v1/models")
    try:
        m = json.loads(body)["data"][0]
        assert m.get("properties", {}).get("context_length", 0) > 0
        assert "tools" in m.get("supported_parameters", [])
        assert m.get("capabilities", {}).get("tools") is True
        ok("models metadata", "ctx=%s" % m["properties"]["context_length"])
    except Exception as e:
        bad("models metadata", "%s %s" % (st, e))

    # 2) CORS 预检
    st, hdrs = c.options("/v1/chat/completions")
    allow = hdrs.get("Access-Control-Allow-Methods", "") or hdrs.get("access-control-allow-methods", "")
    if st in (200, 204) and "POST" in allow.upper():
        ok("CORS preflight", allow)
    else:
        bad("CORS preflight", "%s %r" % (st, allow))

    # 3) chunked 请求体
    b = base_chat()
    b["messages"].append({"role": "user", "content": "Reply with the single word: pong"})
    st, body = c.post_chunked("/v1/chat/completions", b)
    if st == 200:
        ok("chunked request body", "200")
    else:
        bad("chunked request body", "%s %s" % (st, body[:200]))

    # 4) /v1/embeddings
    st, body = c.post("/v1/embeddings", {"input": ["hello", "world"]}, timeout=120)
    try:
        d = json.loads(body)
        assert len(d["data"]) == 2 and len(d["data"][0]["embedding"]) > 0
        ok("embeddings", "dim=%d" % len(d["data"][0]["embedding"]))
    except Exception as e:
        bad("embeddings", "%s %s" % (st, e))

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }]
    tool_prompt = "Use the get_weather tool to look up Shanghai. You must call the tool, not answer in text."

    # 5) 流式 tool_calls 增量协议
    b = base_chat()
    b["messages"].append({"role": "user", "content": tool_prompt})
    b["tools"] = tools
    b["stream"] = True
    b["stream_options"] = {"include_usage": True}
    st, chunks = c.post_stream("/v1/chat/completions", b)
    calls, why = accumulate_tool_calls(chunks)
    content = delta_texts(chunks, "content")
    usage = choices_empty_usage(chunks)
    if st != 200:
        bad("stream tool_calls", "HTTP %s" % st)
    elif not chunks or chunks[-1] is not None:
        bad("stream tool_calls", "missing [DONE]")
    elif why:
        bad("stream tool_calls", why)
    elif not calls:
        bad("stream tool_calls", "no tool_calls in stream; content=%r" % content[:120])
    elif not calls[0]["id"] or calls[0]["type"] != "function" or not calls[0]["name"]:
        bad("stream tool_calls", "first delta missing id/type/name: %r" % (calls[0],))
    elif finish_reason(chunks) != "tool_calls":
        bad("stream tool_calls", "finish_reason=%r" % finish_reason(chunks))
    elif TOOL_OPEN in content or TOOL_CLOSE in content or TAG_ARGK in content or TAG_ARGV in content:
        bad("stream tool_calls", "raw tool markup leaked into content")
    elif usage is None:
        bad("stream tool_calls", "missing usage chunk with choices=[]")
    else:
        try:
            json.loads(calls[0]["arguments"] or "{}")
            ok("stream tool_calls", "id=%s name=%s args=%s" % (calls[0]["id"], calls[0]["name"],
                                                               (calls[0]["arguments"] or "")[:60]))
        except ValueError as e:
            bad("stream tool_calls", "arguments is not valid JSON: %s %r" % (e, calls[0]["arguments"][:120]))

    # 6) 非流式 content=null
    b = base_chat()
    b["messages"].append({"role": "user", "content": tool_prompt})
    b["tools"] = tools
    st, body = c.post("/v1/chat/completions", b)
    try:
        d = json.loads(body)
        msg = d["choices"][0]["message"]
        assert d["choices"][0]["finish_reason"] == "tool_calls"
        assert msg.get("content", "missing") is None
        assert len(msg.get("tool_calls") or []) >= 1
        ok("non-stream content=null", "tool_calls=%d" % len(msg["tool_calls"]))
    except Exception as e:
        bad("non-stream content=null", "%s %s" % (st, e))

    # 7) tool_choice=required
    b = base_chat()
    b["messages"].append({"role": "user", "content": "What time is it in Shanghai?"})
    b["tools"] = tools
    b["tool_choice"] = "required"
    st, body = c.post("/v1/chat/completions", b)
    try:
        d = json.loads(body)
        calls2 = d["choices"][0]["message"].get("tool_calls") or []
        assert d["choices"][0]["finish_reason"] == "tool_calls" and calls2
        assert calls2[0]["function"]["name"] == "get_weather"
        ok("tool_choice=required", "forced %s" % calls2[0]["function"]["name"])
    except Exception as e:
        bad("tool_choice=required", "%s %s" % (st, e))

    # 8) stop 句中截断
    b = base_chat()
    b["messages"].append({"role": "user", "content": "Reply with exactly this text and nothing else: abcdef"})
    b["stop"] = ["cd"]
    st, body = c.post("/v1/chat/completions", b)
    try:
        d = json.loads(body)
        text = d["choices"][0]["message"].get("content") or ""
        assert "cd" not in text, "stop string leaked: %r" % text
        ok("stop mid-sentence", "content=%r" % text[:40])
    except Exception as e:
        bad("stop mid-sentence", "%s %s" % (st, e))

    # 9) 未知 response_format -> 400
    b = base_chat()
    b["messages"].append({"role": "user", "content": "hi"})
    b["response_format"] = {"type": "yaml_mode"}
    st, body = c.post("/v1/chat/completions", b)
    if st == 400:
        ok("unknown response_format=400", "")
    else:
        bad("unknown response_format=400", "HTTP %s %s" % (st, body[:160]))

    # 10) /v1/responses 非流式
    st, body = c.post("/v1/responses", {
        "model": "Spark-X2.6-4B",
        "input": "Reply with the single word: pong",
        "max_output_tokens": 64,
        "temperature": 0,
        "reasoning": {"effort": "none"},
    })
    try:
        d = json.loads(body)
        assert d["object"] == "response" and d["status"] in ("completed", "incomplete")
        assert isinstance(d["output"], list) and d["output"]
        assert d["usage"]["input_tokens"] > 0
        ok("responses non-stream", "output items=%d" % len(d["output"]))
    except Exception as e:
        bad("responses non-stream", "%s %s" % (st, body[:200]))

    # 11) /v1/responses 流式
    st, chunks = c.post_stream("/v1/responses", {
        "model": "Spark-X2.6-4B",
        "input": "Reply with the single word: pong",
        "max_output_tokens": 64,
        "temperature": 0,
        "reasoning": {"effort": "none"},
        "stream": True,
    })
    types = [x.get("type") for x in chunks if isinstance(x, dict)]
    if st == 200 and "response.created" in types and "response.completed" in types:
        ok("responses stream", "events=%d" % len(types))
    else:
        bad("responses stream", "HTTP %s types=%r" % (st, types[:8]))

    # 12) 多轮 KV 槽位续接（同时记录输出用于 on/off 对拍）
    turns = []
    prefix = []
    msgs = [{"role": "system", "content": LONG_SYS}, {"role": "user", "content": "My name is Ada. Remember it. Answer with OK only."}]
    for i, user_text in enumerate(["What is my name? Answer with the name only.", "Repeat the name once more."]):
        b = base_chat()
        b["messages"] = msgs
        b["max_tokens"] = 96
        st, body = c.post("/v1/chat/completions", b)
        try:
            d = json.loads(body)
            ans = d["choices"][0]["message"].get("content") or ""
            u = d.get("usage") or {}
            prefix.append(u.get("kv_prefix_tokens", 0))
            turns.append(ans)
            msgs = msgs + [{"role": "assistant", "content": ans}, {"role": "user", "content": user_text}]
        except Exception as e:
            bad("multi-turn slot turn %d" % (i + 1), "%s %s" % (st, e))
            turns.append("")
            prefix.append(0)
    if all(turns):
        ok("multi-turn output", "turns=%r" % (turns,))
    else:
        bad("multi-turn output", "empty turn: %r" % (turns,))
    if expect_slots:
        if len(prefix) > 1 and prefix[-1] > 0:
            ok("KV slot reuse", "kv_prefix_tokens=%r" % (prefix,))
        else:
            bad("KV slot reuse", "expected second turn to reuse KV, prefix=%r" % (prefix,))
    else:
        if all(p == 0 for p in prefix):
            ok("KV slot off baseline", "prefix=%r" % (prefix,))
        else:
            bad("KV slot off baseline", "prefix should be 0, got %r" % (prefix,))

    record["turns"] = turns
    record["prefix"] = prefix
    if args.save:
        with open(args.save, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print("saved -> %s" % args.save)

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
