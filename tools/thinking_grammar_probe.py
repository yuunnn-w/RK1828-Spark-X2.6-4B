#!/usr/bin/env python3
"""思考段 grammar 的对照实验：能不能用 structural_tag 逼模型写出思考结束标记、给出非空 content。

用法：python3 thinking_grammar_probe.py [--base http://127.0.0.1:8080]

三个用例，都用 temperature 0（贪心，可复现）：
1. fib 题 + grammar 开（chat_template_kwargs.thinking_grammar: true）
2. fib 题 + grammar 关
3. fib 题 + 硬性 json_schema 约束（这条会走"最终格式"约束，用来确认 grammar 引擎本身是活的）

结论（1.2.0 运行时 + Spark-X2.6，见 docs/server.md 4.3）：1 和 2 逐字节相同，都在思考段里选 EOS、
content 为空；3 也没有 JSON —— grammar 只约束最终格式那段，约束不了思考段。
要复现第 4 个观察（把 grammar 要求的结束标记改成无意义字符串，模型不认）需要改 server/cpp/main.cpp
里 build_runtime_grammar 的 t["end"] 再编译。
"""
import argparse
import json
import time
import urllib.request

MODEL = "Spark-X2.6-4B"
FIB = "用 Python 写一个 fib(n) 函数，返回第 n 个斐波那契数。只输出代码，不要解释。"
SCHEMA = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"],
          "additionalProperties": False}


def post(base, name, prompt, extra):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "temperature": 0, "max_tokens": 400}
    body.update(extra)
    t0 = time.time()
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    el = round(time.time() - t0, 2)
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    u = d.get("usage") or {}
    c = msg.get("content") or ""
    rc = msg.get("reasoning_content") or ""
    return {"name": name, "finish": ch.get("finish_reason"), "ct": u.get("completion_tokens"),
            "content": c, "reasoning": rc, "content_len": len(c), "reasoning_len": len(rc), "wall": el}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [
        post(args.base, "fib_grammar_on", FIB, {"chat_template_kwargs": {"thinking_grammar": True}}),
        post(args.base, "fib_grammar_off", FIB, {}),
        post(args.base, "fib_json_schema", FIB, {"response_format": {"type": "json_schema",
                                                                     "json_schema": {"schema": SCHEMA}}}),
    ]
    for r in rows:
        print("%-15s finish=%-11s ct=%-5s wall=%-6s content_len=%-5d reasoning_len=%d" %
              (r["name"], r["finish"], r["ct"], r["wall"], r["content_len"], r["reasoning_len"]), flush=True)
    same = rows[0]["content"] == rows[1]["content"] and rows[0]["reasoning"] == rows[1]["reasoning"]
    print("grammar on 与 off 的输出是否逐字节相同：", same, "（相同 = 这条 grammar 没有起作用）")
    print("加了 json_schema 之后 content 有没有 JSON：", rows[2]["content"].strip().startswith("{"))
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
