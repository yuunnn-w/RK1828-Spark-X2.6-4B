#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 yuunnn-w
# SPDX-License-Identifier: GPL-3.0-only
"""把本服务端 compose_spark() 的等价 Python 镜像与 convert/render_chat_template.py 产出的官方参考 prompt
逐字节比对，输出差异报告。先跑 render，再跑本脚本。"""
import json
import os
import sys

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="results/chat_template_reference.json")
ap.add_argument("--out", default="results/chat_template_diff.txt")
args = ap.parse_args()
REF, OUT = args.ref, args.out

SOS = "<｜start▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
SYS = "<|System|>\n"
USER = "<|User|>"
BOT = "<|Bot|>"
TOOL = "<|Tool|>"


def jdump_compact(o):
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def jdump_spaced(o):
    return json.dumps(o, ensure_ascii=False, separators=(", ", ": "))


def content_to_text(c):
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c)


def tools_block(tools):
    if not tools:
        return ""
    out = "\n\n## Tools\nYou have access to the following functions:\n<tools>"
    for t in tools:
        out += "\n" + jdump_compact(t["function"])
    out += "\n</tools>"
    return out


def tool_calls_text(m):
    out = ""
    for tc in m.get("tool_calls") or []:
        f = tc.get("function") or {}
        out += "<tool_call>" + (f.get("name") or "")
        args = f.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = None
        if isinstance(args, dict):
            for k, v in args.items():
                out += "<arg_key>" + k + "</arg_key><arg_value>" + (v if isinstance(v, str) else jdump_spaced(v)) + "</arg_value>"
        out += "</tool_call>"
    return out


def mine(messages, tools=None, thinking=True, effort=None, default_system="you are a helpful assistant."):
    e = "low" if effort == "low" else "high"
    ri = ""
    if thinking:
        ri = ("\n`thinking_effort` guides on how much to think in your thinking channel "
              "(not including the response channel), supported values include `low` and `high`.\n"
              "Now the system is invoked with `thinking_effort=" + e + "`.")
    sys_body = default_system + ri
    initial = ""
    if messages and messages[0].get("role") == "system":
        initial = content_to_text(messages[0].get("content"))
    system = SOS + SYS + sys_body + tools_block(tools) + (("\n\n" + initial) if initial else "") + EOS

    turns = []
    for i, m in enumerate(messages):
        role = m.get("role", "user")
        content = content_to_text(m.get("content"))
        if role == "system" and i == 0:
            continue
        if role == "system":
            turns.append((SYS, content))
        elif role == "assistant":
            rc = m.get("reasoning_content") or ""
            body = ("<think>" + rc + "</think>" if rc else "</think>") + content + tool_calls_text(m)
            turns.append((BOT, body))
        elif role == "tool":
            wrapped = "<tool_response>" + content + "</tool_response>"
            if turns and turns[-1][0] == TOOL:
                turns[-1] = (TOOL, turns[-1][1] + wrapped)
            else:
                turns.append((TOOL, wrapped))
        else:
            turns.append((USER, content))
    if not turns:
        return None
    text = turns[0][1]
    for marker, body in turns[1:]:
        text += EOS + SOS + marker + body
    prefix = SOS + turns[0][0]
    if turns[-1][0] == BOT:
        postfix = ""
    else:
        postfix = EOS + SOS + BOT + ("<think>" if thinking else "</think>")
    return system + prefix + text + postfix


CASES = {
    "c1_plain":        ([{"role": "user", "content": "1+1 等于几？只回答数字。"}], None, True, None),
    "c2_system":       ([{"role": "system", "content": "你是严谨的数学助教。"},
                         {"role": "user", "content": "鸡兔同笼，头35脚94，各几只？"}], None, True, None),
    "c3_tools_cn":     ([{"role": "user", "content": "旧金山现在天气如何？"}],
                        [{"type": "function", "function": {
                            "name": "get_current_temperature",
                            "description": "获取指定城市的当前温度",
                            "parameters": {"type": "object", "properties": {
                                "location": {"type": "string", "description": "城市名，例如 San Francisco, California, USA"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位"}},
                                "required": ["location"]}}}], True, None),
    "c4_tools_en":     ([{"role": "user", "content": "旧金山现在天气如何？"}],
                        [{"type": "function", "function": {"name": "get_current_temperature", "description": "get temp",
                                                           "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                                                                        "required": ["location"]}}}], True, None),
    "c5_toolcall_hist": ([
        {"role": "user", "content": "旧金山现在天气如何？"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "get_current_temperature",
                "arguments": {"location": "San Francisco, California, USA", "unit": "celsius"}}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18.3}"},
        {"role": "user", "content": "换算成华氏度"}],
        [{"type": "function", "function": {"name": "get_current_temperature", "description": "get temp",
                                           "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                                                        "required": ["location"]}}}], True, None),
    "c6_reasoning_hist": ([{"role": "user", "content": "17*23"},
                           {"role": "assistant", "content": "391", "reasoning_content": "先算 17*20=340，再加 51"}], None, True, None),
    "c7_think_off":    ([{"role": "user", "content": "hello"}], None, False, None),
    "c8_effort_low":   ([{"role": "user", "content": "hello"}], None, True, "low"),
    "c9_multi_system": ([{"role": "system", "content": "第一条 system"},
                         {"role": "user", "content": [{"type": "text", "text": "混合内容 A"}, {"type": "text", "text": "B"}]},
                         {"role": "system", "content": "中途插入的 system"}], None, True, None),
    "c10_last_assistant": ([{"role": "user", "content": "写一首两行诗"},
                            {"role": "assistant", "content": "春风又绿"}], None, True, None),
    "c11_nonstring_args": ([
        {"role": "user", "content": "查天气"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "get_current_temperature",
                "arguments": {"location": "北京", "unit": "celsius", "days": 3, "tags": ["a", "b"]}}}]}],
        [{"type": "function", "function": {"name": "get_current_temperature", "description": "get temp",
                                           "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                                                        "required": ["location"]}}}], True, None),
}

# 官方模板会直接 raise_exception 的用例（我方实现更宽松），单独标注不参与一致性统计
LENIENT = {
    "c12_string_args_must_fail": (
        [{"role": "user", "content": "查天气"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "call_1", "type": "function", "function": {
                 "name": "get_current_temperature", "arguments": '{"location":"北京"}'}}]}],
        None, True, None),
    "c13_effort_invalid_must_fail": ([{"role": "user", "content": "hi"}], None, True, "medium"),
}

ref = json.load(open(REF, encoding="utf-8"))
lines = []
same = 0
for name, (msgs, tools, thinking, effort) in LENIENT.items():
    a = ref.get(name)
    err = a.get("error") if isinstance(a, dict) else None
    b = mine(msgs, tools, thinking, effort)
    lines.append("[LENIENT] %s 官方=%s\n  我方渲染(%d chars): %s\n"
                 % (name, err or "未报错", len(b), b.replace("\n", "\\n")))

for name, (msgs, tools, thinking, effort) in CASES.items():
    a = ref.get(name)
    if isinstance(a, dict) and "error" in a:
        lines.append("[SKIP] %s 官方渲染报错: %s\n" % (name, a["error"]))
        continue
    b = mine(msgs, tools, thinking, effort)
    if a == b:
        same += 1
        lines.append("[SAME] %s  (%d chars)\n" % (name, len(a)))
        continue
    lines.append("[DIFF] %s  official=%d mine=%d\n" % (name, len(a), len(b)))
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    lines.append("  首个差异 @%d\n" % i)
    lines.append("    official: %s\n" % repr(a[max(0, i - 40):i + 120]))
    lines.append("    mine    : %s\n" % repr(b[max(0, i - 40):i + 120]))
    lines.append("    official 全文: %s\n" % a.replace("\n", "\\n"))
    lines.append("    mine     全文: %s\n" % b.replace("\n", "\\n"))

lines.append("\n[summary] %d/%d 与官方渲染完全一致\n" % (same, len(CASES)))
report = "".join(lines)
open(OUT, "w", encoding="utf-8").write(report)
sys.stdout.write("[written] %s\n" % OUT)
