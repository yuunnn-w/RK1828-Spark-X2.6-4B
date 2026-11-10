#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 yuunnn-w
# SPDX-License-Identifier: GPL-3.0-only
"""用官方仓库的 tokenizer + transformers 渲染参考 prompt（chat template 一致性检查的第一步）。

必须在装了 transformers==4.57.1（与模型 config 的 transformers_version 一致）的转换机上跑：
    python3 convert/render_chat_template.py --model-dir model/Spark-X2.6-4B \
            --out results/chat_template_reference.json
产物交给 convert/check_chat_template.py 做逐字节比对。
"""
import argparse
import json
import os

CASES_KEY = "cases"


def tc_msg(args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_current_temperature", "arguments": args}}]}


TOOLS_CN = [{"type": "function", "function": {
    "name": "get_current_temperature",
    "description": "获取指定城市的当前温度",
    "parameters": {"type": "object", "properties": {
        "location": {"type": "string", "description": "城市名，例如 San Francisco, California, USA"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位"}},
        "required": ["location"]}}}]
TOOLS_EN = [{"type": "function", "function": {
    "name": "get_current_temperature",
    "description": "get temp",
    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]

CASES = [
    ("c1_plain", [{"role": "user", "content": "1+1 等于几？只回答数字。"}], None, True, None),
    ("c2_system", [{"role": "system", "content": "你是严谨的数学助教。"},
                   {"role": "user", "content": "鸡兔同笼，头35脚94，各几只？"}], None, True, None),
    ("c3_tools_cn", [{"role": "user", "content": "旧金山现在天气如何？"}], TOOLS_CN, True, None),
    ("c4_tools_en", [{"role": "user", "content": "旧金山现在天气如何？"}], TOOLS_EN, True, None),
    ("c5_toolcall_hist", [
        {"role": "user", "content": "旧金山现在天气如何？"},
        tc_msg({"location": "San Francisco, California, USA", "unit": "celsius"}),
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 18.3}'},
        {"role": "user", "content": "换算成华氏度"}], TOOLS_EN, True, None),
    ("c6_reasoning_hist", [{"role": "user", "content": "17*23"},
                           {"role": "assistant", "content": "391", "reasoning_content": "先算 17*20=340，再加 51"}],
     None, True, None),
    ("c7_think_off", [{"role": "user", "content": "hello"}], None, False, None),
    ("c8_effort_low", [{"role": "user", "content": "hello"}], None, True, "low"),
    ("c9_multi_system", [{"role": "system", "content": "第一条 system"},
                         {"role": "user", "content": [{"type": "text", "text": "混合内容 A"},
                                                      {"type": "text", "text": "B"}]},
                         {"role": "system", "content": "中途插入的 system"}], None, True, None),
    ("c10_last_assistant", [{"role": "user", "content": "写一首两行诗"},
                            {"role": "assistant", "content": "春风又绿"}], None, True, None),
    ("c11_nonstring_args", [{"role": "user", "content": "查天气"},
                            tc_msg({"location": "北京", "unit": "celsius", "days": 3, "tags": ["a", "b"]})],
     TOOLS_EN, True, None),
    # 下面两个官方模板会直接报错，用来记录"官方比我们的服务端更严格"这两件事
    ("c12_string_args_must_fail", [{"role": "user", "content": "查天气"}, tc_msg('{"location":"北京"}')],
     TOOLS_EN, True, None),
    ("c13_effort_invalid_must_fail", [{"role": "user", "content": "hi"}], None, True, "medium"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model/Spark-X2.6-4B")
    ap.add_argument("--out", default="results/chat_template_reference.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer   # 需要 transformers==4.57.1
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    print("tokenizer:", type(tok).__name__, "repo chat_template:", bool(tok.chat_template))

    res = {"transformers": __import__("transformers").__version__}
    for name, msgs, tools, thinking, effort in CASES:
        kw = {"add_generation_prompt": True, "tokenize": False}
        if tools is not None:
            kw["tools"] = tools
        if effort is not None:
            kw["reasoning_effort"] = effort
        try:
            res[name] = tok.apply_chat_template(msgs, enable_thinking=thinking, **kw)
        except Exception as e:
            res[name] = {"error": "%s: %s" % (type(e).__name__, str(e)[:300])}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("wrote", args.out, "cases", len(CASES))
    for k, v in res.items():
        if isinstance(v, dict) and "error" in v:
            print("ERR", k, v["error"])


if __name__ == "__main__":
    main()
