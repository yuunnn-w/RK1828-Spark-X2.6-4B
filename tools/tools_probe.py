#!/usr/bin/env python3
# 工具调用专项探针：用官方 function-call 示例里同一套 tools 定义与同一句 prompt，
# 用来区分"服务端接线问题"与"模型本身不遵循工具协议"。
import json
import sys
import time
import urllib.request

TOOLS = [
    {"type": "function", "function": {
        "name": "get_current_temperature",
        "description": "Get current temperature at a location.",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string", "description": 'The location to get the temperature for, in the format "City, State, Country".'},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": 'The unit to return the temperature in. Defaults to "celsius".'}},
        "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "get_temperature_date",
        "description": "Get temperature at a location and date.",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string", "description": 'The location to get the temperature for, in the format "City, State, Country".'},
            "date": {"type": "string", "description": 'The date to get the temperature for, in the format "Year-Month-Day".'},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": 'The unit to return the temperature in. Defaults to "celsius".'}},
        "required": ["location", "date"]}}},
]

PROMPTS = [
    "What's the temperature in San Francisco now? How about tomorrow? Current Date: 2024-09-30.",
    "北京今天气温多少度？请先调用天气工具查询，再根据工具结果回答。",
]


def ask(base, model, prompt, thinking, tools=True):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512, "stream": False, "temperature": 0}
    if tools:
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
    if not thinking:
        body["chat_template_kwargs"] = {"thinking": False}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read().decode("utf-8"))
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return ch.get("finish_reason"), msg.get("tool_calls"), (msg.get("content") or "")[:220], (time.time() - t0)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    out = []
    for p in PROMPTS:
        for thinking in (True, False):
            for tools in (True, False):
                try:
                    fin, tc, content, sec = ask(base, "Spark-X2.6-4B", p, thinking, tools)
                except Exception as e:
                    fin, tc, content, sec = "ERR", None, repr(e)[:180], 0.0
                hit = bool(tc)
                print("[%s] thinking=%-5s tools=%-5s finish=%-8s tool_calls=%s %.1fs"
                      % ("HIT " if hit else "    ", thinking, tools, fin,
                         json.dumps(tc, ensure_ascii=False)[:150] if tc else "none", sec), flush=True)
                print("      content: %s" % content.replace("\n", " ")[:180], flush=True)
                out.append({"prompt": p[:30], "thinking": thinking, "tools": tools, "finish": fin,
                            "tool_calls": tc, "content": content, "sec": round(sec, 1)})
    json.dump(out, open("results/tools_probe.json", "w"), ensure_ascii=False, indent=1)
    hits = sum(1 for o in out if o["tool_calls"])
    print("[tools_probe] 命中 %d / %d -> results/tools_probe.json" % (hits, len(out)), flush=True)


if __name__ == "__main__":
    main()
