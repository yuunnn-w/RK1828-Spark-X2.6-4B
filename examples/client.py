#!/usr/bin/env python3
"""RK1828 Spark-X2.6-4B 服务的最小可用客户端示例（只用标准库）。

用法:
  python3 examples/client.py                      # 跑全部示例
  python3 examples/client.py --base http://<板子IP>:8080
  python3 examples/client.py --only tools
"""
import argparse
import json
import time
import urllib.request

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "获取指定城市的当前温度",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市名，例如 San Francisco, California, USA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位"},
            },
            "required": ["location"],
        },
    },
}]


def req(base, path, body=None, timeout=1200):
    data = None if body is None else json.dumps(body).encode("utf-8")
    r = urllib.request.Request(base.rstrip("/") + path, data=data,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return f.status, f.read().decode("utf-8", "replace")


def jreq(base, path, body=None, timeout=1200):
    st, raw = req(base, path, body, timeout)
    return st, json.loads(raw)


def demo_health(base):
    st, d = jreq(base, "/health")
    print("GET /health ->", st, d)
    st, d = jreq(base, "/v1/models")
    for m in d.get("data", []):
        print("model:", m["id"], "ctx:", m["meta"]["max_context_len"])
        print("采样默认:", json.dumps(m["meta"]["sampling_defaults"], ensure_ascii=False))


def demo_chat(base):
    body = {"model": "Spark-X2.6-4B",
            "messages": [{"role": "user", "content": "一个笼子里有鸡和兔，头 35 个、脚 94 只，问鸡兔各几只？只给答案。"}],
            "temperature": 0}          # 精度关键的任务建议贪心；不传则用服务端默认（厂商推荐值）
    t0 = time.time()
    st, d = jreq(base, "/v1/chat/completions", body)
    ch = d["choices"][0]
    print("POST /v1/chat/completions ->", st, "finish=", ch["finish_reason"])
    if ch["message"].get("reasoning_content"):
        print("  reasoning:", ch["message"]["reasoning_content"][:180].replace("\n", " "))
    print("  content:", ch["message"]["content"][:300])
    print("  usage:", json.dumps(d["usage"], ensure_ascii=False), "wall %.1fs" % (time.time() - t0))


def demo_stream(base):
    body = {"model": "Spark-X2.6-4B", "stream": True,
            "messages": [{"role": "user", "content": "用两句话解释 KV cache 量化。"}]}
    st, raw = req(base, "/v1/chat/completions", body)
    n = 0
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            break
        d = json.loads(payload)
        delta = (d.get("choices") or [{}])[0].get("delta") or {}
        if delta.get("content"):
            n += 1
    print("流式：收到 %d 个含 content 的增量块" % n)


def demo_tools(base):
    body = {"model": "Spark-X2.6-4B", "tools": TOOLS, "temperature": 0,
            "messages": [{"role": "user", "content": "北京今天多少度？请用工具查。"}]}
    st, d = jreq(base, "/v1/chat/completions", body)
    ch = d["choices"][0]
    print("带工具 -> finish=", ch["finish_reason"])
    calls = ch["message"].get("tool_calls") or []
    for c in calls:
        print("  调用:", c["function"]["name"], "参数:", c["function"]["arguments"])
    if not calls:
        print("  （模型这次没走工具，属正常：工具遵循度由模型决定）")
        return
    # 回灌工具结果：OpenAI 里 arguments 是字符串，历史里给字符串也能被服务端解析
    msgs = body["messages"] + [{"role": "assistant", "content": "", "tool_calls": calls},
                               {"role": "tool", "tool_call_id": calls[0]["id"],
                                "content": json.dumps({"temp": 21.5, "unit": "celsius"})},
                               {"role": "user", "content": "换成华氏度告诉我。"}]
    st, d = jreq(base, "/v1/chat/completions", {"model": "Spark-X2.6-4B", "messages": msgs, "tools": TOOLS})
    print("工具结果续写 ->", d["choices"][0]["message"]["content"][:200])


def demo_longctx(base, target_tokens=40000):
    """把一根 needle 埋进长上下文，再问它 —— 验证长上下文真的可用（不是只装得下）。"""
    unit = "第 %d 段：RK1828 的 NPU 通过 PCIe 与主控通信，链路带宽有限，所以权重常驻在卡上。"
    filler = "\n".join(unit % i for i in range(target_tokens // 14))
    needle = "本次会话的暗号是 BLUE-HARBOUR-42。"
    msgs = [{"role": "system", "content": "你是一个严谨的助手，回答要简短。"},
            {"role": "user", "content": filler + "\n" + needle},
            {"role": "user", "content": "本次会话的暗号是什么？只回答暗号。"}]
    t0 = time.time()
    st, d = jreq(base, "/v1/chat/completions", {"model": "Spark-X2.6-4B", "messages": msgs,
                                               "temperature": 0, "max_tokens": 64}, timeout=1800)
    u = d["usage"]
    hit = "BLUE-HARBOUR-42" in (d["choices"][0]["message"].get("content") or "")
    print("长上下文：prompt_tokens=%s prefill=%.1fs needle=%s"
          % (u.get("prompt_tokens"), (u.get("prefill_ms") or 0) / 1000.0, "命中" if hit else "未命中"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--only", default="all", help="health|chat|stream|tools|longctx")
    ap.add_argument("--needle-tokens", type=int, default=40000)
    args = ap.parse_args()
    steps = {"health": demo_health, "chat": demo_chat, "stream": demo_stream,
             "tools": demo_tools, "longctx": lambda b: demo_longctx(b, args.needle_tokens)}
    if args.only == "all":
        for name, fn in steps.items():
            print("\n===== %s =====" % name)
            try:
                fn(args.base)
            except Exception as e:
                print("  跳过：%s: %s" % (type(e).__name__, e))
    else:
        steps[args.only](args.base)


if __name__ == "__main__":
    main()
