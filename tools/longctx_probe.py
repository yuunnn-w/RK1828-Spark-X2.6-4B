#!/usr/bin/env python3
"""长上下文阶梯实测：逐级加大 prompt，并在中段埋一个"needle"事实，
末尾要求模型回忆 —— 既验证 64K 窗口能真的用上，也保证不出现复读。
每级之间检查卡健康（rknn-smi），异常立刻停止（不做任何 reset）。

用法: python3 -u longctx_probe.py [--steps 2000,8000,16000,32000,48000,64000]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8080"
MODEL = "Spark-X2.6-4B"

FILLER = ("人工智能模型的推理过程分为预填充与解码两个阶段：前者并行处理整段输入序列并建立注意力缓存，"
          "后者逐个 token 生成输出。长上下文窗口的可用性由权重、内部缓冲与分页 KV 缓存共同决定，"
          "而量化位宽与分组残差深度决定了每 token 的缓存成本。")
NEEDLE = "本次测试的密钥数字是 8137。"


def card_state():
    try:
        env = dict(os.environ)
        env["TMPDIR"] = env.get("TMPDIR", "/run/rknn3")
        out = subprocess.run(["rknn-smi", "info"], capture_output=True, text=True, timeout=20, env=env).stdout
        lines = [l for l in out.splitlines() if "Online" in l or "RK1828" in l]
        return " | ".join(l.strip() for l in lines)
    except Exception as e:
        return "smi fail: %s" % e


def ask(prompt, max_tokens, timeout=1800):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d, time.time() - t0


def rep8(t, n=10):
    if len(t) < 2 * n:
        return 0.0
    g = [t[i:i + n] for i in range(len(t) - n + 1)]
    return 1.0 - len(set(g)) / len(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="2000,8000,16000,32000,48000,64000")
    ap.add_argument("--out", default="results/longctx.json")
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",")]

    rows = []
    for target in steps:
        # 组装：前 1/3 填充 + needle + 后 2/3 填充 + 提问
        need_chars = int(target * 1.7)
        head = FILLER * (need_chars // len(FILLER) // 3 + 1)
        tail = FILLER * (need_chars // len(FILLER) * 2 // 3 + 1)
        head = head[:need_chars // 3]
        tail = tail[:need_chars * 2 // 3]
        prompt = head + "\n" + NEEDLE + "\n" + tail + "\n\n问题：上面文本中出现的四位密钥数字是什么？只回答数字。"
        print("=== 目标 %d token 级；prompt 字符数 %d ===" % (target, len(prompt)), flush=True)
        t0 = time.time()
        try:
            d, sec = ask(prompt, 96)
        except Exception as e:
            print("  请求失败: %r" % e, flush=True)
            rows.append({"target": target, "error": repr(e)})
            json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)
            break
        ch = (d.get("choices") or [{}])[0]
        content = ((ch.get("message") or {}).get("content") or "")
        us = d.get("usage", {})
        pt = us.get("prompt_tokens", 0)
        hit = "8137" in content
        print("  prompt_tokens=%d completion=%d e2e=%.1fs finish=%s needle=%s rep8=%.3f"
              % (pt, us.get("completion_tokens", 0), sec, ch.get("finish_reason"),
                 "HIT" if hit else "miss", rep8(content)), flush=True)
        print("  卡状态: %s" % card_state(), flush=True)
        rows.append({"target": target, "prompt_chars": len(prompt), "prompt_tokens": pt,
                     "completion_tokens": us.get("completion_tokens"), "sec": round(sec, 1),
                     "finish": ch.get("finish_reason"), "needle_hit": hit,
                     "prefill_ms": us.get("prefill_ms"), "decode_ms": us.get("decode_ms"),
                     "veto": us.get("degeneration_veto", 0), "content_head": content[:200],
                     "rep8": round(rep8(content), 3)})
        json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)
        if not hit or us.get("completion_tokens", 0) == 0:
            print("  本级未通过，停止继续加压", flush=True)
            break
        time.sleep(4)

    ok = [r for r in rows if r.get("needle_hit")]
    print("\n通过 %d/%d 级" % (len(ok), len(steps)))
    for r in rows:
        print("  %s" % json.dumps(r, ensure_ascii=False)[:220])
    print("-> %s" % args.out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
