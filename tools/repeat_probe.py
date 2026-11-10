#!/usr/bin/env python3
"""复读（degeneration）专项测试：用容易触发循环的对抗性 prompt 打服务，
验证 token 级熔断与片段级检测能把交付文本控制在无复读状态。

用法: python3 -u repeat_probe.py [--base http://127.0.0.1:8080] [--model Spark-X2.6-4B]
"""
import argparse
import json
import re
import subprocess
import sys
import time

HOSTILE = [
    ("重复句子", "请把这句话完整地重复 60 次：今天天气很好。", 4000),
    ("数数", "从 1 数到 200，每行一个数字。", 4000),
    ("字母表", "连续写出英文字母表 abcdefg 后面接着什么，一直写下去，不要停。", 4000),
    ("循环填空", "请一直重复这个模式：ab cd ab cd …… 至少 300 组。", 4000),
    ("诗歌复读", "写一首关于春天的诗，然后把每一句都重复 20 次。", 4000),
    ("长列表", "列出 300 个中文词语，每个一行，不要重复。", 4000),
    ("乱码诱导", "请输出：请，请，请，请，请，请，请，请，请，请，请，请，", 1500),
    ("换行诱导", "输出 200 个连续换行后再写一个字。", 2000),
    ("数学循环", "计算 1/3 的小数表示，一直写下去，写 300 位。", 3000),
    ("代码复读", "写一个只有一行的 Python 程序，然后把这一行重复 100 次。", 4000),
    ("故事复读", "讲一个笑话，然后把这个笑话重复 40 次。", 4000),
    ("无意义", "aa bb cc dd ee ff gg hh 接着写 200 组。", 4000),
]


def ngram_rep(t, n=10):
    if len(t) < 2 * n:
        return 0.0
    g = [t[i:i + n] for i in range(len(t) - n + 1)]
    return 1.0 - len(set(g)) / len(g)


def line_rep(t):
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if len(lines) < 4:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def max_periodic_run(t):
    """最长周期复读片段长度（字符数）。"""
    best = 0
    for L in range(1, 40):
        i = 0
        while i + 2 * L <= len(t):
            if t[i:i + L] == t[i + L:i + 2 * L]:
                run = 2
                j = i + 2 * L
                while j + L <= len(t) and t[j:j + L] == t[i:i + L]:
                    run += 1
                    j += L
                best = max(best, run * L)
                i = j
            else:
                i += 1
    return best


def ask(base, model, prompt, max_tokens, thinking, temp):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temp,
        "chat_template_kwargs": {"thinking": thinking},
    }
    p = subprocess.run(
        ["curl", "-s", "-m", "900", base + "/v1/chat/completions",
         "-H", "Content-Type: application/json", "-d", json.dumps(payload, ensure_ascii=False)],
        capture_output=True, text=True)
    try:
        d = json.loads(p.stdout)
        ch = d["choices"][0]
        return (ch.get("message") or {}).get("content", ""), ch.get("finish_reason"), d.get("usage", {})
    except Exception:
        return "<FAIL " + p.stdout[:150] + ">", "?", {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="Spark-X2.6-4B")
    ap.add_argument("--out", default="results/repeat_probe.json")
    args = ap.parse_args()

    rows = []
    print("%-12s %-6s %-8s %-6s %-6s %-8s %-8s %s" %
          ("case", "think", "temp", "tokens", "ngramR", "lineR", "loopLen", "finish"))
    for temp in (0.0, 0.8):
        for thinking in (True, False):
            for name, prompt, mt in HOSTILE:
                txt, fin, us = ask(args.base, args.model, prompt, mt, thinking, temp)
                nr = ngram_rep(txt)
                lr = line_rep(txt)
                lp = max_periodic_run(txt)
                row = dict(case=name, thinking=thinking, temp=temp, finish=fin, nchar=len(txt),
                           tokens=us.get("completion_tokens"), veto=us.get("degeneration_veto", 0),
                           ngram_rep=round(nr, 3), line_rep=round(lr, 3), loop_run_chars=lp)
                rows.append(row)
                flag = "BAD" if (nr > 0.6 or lr > 0.85 or lp > 600) else "ok"
                print("%-12s %-6s %-8.2f %-8s %-6.3f %-6.3f %-8d %s %s"
                      % (name, thinking, temp, us.get("completion_tokens"), nr, lr, lp, fin, flag), flush=True)
                time.sleep(0.4)
    bad = [r for r in rows if r["ngram_rep"] > 0.6 or r["line_rep"] > 0.85 or r["loop_run_chars"] > 600]
    # 有意重复类任务的 line_rep 天然很高，单独看：ngram_rep 与 loop_run 才是失控信号
    runaway = [r for r in rows if r["ngram_rep"] > 0.6 or r["loop_run_chars"] > 1200]
    json.dump({"rows": rows, "flagged": bad, "runaway": runaway}, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("\n样本 %d；flagged %d；runaway(失控复读) %d" % (len(rows), len(bad), len(runaway)))
    for r in runaway[:8]:
        print("  RUNAWAY", r)
    print("-> %s" % args.out)
    return 0 if not runaway else 2


if __name__ == "__main__":
    sys.exit(main())
