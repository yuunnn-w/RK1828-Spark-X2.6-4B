#!/usr/bin/env python3
"""A/B：思考模式 与 重复惩罚 对答案正确性的影响（板端，打本机 OpenAI 服务）。"""
import itertools
import json
import subprocess
import sys

Q = [
    ("en_mult", "What is 17*23? Answer with just the number.", "391"),
    ("zh_mult", "37乘以48等于多少？只回答数字。", "1776"),
    ("zh_math", "一个笼子里有鸡和兔，共12个头、34只脚。问鸡和兔各几只？最后一行写：答案：鸡a只，兔b只", "鸡7只，兔5只"),
    ("zh_frac", "计算 (3/8 + 5/12) 的结果，用最简分数表示，只回答分数。", "19/24"),
    ("en_word", "一个单词 'extraordinary' 里有几个字母？只回答数字。", "13"),
    ("zh_sq", "121的平方根是多少？只回答数字。", "11"),
]


def ask(msgs, max_tokens, thinking, rp):
    payload = {
        "model": "Spark-X2.6-4B",
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": thinking},
        "extra_body": {"repeat_penalty": rp, "top_k": 1},
    }
    p = subprocess.run(
        ["curl", "-s", "-m", "300", "http://127.0.0.1:8080/v1/chat/completions",
         "-H", "Content-Type: application/json", "-d", json.dumps(payload, ensure_ascii=False)],
        capture_output=True, text=True)
    try:
        d = json.loads(p.stdout)
        ch = d["choices"][0]
        return ch["message"]["content"], ch["finish_reason"], d.get("usage", {})
    except Exception:
        return "<parse fail> " + p.stdout[:200], "?", {}


def main():
    for thinking, rp in itertools.product([True, False], [1.0, 1.05, 1.15]):
        print("=== thinking=%s repeat_penalty=%.2f" % (thinking, rp))
        ok = 0
        for tag, q, gold in Q:
            txt, fin, us = ask([{"role": "user", "content": q}], 1024, thinking, rp)
            hit = gold in txt
            ok += 1 if hit else 0
            print("  %-9s gold=%-12s hit=%s fin=%s tok=%s len=%d" %
                  (tag, gold, "Y" if hit else "n", fin, us.get("completion_tokens"), len(txt)))
            print("     ", txt[:180].replace("\n", "\\n"))
        print("  --> 命中 %d/%d" % (ok, len(Q)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
