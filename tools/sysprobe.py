#!/usr/bin/env python3
# system 消息到底有没有进模型：往 system 轮埋一个只有该轮知道的暗号，然后只向 user 轮提问。
# 判据是行为（能否答出暗号）而不是 prompt_tokens —— 后者只作旁证（system 里的填充词不改变它）。
#
# 背景：RKNN3 1.2.0 的 rknn3_session_set_chat_template() 的 system_prompt 参数对 Spark-X2.6 实测无效，
# 详见 docs/troubleshooting.md §12。本服务默认 --system-in-prompt on 自己把 ChatML 拼进 prompt 文本，
# 跑这个脚本应当 5/5 命中；若只有对照用例命中，说明 system 被 runtime 吃掉了。
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8080"
MODEL = "Spark-X2.6-4B"

CASES = [
    ("sys_fruit", "system", "kumquat",
     "The secret fruit named in these instructions is kumquat.",
     "What is the secret fruit named in your instructions? Answer with the fruit word only."),
    ("sys_animal", "system", "platypus",
     "The secret animal named in these instructions is platypus.",
     "What is the secret animal named in your instructions? Answer with the animal word only."),
    ("sys_number", "system", "7355608",
     "Remember: the magic number in these instructions is 7355608.",
     "What is the magic number stated in your instructions? Answer with digits only."),
    ("sys_word", "system", "banana",
     "The secret word named in these instructions is banana.",
     "What is the secret word named in your instructions? Answer with the word only."),
    # 对照：同样的内容放 user 轮，无论 system 是否生效都应当命中；它不命中说明脚本本身不可信
    ("ctrl_user_number", "user", "4195832",
     "Remember: the magic number in this instruction is 4195832.",
     "What is the magic number stated above? Answer with digits only."),
]


def chat(messages, max_tokens=120):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    m = d["choices"][0]["message"]
    return d["usage"]["prompt_tokens"], (m.get("content") or "")


def main(argv):
    global BASE, MODEL
    args = list(argv)
    while args:
        if args[0] == "--base" and len(args) > 1:
            BASE = args[1]
            del args[:2]
        elif args[0] == "--model" and len(args) > 1:
            MODEL = args[1]
            del args[:2]
        else:
            print("usage: sysprobe.py [--base URL] [--model NAME]")
            return 2

    hits = 0
    for name, where, needle, secret, question in CASES:
        msgs = []
        if where == "system":
            msgs.append({"role": "system", "content": secret})
            msgs.append({"role": "user", "content": question})
        else:
            msgs.append({"role": "user", "content": secret + " " + question})
        try:
            pt, ans = chat(msgs)
        except Exception as e:
            print("%-18s ERR %s" % (name, repr(e)))
            continue
        hit = needle in ans
        hits += 1 if hit else 0
        print("%-18s %s pt=%-5d needle=%-9s %s"
              % (name, "HIT " if hit else "MISS", pt, needle, ans.strip().replace("\n", " / ")[:120]))
        time.sleep(1)

    total = len(CASES)
    print("HIT=%d/%d" % (hits, total))
    return 0 if hits == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
