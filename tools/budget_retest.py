#!/usr/bin/env python3
# 生成预算放宽后的正确率复测：评测时生成预算有限（例如 max_tokens=1200/1600），一批题会撞上限判错。
# 这里只挑"撞上限且判错"的样本，用更大的 max_tokens 重问一遍，用同一套判分函数复判，
# 得到一个更接近真实水平的估计（抽样，不是全量）。
import argparse
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/root")
from eval_llm import judge, extract_answer, degeneracy, SYS  # noqa: E402


def pick(items, per_domain):
    by = {}
    for it in items:
        by.setdefault(it.get("_domain", "?"), []).append(it)
    picked = []
    for dom in sorted(by):
        picked += by[dom][: per_domain]
    return picked


def ask(base, model, prompt, max_tokens, timeout=1800):
    body = {"model": model,
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    ch = d.get("choices", [{}])[0]
    msg = ch.get("message") or {}
    txt = (msg.get("content") or "")
    if msg.get("reasoning_content"):
        txt = msg["reasoning_content"] + "\n" + txt
    return txt, d.get("usage", {}), ch.get("finish_reason")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/evalv4_spark_think.json")
    ap.add_argument("--eval", default="datasets/eval_real.json")
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="Spark-X2.6-4B")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--sample-per-domain", type=int, default=4)
    ap.add_argument("--cap-at", type=int, default=1590)   # 判定“撞上限”的 completion_tokens 阈值（取评测那轮 max_tokens-10）
    ap.add_argument("--out", default="results/budget_retest.json")
    args = ap.parse_args()

    d = json.load(open(args.src))
    picked = pick(json.load(open(args.eval)), int(d.get("per_domain") or 100))
    groups = {}
    for x in d["results"]:
        u = x.get("usage") or {}
        if u.get("completion_tokens", 0) >= args.cap_at and x.get("verdict") not in ("exact", "numeric", "letter", "unscored"):
            groups.setdefault(x.get("domain", "?"), []).append(x)
    sel = []
    for dom in sorted(groups):
        xs = sorted(groups[dom], key=lambda r: -((r.get("usage") or {}).get("completion_tokens") or 0))
        sel += xs[: args.sample_per_domain]
    print(f"[retest] 撞上限且判错 {sum(len(v) for v in groups.values())} 条，抽样复测 {len(sel)} 条 "
          f"max_tokens={args.max_tokens}", flush=True)

    rows = []
    t0 = time.time()
    fixed = 0
    still = 0
    again_capped = 0
    for n, x in enumerate(sel, 1):
        i = int(x["i"])
        prompt = picked[i].get("input", "")
        try:
            ans, usage, finish = ask(args.base, args.model, prompt, args.max_tokens)
        except Exception as e:
            ans, usage, finish = "", {}, "error"
        v = judge(x.get("target", ""), extract_answer(ans))
        deg = degeneracy(ans)
        capped = (usage.get("completion_tokens") or 0) >= args.max_tokens - 10
        if capped:
            again_capped += 1
        if v in ("exact", "numeric", "letter"):
            fixed += 1
        else:
            still += 1
        rows.append({"domain": x.get("domain"), "i": i, "old_verdict": x.get("verdict"),
                     "new_verdict": v, "new_degen": round(deg, 3), "finish": finish,
                     "capped_again": capped,
                     "completion_tokens": usage.get("completion_tokens"),
                     "degeneration_guard": usage.get("degeneration_guard", 0),
                     "degeneration_loop": usage.get("degeneration_loop", 0),
                     "guard_hits": usage.get("degeneration_veto", 0)})
        print(f"  {n}/{len(sel)} {x.get('domain')} {x.get('verdict')} -> {v} "
              f"ct={usage.get('completion_tokens')} capped={capped} guard={usage.get('degeneration_guard', 0)} "
              f"degen={deg:.2f}", flush=True)
        json.dump({"n": len(rows), "fixed": fixed, "still_wrong": still, "capped_again": again_capped,
                   "max_tokens": args.max_tokens, "rows": rows}, open(args.out, "w"), ensure_ascii=False)

    print(f"[retest] 完成：判错被修复 {fixed}，仍错 {still}，又撞 {args.max_tokens} 上限 {again_capped}，"
          f"用时 {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
