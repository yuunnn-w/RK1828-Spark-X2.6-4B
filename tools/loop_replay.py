#!/usr/bin/env python3
# 复读回归验证：把评测里"疑似复读/被生成上限截断"的样本重放到新服务上，
# 检查新的长块复读熔断是否能把它们收敛掉（finish 提前于 max_tokens 即视为熔断生效）。
import argparse
import json
import sys
import time
import urllib.request

SYS = ("你是一个严谨的助手。请先简要推理，然后必须在最后一行单独输出『答案：<最终答案>』。"
       "数值题只写数值；选择题只写字母选项。")


def degeneracy(text):
    # 与 eval_llm.py 同一套修正过的指标：文本短于一个 24 字符窗口时窗口集合为空，
    # 旧实现会返回 1.0，把"答案很短/提前收尾"误判成复读崩坏。短文本直接判 0。
    if not text or len(text) < 48:
        return 0.0
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lr = 1 - len(set(lines)) / max(1, len(lines))
    grams = [text[i:i + 24] for i in range(0, len(text) - 24, 12)]
    gr = 1 - len(set(grams)) / max(1, len(grams))
    return max(lr, gr)


def ask(base, model, prompt, max_tokens, timeout=1200):
    body = {"model": model,
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    ch = d.get("choices", [{}])[0]
    return (ch.get("message", {}) or {}).get("content", "") or "", d.get("usage", {}), ch.get("finish_reason")


def pick(items, per_domain):
    """复刻 eval_llm.py 的取样逻辑，使 i 能对上。"""
    by_dom = {}
    for it in items:
        by_dom.setdefault(it.get("_domain", "?"), []).append(it)
    picked = []
    for dom in sorted(by_dom):
        picked += by_dom[dom][: per_domain]
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/evalv4_spark_think.json")
    ap.add_argument("--eval", default="datasets/eval_real.json")
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="Spark-X2.6-4B")
    ap.add_argument("--max-tokens", type=int, default=1600)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--per-domain-pick", type=int, default=5)
    ap.add_argument("--out", default="results/loop_replay.json")
    args = ap.parse_args()

    d = json.load(open(args.src))
    per_domain = int(d.get("per_domain") or 100)
    picked = pick(json.load(open(args.eval)), per_domain)
    cand_all = []
    for x in d["results"]:
        u = x.get("usage", {}) or {}
        comp = u.get("completion_tokens", 0)
        if x.get("verdict") == "degenerate" or x.get("degen", 0) >= 0.20 or comp >= 1590:
            i = int(x.get("i", -1))
            if 0 <= i < len(picked):
                x = dict(x)
                x["input"] = picked[i].get("input", "")
                cand_all.append(x)
    # 按领域分层取样，避免全部候选都落在前面的数学领域
    groups = {}
    for x in cand_all:
        groups.setdefault(x.get("domain", "?"), []).append(x)
    cand = []
    for dom in sorted(groups):
        xs = sorted(groups[dom], key=lambda r: (-(r.get("degen") or 0),
                                                -((r.get("usage") or {}).get("completion_tokens") or 0)))
        cand += xs[: args.per_domain_pick]
    cand = cand[: args.limit]
    per_dom = ", ".join(["%s:%d" % (dm, min(len(v), args.per_domain_pick)) for dm, v in sorted(groups.items())])
    print("[loop_replay] 重放 %d 条（候选 %d，来自 %s） 各领域 %s"
          % (len(cand), len(cand_all), args.src, per_dom), flush=True)

    rows = []
    t0 = time.time()
    guarded = 0
    worst = 0.0
    for n, x in enumerate(cand, 1):
        try:
            ans, usage, finish = ask(args.base, args.model, x["input"] if "input" in x else "", args.max_tokens)
        except Exception as e:
            ans, usage, finish = f"<<ERROR {type(e).__name__}: {e}>>", {}, "error"
        deg = degeneracy(ans)
        u = {"completion_tokens": usage.get("completion_tokens"),
             "prompt_tokens": usage.get("prompt_tokens"),
             "prefill_ms": usage.get("prefill_ms"),
             "decode_ms": usage.get("decode_ms"),
             "tokens_per_second": usage.get("tokens_per_second"),
             "degeneration_veto": usage.get("degeneration_veto", 0),
             "degeneration_loop": usage.get("degeneration_loop", 0),
             "degeneration_guard": usage.get("degeneration_guard", 0)}
        fused = bool(u["degeneration_guard"]) or (u["completion_tokens"] or 0) < args.max_tokens
        if u["degeneration_guard"]:
            guarded += 1
        worst = max(worst, deg)
        rows.append({"domain": x.get("domain"), "old_verdict": x.get("verdict"),
                     "old_degen": x.get("degen"), "old_completion": (x.get("usage") or {}).get("completion_tokens"),
                     "new_finish": finish, "new_degen": round(deg, 3), "new_len_chars": len(ans),
                     "fused_or_early_stop": fused, "usage": u, "tail": ans[-160:]})
        print(f"  {n}/{len(cand)} {x.get('domain')} old={x.get('verdict')}/{x.get('degen')} "
              f"new={finish}/{u['completion_tokens']} guard={u['degeneration_guard']} "
              f"loop={u['degeneration_loop']} degen={deg:.2f}", flush=True)
        json.dump({"n": len(rows), "guarded": guarded, "max_degen": worst, "rows": rows},
                  open(args.out, "w"), ensure_ascii=False)

    early = sum(1 for r in rows if r["fused_or_early_stop"])
    print(f"[loop_replay] 完成 {len(rows)} 条 用时 {time.time()-t0:.0f}s "
          f"熔断触发={guarded} 提前收尾={early} 最大degen={worst:.2f} -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
