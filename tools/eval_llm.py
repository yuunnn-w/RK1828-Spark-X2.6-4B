#!/usr/bin/env python3
"""板端 LLM 精度评测 v2：向 rkllm3-server(OpenAI 兼容) 批量提问并自动判分。

与 v1 的差别（v1 的 17% 命中率有很大一部分是判分/预算问题）：
  * 默认不传 temperature（用 server 默认采样）。实测 temperature=0.1 会让 w4a16 的 4B 模型陷入复读崩坏。
  * 默认 max_tokens=1600：这类模型带 <think> 通道，400 token 根本到不了最终答案。
  * 答案抽取放宽：答案：/ #### / 末尾数字 / 末尾字母选项，四种回退。
  * 崩坏检测：行级 + 4-gram 复读率 > 0.6 记 verdict=degenerate，不与"答错"混在一起。
  * 保存完整回答（最多 6000 字）便于人工复核。
"""
import argparse, json, os, re, sys, time, urllib.request
from collections import defaultdict

SYS = ("你是一个严谨的解题助手。请先简要推理，然后必须在最后一行单独输出『答案：<最终答案>』。"
       "数值题只写数值；选择题只写字母选项。")
SYS_CONCISE = ("你是一个解题助手。不要长篇推理，最多用一两句话给出关键依据，然后立刻在最后一行单独输出"
               "『答案：<最终答案>』。数值题只写数值；选择题只写字母选项。")
ANS_RE = re.compile(r"答案\s*[:：]\s*(.+)")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
LET_RE = re.compile(r"\b([A-E])\b")
FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def to_float(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None


def frac_expand(t):
    def rep(m):
        a, b = to_float(m.group(1)), to_float(m.group(2))
        if a is None or not b:
            return m.group(0)
        v = a / b
        return f"{v:g} {m.group(1)}/{m.group(2)}"
    return FRAC_RE.sub(rep, t)


def candidates(t):
    if t is None:
        return set(), set(), ""
    t = frac_expand(str(t))
    m = re.search(r"####\s*(.+)", t)
    if m:
        t = m.group(1)
    nums = set()
    for n in NUM_RE.findall(t):
        v = to_float(n)
        if v is not None:
            nums.add(round(v, 6))
    letters = set(LET_RE.findall(t.upper()))
    norm = re.sub(r"\s+", "", t).strip().lower()
    return nums, letters, norm


def extract_answer(text):
    """四级回退：答案：行 → #### 行 → 结尾 500 字里最后一个独立数字 → 最后一个字母选项。"""
    if not text:
        return ""
    tail = text.rstrip()
    m = ANS_RE.findall(tail)
    if m:
        return m[-1].strip()[:200]
    m = re.findall(r"####\s*([^\n]{1,80})", tail)
    if m:
        return m[-1].strip()
    last = tail[-500:]
    nums = NUM_RE.findall(last)
    letters = LET_RE.findall(last.upper())
    if nums and (not letters or len(nums) >= len(letters)):
        return nums[-1]
    if letters:
        return letters[-1]
    return tail.splitlines()[-1].strip()[:200] if tail else ""


def judge(target, got):
    tn, tl, ts = candidates(target)
    gn, gl, gs = candidates(got)
    if ts and gs and (ts == gs or ts in gs or gs in ts):
        return "exact"
    if tn and gn:
        for a in gn:
            for b in tn:
                if abs(a - b) <= 1e-6 * max(1.0, abs(b)):
                    return "numeric"
    if tl and gl and (tl == gl or (len(tl) == 1 and tl & gl)):
        return "letter"
    if not tn and not tl and not ts:
        return "unscored"
    return "wrong"


def degeneracy(text):
    # 注意：文本短于一个 24 字符窗口时所有窗口都为空，旧实现会返回 1.0，
    # 把"答案很短"误判成复读崩坏。短文本直接判 0。
    if not text or len(text) < 48:
        return 0.0
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lr = 1 - len(set(lines)) / max(1, len(lines))
    grams = [text[i:i + 24] for i in range(0, len(text) - 24, 12)]
    gr = 1 - len(set(grams)) / max(1, len(grams))
    return max(lr, gr)


def ask(base, model, prompt, max_tokens, temperature, timeout=900, thinking=None, effort=None, inject=None):
    sys_prompt = SYS_CONCISE if os.environ.get("EVAL_STYLE") == "concise" else SYS
    body = {"model": model,
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": False}
    if temperature is not None:
        body["temperature"] = temperature
    if thinking is not None:
        body["thinking"] = thinking
    if effort is not None:
        body["reasoning_effort"] = effort
    if inject is not None:
        body["chat_template_kwargs"] = {"thinking_effort_inject": inject}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    ch = d.get("choices", [{}])[0]
    return (ch.get("message", {}) or {}).get("content", "") or "", d.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="spark-x2.6-4b")
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--per-domain", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=1600)
    ap.add_argument("--temperature", type=float, default=None, help="不传则用 server 默认采样")
    ap.add_argument("--thinking", default=None, help="on|off：请求级覆盖思考开关（不传则用 server 默认）")
    ap.add_argument("--effort", default=None, help="high|low|none|minimal：请求级 reasoning_effort")
    ap.add_argument("--inject", default=None, help="on|off：是否注入官方 thinking_effort 文本（需服务端支持）")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--server-args", default="", help="仅记录到结果里，便于对照")
    args = ap.parse_args()

    items = json.load(open(args.eval))
    by_dom = defaultdict(list)
    for it in items:
        by_dom[it.get("_domain", "?")].append(it)
    picked = []
    for dom, xs in sorted(by_dom.items()):
        picked += xs[: args.per_domain]
    print(f"[{args.tag}] 评测 {len(picked)}/{len(items)} 条；" +
          ", ".join(f"{d}:{min(len(xs), args.per_domain)}" for d, xs in sorted(by_dom.items())), flush=True)

    results, stats = [], defaultdict(lambda: [0, 0, 0, 0])   # dom -> [对, 错, 未判定, 崩坏]
    t0 = time.time()
    for i, it in enumerate(picked):
        t1 = time.time()
        try:
            ans, usage = ask(args.base, args.model, it["input"], args.max_tokens, args.temperature,
                             thinking=(None if args.thinking is None else args.thinking == "on"),
                             effort=args.effort,
                             inject=(None if args.inject is None else args.inject == "on"))
        except Exception as e:
            ans, usage = f"<<ERROR {type(e).__name__}: {e}>>", {}
        deg = degeneracy(ans)
        verdict = judge(it.get("target", ""), extract_answer(ans))
        if deg > 0.6 and verdict != "unscored":
            verdict = "degenerate"
        dom = it.get("_domain", "?")
        if verdict in ("exact", "numeric", "letter"):
            stats[dom][0] += 1
        elif verdict == "unscored":
            stats[dom][2] += 1
        elif verdict == "degenerate":
            stats[dom][3] += 1
            stats[dom][1] += 1
        else:
            stats[dom][1] += 1
        results.append({"i": i, "domain": dom, "src": it.get("_src"), "verdict": verdict,
                        "degen": round(deg, 2), "target": it.get("target", ""),
                        "got_line": extract_answer(ans)[:200], "answer_full": ans[:6000],
                        "echo_effort": ("thinking_effort" in ans),
                        "sec": round(time.time() - t1, 1), "usage": usage})
        if (i + 1) % 10 == 0 or i + 1 == len(picked):
            ok = sum(v[0] for v in stats.values())
            done = sum(v[0] + v[1] for v in stats.values())
            print(f"  {i+1}/{len(picked)} 正确 {ok}/{done} 用时 {time.time()-t0:.0f}s", flush=True)
        json.dump({"tag": args.tag, "model": args.model, "server_args": args.server_args,
                   "per_domain": args.per_domain, "max_tokens": args.max_tokens,
                   "temperature": args.temperature,
                   "cfg": {"thinking": args.thinking, "effort": args.effort, "inject": args.inject},
                   "stats": {k: v for k, v in stats.items()},
                   "results": results}, open(args.out, "w"), ensure_ascii=False)

    ok = sum(v[0] for v in stats.values())
    bad = sum(v[1] for v in stats.values())
    uns = sum(v[2] for v in stats.values())
    print(f"[{args.tag}] 正确 {ok} 含错 {bad} 未判定 {uns} 用时 {time.time()-t0:.0f}s -> {args.out}")
    print("   领域           对/评  未判定 崩坏")
    for d, v in sorted(stats.items()):
        print(f"   {d:14s} {v[0]:3d}/{v[0]+v[1]:3d}  {v[2]:4d} {v[3]:4d}")


if __name__ == "__main__":
    main()
