#!/bin/bash
# 从 HuggingFace（走 hf-mirror，不需要代理）重新构建 GRQ 校准集与留出评测集。
# 输出：<WORKDIR>/calib_real.json、eval_real.json、calib_real.provenance.json
# 依赖：pip install datasets（转换机环境即可，不需要 GPU）
# 说明：数据集来源逐条写进 provenance；上游许可各异，使用前请自行核对（见 datasets/README.md）。
set -u
C=${WORKDIR:-$PWD/datasets_build}
mkdir -p "$C"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_PROGRESS_BARS=1
python - <<'PY'
import json, os, random, time, re
from datasets import load_dataset, get_dataset_config_names

C = os.environ.get("WORKDIR", os.getcwd() + "/datasets_build")
os.makedirs(C, exist_ok=True)
random.seed(20261105)
MAXIN = 900          # 单条校准输入字符上限
MAXTOK = 260000      # 校准集总字符预算（约 12 万 token）

NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def load_retry(name, cfg, split, tries=4):
    last = None
    for i in range(tries):
        try:
            return load_dataset(name, cfg, split=split)
        except Exception as e:
            last = e
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                time.sleep(8 * (i + 1))
            else:
                time.sleep(1)
    raise last

def avail_cfgs(name):
    try:
        return get_dataset_config_names(name)
    except Exception:
        return []

def pick(name, want_cfgs, want_splits, tag):
    """返回 (dataset, cfg, split)；按候选优先，再退回全部可用 config。"""
    cfgs = avail_cfgs(name)
    order = [c for c in want_cfgs if not cfgs or c in cfgs]
    order += [c for c in cfgs if c not in order]
    if not order:
        order = ["default"]
    errs = []
    for c in order[:8]:
        for sp in want_splits:
            try:
                d = load_retry(name, c, sp)
                if len(d):
                    print(f"  OK   {name} cfg={c} split={sp} rows={len(d)} cols={list(d.features.keys())[:12]}")
                    return d, c, sp
            except Exception as e:
                errs.append(f"{c}/{sp}: {type(e).__name__} {str(e)[:70]}")
    # 无 config 版本
    for sp in want_splits:
        try:
            d = load_retry(name, None, sp)
            if len(d):
                print(f"  OK   {name} (no cfg) split={sp} rows={len(d)}")
                return d, None, sp
        except Exception as e:
            errs.append(f"nocfg/{sp}: {type(e).__name__} {str(e)[:70]}")
    print(f"  FAIL {name} :: " + " | ".join(errs[:4]))
    return None, None, None

def s(v, n=4000):
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        v = v.strip()
        return v[:n] if v else None
    if isinstance(v, list):
        parts = [s(x, n // max(1, len(v))) for x in v]
        parts = [p for p in parts if p]
        return "\n".join(parts)[:n] if parts else None
    if isinstance(v, dict):
        return s(list(v.values()), n)
    return None

def first(row, keys):
    for k in keys:
        if k in row:
            v = s(row[k])
            if v:
                return v
    return None

def from_turns(row):
    """livebench / bfcl / 多轮格式：
       livebench -> turns = [str, ...]
       bfcl_v3   -> turns = [[{role, content}, ...], ...]
    """
    t = row.get("turns")
    if not isinstance(t, list) or not t:
        return None
    out = []
    def handle(x):
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            role = str(x.get("role", "")).lower()
            if role in ("system", "tool", "assistant"):
                return
            for k in ("question", "user_query", "query", "prompt", "instruction", "content", "input"):
                if k in x:
                    v = s(x[k], 2000)
                    if v:
                        out.append(v)
                    return
            if "function_call" in x:
                out.append("调用工具：" + s(x.get("function_call"), 900))
        elif isinstance(x, list):
            for y in x:
                handle(y)
    for turn in t:
        handle(turn)
    if not out:
        return None
    # bfcl 的可用工具定义也进 prompt（截短，避免整条超限）
    tools = s(row.get("tools"), 420)
    body = "\n\n".join(out)[:1100]
    if tools and row.get("tools") is not None:
        body = "可用工具：\n" + tools + "\n任务：\n" + body
    return body

QKEYS = ["question", "problem", "prompt", "Question", "query", "input", "text", "instruction", "task_prompt", "question_text"]
AKEYS = ["answer", "Answer", "final_answer", "ground_truth", "solution", "golden", "target", "expected_answer", "completion"]

def make_item(row, extra_prefix_keys=(), qkeys=None):
    q = first(row, qkeys or QKEYS) or from_turns(row)
    if not q:
        return None
    pre = []
    for k in extra_prefix_keys:
        v = first(row, [k])
        if v:
            pre.append(v)
    if pre:
        q = "\n".join(pre) + "\n" + q
    a = first(row, AKEYS)
    return {"input": q.strip(), "target": (a or "").strip()[:600]}

def harvest(name, cfgs, splits, want, domain, maxin=None, prefix=(), qkeys=None):
    maxin = maxin or MAXIN
    d, c, sp = pick(name, cfgs, splits, domain)
    if d is None:
        return []
    items = []
    seen = set()
    for i, row in enumerate(d):
        it = make_item(row, prefix, qkeys)
        if not it:
            continue
        q = it["input"]
        if len(q) < 25 or len(q) > maxin:
            continue
        k = None
        for ik in ("id", "question_id", "problem_idx", "key", "idx", "Unique_id"):
            if ik in row:
                k = f"{name}|{c}|{ik}={row[ik]}"
                break
        k = k or f"{name}|{c}|{i}|{hash(q)}"
        if k in seen:
            continue
        seen.add(k)
        it["_src"] = f"{name}:{c}:{sp}"
        it["_domain"] = domain
        it["_key"] = k
        items.append(it)
    random.shuffle(items)
    print(f"       {name} 合格 {len(items)} 条，取 {min(want, len(items))}")
    return items[:want]

calib, evl = [], []

# ---------- 校准（数学EN / 数学ZH / 推理 / 代码 / 工具调用 / 指令跟随） ----------
calib += harvest("madrylab/gsm8k-platinum", ["main"], ["test"], 150, "math_en")
calib += harvest("weitianwen/cmath", ["default"], ["test", "train"], 130, "math_zh")
calib += harvest("Hothan/OlympiadBench",
                 ["OE_MM_maths_zh_CEE", "OE_MM_maths_zh_COMP", "OE_TO_maths_zh_CEE", "OE_MM_maths_en_COMP"],
                 ["train"], 110, "math_bil")
calib += harvest("simplescaling/s1K-1.1", ["default"], ["train"], 100, "reason")
calib += harvest("TIGER-Lab/TheoremQA", ["default"], ["test"], 90, "reason")
calib += harvest("bigcode/bigcodebench-hard", ["default"], ["v0.1.4", "v0.1.3", "v0.1.2", "v0.1.1", "v0.1.0_hf"],
                 120, "code", maxin=1500,
                 qkeys=("complete_prompt", "code_prompt", "instruct_prompt", "question", "doc_struct"))
calib += harvest("llamastack/bfcl_v3", ["default"], ["train"], 120, "tool", maxin=1500)
calib += harvest("google/IFEval", ["default"], ["train"], 80, "instr")

# ---------- 留出评测（绝不与校准重叠：按 _key 去重） ----------
evl += harvest("MathArena/aime_2025", ["default"], ["train", "test"], 30, "math_hard", maxin=2000)
evl += harvest("livebench/math", ["default"], ["test"], 90, "math_live", maxin=2500)
evl += harvest("livebench/reasoning", ["default"], ["test"], 60, "reason_live", maxin=2500)
evl += harvest("livebench/logic", ["default"], ["test"], 60, "logic_live", maxin=2500)
evl += harvest("livebench/instruction_following", ["default"], ["test"], 60, "instr_live", maxin=2500)
evl += harvest("RUCAIBox/gaokao-bench",
               ["2010-2022_Math_I_MCQs", "2010-2022_Math_II_MCQs", "2010-2022_Chinese_Lang_and_Usage_MCQs"],
               ["test"], 100, "zh_exam", maxin=2000)
evl += harvest("TIGER-Lab/TheoremQA", ["default"], ["test"], 60, "reason", maxin=2000)

used = {it.get("_key") for it in calib}
evl2 = []
for it in evl:
    if it["_key"] in used:
        continue
    used.add(it["_key"])
    evl2.append(it)
evl = evl2

tot = 0
calib2 = []
for it in calib:
    tot += len(it["input"])
    if tot > MAXTOK:
        break
    calib2.append(it)
calib = calib2

for it in calib + evl:
    it.pop("_key", None)

def dom(xs):
    d = {}
    for x in xs:
        d[x["_domain"]] = d.get(x["_domain"], 0) + 1
    return d

json.dump(calib, open(f"{C}/calib_real.json", "w"), ensure_ascii=False, indent=1)
json.dump(evl, open(f"{C}/eval_real.json", "w"), ensure_ascii=False, indent=1)
prov = [{"_src": it["_src"], "_domain": it["_domain"]} for it in calib + evl]
json.dump(prov, open(f"{C}/calib_real.provenance.json", "w"), ensure_ascii=False, indent=1)

print("== 校准集 ==", len(calib), "条；分布", dom(calib))
print("== 评测集 ==", len(evl), "条；分布", dom(evl))
print("校准 input 平均字符", sum(len(x["input"]) for x in calib) // max(1, len(calib)),
      "总计", round(sum(len(x["input"]) for x in calib) / 1e6, 2), "M 字符")
print("样例1:", json.dumps(calib[0], ensure_ascii=False)[:300])
print("样例2:", json.dumps(calib[len(calib)//2], ensure_ascii=False)[:300])
print("样例3:", json.dumps(evl[0], ensure_ascii=False)[:300])
print("CALIB_BUILD_DONE")
PY
ls -l "$C/calib_real.json" "$C/eval_real.json"
