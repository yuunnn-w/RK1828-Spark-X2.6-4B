#!/usr/bin/env python3
"""GRQ 校准质量测量：同一批 held-out 题目上，量化前(FP) vs 量化后(GRQ fake-quant) 的
   答案困惑度(PPL)、平均 NLL、以及逐 token top-1 一致率。按领域分解。
   不生成文本，只做 teacher-forced 前向，所以与采样无关。"""
import json, math, os, sys, copy
import torch
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, os.environ.get("SPARK_PY_DIR", os.getcwd() + "/.model-zoo/examples/Spark_X2_5/python"))
from spark2_5.modeling_spark import Spark2_5ForCausalLM

MP     = os.environ.get("MODEL_DIR",  os.getcwd() + "/model/Spark-X2.6-4B")
EVAL   = os.environ.get("EVAL_SET",   os.getcwd() + "/datasets/eval_real.json")
CALIB  = os.environ.get("CALIB_SET",  os.getcwd() + "/datasets/calib_real.json")
OUT    = os.environ.get("OUT",        os.getcwd() + "/results/quant_eval_grq.json")
MAXLEN = int(os.environ.get("QE_MAXLEN", "1024"))
NDATASET = os.environ.get("QE_DATASET", EVAL)
CALI   = os.environ.get("QE_CALI", CALIB)
DTYPE  = torch.bfloat16

tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(MP, trust_remote_code=True)
print("[cfg] use_cache ->", end=" ")
if getattr(cfg, "use_cache", None) is not None:
    cfg.use_cache = False
print(cfg.use_cache)

items = json.load(open(NDATASET, encoding="utf-8"))
print(f"[data] {NDATASET} 条数={len(items)}")

def enc(model, it):
    msgs_u = [{"role": "user", "content": it["input"]}]
    msgs_a = msgs_u + [{"role": "assistant", "content": str(it["target"])}]
    pre = tok.apply_chat_template(msgs_u, add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(msgs_a, add_generation_prompt=False, tokenize=False)
    e1 = tok(pre, add_special_tokens=False)["input_ids"]
    e2 = tok(full, add_special_tokens=False)["input_ids"]
    if len(e2) > MAXLEN: e2 = e2[:MAXLEN]
    n_prompt = min(len(e1), len(e2) - 1)
    return e2, n_prompt

@torch.no_grad()
def measure(model, tag, ref_top=None, store_top=None):
    model.eval()
    tot_nll, tot_n, agree, agree_tot = 0.0, 0, 0, 0
    per = {}
    for i, it in enumerate(items):
        ids, np_ = enc(model, it)
        if len(ids) - np_ < 1:
            continue
        x = torch.tensor([ids], device="cuda")
        y = model(x).logits.float()
        m = mask_of(x, np_)
        sel_pred = y[:, :-1, :][m]
        sel_true = x[:, 1:][m]
        if sel_true.numel() == 0:
            continue
        topk = sel_pred.argmax(-1)
        if store_top is not None: store_top[i] = topk.cpu()
        if ref_top is not None and i in ref_top:
            agree += int((topk == ref_top[i].to(topk.device)).sum().item())
        agree_tot += sel_true.numel()
        nll = torch.nn.functional.cross_entropy(sel_pred, sel_true, reduction="sum").item()
        tot_nll += nll; tot_n += sel_true.numel()
        dom = it.get("_domain", "?")
        a, b = per.get(dom, (0.0, 0))
        per[dom] = (a + nll, b + sel_true.numel())
        if (i + 1) % 50 == 0:
            print(f"  [{tag}] {i+1}/{len(items)} ppl={math.exp(tot_nll/max(tot_n,1)):.3f}", flush=True)
    ppl = math.exp(tot_nll / max(tot_n, 1))
    if agree_tot and ref_top is not None:
        print(f"[{tag}] 与 FP 的 top-1 一致率 = {100*agree/agree_tot:.2f}% (token {agree_tot})")
    print(f"[{tag}] 平均 NLL={tot_nll/max(tot_n,1):.4f}  PPL={ppl:.4f}  评测 token={tot_n}")
    for d, (a, b) in sorted(per.items()):
        print(f"   {d:12s} nll={a/b:.4f} ppl={math.exp(a/b):.4f} tokens={b}")
    return tot_nll / max(tot_n, 1), ppl, {d: v[0]/v[1] for d, v in per.items()}

def mask_of(x, np_):
    m = torch.ones_like(x[:, 1:], dtype=torch.bool)
    m[:, :np_] = False
    return m

model = Spark2_5ForCausalLM.from_pretrained(MP, config=cfg, trust_remote_code=True,
                                           attn_implementation="eager", dtype=DTYPE).to("cuda")

fp_top = {}
nll_fp, ppl_fp, per_fp = measure(model, "fp", store_top=fp_top)

print("[grq] 开始 GRQ 量化 ...", flush=True)
from rknn.quantization.api import RKQuantizer
qt = RKQuantizer(verbose=True)
r = qt.load_model(model=model, tokenizer=tok, device="cuda",
                  system_prompt=None, tools=None, system_role="system", user_role="user")
print("[grq] load_model rc=", r)
model = qt.quantize(quantized_dtype="w4a16", quantized_method="group32",
                    quantized_algorithm="grq", dataset=CALI)
model = model.to("cuda")          # GRQ 会把部分模块留在 cpu，测前先挪回 cuda
torch.cuda.empty_cache()
nll_q, ppl_q, per_q = measure(model, "grq", ref_top=fp_top)

print(f"\n===== 结论 =====")
print(f"FP  PPL={ppl_fp:.4f}   GRQ PPL={ppl_q:.4f}   增量 PPL={ppl_q-ppl_fp:+.4f}  相对升幅={100*(ppl_q/ppl_fp-1):.2f}%")
print(f"FP  NLL={nll_fp:.4f}   GRQ NLL={nll_q:.4f}   ΔNLL={nll_q-nll_fp:+.4f}")
res = {"fp_nll": nll_fp, "grq_nll": nll_q, "fp_ppl": ppl_fp, "grq_ppl": ppl_q,
       "per_fp": per_fp, "per_grq": per_q, "dataset": NDATASET, "calib": CALI,
       "n_items": len(items), "maxlen": MAXLEN}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", OUT)
