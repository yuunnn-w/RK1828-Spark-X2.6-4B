# 实测成绩与测法

统一环境：OrangePi5Max（RK3588，aarch64 Linux）+ RK1828 M.2 卡；
固件 1.2.0 / rknn3 API 1.2.0 / PCIe 驱动 3.3.1 / `rknn-smi 1.3.0`；
模型 `Spark-X2.6-4B-73728-kvi4`（GRQ w4a16 group32 权重，KV `Int4_to_F16/GroupQuant/group16/resid64`）；
8 核 `0xff`；服务端默认采样 = 模型厂商 `generation_config.json`。
所有脚本在 `tools/`，可复现。

---

## 1. 量化质量（转换机侧，GPU）

`convert/quant_eval_grq.py`：同一批留出题（392 条，不参与校准），teacher-forced 只算答案 span 的 NLL/PPL，
并比较逐 token top-1。

| 模型 | PPL | NLL |
|---|---|---|
| bf16 原始 | 1.2536 | 0.2260 |
| GRQ w4a16 + group32（884 条校准） | **1.2451** | 0.2192 |

与 bf16 的逐 token top-1 一致率 **96.16 %**（2393 个答案 token）。
结论：**权重 int4 不是精度瓶颈**；量化侧可动的旋钮（`residual_depth`、head 位宽）收益空间有限。

## 2. 板端精度（392 题留出集）

`tools/eval_llm.py`（thinking on、贪心 `temperature 0`、`max_tokens 1600`）：

| 领域 | 对/评 | 未判定 |
|---|---|---|
| zh_exam | 35/100 | 0 |
| math_live | 32/90 | 0 |
| reason | 14/52 | 0 |
| reason_live | 13/60 | 0 |
| math_hard（AIME 级） | 5/30 | 0 |
| instr_live | 0/60 | 60（无标准答案，不参与判分） |
| **合计** | **99/332 = 29.8 %** | 392 条中 332 可判分 |

- **真实复读 0 条**（用修正后的 `degeneracy()` 对全部已存答案重算；早期记录的"3 条崩坏"是指标 bug：
  短于 24 字符的文本被误判成 1.0）。
- 熔断实际触发 14/392 请求。
- 生成预算不是主因：把撞上限判错的 20 条放宽到 4000 token 复测（`tools/budget_retest.py`），
  **只救回 2 条**，18 条仍错 → 29.8 % 基本为真。

### 采样默认的影响（同 30 题、`--per-domain 5`）

| 采样 | 可判分正确率 |
|---|---|
| 厂商默认（temp 1.0 / top_p 0.95 / top_k 不截断） | 5/25 = 20 % |
| 贪心（客户端传 `temperature 0`） | **7/25 = 28 %** |

→ 服务端默认保持厂商值（用户可预测），但**精度关键的调用应显式传 `temperature 0`**。

## 3. 性能

`tools/perf_probe.py`（同一模型，短/中/长三档 prompt）：

| 档 | prompt tokens | prefill | decode |
|---|---|---|---|
| 短 | 88 | 137 ms（642–647 tok/s） | 375 ms（**50.7–52.3 tok/s**） |
| 中 | 4291 | 5443 ms（788 tok/s） | 635–639 ms（48.5–48.8 tok/s） |
| 长 | 14791 | 21609–21624 ms（684–685 tok/s） | 728–738 ms（42.0–42.6 tok/s） |

超长上下文（`tools/longctx_probe.py`）：38,006 token prefill 72.0 s（≈460 tok/s）、
53,166 token prefill 115.6 s；decode 分别降到 35 / 32 tok/s。

两个被否定的假设：

- **"每个请求有约 10 s 固定开销"** —— `timing_probe.py` / `mt_sweep.py` 证明 `decode_ms` 只随实际生成
  token 数走，与 `max_tokens` 无关（mt=16/64/256/1024/4096 时 decode 284–382 ms）。
- **"换 KV cache 运行时策略能提速"** —— 默认 / `--kvcache-policy 0`(RECURRENT) / `1`(NORMAL) 三档差异在噪声内；
  `--kvcache-policy 0 --kv-n-keep 2048` 直接起不来（HEALTH_FAIL）。

## 4. 复读与熔断

- **对抗性专项**（`tools/repeat_probe.py`，24 组 prompt × thinking on/off × temp 0/0.8 = 48 组）：
  runaway **2/48**，且这 2 条是 prompt 本身要求"重复这段话"（服从指令），并在 46 token 处自然 stop、
  熔断已介入（`veto=7`）。趋势：12/48（无 n-gram 熔断）→ 8/48（加 token 级 n-gram）→ 2/48（加官方模板注入后调优）。
- **重放专项**（`tools/loop_replay.py`，从评测里挑疑似复读/被截断的样本重放）：
  25 条 → 熔断触发 4、**全部在撞上限之前自然收尾**、最大 degen 0.47。
- 熔断三层（默认开启）：token 级周期检测（周期 ≤400 的三连块）、token 级 8-gram 计数、
  文本级 120 字节尾部长块出现 ≥3 次。旋钮：`--veto-period-max`、`--ngram-len`、`--ngram-ban-at`、
  `--loop-max-occurrences`、`--veto-limit`、总开关 `--fuse-off`。

## 5. 压力与 OpenAI 兼容性

`tools/run_stress_suite.sh` = 复读专项 + `tools/stress_server.py` 十一阶段（31 项检查）。
最佳记录：**失败 1 / 31**。通过项包括：并发 24 全成功（wall 18.2 s）、流式 SSE 合法、客户端硬断开后服务仍可用、
`response_format=json_object` 可解析、120 条浸泡零客户端错误 / 零复读 / 零服务端错误、
HEAD `/health`、`/v1/models/{id}`、未知 id 404、`n>1` → 400、reasoning 分离、`role:"tool"` 续写、
`/v1/completions`、`logprobs`/`top_logprobs`、`/stats` 队列水位与 429。

唯一失败项 **P3（fib 题）**：见下一节，性质是模型行为。

## 6. `thinking` 配置对照（`tools/fib_probe.py`）

题目："用 Python 写一个函数 `fib(n)` 返回第 n 个斐波那契数，只输出代码。"

| 配置 | 正式回答里有无 `def fib` | completion |
|---|---|---|
| thinking on + 官方注入 + `temperature 0`（生产默认） | ❌ MISS | 155（复述任务后提前 EOS，没写代码） |
| 同上，`max_tokens 2500` / 不限制 | ❌ MISS（同上，说明不是预算不够） | 155 |
| `reasoning_effort:"none"` | ✅ HIT | 74（1.8 s，代码正确） |
| `chat_template_kwargs.enable_thinking:false` | ✅ HIT | 74 |
| 厂商采样（temp 1.0）+ thinking on | ❌ MISS | 155 |

→ 服务端因此提供 `--thinking-effort-inject on|off` 与请求级 `chat_template_kwargs.thinking_effort_inject`。

（待补：`cfgab_*` 那轮 60 题 A/B —— `effort low` / 注入关闭 / 思考关闭 对整体正确率的影响。）

## 7. KV 精度 A/B（fp16 KV vs int4 KV）

同一批 60 题（每领域 10 条，同一顺序）、同一生成预算 `max_tokens 1000`、同一采样（贪心 `temperature 0`）、
同一服务端与同一权重，只换 KV 档位：

| 档位 | ctx | 卡内存 | 可判分正确率 | 复读崩坏 | 单题用时 |
|---|---|---|---|---|---|
| `Int4_to_F16 / GroupQuant / g16 / r64`（生产） | 73728 | 4903/5120 MB | **14/50 = 28.0 %** | 0 | avg 15.2 s |
| `Float16 / Normal` | 8192 | 3191/5120 MB | 13/50 = 26.0 % | 0 | avg 16.2 s |

**结论：把 KV 从 int4 提到 fp16 换不回精度**（int4 略好 1 题，差 2 个百分点在 50 题样本上是噪声级别）。
分领域看也没有系统性方向：int4 在 zh_exam(5 vs 4)、reason(4 vs 3)、math_hard(1 vs 0) 略好，
fp16 在 math_live(3 vs 2)、reason_live(3 vs 2) 略好。

结合 §1（权重 int4 近乎无损）与 `docs/conversion.md` 的内存表（int8 KV 在 65536 token 必然超卡内存上限），
得到本项目的定论：**要 64K 以上上下文，int4 KV + resid64 就是正确档位，精度上没有可换回的东西**。
唯一还没试过的精度旋钮是 `residual_depth 128/256`（更准也更占内存）与 `llm_head` 显式指定张量名拿 w6a16 ——
前者在 73728 档只剩 198 MB 余量、后者尚未找到被 toolkit 接受的张量名。

## 8. 卡健康与温度

- 连续评测 + 压测期间的温度采样（`tools/temp_watch.sh`，60 s 一次）：卡 **29–56 °C**，
  SoC 七个热区 28–33 °C；trip point 85 °C（软）/ 115 °C（硬），`dmesg` / `journalctl` 无热节流事件。
- 每轮测试前后对比：`Online / OK`、显存 4903/5120 MB、`LnkSta: 5GT/s x1` 不变；
  `lspci` 侧 `CorrErr- FatalErr-`，卡侧 `pcie_err` 计数为 0。
