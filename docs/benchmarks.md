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

`tools/eval_llm.py`（thinking on、贪心 `temperature 0`、`max_tokens 1600`）。

⚠️ 这一轮跑在**修复 system 送达路径之前**（`--system-in-prompt`，见 §6.2）：当时客户端的 system 内容不进模型。
同题集的 60 题抽样在修复后的默认配置下是 12/50 = 24 %（预算 1200），两个数字不能直接比较。


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
  修复 system 送达路径之后那一轮是 runaway **4/48**（`字母表` 与 `循环填空` 两组 × temp 0/0.8），四条的 prompt
  本身就是"把字母表 / 填空循环重复下去"（模型在服从指令），且都在几十 token 内自然 `stop`、熔断已介入（veto 1、7）。
  修复前那一轮是 2/48，同样是这两类 case。趋势：12/48（无 n-gram 熔断）→ 8/48（加 token 级 n-gram）
  → 2–4/48（剩下的全是"prompt 明确要求重复"，不该被熔断打断）。
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

## 6. 推理配置对照

### 6.1 同一道"只输出代码"的题（`tools/fib_probe.py`）

题目："用 Python 写一个函数 `fib(n)`，返回第 n 个斐波那契数，**只输出代码**。"
下表是 `--system-in-prompt on`（system 真正生效）之后的 `fib_probe3`：

| 配置 | 结果 | completion | 正式回答里有无 `def fib` |
|---|---|---|---|
| thinking on + 注入 on + `temperature 0`（生产默认） | ❌ MISS | 88（12.1 s） | 无——回答通道里写的是"用户要求…首先斐波那契数列的定义…需要考虑 n 的范围…"，写完就 EOS，代码没出现 |
| 同上，预算不限制 / 2500 | ❌ MISS | 88 | 与上一行**逐字节相同** → 不是预算问题，是收尾行为 |
| `reasoning_effort:"none"` | ✅ HIT | 75（1.8 s） | `def fib(n): …` 完整正确 |
| `chat_template_kwargs.enable_thinking:false` | ✅ HIT | 75 | 同上 |
| 厂商采样（temp 1.0）+ thinking on | ❌ MISS | 88 | 同第一行 |

→ 命中 **2/6**。`thinking on` 对"只输出代码"这类短任务是负资产，对数学/推理题是正资产（§6.2）。

⚠️ 修复 system 送达路径之前这张表的数字是 155/74，而且 MISS 的回答里带一句
"并且系统提示 `thinking_effort=high`，需要在思考通道中体现高程度的思考"。**那句复述不是注入文本被抄出来的**，
而是模型当时根本看不到预期的 system 块，把训练分布里的话凭空补写了出来。修复后这句消失、回答变短（155→88），
但结论方向不变。

### 6.2 system 块到底进不进模型（`--system-in-prompt`）

同一批 60 题（每领域 10 题）、`temperature 0`、`max_tokens 1200`、同一判分函数、同一个服务实例（`tools/eval_llm.py`）：

| 配置 | 正确率 | ct avg | ct 合计 | pt avg | 撞 1200 上限且判错 | 回答含"答案：" | 复读崩坏 |
|---|---|---|---|---|---|---|---|
| `--system-in-prompt off`（旧：客户端 system 不进模型） | **16/50 = 32.0 %** | 596 | 29814 | 237 | 13 | 2/50 | 0 |
| 同上 + `thinking off` | 8/50 = 16.0 % | 609 | 30451 | 237 | 13 | 2/50 | 0 |
| `--system-in-prompt on` + 注入 off | 10/50 = 20.0 % | 680 | 33982 | 293 | 18 | 8/50 | 0 |
| `--system-in-prompt on` + 注入 on（现默认） | 12/50 = 24.0 % | 833 | 41648 | 344 | 18 | 13/50 | 0 |

（`off` 那一行与更早两轮 `cfgab_low` / `cfgab_noinj` **逐字节相同**（连 ct 合计 29814 都一样），当同一配置看。
那两轮曾被误读成"`reasoning_effort=low` 更准"——其实当时"改 system 文本"的开关全是空操作。）

读数：

1. **注入句不是元凶**：`on` 模式下把官方那句 `thinking_effort` 文本关掉，正确率从 24 % 降到 20 %，没有回升。
   `prompt_tokens` 平均差正好 51（344 vs 293），与 token 差分探针一致（`prompt_tokens` 167 → 116）。
2. 真正变化的是**客户端 system 第一次进了模型**：模型开始照评测 system 的要求写 `答案：…`（2/50 → 13/50），
   平均输出多 40 %，于是撞 1200 上限的判错题从 13 增到 18。
3. 预算复测（`tools/budget_retest.py --cap-at 1190 --max-tokens 2600`）：18 条撞 1200 上限的判错题放宽到 2600 token 重问：**救回 5 条**、13 条仍错、5 条又撞上新上限（793 s）。把这 5 条计回去，臂 A 相当于 17/50 = 34 %，**不低于**旧配置在同一题集同一预算下的 16/50 = 32 % → 那 8 个百分点来自预算与格式指令，不是模型变差。
4. `thinking on/off` 仍是最大的单一杠杆：同一送达方式下 32 %（on） vs 16 %（off）。
5. **结论：必须让 system 生效**。否则 OpenAI 客户端的 system / 工具指令静默丢失，Agent 场景不可接受。
   留出集上的下降来自评测协议（固定预算 + 格式指令），不是模型变笨；生产配置 `MAX_NEW=0` 不设生成上限，
   这个惩罚在真实使用里不存在。

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
