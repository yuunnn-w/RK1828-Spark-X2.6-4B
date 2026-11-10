# 测试与验证工具

所有脚本在 `tools/`，都是纯标准库 Python 3（板端 python3 直接能跑）+ shell，
默认打 `http://127.0.0.1:8080`，用 `--base http://<host>:8080` 可从任意机器跑。

| 脚本 | 查什么 | 关键参数 |
|---|---|---|
| `run_stress_suite.sh` | **一键全套**：先跑复读专项，再跑十一阶段压力，前后各拍一次卡状态并抓 journal 错误 | 无（内部固定）；产物 `stress_report.json`、`repeat_probe.json` |
| `stress_server.py` | 协议与鲁棒性：顺序正确性、并发、流式、客户端硬断开、长 prompt 阶梯、工具、`response_format`、浸泡、OpenAI 语义检查（HEAD、`/v1/models/{id}`、未知 id 404、`n>1`→400、reasoning 分离、`role:"tool"`、`/v1/completions`、logprobs、`/stats` 队列水位） | `--threads 24 --soak 120` |
| `repeat_probe.py` | 24 组**对抗性** prompt（要求重复句子/数数/循环填空/代码复读/诗歌复读…）× thinking on/off × temp 0/0.8 = 48 组，判 `flagged` / `runaway` | `--out` |
| `longctx_probe.py` | 长上下文阶梯：逐级加大 prompt 并埋 needle 检查回忆，每级后复查卡健康 | `--steps 40000,56000` |
| `eval_llm.py` | 留出集评测（正确率 / 复读指标 / 用时 / usage），可请求级覆盖 `--thinking`、`--effort`、`--inject`、`--temperature` | `--eval datasets/eval_real.json --per-domain 10 --max-tokens 1200` |
| `eval_summary.py` | 把上面那份 JSON 汇总成表（含复读率最高的 5 条） | `<json>` |
| `budget_retest.py` | 把"撞生成上限且判错"的样本放宽预算复测，判断正确率是被预算压低还是真的不会 | `--sample-per-domain 10 --max-tokens 2600 --cap-at 1190`（`--cap-at` 要取评测那轮 `max_tokens-10`，否则挑不到样本） |
| `loop_replay.py` | 从评测结果里挑出疑似复读/被截断的样本重放，验证熔断是否生效 | `--limit 20` |
| `tools_probe.py` | 工具调用链路（官方 function-call 示例那套工具与 prompt），检查 `finish_reason=tool_calls`、`arguments` 可 JSON 解析、不给 tools 时不伪造 | — |
| `defaults_probe.py` | 服务端默认采样是否等于模型厂商推荐；`max_tokens` 不限制/截断/`-1` 语义 | — |
| `sysprobe.py` | **system 消息有没有真的进模型**：往 system/developer 轮埋暗号、只向 user 轮提问（判 `HIT=n/6`，含一条 user 轮对照）。修复 `--system-in-prompt` 前后分别是 6/6 与 1/6 | `--base http://<host>:8080` |
| `feat_probe.py` | 新增 OpenAI 特性（`logit_bias`、`top_logprobs`、`reasoning_effort`、429 + `Retry-After`） | — |
| `perf_probe.py` / `timing_probe.py` / `mt_sweep.py` | prefill/decode 吞吐与"是否存在每请求固定开销"的定量判定 | — |
| `fib_probe.py` | 同一道"只输出代码"的题在 6 种推理配置下的对照（结论见 `docs/benchmarks.md`） | — |
| `kv_sweep.sh` | KV cache 运行时策略扫描（默认 / RECURRENT / NORMAL / `n_keep`） | 会临时改 env 并在结束时恢复 |
| `temp_watch.sh` | 每 60 s 记录 SoC 热区 + 卡温度/利用率/显存，用于峰值统计 | `<样本数>` |
| `gguf_meta.py` | 从 `.tokenizer.gguf` 里导出全部元数据（含 `tokenizer.chat_template` 逐字） | `<gguf>` |

## 结果解读注意

- **判成败看汇总，不要看脚本退出码**：早期版本的 `run_stress_suite.sh` 用 `cmd | tail` 取 `$?`，
  永远得到 0。现在先把输出落盘再取退出码；仍建议以日志里的 `失败 N / 共 M` 与 `stress_report.json` 为准。
- **复读指标 `degeneracy()`**：取"行重复率"与"24 字符滑窗重复率"的最大值。
  注意文本短于一个窗口时窗口集合为空 —— 旧实现会返回 1.0 把"答案很短"误判成复读崩坏（本仓库已修：短于 48 字符直接判 0）。
  如果你自己改这个指标，别重犯。
- **评测公平性**：对比两个配置时必须用**同一批题**（`eval_llm.py` 按 `_domain` 排序取前 N 条，`--per-domain` 相同即同题集）、
  同一 `--max-tokens`、同一 `--temperature`。
- **长上下文测试最后跑**：单级 prefill 很大，历史上有把卡打挂的先例（`docs/troubleshooting.md` 第 3 条）。
