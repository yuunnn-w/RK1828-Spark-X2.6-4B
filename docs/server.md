# rknn_spark_server — RK1828 板端 OpenAI 兼容推理服务

自研 C++ 服务（GPL-3.0），把 Rockchip RK1828 M.2 算力卡上的 **Spark-X2.6-4B**（RKNN3 V1.2.0 转换产物）
以 OpenAI 兼容 HTTP API 的形式常驻对外提供。

存在的理由：官方 `rkllm3-server` 1.2.0 对 `model_type=spark2_5` 的输出链路是坏的
—— 任何 prompt / chat template / `--reasoning on|off` / 采样组合下，输出从第一个 token 就开始崩坏
（`We need answer answer answer…`、整片换行），而 model-zoo 的 Spark C++ demo 路径（`rknn3_session_set_chat_template`
+ `rknn3_session_run`）用**同一份模型文件**完全正常。本服务就是把这条已验证可用的路径常驻化 + HTTP 化，
并补上生产所需的协议面与守护。证据与复现步骤见 `docs/troubleshooting.md`。

- 源码：`server/cpp/{engine.h,engine.cpp,httpd.h,httpd.cpp,main.cpp}`（engine = 推理引擎，httpd = HTTP/1.1 + SSE，
  main = OpenAI 协议、chat template 组装、运维端点）
- 编译：`scripts/build_server.sh`（产物 `server/build/rknn_spark_server`）
- 部署：`deploy/install_service.sh`（装成 systemd 服务 + 健康守护，默认前缀 `/opt/rk1828-spark`）

## 1. 编译

```bash
scripts/get_sdk.sh          # 取 RKNN3 SDK 头文件与运行时库（不在本仓库里，见 docs/build.md）
scripts/build_server.sh     # cmake + make，产物 server/build/rknn_spark_server
```

要点：

- 需要 **1.2.0 版**的 `rknn3_api.h`：model-zoo 自带的是 1.1.x，`rknn3_session_clear_kvcache` 等签名不同，
  直接编译会报错。所以 CMake 优先从 `RKNN3_SDK_ROOT` 取头文件。
- `libtokenizer.a`（model-zoo 预编译，Apache-2.0）已随仓库提供；也可自行从 model-zoo 的 `tokenizer/` 源码重建。
- `thirdparty/utils` 只保留 `fileutils / cnpy / timeutils` 三个小库（本服务不需要图像/音频部分）。

## 2. 启动

直接用部署脚本装好的服务（推荐，见第 5 节）。手工跑法：

```bash
env TMPDIR=/run/rknn3 /opt/rk1828-spark/bin/rknn_spark_server \
    /opt/rk1828-spark/model/Spark-X2.6-4B-73728-kvi4.rknn \
    /opt/rk1828-spark/model/Spark-X2.6-4B-73728-kvi4.weight \
    0xff \
    /opt/rk1828-spark/model/Spark-X2.6-4B.tokenizer.gguf \
    /opt/rk1828-spark/model/Spark-X2.6-4B.embed.bin \
    73728 \
    /opt/rk1828-spark/model/Spark-X2.6-4B-73728-kvi4.safetensors \
    --host 0.0.0.0 --port 8080 --model-alias Spark-X2.6-4B \
    --thinking on --temp 1.0 --top-k 0 --top-p 0.95 \
    --repeat-penalty 1.0 --frequency-penalty 0.0 --presence-penalty 0.0 \
    --max-new-tokens 0 --max-new-tokens-cap 0 \
    --request-timeout-sec 1200 --max-connections 128 --max-queue 64 \
    --reasoning-format deepseek
```

位置参数顺序固定：`<llm.rknn> <llm.weight> <core_mask(十六进制)> <tokenizer.gguf> <embed.bin>
<max_context_len> <rope.safetensors>`。模型加载约 45 s，期间 `/health` 返回 503。
`TMPDIR` 必须可写（运行时在这里放临时文件），systemd 单元已设好。

也可用 JSON 配置（与官方 rkllm3-server 风格一致）：`--params-file params.json`。
可选鉴权：`--api-key K` → 客户端必须带 `Authorization: Bearer K`。
完整参数列表：`rknn_spark_server --help`。

## 3. 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/chat/completions` | 主端点。`messages`（string / 多模态数组均可，`role` 支持 `system`/`user`/`assistant`/`tool`，`assistant.tool_calls` 会回填进上下文，`role:"tool"` 渲染成 Spark 的 `tool_call_result` 块）、`stream`、`stream_options.include_usage`、`max_tokens`/`max_completion_tokens`/`n_predict`、`temperature`、`top_p`、`top_k`、`frequency_penalty`、`presence_penalty`、`repeat_penalty`(或 `extra_body.repeat_penalty`)、`stop`、`seed`、`logprobs`、`tools`+`tool_choice`+`parallel_tool_calls`、`response_format`(`text`/`json_object`/`json_schema`)、`chat_template_kwargs.thinking`/`enable_thinking`、`chat_template_kwargs.thinking_effort_inject`(bool，关掉官方 `thinking_effort` 文本注入)、`reasoning_format`(`deepseek`)、`reasoning_effort`(`none`/`minimal` 关闭思考，`high`/`low` 写入 system 文本)、`logit_bias`（token id → −100…+5，≤−100 即屏蔽）、`top_logprobs`（1..8）；`n>1` 明确返回 400 |
| `POST` | `/v1/completions` | 传统补全，`prompt` 原样送入（不做 chat 包装），用于评测与低层实验 |
| `GET` | `/v1/models`, `/v1/models/{id}` | 模型列表，`meta` 内含 `model_type`、`max_context_len`、`vocab_size`、量化方式 |
| `GET`/`HEAD` | `/health`, `/healthz` | 就绪探针（模型未加载完 503） |
| `GET` | `/metrics` | Prometheus 文本；`/stats` 为同一数据的 JSON |
| `GET` | `/v1/slots` | 单 slot 状态（idle/running） |
| `POST` | `/slots/{id}/pause\|resume\|interrupt` | 对应 `rknn3_session_pause/resume/stop` |
| `POST` | `/slots/{id}/kvcache/save\|load`、`/v1/kvcache/save\|load` | 对应 `rknn3_session_save_kvcache` / `load_kvcache_from_path`，默认目录 `/run/rknn3/kvcache` |
| `OPTIONS` | 任意 | CORS 预检 |

响应为 OpenAI 形状：`id/object/created/model/choices[]/usage`，
`finish_reason ∈ stop|length|tool_calls|error`；流式为 SSE（`data: {...}` + `data: [DONE]`），
首块带 `delta.role="assistant"`，尾块带 `finish_reason`，`include_usage` 时最后一块带 `usage`。
`finish_reason` 的语义：`stop` = 采样器自己选中了 EOS / 命中 `stop` 串 / 熔断收尾；`length` = 生成被
`max_tokens` 截断；`tool_calls` = 解析出工具调用；`error` = 推理失败。
（runtime 的 `n_decode_tokens` 与请求上限存在 off-by-one —— 实测 `max_new_tokens=64` 时它只报 63，
所以判 `length` 用"采样器是否选了 EOS"而不是 token 数比较；用 `--sampler runtime` 时没有采样回调，退回按 token 数判断。）
`usage` 额外给出 `prefill_ms/decode_ms/e2e_ms/tokens_per_second`，触发过复读熔断时给
`degeneration_veto` / `degeneration_loop` / `degeneration_guard`。
`reasoning_format=deepseek`（或启动参数 `--reasoning-format deepseek`，本部署默认开启）时，
非流式在 `message` 里同时给 `reasoning_content`（思考）与 `content`（正式回答），流式给 `delta.reasoning_content`；
模型没写出 think 闭合标记时不硬拆，整段留在 `content` 里。

错误同样走 OpenAI 形状：400（请求不合法）、401（api-key）、404（无路由）、413（body 过大）、
429（推理排队超过 `--max-queue`）、500（推理失败）、503（模型未就绪/连接过多）。

## 4. 采样与"不出现复读"的实现

**默认采样参数 = 模型厂商官方推荐**（`XHToken/Spark-X2.6-4B` 的 `generation_config.json`）：
`temperature=1.0`、`top_p=0.95`、`top_k=0`（=不截断，对应官方 `-1`）、
**`repeat_penalty=1.0`、`frequency_penalty=0`、`presence_penalty=0`——本部署刻意不加任何重复惩罚**：
用途是长上下文 Agent 任务与编码，重复惩罚会误伤代码里合法的重复结构。
`max_new_tokens=0` 表示**输出长度不限制**，生成到 EOS 或上下文用完为止（`max_new_tokens_cap=0` = 不给客户端设天花板）。

> 对照：`rkllm3-server --help` 的默认（temp 0.8 / top_k 1 / top_p 0.9）是 Rockchip 运行时默认，不是模型厂商推荐。
> `temperature>0` 时走 top-k/top-p 截断 + softmax 抽样（`seed` 可复现）；`top_k=0` 且 `top_p<1` 时
> 实现里把候选裁到 4096（等价于官方 `top_k=-1` 配核采样，省掉整表 13 万次的排序/softmax）。

复读防线只保留"熔断"，不做逐 token 惩罚（自定义采样器 `--sampler custom`，默认开启）：

1. **token 级周期熔断**：每步在选 token 前检测已生成序列末尾是否存在周期 `L≤400` 的三连块；
   若存在，把"延续该周期的下一个 token"列入禁止表，改选次优 token。（旧版上限是 24，抓不到
   "整块自我复述"这种上百 token 周期的失效模式。）
2. **换行连击熔断**：连续 ≥3 个换行 token 直接禁止继续选换行。
3. **token 级 n-gram 熔断**：候选 token 若会让某个已出现 ≥`ngram_ban_at`（默认 3）次的
   `ngram_len`（默认 8）-gram 再次闭合，就禁止它并重采样（最多重试 3 次）。
4. **文本级长块熔断**：输出每增长 64 字节，就取尾部 120 字节窗口在全文里计数，出现 ≥3 次即判定复读并强制收尾
   （`loop_window_bytes` / `loop_check_min_bytes` / `loop_check_stride_bytes` / `loop_max_occurrences`，见 `engine.h`）。
5. **片段级检测**：同一文本片段连续重复 ≥6 次也强制收尾。
6. **熔断次数上限**（`--veto-limit`，默认 6）：超限即返回 EOS 强制收尾，
   `usage.degeneration_veto` / `usage.degeneration_loop` / `usage.degeneration_guard` 与
   `/stats` 的 `requests_degeneration_guard` 会计数，服务端日志每请求打印 `veto=… guard=… loop=…`。

旋钮（都可经 `/etc/rknn3-llm.env` 的 `EXTRA_ARGS` 传入）：`--veto-period-max`、`--ngram-len`、
`--ngram-ban-at`、`--loop-max-occurrences`、`--veto-limit`；**`--fuse-off` 把三层熔断整个关掉**
（阈值置 0 即短路，只剩 EOS 与上下文窗口收尾）。

> 编码/Agent 场景提醒：第 3、4 层是"完全相同片段重复 N 次"的硬匹配，代码里重复的 `}`、导入行、
> 样板块理论上可能触发。若发现误伤，先 `--ngram-ban-at 6 --loop-max-occurrences 4`，仍不满意再 `--fuse-off`。
> 目前 392 题评测里没有误伤记录（熔断 14 次全部是真复读）。

采样器可退回 runtime 内置实现：`--sampler runtime`（此时没有熔断，仅用 llm_param 的惩罚）。

### 4.1 思考通道与 `thinking_effort` 注入开关

官方模板在 `enable_thinking=true` 时，会把这段话追加进 system 块（`high`/`low` 两档）：

> `` `thinking_effort` guides on how much to think in your thinking channel … Now the system is invoked with `thinking_effort=high`. ``

本服务默认照做（与官方逐字节一致），但提供开关：

- 服务端：`--thinking-effort-inject on|off`（默认 on）
- 请求级：`chat_template_kwargs.thinking_effort_inject: false`
- 关闭思考：`reasoning_effort: "none" | "minimal"`，或 `chat_template_kwargs.enable_thinking: false`，或 `--thinking off`
- `/v1/models` 的 `meta.sampling_defaults.thinking_effort_inject` 会回显当前状态

**为什么需要这个开关（实测 `fib_probe`）**：同一道"用 Python 写 `fib(n)`，只输出代码"（`/root/fib_probe.json`）：

| 配置 | 结果 | completion | 正式回答里有没有 `def fib` |
|---|---|---|---|
| thinking on + 注入 on + `temperature 0`（当前生产默认） | MISS | 155 | 无——模型把"用户要求…并且系统提示 `thinking_effort=high`，需要在思考通道中体现高程度的思考…思考过程：…"写在**回答通道**里，写完 155 token 就 EOS，**没有给出代码** |
| 同上，预算不限制 / 2500 | MISS | 155 | 同上（说明不是预算不够，是收尾行为） |
| `reasoning_effort:"none"` | **HIT** | 74 | `def fib(n): …` 完整正确代码，1.8 s |
| `chat_template_kwargs.enable_thinking:false` | **HIT** | 74 | 同上 |
| 厂商采样（temp 1.0）+ thinking on | MISS | 155 | 同第一行（英文复述） |

即：**注入文本会被部分 prompt 复述**，而"要求只输出代码"这类短任务在 thinking on 下更容易出现"只复述不作答"。
官方 README 给的权衡数字是 high→low 平均 −1.7 分、输出长度 −22 %（4B）。
→ 结论要按任务类型选：数学/推理类 thinking on 更准（A/B 5/6 vs 4/6），"只输出代码"类 thinking off 明显更好；
`cfgab_*` 那轮 60 题 A/B（effort low / 注入关闭 / 思考关闭）给出量化结果后回填本节。

## 5. 开机自启与进程守护

`deploy/install_service.sh` 一次装好并 enable 下面这套（unit 模板里的 `__PREFIX__` 由脚本替换，默认 `/opt/rk1828-spark`）：

| 文件 | 作用 |
|---|---|
| `/etc/systemd/system/rk1828-spark.service` | 常驻服务：`Restart=always RestartSec=8`、`After/Requires=rknn3.service`、`StartLimitIntervalSec=0`（崩溃后无限重试）、`KillSignal=SIGTERM` + `TimeoutStopSec=90`（优雅收尾：先停 accept、等活跃连接结束、再销毁 session/context，避免把脏 session 留在卡上）、`LimitNOFILE=8192` |
| `/etc/systemd/system/rk1828-spark-health.timer` | 开机 120 s 后每 20 s 触发一次健康检查 |
| `/etc/systemd/system/rk1828-spark-health.service` | 运行 `/opt/rk1828-spark/rk1828-spark-health.sh` |
| `/opt/rk1828-spark/rk1828-spark-health.sh` | `/health` 连续失败 4 次（≈80 s）才 `systemctl restart`。两个防自激设计：unit 处于 `activating/starting/reloading`、或主进程启动不足 240 s，都视为"模型正在加载"而不累计失败。卡片异常只记录，**绝不做 reset**（这张卡的复位手段是整机断电） |
| `/etc/rk1828-spark.env` | 全部参数集中在这里（模型档位、端口、采样、熔断、鉴权） |
| `/opt/rk1828-spark/start.sh` | 启动包装：读 env 再 `exec` 二进制 —— 改参数不必动 unit |

```bash
PREFIX=/opt/rk1828-spark deploy/install_service.sh
systemctl status rk1828-spark.service
journalctl -u rk1828-spark -f
```

## 6. 已知限制（诚实清单）

- 单 session（模型 KV 已占 4854/5052 MB），所有推理请求**串行排队**；并发只体现在 HTTP 层与排队上
  （排队深度超过 `--max-queue` 返回 429，`/stats` 暴露 `queued_now` / `queue_limit`）。
- **明确拒绝**：`n>1`（返回 400，不会静默只回一条）。**已实现**：`logit_bias`、`top_logprobs`（1..8，贪心路径给候选排名）、
  `reasoning_effort`、`logprobs`、`/v1/completions`、HEAD、`/v1/models/{id}`、429+`Retry-After`。
  **未实现**：`beam_search`、多模态输入（图片/音频）、`/v1/embeddings`、`audio`/`modalities`、
  `--run-mode async`（`rknn3_session_run_async` 重叠取样）。
- **性能事实（已定量）**：单流 decode ≈ 50 tok/s（短上下文，19–20 ms/token），上下文到 38k/53k token 时降到
  35 / 32 tok/s；prefill 135 ms（88 token）→ 789 tok/s（4.3k）→ 684 tok/s（14.8k）→ 约 460 tok/s（38k）→
  53166 token 用 115.6 s。**不存在"每请求固定 10 s 开销"**——`timing_probe.py` / `mt_sweep.py` 证明
  `decode_ms` 只随实际生成 token 数走，与请求的 `max_tokens` 无关；早期评测日志里那个 10 s 平台期在该构建上不可复现，
  已结案（遗留标注为当时那个实例的现象，非架构问题）。
- **KV cache 策略不是杠杆**：默认 / 显式 RECURRENT / NORMAL 三档实测 prefill 与 decode 差异在噪声内；
  `--kv-n-keep 2048` 直接起不来（HEALTH_FAIL）。`SAVE_CHECKPOINT` 策略未试（Spark 是滑窗注意力，官方注释说它对
  线性/滑窗模型才生效）。
- `tools` 已按官方 chat_template 完整打通：服务端把工具清单渲染进 system 块（`## Tools` + `<tools>` 包裹
  每个 `tool.function` 紧凑 JSON），模型输出 `<tool_call>名称 <arg_key>k</arg_key><arg_value>v</arg_value>`，
  服务端解析成规范 OpenAI `tool_calls`（`name` + 可解析的 `arguments` JSON），`finish_reason="tool_calls"`；
  工具结果走 `<|Tool|>` 轮 + `tool_response` 成对标记回填。实测（官方 function-call 示例那套工具与 prompt）
  命中 4/8，`tools` 不给时不会伪造 tool_calls。模型自身的工具遵循度决定命中率。
- `response_format` 走 runtime 的 grammar（`json_object` → BUILTIN_JSON，`json_schema` → JSON_SCHEMA）。
- 流式响应无法在同一条连接上继续 keep-alive（没有 Content-Length，靠关连接收尾）。
- 长上下文 prefill：实测 **38006 token / 72.0 s** 与 **53166 token / 115.6 s** 单次 prefill 均成功且 needle 命中，
  逐级加压时卡健康（Online/OK、显存、温度、LnkSta）每次都复查通过。更早的"长 prefill 打挂"记录发生在
  旧固件 + 未外接 12 V 供电的条件下，不要据此自我设限；但仍建议阶梯式加压并在每级后检查卡状态。

## 7. 长上下文 Agent / 编码的推荐配置

用途设定：长上下文 Agent 任务与代码生成。据此定下的生产配置（`/etc/rknn3-llm.env`，改完 `systemctl restart rknn3-llm.service`）：

| 项 | 值 | 理由 |
|---|---|---|
| 模型档位 | `Spark-X2.6-4B-73728-kvi4` | ctx 73728 ≥ 64K，KV int4；卡常驻约 4903/5120 MB |
| `THINKING` | `on` | thinking=on 在算术/推理题上命中更高（A/B：5/6 vs 4/6），且 `reasoning_effort=high/low` 注入依赖它 |
| `TEMP / TOP_K / TOP_P` | `1.0 / 0 / 0.95` | 模型厂商 `generation_config.json` 的推荐值 |
| `REPEAT_PENALTY / FREQ / PRESENCE` | `1.0 / 0.0 / 0.0` | **不加任何重复惩罚**：代码里合法的重复结构会被惩罚器误伤 |
| `MAX_NEW / MAX_NEW_CAP` | `0 / 0` | 输出长度不限制，靠 EOS / 上下文窗口 / 请求超时收尾 |
| `EXTRA_ARGS` | `--reasoning-format deepseek` | 思考内容与正式回答分离，Agent 客户端只消费 `content` |
| 熔断 | 默认三层（周期 / n-gram / 长块） | 392 题评测真实复读 0 条，熔断 14 次全部为真复读；误伤时按 §4 的旋钮降级 |

实践要点（已实测）：

- 客户端**不要**传 `repeat_penalty`；要控制长度用 `max_tokens`，要关思考用 `reasoning_effort:"none"`（或
  `chat_template_kwargs.enable_thinking=false`）。
- 工具调用必须走 `tools` 字段（服务端负责渲染进 system），不要自己把工具定义拼进 system 文本。
- Agent 多轮：`role:"tool"` 的消息被渲染成 Spark 的 `<|Tool|>` 轮；assistant 历史里的 `tool_calls` 会被回填成
  `<tool_call>…<arg_key>…</arg_key><arg_value>…</arg_value>` 形态。
- 超长上下文别指望并发：单 session 串行，38k token 的 prefill 要 70 s，期间其他请求在队列里（超过 `--max-queue` 收 429）。
