# rknn_spark_server：RK1828 板端 OpenAI 兼容推理服务

自研 C++ 服务（GPL-3.0）。它把 RK1828 卡上的 **Spark-X2.6-4B**（RKNN3 1.2.0 转换产物）包成
OpenAI 兼容的 HTTP 接口，常驻运行。

先约定几个词，后面会反复用到：

- **KV cache**：模型记住前文的中间结果。多轮对话有它就不用把历史重算一遍。
- **prefill / decode**：prefill 是把整段提示词读进模型，decode 是一个 token 一个 token 往下写。
- **槽位（slot）**：某个会话的 KV 快照，存在主机内存里；卡上同一时刻只放一个。
- **grammar**：runtime 的结构约束，用来强制输出 JSON 或工具调用的外层格式。
- **SSE**：流式响应，一行一个 `data: {...}`。

为什么不用官方 `rkllm3-server`：RKNN3 1.2.0 的官方服务对 `model_type=spark2_5` 输出是坏的。
换 prompt、换 chat template、开关 `--reasoning`、改采样，穷举 6 种配置后输出仍从第一个 token 开始崩；
同一份模型走 model-zoo 的 C++ demo 就正常。本服务就是把那条可用路径做成常驻 + HTTP。
证据见 `docs/troubleshooting.md` 第 4 条。

- 源码：`server/cpp/{engine.h,engine.cpp,httpd.h,httpd.cpp,main.cpp,kvslots.h,kvslots.cpp}`
- 编译：`scripts/build_server.sh`，产物 `server/build/rknn_spark_server`
- 部署：`deploy/install_service.sh`，默认前缀 `/opt/rk1828-spark`

## 1. 编译

```bash
scripts/get_sdk.sh          # 取 RKNN3 SDK 头文件和运行时库（不在本仓库，见 docs/build.md）
scripts/build_server.sh     # cmake + make
```

三个注意点：

- 必须用 **1.2.0** 的 `rknn3_api.h`。model-zoo 自带的是 1.1.x，`rknn3_session_clear_kvcache` 等签名不同，
  直接编译会报错。CMake 会优先从 `RKNN3_SDK_ROOT` 取头文件。
- `libtokenizer.a` 是 model-zoo 预编译的（Apache-2.0），仓库里已经带了；也可以自己从 model-zoo 源码重建。
- `thirdparty/utils` 只保留 `fileutils / cnpy / timeutils` 三个小库，本服务不需要图像/音频部分。

## 2. 启动

装好服务后建议用 systemd 启动（见第 5 节）。手工启动：

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
    --kv-slots 4 --kv-slot-dir /run/rk1828-kv --kv-slot-min-tokens 1024 \
    --stream-tools on --tool-grammar on \
    --reasoning-format deepseek
```

位置参数顺序固定：`<llm.rknn> <llm.weight> <core_mask（十六进制）> <tokenizer.gguf> <embed.bin>
<max_context_len> <rope.safetensors>`。`core_mask` 表示用哪几个 NPU 核，`0xff` 是 8 核全用。

模型加载大约 45 s，期间 `/health` 返回 503，加载完变 200。`TMPDIR` 必须可写，systemd 单元里已经设好。

另外两种配置方式：

- JSON 配置文件（和官方 rkllm3-server 风格一致）：`--params-file params.json`
- 鉴权：`--api-key K`，之后客户端要带 `Authorization: Bearer K`（`/health` 除外）

完整参数列表：`rknn_spark_server --help`。

KV 槽位的镜像放在内存盘里。`deploy/start.sh` 会检查 `KV_SLOT_DIR`：如果它不是 tmpfs，就按
`KV_SLOT_TMPFS_SIZE`（默认 11G）挂一块；挂载失败会退回磁盘（能用，但保存/加载慢一个数量级）。
镜像大小约 `78 MB + 31 KB × token 数`：1.2k token 约 110 MB，64K 约 2.4 GB。四个槽都跑满 64K
约 9.6 GB，所以内存盘给 11G、总预算给 10240 MB。

## 3. 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/chat/completions` | 主接口，参数见下方列表；`n>1` 返回 400 |
| `POST` | `/v1/completions` | 传统补全。`prompt` 原样送进模型，不做 chat 包装，适合评测和底层实验 |
| `POST` | `/v1/responses` | OpenAI Responses API（流式 + 非流式）。`instructions`、`input`（字符串或 item 数组）、扁平 `tools`、`tool_choice`、`max_output_tokens`、`reasoning.effort`、`text.format` |
| `POST` | `/v1/embeddings` | 词嵌入表均值池化 + L2 归一化（本模型 2560 维）。只是近似嵌入，不是训练过的 embedding 模型 |
| `GET` | `/v1/models`, `/v1/models/{id}` | 模型列表。`properties.context_length`、`capabilities`、`supported_parameters`，`meta` 里有量化方式和 KV 槽位配置 |
| `GET`/`HEAD` | `/health`, `/healthz` | 就绪探针。模型没加载完返回 503；不需要 api-key |
| `GET` | `/metrics` | Prometheus 文本；`/stats` 是同一份数据的 JSON |
| `GET` | `/v1/slots` | KV 槽位表：空闲/在卡/已换出、镜像大小、命中次数。没开槽位时给单 slot 状态 |
| `POST` | `/slots/{id}/pause\|resume\|interrupt` | 对应 `rknn3_session_pause/resume/stop` |
| `POST` | `/slots/{id}/kvcache/save\|load`、`/v1/kvcache/save\|load` | 对应 `rknn3_session_save_kvcache` / `load_kvcache_from_path`，默认目录 `/run/rknn3/kvcache` |
| `OPTIONS` | 任意 | CORS 预检 |

`/v1/chat/completions` 支持的参数：

- 对话：`messages`（`system/developer`、`user`、`assistant`、`tool/function`）
- 流式：`stream`、`stream_options.include_usage`
- 长度与停止：`max_tokens` / `max_completion_tokens` / `n_predict`、`stop`
- 采样：`temperature`、`top_p`、`top_k`、`frequency_penalty`、`presence_penalty`、`repeat_penalty`、`seed`
- 概率与偏置：`logprobs`、`top_logprobs`、`logit_bias`
- 工具与格式：`tools`、`tool_choice`、`parallel_tool_calls`、`response_format`
- 思考控制：`reasoning_format`、`reasoning_effort`、`chat_template_kwargs.{thinking,enable_thinking,thinking_effort_inject}`

响应是 OpenAI 的形状：`id/object/created/model/choices[]/usage`。流式用 SSE（`data: {...}`，最后
`data: [DONE]`），第一块带 `delta.role="assistant"`，结束块带 `finish_reason`，`include_usage` 时
最后再单独发一块 `choices: []` + `usage`。

`finish_reason` 的含义：

- `stop`：采样器自己选了 EOS、命中 `stop` 串，或触发熔断收尾。
- `length`：被 `max_tokens` 截断。
- `tool_calls`：解析出了工具调用。
- `error`：推理失败。

runtime 的 `n_decode_tokens` 和请求上限差一个（实测 `max_new_tokens=64` 它只报 63），所以判 `length`
看的是"采样器有没有选 EOS"，不是比较 token 数。用 `--sampler runtime` 时没有采样回调，只能按 token 数判断。

`usage` 里额外有 `prefill_ms/decode_ms/e2e_ms/tokens_per_second`；触发过复读熔断时给
`degeneration_veto` / `degeneration_loop` / `degeneration_guard`；开槽位时还有
`prompt_tokens_details.cached_tokens`、`kv_slot`、`kv_prefix_tokens`、`kv_image_loaded`。

开 `reasoning_format=deepseek`（本部署默认开）后：

- 非流式：`message.reasoning_content` 是思考，`message.content` 是正式回答。
- 流式：思考走 `delta.reasoning_content`，正文走 `delta.content`。
- 关掉思考时不会产生 `reasoning_content`。
- 模型没写出 `</think>`（常见于 `max_tokens` 截断）时，整段按思考内容放进 `reasoning_content`，
  `content` 为空，和流式路径保持一致。

错误也走 OpenAI 形状：400（请求不合法）、401（api-key）、404（无路由）、413（body 过大）、
429（排队超过 `--max-queue`）、500（推理失败）、503（模型没就绪或连接过多）。

### 3.1 工具调用

- `tool_choice`：`auto`（默认）、`none`、`required`、`{"type":"function","function":{"name":...}}`。
  `parallel_tool_calls:false` 会在工具说明里加一句"只发一个调用"。
- 流式调用按 OpenAI 规范分片：第一片带 `index/id/type/function.name`（`arguments` 是空串），
  之后每片只有 `index` + `arguments` 片段。客户端把片段拼起来再 `JSON.parse`。
- 有工具调用时，非流式 `message.content` 是 `null`，`finish_reason` 是 `tool_calls`。
- 工具标记不会漏进 `content`；解析只作用于答案通道，思考里转述的工具语法不会被当成调用。
- `response_format` 只接受 `text`/`json_object`/`json_schema`，其它类型返回 400。
  优先级按官方的 `json_schema > structural_tag > grammar`。思考模式下用 `structural_tag` 把
  "思考段 + 约束段"合成一份 grammar；`tool_choice=required` 或指定函数时也一样，
  失败会退化成提示词约束并写日志（`--tool-grammar off` 可以关掉）。

### 3.2 KV 槽位（多会话切换）

`--kv-slots 4` 开启后，服务在主机内存里保存多个会话的 KV，卡上同一时刻只放一个。

一次请求的处理顺序：

1. 按到达顺序排队（FIFO 票据），拿到卡后继续。
2. 在槽表里找"新提示词以某槽的『上次提示词 + 原始生成文本』开头"的槽。
3. 命中且该槽就在卡上：直接续接，只把新增的这一轮文本送进 runtime。
4. 命中但卡上是别的槽：先把卡上这个槽存成镜像，再 load 目标槽镜像（接在镜像末尾）。
5. 没命中：槽没满就新开一个；满了就按 LRU 清掉最不活跃的槽，然后冷启动（清 KV、整段重算）。

几条实测的坑：

- **客户端必须原样回放上一轮的 `content` 和 `reasoning_content`**。开槽位时服务端不做裁剪；
  客户端丢了 reasoning 或改写了内容，就会退回第 5 步的冷启动。结果仍然正确，只是慢。
- Spark 是滑窗注意力，**续接点必须正好在镜像末尾**。截断到中间位置需要 checkpoint，1.2.0 默认策略
  下没有可用 checkpoint，runtime 会报 `can not find the kvcache_checkpoint`。
- **每轮重设 chat template / 工具 / grammar 会清空 KV**。服务只在内容变化时下发这些设置，改这里要小心。
- 镜像目录是 `/run/rk1828-kv`（11G tmpfs，掉电即失）。单槽上限 `--kv-slot-max-image-mb`（默认 2560），
  总预算 `--kv-slot-max-total-mb`（默认 10240）；超了按 LRU 丢镜像，不会把内存盘写满。
- 启动时会做一次 save→load→续接自检。如果 runtime 报告 `n_reuse_tokens == 0`，槽位自动关闭，
  `/v1/slots` 和 `/v1/models` 的 `kv_reuse_probe` 会显示 `unsupported`。

效果：同一会话第二轮提示词 1214 token 时，prefill 从 1214 token 降到 18 token；保存镜像 0.4–1.0 s，
加载约 0.2 s。槽位开、关两种配置下，贪心输出逐字节一致。

## 4. 采样与复读

默认采样值来自**模型厂商**的 `generation_config.json`：`temperature=1.0`、`top_p=0.95`、
`top_k=0`（不截断，对应官方 `-1`）、各种惩罚全 0。输出长度默认不限制（`max_new_tokens=0`），
生成到 EOS 或上下文用完为止。

`temperature>0` 时走 top-k/top-p 截断 + softmax 抽样（`seed` 可以复现）。

抽样只看分数最高的 4096 个候选。客户端没给 `top_k` 时这个上限就生效，和 `top_p` 是多少无关。
被裁掉的候选都不大于第 4096 名，所以它们的概率和有一个上界：`剩余个数 × exp((第 4096 名 − 第 1 名) ÷ 温度)`。
上界超过保留质量的 1% 时按 4 倍扩大候选，最多回到整表。正常输出不会触发这个扩大：`tools/cap_check.cpp` 用
合成分布实测，常见的"前几名很尖、后面一条低尾巴"裁掉的质量在 1e-7 量级，重尾情形 1e-4；只有分布非常平
（`temperature=2.0` 的重尾、或 logits 几乎相等）才会扩到整表。客户端自己给了 `top_k` 就按它的值截断，不扩大。

为什么要有上限：整表是 131072 个分数，排序加 softmax 每个 token 都要做一遍，只看前 4096 个把排序量缩小 32 倍。
早期的条件是"`top_k` 关闭并且 `top_p<1`"，于是客户端按 OpenAI 默认传 `top_p=1.0` 就退回整表路径：同样生成
63 个 token 用 1.36 s 变 2.45 s（21 ms/token 变 39 ms/token）。条件改成与 `top_p` 无关后，两条路径都是
约 21 ms/token。

复读防线只做"熔断"，不做逐 token 惩罚（默认 `--sampler custom`）：

1. **周期熔断**：选 token 前检查已生成序列末尾有没有周期不超过 400 的三连块。有的话，
   把"延续这个周期的下一个 token"禁掉，改选次优。
2. **换行连击**：连续 3 个以上换行 token，禁止继续选换行。
3. **n-gram 熔断**：候选 token 如果会让某个已经出现 3 次以上的 8-gram 再次闭合，就禁掉并重采样（最多重试 3 次）。
4. **长块熔断**：输出每增长 64 字节，取尾部 120 字节在全文里数一数，出现 3 次以上就判定复读并强制收尾。
5. **片段复读**：同一段文本连续重复 6 次以上也强制收尾。
6. **次数上限**：熔断累计超过 `--veto-limit`（默认 6）就直接返回 EOS。

熔断的计数会出现在 `usage.degeneration_veto` / `degeneration_loop` / `degeneration_guard` 和
`/stats` 的 `requests_degeneration_guard` 里，日志每请求打印 `veto=… guard=… loop=…`。

可调参数：`--veto-period-max`、`--ngram-len`、`--ngram-ban-at`、`--loop-max-occurrences`、`--veto-limit`。
`--fuse-off` 把三层熔断全关掉，只剩 EOS 和上下文窗口收尾。

编码/Agent 场景提醒：第 3、4 层是"完全相同片段重复 N 次"的硬匹配，代码里重复的 `}`、导入行、
样板代码理论上可能触发。如果误伤，先把 `--ngram-ban-at` 调到 6、`--loop-max-occurrences` 调到 4；
还不满意再 `--fuse-off`。目前 392 题评测没有误伤记录（14 次熔断全部是真复读）。

采样器也可以退回 runtime 内置实现（`--sampler runtime`），这时没有熔断，只用 llm_param 里的惩罚参数。

### 4.1 思考通道与 `thinking_effort`

官方模板在 `enable_thinking=true` 时，会往 system 块追加一段文本：

> `thinking_effort` guides on how much to think in your thinking channel … Now the system is invoked with `thinking_effort=high`.

本服务默认照做（和官方逐字节一致），但给了开关：

- 服务端：`--thinking-effort-inject on|off`（默认 on）
- 请求级：`chat_template_kwargs.thinking_effort_inject: false`
- 关思考：`reasoning_effort: "none" | "minimal"`，或 `chat_template_kwargs.enable_thinking: false`，或 `--thinking off`
- 当前状态可以在 `/v1/models` 的 `meta.sampling_defaults.thinking_effort_inject` 里看到

实测（`tools/fib_probe.py`，"用 Python 写 `fib(n)`，只输出代码"）：

| 配置 | 结果 | completion | 回答里有没有 `def fib` |
|---|---|---|---|
| thinking on + 注入 on + `temperature 0`（生产默认） | MISS | 88 | 没有。回答通道写的是"用户要求…斐波那契的定义…需要考虑 n 的范围…"，写完就 EOS |
| 同上，预算不限制 / 2500 | MISS | 88 | 和上一行逐字节相同，说明不是预算不够，是收尾行为 |
| `reasoning_effort:"none"` | **HIT** | 75 | 完整正确的 `def fib(n): …` |
| `chat_template_kwargs.enable_thinking:false` | **HIT** | 75 | 同上 |
| 厂商采样（temp 1.0）+ thinking on | MISS | 88 | 同第一行 |

所以"只复述不作答"不是注入文本造成的，而是 thinking on 下短任务的收尾行为。按任务选：
数学/推理类开着思考更准（60 题 32 % vs 16 %），"只输出代码"类关掉明显更好。

### 4.2 system 块怎么送进模型：`--system-in-prompt`

| 取值 | 行为 |
|---|---|
| `on`（默认） | system 块写进 prompt 文本；客户端的 system 生效；`prompt_tokens` 包含 system |
| `off` | 旧行为：system 交给 `set_chat_template`（实测内容不会进模型），只用于对照实验 |

推论：`--thinking-effort-inject`、`reasoning_effort` 这类"改 system 文本"的开关，只有在 `on` 模式下
才可能真正落到模型看到的 token 序列上。

## 5. 常驻服务

`deploy/install_service.sh` 会装好并 enable 这一套（unit 模板里的 `__PREFIX__` 会替换成实际前缀，默认 `/opt/rk1828-spark`）：

| 文件 | 作用 |
|---|---|
| `/etc/systemd/system/rk1828-spark.service` | 常驻服务。`Restart=always`、崩溃后无限重试、`SIGTERM` 优雅收尾（先停 accept、等连接结束、再销毁 session，避免把脏 session 留在卡上） |
| `/etc/systemd/system/rk1828-spark-health.timer` | 开机 120 s 后，每 20 s 触发一次健康检查 |
| `/etc/systemd/system/rk1828-spark-health.service` | 运行 `/opt/rk1828-spark/rk1828-spark-health.sh` |
| `/opt/rk1828-spark/rk1828-spark-health.sh` | `/health` 连续失败 4 次（约 80 s）才重启服务。模型加载期间不累计失败；卡状态异常只记录，**绝不做 reset**（这张卡复位要整机断电） |
| `/etc/rk1828-spark.env` | 全部参数集中在这里 |
| `/opt/rk1828-spark/start.sh` | 启动包装：先读 env 再 `exec` 二进制，改参数不用动 unit |

```bash
PREFIX=/opt/rk1828-spark deploy/install_service.sh
systemctl status rk1828-spark.service
journalctl -u rk1828-spark -f
```

## 6. 已知限制

- **单 session，请求串行**：模型 KV 已占 4854/5052 MB，所有推理请求排队执行。并发只体现在 HTTP 层
  和排队上（超过 `--max-queue` 返回 429，`/stats` 有 `queued_now` / `queue_limit`）。
  4 个 KV 槽位解决的是会话切换时的重复 prefill，不是并行推理。
- **`n>1` 直接拒绝**（400）。已实现：`logit_bias`、`top_logprobs`（1..8）、`reasoning_effort`、`logprobs`、
  `/v1/completions`、HEAD、`/v1/models/{id}`、429 + `Retry-After`、`/v1/responses`、`/v1/embeddings`
  （均值池化近似）。未实现：`beam_search`、图片/音频输入、`audio`/`modalities`、`rknn3_session_run_async`。
- **KV 能不能复用，取决于续接点**：策略（默认 / RECURRENT / NORMAL）本身不是杠杆，三种策略的
  prefill/decode 差异在噪声内；`--kv-n-keep 2048` 起不来（HEALTH_FAIL）。真正的关键是续接点必须在
  镜像末尾，而且不要每轮重设 chat template / 工具 / grammar（细节见 §3.2）。
- **工具调用**按官方 chat_template 打通：工具清单渲染进 system 块，模型输出
  `<tool_call>名称 <arg_key>k</arg_key><arg_value>v</arg_value>`，服务端解析成 OpenAI `tool_calls`。
  实测官方 function-call 示例 8 题命中 4；不给 `tools` 时不会伪造调用，但模型可能在正文里自己编
  一段"已调用工具查询到…"的叙述，Agent 侧要自己校验。命中率最终由模型本身的工具遵循度决定。
- `response_format` 走 runtime grammar（`json_object` → BUILTIN_JSON，`json_schema` → JSON_SCHEMA）。
  未知类型返回 400；思考模式下用 `structural_tag` 把思考段和约束段合成一份 grammar。
- 流式响应不能在同一条连接上 keep-alive（没有 Content-Length，只能靠关连接收尾）。
- 长上下文 prefill 实测：38,006 token 用 72.0 s，53,166 token 用 115.6 s，都成功且 needle 能召回。
  逐级加压时每次复查卡状态都正常。早期"长 prefill 打挂"的记录发生在旧固件 + 没接 12 V 供电时，
  不要据此自我设限，但仍建议阶梯加压、每级检查卡状态。

## 7. 长上下文 Agent / 编码的推荐配置

生产配置在 `/etc/rknn3-llm.env`（改完 `systemctl restart rknn3-llm.service`）：

| 项 | 值 | 理由 |
|---|---|---|
| 模型档位 | `Spark-X2.6-4B-73728-kvi4` | ctx 73728 ≥ 64K，KV int4；卡常驻约 4903/5120 MB |
| `THINKING` | `on` | 算术/推理留出集上命中明显更高（60 题 32 % vs 16 %）；只有"只输出代码"这类短任务该关（§4.1） |
| `EFFORT_INJECT` | `on` | 与官方模板逐字一致；实测关掉不会提高精度（20 % vs 24 %），只省 51 token |
| `TEMP / TOP_K / TOP_P` | `1.0 / 0 / 0.95` | 模型厂商 `generation_config.json` 的推荐值 |
| `REPEAT_PENALTY / FREQ / PRESENCE` | `1.0 / 0.0 / 0.0` | 不加任何重复惩罚：代码里合法的重复结构会被误伤 |
| `MAX_NEW / MAX_NEW_CAP` | `0 / 0` | 输出长度不限制，靠 EOS / 上下文窗口 / 请求超时收尾 |
| `EXTRA_ARGS` | `--reasoning-format deepseek` | 思考和正式回答分开，Agent 只消费 `content` |
| `KV_SLOTS / KV_SLOT_DIR` | `4 / /run/rk1828-kv` | 4 个内存镜像；64K×4 约 9.6 GB，tmpfs 给 11G |
| `KV_SLOT_TMPFS_SIZE` | `11G` | `start.sh` 负责挂载；失败退回磁盘 |
| `KV_SLOT_MAX_IMAGE_MB / TOTAL_MB` | `2560 / 10240` | 单槽和总量上限，超了按 LRU 丢镜像 |
| 熔断 | 默认三层（周期 / n-gram / 长块） | 392 题评测真实复读 0 条，14 次熔断全部是真复读；误伤时按 §4 调 |

实践要点：

- 客户端**不要**传 `repeat_penalty`。控长度用 `max_tokens`，关思考用 `reasoning_effort:"none"`
  （或 `chat_template_kwargs.enable_thinking=false`）。
- 工具调用走 `tools` 字段，不要自己把工具定义拼进 system 文本。
- Agent 多轮：`role:"tool"` 的消息会渲染成 Spark 的 `<|Tool|>` 轮；assistant 历史里的 `tool_calls`
  会回填成 `<tool_call>…<arg_key>…</arg_key><arg_value>…</arg_value>` 形态。
- 超长上下文别指望并发：单 session 串行，38k token 的 prefill 要 70 s，期间其他请求都在队列里。
