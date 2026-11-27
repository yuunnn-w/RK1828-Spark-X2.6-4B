# Changelog

## [未发布]

### 新增（实验开关，默认关闭）

- **`--thinking-grammar on|off` 与 `chat_template_kwargs.thinking_grammar`**：按官方 SDK 的写法给一份
  `structural_tag`，要求"思考段 → 思考结束标记 → 自由回答"，用来消除"模型在思考通道里直接结束、`content`
  为空"的轮次。**在 RK1828 + 固件/运行时 1.2.0 + Spark-X2.6 上实测无效，所以默认 off**：语法引擎确实在屏蔽
  EOS（日志里能看到 `AcceptToken rejected token_id=1`），但它只作用在"最终格式"那一段，思考段的前缀不被匹配——
  把 grammar 要求的结束标记换成无意义字符串，模型照旧写它自己的标记；给 fib 那道题加硬性 `json_schema` 约束，
  仍然是 157 token 纯思考后结束、没有 JSON。现象、四个对照和结论写在 `docs/server.md` §4.3，开关保留是为了
  换固件后能直接重试。

### 修复

- **`top_k` 关闭时的候选上限不再取决于 `top_p`**：客户端按 OpenAI 默认传 `top_p=1.0` 时，旧实现会退回整表
  131072 个分数排序，解码慢一倍。同一实例上对照（`tools/sampling_regression.py`，同样生成 63 个 token）：
  默认 `top_p=0.95` 1364 ms、显式 `top_p=1.0` 2435 ms → 现在两条都是 1317 / 1378 ms（21 ms/token）。
  默认采样值与默认路径的排序、softmax 都没变；`tools/protocol_check.py` 13/13 仍通过。同一实例上同一 seed 的
  采样输出可逐字节复现（改前改后各跑两次都自洽）；改后与旧二进制在同条件下的 5 项对照里 3 项逐字节一致，
  另 2 项不一致 —— 那两项在旧二进制自己重启前后也不一致，是温度 1.0 采样对运行时状态敏感，不是这次改动带来的。
- 新增 `tools/cap_check.cpp`：不碰卡、不加载模型，单独验证候选上限的算术——常见分布停在 4096 且被裁掉的
  概率约 1e-7，重尾 1e-4，分布很平（`temperature=2.0` 的重尾、logits 全相等）时扩到整表，客户端给了
  `top_k` 时按它的值截断。机制说明见 `docs/server.md` §4，实测见 `docs/benchmarks.md` §3。

## v1.1.0 — 2026-11-27

面向 Agent 客户端的协议对齐、`/v1/responses` 和 4 槽 KV 复用。数字仍来自真机（OrangePi5Max + RK1828，固件 1.2.0）。

### 新增

- **流式工具调用按 OpenAI 规范增量下发**：首片带 `index/id/type/function.name`（`arguments` 为空串），
  后续片只带 `index` + `arguments` 片段；工具标记不再漏进 `content`，有调用时非流式 `content` 为 `null`。
  工具解析只作用于答案通道（`reasoning_content` 里转述的工具语法不会被误判）。
  `content`/`reasoning_content` 里的模板标记在开启槽位时不做裁剪（见下面"KV 槽位"）。
- **`tool_choice` 生效**：`required`、`{"type":"function","function":{"name":...}}` 用 `structural_tag`
  约束工具调用外层结构（可用 `--tool-grammar on|off` 关闭）；`parallel_tool_calls:false` 会在工具说明里
  加"只发一个调用"的约束。
- **`/v1/responses`**（流式 + 非流式）：`instructions`、`input`（字符串或 item 数组）、扁平 `tools`、
  `max_output_tokens`、`reasoning.effort`、`text.format`；事件包括 `response.created`、
  `response.output_item.added/done`、`response.output_text.delta/done`、`response.function_call_arguments.delta/done`、
  `response.completed`。DeepSeek Harness 的 `openai-responses` 路由可直接接。
- **`/v1/embeddings`**：词嵌入表均值池化 + L2 归一化（本模型 2560 维）。只是近似嵌入，不是训练过的
  embedding 模型，够做粗粒度检索/去重。
- **HTTP 细节**：支持 chunked 请求体；`OPTIONS` 带 `Allow-Methods/Allow-Headers/Max-Age`；
  `/health`、`/healthz` 不要求 api-key（其余端点仍校验）。
- **`/v1/models` 元数据**：补 `properties.context_length`、`capabilities`、`supported_parameters`、
  `system_fingerprint`，`meta` 里带 KV 槽位配置与 `kv_reuse_probe`。
- **KV 槽位分页（`--kv-slots 4`）**：每个会话的 KV 存成主机内存镜像（`/run/rk1828-kv`，11G tmpfs），
  卡上只放活跃槽。命中条件是"新提示词以该槽的上次提示词 + 原始生成文本开头"，命中后只送新增的一轮；
  卡上是别的槽时先存后 load；未命中且槽满按 LRU 清理；单槽 2560 MB、总量 10240 MB 预算，超出丢镜像。
  启动时做一次 save→load→续接自检，runtime 不支持就自动关闭槽位。
- **FIFO 排队**：请求按到达顺序拿卡（标准 `std::mutex` 不保证公平）。
- **`tools/protocol_check.py`**：13 项协议验收（流式工具调用、`content=null`、`tool_choice`、`stop` 句中截断、
  `response_format` 400、chunked、CORS、`/v1/responses`、多轮槽位），支持 `--save` / `--compare`。

### 修复

- **每轮重设 chat template / 工具 / grammar 会清空 runtime 的 KV**，导致 `load_kvcache` 后 `n_reuse_tokens`
  恒为 0。现在这些设置只在内容变化时下发，KV 复用才成立（同一会话第二轮 prefill 1214 → 18 token）。
- `stop` 从"只看输出末尾"改成整段任意位置匹配，跨 SSE 片段也能命中。
- `tool_call.id` 改为全请求唯一（`call_<seq>_<12hex>`）。
- prompt 预算改用真实词表计数，超窗返回 400 + `context_length_exceeded`，不再按字符数估算。
- `include_usage` 的尾块改为 `choices: []` + `usage`（OpenAI 规范）。
- `response_format` 未知 `type` 返回 400；思考模式下用 `structural_tag` 把思考段和约束段合成一份 grammar。
- 带 `tools` 时放宽复读熔断（工具参数里重复的键/括号不再误触发）。

### 实测

- 协议验收 13/13 通过（槽位开、关各跑一遍）。
- 多轮 KV：同一会话第二轮 `reuse=1214, prefill=18`；A/B 两个会话交替切换答案正确；槽位开/关的贪心输出
  逐字节一致。
- 镜像大小 ≈ 78 MB + 31 KB/token（1.2k token 约 110 MB，5.9k 约 261 MB，64K 约 2.4 GB）；
  保存 0.4–1.0 s，加载约 0.2 s（tmpfs）。
- 滑窗 KV 的续接点必须正好在镜像末尾；截断到中间位置需要 checkpoint，1.2.0 默认策略下会报
  `can not find the kvcache_checkpoint`。

### 已知限制

- 单 session，推理仍严格串行；4 个槽位解决的是会话切换的重复 prefill，不是并行推理。
- 槽位命中要求客户端原样回放上一轮的 `content` 与 `reasoning_content`；回放不一致会冷启动（结果正确，只是慢）。
- 工具调用轮次如果客户端回放的 `tool_calls` 重新序列化后与原始生成文本不一致，也会冷启动。
- `/v1/embeddings` 是均值池化近似，不保证语义效果。

## v1.0.0 — 2026-11-10

首个公开发布版本。全部数字来自真机（OrangePi5Max + RK1828 M.2，固件 1.2.0）。

### 新增

- **`server/`：OpenAI 兼容推理服务**（C++11，单文件级依赖）
  - 端点：`/v1/chat/completions`（非流式与 SSE 流式）、`/v1/completions`、`/v1/models`、`/v1/models/{id}`、
    `/health`(+HEAD)、`/healthz`、`/metrics`(Prometheus)、`/stats`、`/v1/slots`、
    `/slots/{id}/pause|resume|interrupt`、`/slots/{id}/kvcache/save|load`、`/v1/kvcache/save|load`、CORS 预检
  - 请求参数：`messages`（含 `system/user/assistant/tool` 角色与 `assistant.tool_calls` 历史）、
    `stream`、`stream_options.include_usage`、`max_tokens`/`max_completion_tokens`/`n_predict`、
    `temperature`、`top_p`、`top_k`、`frequency_penalty`、`presence_penalty`、`repeat_penalty`、`stop`、`seed`、
    `logprobs` + `top_logprobs`、`logit_bias`、`tools` + `tool_choice` + `parallel_tool_calls`、
    `response_format`(`text`/`json_object`/`json_schema`，走 runtime grammar)、
    `reasoning_format=deepseek`(`reasoning_content` 分离)、`reasoning_effort`、
    `chat_template_kwargs.{thinking,enable_thinking,thinking_effort_inject}`
  - 语义细节：`n>1` 明确返回 400；队列超过 `--max-queue` 返回 429 + `Retry-After: 2`；
    模型未就绪 503；连接过多 503；`finish_reason` 用"采样器是否选中 EOS"判截断（避开 runtime 的 off-by-one）
  - 复读防线：三层熔断（token 级周期 / token 级 8-gram / 文本级长块），旋钮
    `--veto-period-max`、`--ngram-len`、`--ngram-ban-at`、`--loop-max-occurrences`、`--veto-limit`、`--fuse-off`；
    `usage` 里回显 `degeneration_veto/loop/guard`
  - 健壮性：客户端硬断开不影响服务；输出里的非法 UTF-8 做修复与跨片段续接，出站 JSON 一律 `replace` 模式
- **`convert/`：转换流水线**
  - `run_conversion.sh` 一键：GRQ(w4a16/group32) 导出 ONNX + `.tokenizer.gguf` + `.embed.bin` → RKNN
    （`--ctx` 与 `--kv int4|int8|fp16` 可选）
  - `export_rknn.py` 暴露 KV 档位、`llm_head`、tie-embedding、`seq`、`core_num` 等旋钮
  - `build_datasets.sh` 从公开数据集重建 884 条校准集与 392 条留出集；`quant_eval_grq.py` 量化质量自检
  - `render_chat_template.py` + `check_chat_template.py`：与官方模板做**逐字节**一致性比对（8/11 一致，其余为有意偏差）
- **`deploy/`：常驻化**：systemd 单元模板（`Restart=always`、优雅 SIGTERM 收尾）、20 s 健康守护 timer
  （含"模型加载期不累计失败"的防自激设计）、`/etc/rk1828-spark.env` 集中配置、`start.sh` 包装、`install_service.sh`
- **`tools/`：19 个验证脚本**（评测、复读专项、长上下文阶梯、十一阶段压力、KV 策略扫描、温度采样等）
- **`docs/`：build / conversion / server / chat_template / benchmarks / testing / troubleshooting**

### 实测（详见 `docs/benchmarks.md`）

- 上下文：`ctx 73728` 档位可用（单次 prefill 53,166 token / 115.6 s，needle 命中）；int4 KV 硬上限 ≈ 79,900 token
- 性能：decode 50–52 tok/s（短上下文）；prefill 647 → 788 → 685 tok/s（88 / 4.3k / 14.8k token）
- 精度：392 题留出集 99/332 = 29.8 %；GRQ w4a16 与 bf16 top-1 一致率 96.16 %
- 复读：392 题真实复读 0 条；对抗性 48 组 runaway 2（均为 prompt 明确要求重复）
- 压力：31 项套件失败 1（模型行为，非服务缺陷）
- KV 精度 A/B：int4(28.0 %) 不比 fp16(26.0 %) 差 → 长上下文场景直接用 int4

### 修复

- **system 消息此前不会进入模型**：RKNN3 1.2.0 的 `rknn3_session_set_chat_template()` 不吃 `system_prompt`，
  客户端 `system` 被静默丢弃（`role:"developer"` 也一律降级成 user）。新增 `--system-in-prompt on`（默认）——
  整段 ChatML 由服务端拼进 prompt 文本，`tools/sysprobe.py` 从 1/6 变 6/6，`prompt_tokens` 开始包含 system。
  同时 `role:"developer"` 归入 system、遗留 `role:"function"` 归入 tool。
  详见 `docs/troubleshooting.md` §12、`docs/server.md` §4.2、`docs/benchmarks.md` §6.2。
- `finish_reason` 判截断改用"采样器是否选中 EOS"（避开 runtime 的 off-by-one，`max_tokens=64` 现在报 `ct=63/length`）。
- 归因更正：早期"`reasoning_effort=low` 比 `high` 高约 4 个百分点"实际是同一配置在 `max_tokens 1000 → 1200`
  下的差异，不是档位效果（当时改 system 文本的开关全是空操作）。

### 已知限制

- 单 session 串行；`n>1` 拒绝；`beam_search`、多模态输入、`/v1/embeddings`、audio、异步重叠取样未实现
- `llm_head` 的 w6a16 在 RKNN3 1.2.0 上匹配失败（`Match llm_head with 'auto' failed`），输出头实际按 w4a16 编译
- `thinking on` 时模型可能"只复述不作答"（`fib_probe` 命中 2/6）；同一送达方式下 thinking on 在留出集 32 % vs thinking off 16 %
