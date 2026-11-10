# Changelog

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

### 已知限制

- 单 session 串行；`n>1` 拒绝；`beam_search`、多模态输入、`/v1/embeddings`、audio、异步重叠取样未实现
- `llm_head` 的 w6a16 在 RKNN3 1.2.0 上匹配失败（`Match llm_head with 'auto' failed`），输出头实际按 w4a16 编译
- `thinking on` 时模型可能"只复述不作答"；`reasoning_effort=low` 在留出集上比 `high` 高约 4 个百分点
