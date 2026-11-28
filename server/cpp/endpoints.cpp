// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// endpoints.cpp — /v1/chat/completions、/v1/completions、/v1/responses、/v1/embeddings、运维端点与路由
#include "endpoints.h"

#include <cctype>
#include <sstream>

#include "openai.h"
#include "prompt.h"
#include "sse.h"

static void err_json(ResponseWriter& w, int status, const std::string& msg, const std::string& type,
                     const std::string& code) {
  json e;
  e["error"]["message"] = msg;
  e["error"]["type"] = type;
  if (!code.empty()) {
    e["error"]["code"] = code;
  }
  w.write_full(status, safe_dump(e), "application/json");
}

// 队列已满：直接 429，不进入排队
static bool reject_if_queue_full(ResponseWriter& w) {
  if (g_waiting.load() < g_max_queue) {
    return false;
  }
  err_json(w, 429, "inference queue is full, retry later", "rate_limit_error", "");
  return true;
}

// 采样与生成长度的统一下发（chat / completions / responses 共用）
static void apply_gen_params(const ChatParams& cp, GenRequest& gr) {
  gr.max_new_tokens = cp.max_new;
  gr.temperature = cp.temperature;
  gr.top_k = cp.top_k;
  gr.top_p = cp.top_p;
  gr.repeat_penalty = cp.repeat_penalty;
  gr.frequency_penalty = cp.frequency_penalty;
  gr.presence_penalty = cp.presence_penalty;
  gr.seed = cp.seed;
  gr.thinking = cp.thinking;
  gr.use_runtime_sampler = g_cfg.use_runtime_sampler;
  gr.stop_strings = cp.stops;
  gr.timeout_ms = g_cfg.request_timeout_sec * 1000;
}

// 用真实词表计数并夹紧生成预算；超窗时写 400 并返回 false
static bool check_prompt_budget(const PromptParts& pp, ChatParams& cp, ResponseWriter& w) {
  std::vector<int32_t> ptoks;
  int ntok = g_engine.encode_tokens(pp.text, ptoks);
  int max_ctx = g_engine.model_max_ctx();
  if (max_ctx <= 0 || ntok <= 0) {
    return true;
  }
  if (ntok >= max_ctx - 8) {
    err_json(w, 400,
             "prompt is " + std::to_string(ntok) + " tokens, exceeding the loaded context window (" +
                 std::to_string(max_ctx) + " tokens)",
             "invalid_request_error", "context_length_exceeded");
    return false;
  }
  if (cp.max_new > 0 && (long)ntok + cp.max_new > (long)max_ctx - 8) {
    cp.max_new = (int)((long)max_ctx - 8 - ntok);
    if (cp.max_new < 1) {
      cp.max_new = 1;
    }
  }
  return true;
}

// chat.completion 的 usage（流式与非流式相同）
static json chat_usage(const GenResult& res, const SlotPlan& plan, long t0) {
  json usage;
  usage["prompt_tokens"] = res.prompt_tokens;
  usage["completion_tokens"] = res.completion_tokens;
  usage["total_tokens"] = res.prompt_tokens + res.completion_tokens;
  usage["prompt_tokens_details"]["cached_tokens"] = res.reused_tokens;
  usage["prefill_ms"] = (long)(res.prefill_ms + 0.5);
  usage["decode_ms"] = (long)(res.decode_ms + 0.5);
  usage["e2e_ms"] = (long)((double)((long)getCurrentTimeUs() - t0) / 1000.0 + 0.5);
  usage["tokens_per_second"] =
      res.decode_ms > 0 && res.completion_tokens > 0 ? (double)res.completion_tokens / (res.decode_ms / 1000.0) : 0.0;
  if (plan.slot >= 0) {
    usage["kv_slot"] = plan.slot;
    usage["kv_prefix_tokens"] = plan.prefix_tokens;
    usage["kv_image_loaded"] = plan.loaded ? 1 : 0;
  }
  if (res.veto_hits > 0) {
    usage["degeneration_veto"] = res.veto_hits;
  }
  if (res.loop_hits > 0) {
    usage["degeneration_loop"] = res.loop_hits;
  }
  if (res.degeneration_guard) {
    usage["degeneration_guard"] = 1;
  }
  return usage;
}

static json responses_usage(const GenResult& res) {
  json u;
  u["input_tokens"] = res.prompt_tokens;
  u["output_tokens"] = res.completion_tokens;
  u["total_tokens"] = res.prompt_tokens + res.completion_tokens;
  u["input_tokens_details"]["cached_tokens"] = res.reused_tokens;
  return u;
}

// 排队 + 槽位换入 + 生成 + 槽位提交，全部在 FIFO 闸门内完成
static int run_core(const PromptParts& pp, GenRequest& gr, StreamParser& parser, GenResult& res, SlotPlan& plan) {
  int ret = -1;
  g_waiting++;
  {
    FifoGuard guard(g_engine_gate);
    g_waiting--;
    g_m.active++;
    if (g_slots.enabled()) {
      plan = g_slots.acquire(g_engine, pp.text);
      if (plan.slot >= 0) {
        gr.cached_prefix_tokens = plan.prefix_tokens;
        gr.keep_kv_after = true;
        if (!plan.send_text.empty()) {
          gr.text = plan.send_text;
        }
      }
    }
    ret = g_engine.generate(gr, res, &parser);
    if (g_slots.enabled() && plan.slot >= 0) {
      if (ret == RKNN3_SUCCESS) {
        g_slots.commit(plan.slot, pp.text, res.prompt_tokens, res.text, res.completion_tokens,
                       res.kv_total_tokens, res.prompt_token_ids, res.prompt_ids_suffix, plan.prefix_tokens);
      } else {
        g_slots.invalidate(g_engine, plan.slot);
      }
    }
    g_m.active--;
  }
  parser.finish();
  return ret;
}

static void do_chat(const HttpRequest& rq, ResponseWriter& w) {
  json body;
  try {
    body = json::parse(rq.body);
  } catch (const std::exception& e) {
    err_json(w, 400, std::string("invalid json: ") + e.what(), "invalid_request_error", "");
    return;
  }
  if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
    err_json(w, 400, "'messages' must be a non-empty array", "invalid_request_error", "");
    return;
  }
  ChatParams cp = parse_params(body);
  if (cp.err) {
    err_json(w, 400, cp.err_msg, "invalid_request_error", cp.err_code);
    return;
  }
  json tools_arg = (body.contains("tools") && body["tools"].is_array()) ? body["tools"] : json::array();
  bool inject_effort = (cp.inject_effort >= 0) ? (cp.inject_effort == 1) : g_cfg.thinking_effort_inject;
  PromptParts pp = compose_spark(body["messages"], cp.thinking, g_cfg.system_prompt, tools_arg, cp.effort,
                                 inject_effort, g_cfg.system_in_prompt, cp.forced_tool, cp.require_tool,
                                 cp.single_tool);
  if (!pp.ok) {
    err_json(w, 400, pp.err, "invalid_request_error", "");
    return;
  }

  if (body.contains("n") && body["n"].is_number() && body["n"].get<int>() > 1) {
    err_json(w, 400,
             "n>1 is not supported by this server (single NPU session per request); use n=1",
             "invalid_request_error", "");
    return;
  }

  // prompt 预算：用真实词表计数，超窗直接 400（不能靠字符数估）
  if (!check_prompt_budget(pp, cp, w)) {
    return;
  }

  GenRequest gr;
  apply_gen_params(cp, gr);
  gr.system_prompt = pp.system;
  gr.prompt_prefix = pp.prefix;
  gr.prompt_postfix = pp.postfix;
  gr.text = pp.text;
  gr.role = (pp.last_role == "tool") ? "tool" : "user";
  gr.use_tools = cp.use_tools;
  gr.tools_json = cp.tools_json;
  gr.relax_guard = cp.use_tools;
  gr.want_logprobs = cp.want_logprobs;
  gr.top_logprobs = cp.top_logprobs;
  gr.logit_bias_ids = cp.lb_ids;
  gr.logit_bias_vals = cp.lb_vals;
  bool think_grammar_on = (cp.thinking_grammar >= 0) ? (cp.thinking_grammar == 1) : g_cfg.thinking_grammar;
  if (build_runtime_grammar(body.contains("response_format") ? body["response_format"] : json(), cp.use_grammar,
                            cp.thinking, cp.forced_tool, cp.require_tool, g_cfg.tool_grammar, think_grammar_on,
                            gr.grammar_str, gr.grammar_type)) {
    gr.use_grammar = true;
  }

  std::string id = rand_id();
  long rid = ++g_req_id;
  bool streaming = cp.stream;
  bool json_mode = body.contains("response_format") &&
                   (body["response_format"].value("type", std::string("")) == "json_object" ||
                    body["response_format"].value("type", std::string("")) == "json_schema");
  if (json_mode && cp.use_tools) {
    // 两种约束没法同时满足；按官方优先级，JSON 模式优先，工具仅保留定义
    LOGF("REQ %ld: response_format=%s 与 tools 同时出现，按 JSON 模式约束输出", rid,
         body["response_format"].value("type", std::string("")).c_str());
  }

  // 流式通道：思考分离只在 thinking 打开时生效
  bool split_reason = (g_cfg.reasoning_format == "deepseek") && cp.thinking;
  if (body.contains("reasoning_format") && body["reasoning_format"].is_string()) {
    split_reason = (body["reasoning_format"].get<std::string>() == "deepseek") && cp.thinking;
  }

  if (reject_if_queue_full(w)) {
    return;
  }

  long t0 = (long)getCurrentTimeUs();
  GenResult res;
  SlotPlan plan;
  int ret = -1;

  if (streaming) {
    if (!w.write_sse_start()) {
      return;
    }
    ChatSse csse(w, id, g_cfg.model_alias, g_cfg.stream_tool_parse);
    csse.emit_role(cp.use_tools);
    StreamParser parser(csse, split_reason, cp.use_tools);
    ret = run_core(pp, gr, parser, res, plan);
    bool ok = (ret == RKNN3_SUCCESS);
    std::string finish = ok ? (parser.tool_count() > 0 ? std::string("tool_calls") : res.finish_reason)
                            : std::string("error");
    json usage = chat_usage(res, plan, t0);
    g_m.record((double)((long)getCurrentTimeUs() - t0) / 1000.0, res.prompt_tokens, res.completion_tokens, ok,
               res.degeneration_guard, res.veto_hits, true, parser.tool_count() > 0);
    LOGF("REQ %ld stream prompt=%ld completion=%ld finish=%s prefill=%.1fms decode=%.1fms slot=%d reused=%ld "
         "tools=%d ret=%d",
         rid, res.prompt_tokens, res.completion_tokens, finish.c_str(), res.prefill_ms, res.decode_ms, plan.slot,
         res.reused_tokens, parser.tool_count(), ret);
    if (!ok) {
      ordered_json e;
      e["error"]["message"] = "inference failed ret=" + std::to_string(ret);
      e["error"]["type"] = "server_error";
      e["error"]["code"] = 500;
      w.write_sse_data(safe_dump(e));
    } else {
      csse.emit_final(finish, cp.include_usage ? &usage : NULL);
    }
    w.write_sse_end();
    return;
  }

  AccHandler acc;
  StreamParser parser(acc, split_reason, cp.use_tools);
  ret = run_core(pp, gr, parser, res, plan);
  bool ok = (ret == RKNN3_SUCCESS);

  json usage = chat_usage(res, plan, t0);
  g_m.record((double)((long)getCurrentTimeUs() - t0) / 1000.0, res.prompt_tokens, res.completion_tokens, ok,
             res.degeneration_guard, res.veto_hits, false, !acc.calls.empty());
  LOGF("REQ %ld json prompt=%ld completion=%ld finish=%s prefill=%.1fms decode=%.1fms slot=%d reused=%ld tools=%d ret=%d",
       rid, res.prompt_tokens, res.completion_tokens, res.finish_reason.c_str(), res.prefill_ms, res.decode_ms,
       plan.slot, res.reused_tokens, (int)acc.calls.size(), ret);

  if (!ok) {
    err_json(w, 500, "inference failed ret=" + std::to_string(ret), "server_error", "");
    return;
  }

  std::string content = acc.text;
  std::string reasoning = acc.reasoning;
  // 槽位续接依赖"客户端回放的内容 == 原始生成文本"，所以开启槽位时不做任何裁剪
  if (!g_slots.enabled()) {
    trim_front(content);
    endswith_trim_special(content);
    trim_front(reasoning);
    if (reasoning.compare(0, strlen(TAG_THINK_OPEN), TAG_THINK_OPEN) == 0) {
      reasoning.erase(0, strlen(TAG_THINK_OPEN));
    }
  }
  if (!split_reason) {
    reasoning.clear();  // 没有思考通道时模型正文就是 content
  }
  bool has_tools = !acc.calls.empty();
  std::string finish = has_tools ? std::string("tool_calls") : res.finish_reason;

  ordered_json resp;
  resp["id"] = id;
  resp["object"] = "chat.completion";
  resp["created"] = (long)time(NULL);
  resp["model"] = body.contains("model") && body["model"].is_string() ? body["model"].get<std::string>()
                                                                      : g_cfg.model_alias;
  resp["system_fingerprint"] = g_fingerprint;
  ordered_json msg;
  msg["role"] = "assistant";
  msg["content"] = has_tools ? ordered_json(nullptr) : ordered_json(content);
  if (split_reason && !reasoning.empty()) {
    msg["reasoning_content"] = reasoning;
  }
  if (has_tools) {
    ordered_json calls = ordered_json::array();
    for (size_t i = 0; i < acc.calls.size(); ++i) {
      calls.push_back(tool_call_json(acc.calls[i].id, acc.calls[i].name, acc.calls[i].args));
    }
    msg["tool_calls"] = calls;
  }
  ordered_json choice;
  choice["index"] = 0;
  choice["message"] = msg;
  choice["logprobs"] = nullptr;
  if (cp.want_logprobs && !res.logprobs.empty()) {
    ordered_json lp;
    ordered_json content_lp = ordered_json::array();
    for (size_t i = 0; i < res.logprobs.size(); ++i) {
      ordered_json item;
      int tid = (i < res.gen_tokens.size()) ? res.gen_tokens[i] : -1;
      item["token"] = tid >= 0 ? g_engine.token_piece(tid) : "";
      item["token_id"] = tid;
      item["logprob"] = res.logprobs[i].empty() ? 0.0f : res.logprobs[i][0].logprob;
      if (res.logprobs[i].size() > 1) {
        ordered_json tl = ordered_json::array();
        for (size_t k = 0; k < res.logprobs[i].size(); ++k) {
          ordered_json t;
          int id2 = res.logprobs[i][k].token_id;
          t["token"] = g_engine.token_piece(id2);
          t["token_id"] = id2;
          t["logprob"] = res.logprobs[i][k].logprob;
          tl.push_back(t);
        }
        item["top_logprobs"] = tl;
      }
      content_lp.push_back(item);
    }
    lp["content"] = content_lp;
    choice["logprobs"] = lp;
  }
  choice["finish_reason"] = finish;
  resp["choices"] = ordered_json::array({choice});
  resp["usage"] = usage;
  w.write_full(200, safe_dump(resp), "application/json");
}

// /v1/completions（传统补全）：prompt 原样作为用户轮文本
static void do_completions(const HttpRequest& rq, ResponseWriter& w) {
  json body;
  try {
    body = json::parse(rq.body);
  } catch (...) {
    err_json(w, 400, "invalid json", "invalid_request_error", "");
    return;
  }
  if (!body.contains("prompt")) {
    err_json(w, 400, "'prompt' is required", "invalid_request_error", "");
    return;
  }
  std::string prompt;
  if (body["prompt"].is_string()) {
    prompt = body["prompt"].get<std::string>();
  } else if (body["prompt"].is_array() && body["prompt"].size() == 1 && body["prompt"][0].is_string()) {
    prompt = body["prompt"][0].get<std::string>();
  } else {
    err_json(w, 400, "'prompt' must be a string (or 1-element array)", "invalid_request_error", "");
    return;
  }

  ChatParams cp = parse_params(body);
  if (cp.err) {
    err_json(w, 400, cp.err_msg, "invalid_request_error", cp.err_code);
    return;
  }
  GenRequest gr;
  gr.system_prompt.clear();  // 空 system：prompt 完全由客户端控制
  gr.prompt_prefix.clear();
  gr.prompt_postfix.clear();
  gr.text = prompt;
  apply_gen_params(cp, gr);
  gr.use_grammar = cp.use_grammar;
  gr.grammar_str = cp.grammar_str;
  gr.grammar_type = cp.grammar_type;

  if (reject_if_queue_full(w)) {
    return;
  }
  GenResult res;
  SlotPlan plan;
  int ret;
  long t0 = (long)getCurrentTimeUs();
  g_waiting++;
  {
    FifoGuard guard(g_engine_gate);
    g_waiting--;
    g_m.active++;
    ret = g_engine.generate(gr, res, NULL);
    g_m.active--;
  }
  double e2e_ms = (double)((long)getCurrentTimeUs() - t0) / 1000.0;
  g_m.record(e2e_ms, res.prompt_tokens, res.completion_tokens, ret == RKNN3_SUCCESS, res.degeneration_guard,
             res.veto_hits, false, false);
  if (ret != RKNN3_SUCCESS) {
    err_json(w, 500, "inference failed ret=" + std::to_string(ret), "server_error", "");
    return;
  }
  std::string text = res.text;
  endswith_trim_special(text);
  ordered_json resp;
  resp["id"] = "cmpl-" + std::to_string((long)time(NULL));
  resp["object"] = "text_completion";
  resp["created"] = (long)time(NULL);
  resp["model"] = g_cfg.model_alias;
  resp["system_fingerprint"] = g_fingerprint;
  ordered_json item;
  item["text"] = text;
  item["index"] = 0;
  item["finish_reason"] = res.finish_reason;
  resp["choices"] = ordered_json::array({item});
  resp["usage"] = {{"prompt_tokens", res.prompt_tokens},
                   {"completion_tokens", res.completion_tokens},
                   {"total_tokens", res.prompt_tokens + res.completion_tokens}};
  w.write_full(200, safe_dump(resp), "application/json");
}

// ------------------------------------------------------------------ 其它端点
static void do_models(ResponseWriter& w, const std::string& req_id) {
  ordered_json item;
  item["id"] = g_cfg.model_alias;
  item["object"] = "model";
  item["created"] = g_start_time;
  item["owned_by"] = "rk1828";
  item["context_length"] = g_engine.model_max_ctx();
  item["max_context_length"] = g_engine.model_max_ctx();
  item["properties"] = ordered_json{{"context_length", g_engine.model_max_ctx()},
                                    {"max_tokens", g_cfg.max_new_tokens},
                                    {"max_output_tokens", g_cfg.max_new_tokens}};
  item["capabilities"] = ordered_json{{"chat", true},
                                      {"completions", true},
                                      {"responses", true},
                                      {"tools", true},
                                      {"tool_choice", true},
                                      {"response_format", true},
                                      {"stream", true},
                                      {"stream_options", true},
                                      {"logprobs", true},
                                      {"seed", true},
                                      {"thinking", true},
                                      {"embeddings", g_cfg.embeddings}};
  item["supported_parameters"] = ordered_json::array(
      {"messages", "model", "stream", "stream_options", "max_tokens", "max_completion_tokens", "temperature", "top_p",
       "top_k", "stop", "seed", "frequency_penalty", "presence_penalty", "repeat_penalty", "logprobs", "top_logprobs",
       "logit_bias", "tools", "tool_choice", "parallel_tool_calls", "response_format", "reasoning_effort", "thinking",
       "chat_template_kwargs", "n"});
  ordered_json meta;
  meta["model_type"] = g_engine.model_type();
  meta["max_context_len"] = g_engine.model_max_ctx();
  meta["vocab_size"] = g_engine.vocab_size();
  meta["quant"] = "w4a16 + GRQ + group32 (weights), Int4_to_F16/GroupQuant/group16/resid64 (KV cache)";
  meta["kv_slots"] = g_cfg.kv_slots;
  meta["kv_slot_dir"] = g_cfg.kv_slot_dir;
  meta["kv_slot_min_tokens"] = g_cfg.kv_slot_min_tokens;
  meta["kv_slot_max_image_mb"] = g_cfg.kv_slot_max_image_mb;
  meta["kv_slot_max_total_mb"] = g_cfg.kv_slot_max_total_mb;
  meta["kv_reuse_probe"] = g_kv_probe;
  meta["sampling_defaults"] = {{"temperature", g_cfg.temperature},
                               {"top_k", g_cfg.top_k},
                               {"top_p", g_cfg.top_p},
                               {"repeat_penalty", g_cfg.repeat_penalty},
                               {"frequency_penalty", g_cfg.frequency_penalty},
                               {"presence_penalty", g_cfg.presence_penalty},
                               {"max_new_tokens", g_cfg.max_new_tokens},
                               {"max_new_tokens_cap", g_cfg.max_new_tokens_cap},
                               {"thinking", g_cfg.default_thinking},
                               {"reasoning_format", g_cfg.reasoning_format},
                               {"thinking_effort_inject", g_cfg.thinking_effort_inject},
                               {"thinking_grammar", g_cfg.thinking_grammar},
                               {"system_in_prompt", g_cfg.system_in_prompt}};
  if (!g_cfg.reasoning_effort.empty()) {
    meta["sampling_defaults"]["reasoning_effort"] = g_cfg.reasoning_effort;
  }
  meta["sampling_defaults_note"] =
      "默认取自模型厂商 XHToken/Spark-X2.6-4B 的 generation_config.json（temp 1.0 / top_p 0.95 / top_k -1 / "
      "penalties 关闭 / do_sample true / max_tokens=1048576）；本部署按用途刻意保持 repeat_penalty=1.0（不加重复惩罚）；"
      "max_new_tokens=0 表示不限制输出长度，生成到 EOS 或上下文用完为止";
  item["meta"] = meta;
  if (!req_id.empty()) {
    // GET /v1/models/{id}：OpenAI 语义是返回单个 model 对象，不认识的就 404
    if (req_id != g_cfg.model_alias) {
      err_json(w, 404, "model not found: " + req_id, "invalid_request_error", "model_not_found");
      return;
    }
    w.write_full(200, safe_dump(item), "application/json");
    return;
  }
  ordered_json resp;
  resp["object"] = "list";
  resp["data"] = ordered_json::array({item});
  w.write_full(200, safe_dump(resp), "application/json");
}


static void do_health(ResponseWriter& w, bool head_only) {
  if (!g_ready.load()) {
    if (head_only) {
      w.write_status_only(503, "application/json");
    } else {
      w.write_full(503, "{\"error\":{\"code\":503,\"message\":\"Loading model\",\"type\":\"unavailable_error\"}}",
                   "application/json");
    }
    return;
  }
  if (head_only) {
    w.write_status_only(200, "application/json");
  } else {
    w.write_full(200, "{\"status\":\"ok\"}", "application/json");
  }
}

static void do_metrics(ResponseWriter& w) { w.write_full(200, g_m.snapshot().dump(2), "application/json"); }

static void do_metrics_text(ResponseWriter& w) {
  json j = g_m.snapshot();
  std::ostringstream o;
  o << "# rknn_spark_server metrics\n";
  for (json::iterator it = j.begin(); it != j.end(); ++it) {
    if (it.value().is_number()) {
      o << "rkllm_" << it.key() << " " << it.value() << "\n";
    }
  }
  w.write_full(200, o.str(), "text/plain; version=0.0.4");
}

static void do_slots(ResponseWriter& w) {
  std::vector<SlotState> snap = g_slots.snapshot();
  ordered_json resp;
  resp["object"] = "list";
  resp["kv_enabled"] = g_slots.enabled();
  resp["kv_reuse_probe"] = g_kv_probe;
  resp["kv_dir"] = g_slots.dir();
  resp["image_bytes_total"] = g_slots.image_bytes();
  ordered_json arr = ordered_json::array();
  if (snap.empty()) {
    ordered_json slot;
    slot["id"] = 0;
    slot["n_ctx"] = g_engine.model_max_ctx();
    slot["state"] = g_m.active.load() > 0 ? "running" : "idle";
    arr.push_back(slot);
  } else {
    for (size_t i = 0; i < snap.size(); ++i) {
      const SlotState& s = snap[i];
      ordered_json slot;
      slot["id"] = s.id;
      slot["n_ctx"] = g_engine.model_max_ctx();
      slot["state"] = s.resident ? (g_m.active.load() > 0 ? "running" : "resident") : (s.saved ? "swapped" : "empty");
      slot["prompt_tokens"] = s.prompt_tokens;
      slot["gen_tokens"] = s.gen_tokens;
      slot["hits"] = s.hits;
      slot["last_used"] = s.last_used;
      slot["image_saved"] = s.saved;
      slot["image_bytes"] = s.saved_bytes;
      slot["path"] = s.path;
      slot["save_ms"] = s.save_ms;
      slot["load_ms"] = s.load_ms;
      arr.push_back(slot);
    }
  }
  resp["data"] = arr;
  w.write_full(200, safe_dump(resp), "application/json");
}

// /slots/{id}/{action}
static void do_slot_action(const HttpRequest& rq, ResponseWriter& w) {
  // path: /slots/{id}/pause|resume|interrupt 或 /slots/{id}/kvcache/save|load
  std::vector<std::string> parts;
  {
    std::string seg;
    for (size_t k = 1; k < rq.path.size(); ++k) {
      if (rq.path[k] == '/') {
        if (!seg.empty()) {
          parts.push_back(seg);
        }
        seg.clear();
      } else {
        seg += rq.path[k];
      }
    }
    if (!seg.empty()) {
      parts.push_back(seg);
    }
  }
  std::string action = parts.empty() ? "" : parts.back();
  if (action != "pause" && action != "resume" && action != "interrupt" && action != "save" && action != "load") {
    err_json(w, 404, "unknown slot action", "not_found_error", "");
    return;
  }
  json body;
  if (!rq.body.empty()) {
    try {
      body = json::parse(rq.body);
    } catch (...) {
      body = json::object();
    }
  }
  int ret = -1;
  json out;
  out["id_slot"] = 0;
  {
    FifoGuard guard(g_engine_gate);
    if (action == "pause") {
      ret = g_engine.pause_session();
      out["state"] = "paused";
    } else if (action == "resume") {
      ret = g_engine.resume_session();
      out["state"] = "resumed";
    } else if (action == "interrupt") {
      ret = g_engine.stop_generation();
      out["state"] = "idle";
    } else if (action == "save") {
      std::string path = body.value("path", g_cfg.kvcache_dir + "/slot0.kv");
      ret = g_engine.save_kvcache(path);
      out["path"] = path;
      g_slots.reset(g_engine);  // 手工存盘后槽位记录不再可信
    } else {
      std::string path = body.value("path", std::string(""));
      if (path.empty()) {
        err_json(w, 400, "'path' is required", "invalid_request_error", "");
        return;
      }
      ret = g_engine.load_kvcache(path);
      out["path"] = path;
      g_slots.reset(g_engine);
    }
  }
  if (ret != 0) {
    err_json(w, 500, action + " failed ret=" + std::to_string(ret), "server_error", "");
    return;
  }
  out["success"] = true;
  w.write_full(200, safe_dump(out), "application/json");
}

// ------------------------------------------------------------------ /v1/responses
static std::string responses_text(const json& content) {
  if (content.is_string()) {
    return content.get<std::string>();
  }
  if (content.is_array()) {
    std::string out;
    for (size_t i = 0; i < content.size(); ++i) {
      const json& p = content[i];
      if (p.is_string()) {
        out += p.get<std::string>();
      } else if (p.is_object() && p.contains("text") && p["text"].is_string()) {
        out += p["text"].get<std::string>();
      }
    }
    return out;
  }
  if (content.is_object() && content.contains("text") && content["text"].is_string()) {
    return content["text"].get<std::string>();
  }
  return "";
}

static bool responses_input_to_messages(const json& input, ordered_json& messages, std::string& err) {
  messages = ordered_json::array();
  if (input.is_string()) {
    messages.push_back(ordered_json{{"role", "user"}, {"content", input.get<std::string>()}});
    return true;
  }
  if (input.is_null()) {
    return true;
  }
  if (!input.is_array()) {
    err = "'input' must be a string or an array of items";
    return false;
  }
  for (size_t i = 0; i < input.size(); ++i) {
    const json& it = input[i];
    if (it.is_string()) {
      messages.push_back(ordered_json{{"role", "user"}, {"content", it.get<std::string>()}});
      continue;
    }
    if (!it.is_object()) {
      continue;
    }
    std::string type = it.value("type", std::string(""));
    if (type == "function_call") {
      std::string call_id = it.value("call_id", it.value("id", std::string("call_" + rand_hex(12))));
      std::string name = it.value("name", std::string(""));
      std::string args = it.contains("arguments")
                             ? (it["arguments"].is_string() ? it["arguments"].get<std::string>()
                                                            : safe_dump(it["arguments"]))
                             : std::string("{}");
      ordered_json call;
      call["id"] = call_id;
      call["type"] = "function";
      call["function"]["name"] = name;
      call["function"]["arguments"] = args;
      if (!messages.empty() && messages.back().value("role", std::string()) == "assistant" &&
          messages.back().contains("tool_calls")) {
        messages.back()["tool_calls"].push_back(call);
      } else {
        messages.push_back(ordered_json{
            {"role", "assistant"}, {"content", ""}, {"tool_calls", ordered_json::array({call})}});
      }
    } else if (type == "function_call_output") {
      std::string call_id = it.value("call_id", std::string(""));
      std::string out = responses_text(it.contains("output") ? it["output"] : json());
      messages.push_back(ordered_json{{"role", "tool"}, {"tool_call_id", call_id}, {"content", out}});
    } else if (type == "reasoning") {
      continue;  // 推理项不回放（模型侧已由思考通道重新生成）
    } else if (it.contains("role")) {
      std::string role = it["role"].get<std::string>();
      json content = it.contains("content") ? it["content"] : json("");
      if (content.is_array()) {
        std::string text;
        for (size_t k = 0; k < content.size(); ++k) {
          text += responses_text(content[k]);
        }
        messages.push_back(ordered_json{{"role", role}, {"content", text}});
      } else {
        messages.push_back(ordered_json{{"role", role}, {"content", responses_text(content)}});
      }
    } else if (type == "input_text" || type == "text") {
      messages.push_back(ordered_json{{"role", "user"}, {"content", responses_text(it)}});
    }
  }
  return true;
}

static ordered_json responses_tools_to_chat(const json& tools) {
  ordered_json out = ordered_json::array();
  if (!tools.is_array()) {
    return out;
  }
  for (size_t i = 0; i < tools.size(); ++i) {
    const json& t = tools[i];
    if (!t.is_object()) {
      continue;
    }
    if (t.value("type", std::string("")) != "function") {
      continue;  // 内置工具（web_search 等）不支持
    }
    ordered_json f;
    f["type"] = "function";
    f["function"]["name"] = t.value("name", t.value("function", json::object()).value("name", std::string("")));
    f["function"]["description"] = t.value("description", std::string(""));
    if (t.contains("parameters")) {
      f["function"]["parameters"] = t["parameters"];
    } else if (t.contains("function") && t["function"].is_object() && t["function"].contains("parameters")) {
      f["function"]["parameters"] = t["function"]["parameters"];
    } else {
      f["function"]["parameters"] = ordered_json{{"type", "object"}, {"properties", ordered_json::object()}};
    }
    out.push_back(f);
  }
  return out;
}

static void do_responses(const HttpRequest& rq, ResponseWriter& w) {
  json body;
  try {
    body = json::parse(rq.body);
  } catch (const std::exception& e) {
    err_json(w, 400, std::string("invalid json: ") + e.what(), "invalid_request_error", "");
    return;
  }

  ordered_json messages;
  std::string err;
  if (!responses_input_to_messages(body.contains("input") ? body["input"] : json(), messages, err)) {
    err_json(w, 400, err, "invalid_request_error", "");
    return;
  }
  if (body.contains("instructions") && body["instructions"].is_string() &&
      !body["instructions"].get<std::string>().empty()) {
    ordered_json sys;
    sys["role"] = "system";
    sys["content"] = body["instructions"].get<std::string>();
    ordered_json with_sys = ordered_json::array();
    with_sys.push_back(sys);
    for (size_t i = 0; i < messages.size(); ++i) {
      with_sys.push_back(messages[i]);
    }
    messages = with_sys;
  }
  if (messages.empty()) {
    err_json(w, 400, "'input' must contain at least one message", "invalid_request_error", "");
    return;
  }

  // 拼一份等价的 chat 请求，复用同一条生成链路
  json cb;
  cb["messages"] = messages;
  cb["model"] = body.value("model", g_cfg.model_alias);
  cb["stream"] = body.value("stream", false);
  if (body.contains("temperature")) cb["temperature"] = body["temperature"];
  if (body.contains("top_p")) cb["top_p"] = body["top_p"];
  if (body.contains("max_output_tokens") && body["max_output_tokens"].is_number()) {
    cb["max_tokens"] = body["max_output_tokens"];
  }
  if (body.contains("parallel_tool_calls")) cb["parallel_tool_calls"] = body["parallel_tool_calls"];
  if (body.contains("tool_choice")) {
    const json& tc = body["tool_choice"];
    if (tc.is_object() && tc.value("type", std::string("")) == "function" && tc.contains("name")) {
      cb["tool_choice"] = ordered_json{{"type", "function"}, {"function", ordered_json{{"name", tc["name"]}}}};
    } else {
      cb["tool_choice"] = tc;
    }
  }
  if (body.contains("tools")) cb["tools"] = responses_tools_to_chat(body["tools"]);
  if (body.contains("reasoning") && body["reasoning"].is_object()) {
    std::string effort = body["reasoning"].value("effort", std::string(""));
    if (!effort.empty()) cb["reasoning_effort"] = effort;
  }
  // Responses 的文本格式在 text.format；映射回 chat 的 response_format
  if (body.contains("text") && body["text"].is_object() && body["text"].contains("format") &&
      body["text"]["format"].is_object()) {
    const json& f = body["text"]["format"];
    std::string ft = f.value("type", std::string(""));
    if (ft == "json_object") {
      cb["response_format"] = ordered_json{{"type", "json_object"}};
    } else if (ft == "json_schema") {
      ordered_json rf;
      rf["type"] = "json_schema";
      rf["json_schema"]["schema"] = f.contains("schema") ? f["schema"] : json::object();
      if (f.contains("name")) rf["json_schema"]["name"] = f["name"];
      cb["response_format"] = rf;
    } else if (!ft.empty() && ft != "text") {
      err_json(w, 400, "unsupported text.format.type: " + ft, "invalid_request_error", "");
      return;
    }
  }
  if (body.contains("metadata")) cb["metadata"] = body["metadata"];

  ChatParams cp = parse_params(cb);
  if (cp.err) {
    err_json(w, 400, cp.err_msg, "invalid_request_error", cp.err_code);
    return;
  }
  json tools_arg = cb.contains("tools") && cb["tools"].is_array() ? cb["tools"] : json::array();
  bool inject_effort = (cp.inject_effort >= 0) ? (cp.inject_effort == 1) : g_cfg.thinking_effort_inject;
  PromptParts pp = compose_spark(messages, cp.thinking, g_cfg.system_prompt, tools_arg, cp.effort, inject_effort,
                                 g_cfg.system_in_prompt, cp.forced_tool, cp.require_tool, cp.single_tool);
  if (!pp.ok) {
    err_json(w, 400, pp.err, "invalid_request_error", "");
    return;
  }

  if (!check_prompt_budget(pp, cp, w)) {
    return;
  }

  GenRequest gr;
  apply_gen_params(cp, gr);
  gr.system_prompt = pp.system;
  gr.prompt_prefix = pp.prefix;
  gr.prompt_postfix = pp.postfix;
  gr.text = pp.text;
  gr.role = (pp.last_role == "tool") ? "tool" : "user";
  gr.use_tools = cp.use_tools;
  gr.tools_json = cp.tools_json;
  gr.relax_guard = cp.use_tools;
  bool think_grammar_on = (cp.thinking_grammar >= 0) ? (cp.thinking_grammar == 1) : g_cfg.thinking_grammar;
  if (build_runtime_grammar(cb.contains("response_format") ? cb["response_format"] : json(), cp.use_grammar,
                            cp.thinking, cp.forced_tool, cp.require_tool, g_cfg.tool_grammar, think_grammar_on,
                            gr.grammar_str, gr.grammar_type)) {
    gr.use_grammar = true;
  }

  std::string rid = "resp_" + rand_hex(24);
  bool streaming = cp.stream;
  bool split_reason = (g_cfg.reasoning_format == "deepseek") && cp.thinking;

  if (reject_if_queue_full(w)) {
    return;
  }

  long t0 = (long)getCurrentTimeUs();
  GenResult res;
  SlotPlan plan;
  int ret;

  if (streaming) {
    if (!w.write_sse_start()) {
      return;
    }
    ResponsesSse rsse(w, rid, g_cfg.model_alias);
    rsse.emit_created();
    StreamParser parser(rsse, split_reason, cp.use_tools && g_cfg.stream_tool_parse);
    ret = run_core(pp, gr, parser, res, plan);
    bool ok = (ret == RKNN3_SUCCESS);
    json usage = responses_usage(res);
    if (ok) {
      rsse.finish(usage, "completed");
    }
    g_m.record((double)((long)getCurrentTimeUs() - t0) / 1000.0, res.prompt_tokens, res.completion_tokens, ok,
               res.degeneration_guard, res.veto_hits, true, rsse.tool_count() > 0);
    LOGF("RESP %s stream prompt=%ld completion=%ld slot=%d tools=%d ret=%d", rid.c_str(), res.prompt_tokens,
         res.completion_tokens, plan.slot, rsse.tool_count(), ret);
    w.write_sse_end();
    return;
  }

  AccHandler acc;
  StreamParser parser(acc, split_reason, cp.use_tools);
  ret = run_core(pp, gr, parser, res, plan);
  bool ok = (ret == RKNN3_SUCCESS);
  if (!ok) {
    err_json(w, 500, "inference failed ret=" + std::to_string(ret), "server_error", "");
    return;
  }
  std::string content = acc.text;
  trim_front(content);
  endswith_trim_special(content);
  g_m.record((double)((long)getCurrentTimeUs() - t0) / 1000.0, res.prompt_tokens, res.completion_tokens, ok,
             res.degeneration_guard, res.veto_hits, false, !acc.calls.empty());
  LOGF("RESP %s json prompt=%ld completion=%ld slot=%d tools=%d ret=%d", rid.c_str(), res.prompt_tokens,
       res.completion_tokens, plan.slot, (int)acc.calls.size(), ret);

  ordered_json output = ordered_json::array();
  for (size_t i = 0; i < acc.calls.size(); ++i) {
    ordered_json item;
    item["type"] = "function_call";
    item["id"] = "fc_" + rand_hex(16);
    item["call_id"] = acc.calls[i].id;
    item["name"] = acc.calls[i].name;
    item["arguments"] = acc.calls[i].args.empty() ? std::string("{}") : acc.calls[i].args;
    item["status"] = "completed";
    output.push_back(item);
  }
  if (!content.empty() || acc.calls.empty()) {
    ordered_json item;
    item["type"] = "message";
    item["id"] = "msg_" + rand_hex(16);
    item["status"] = "completed";
    item["role"] = "assistant";
    item["content"] = ordered_json::array(
        {ordered_json{{"type", "output_text"}, {"text", content}, {"annotations", ordered_json::array()}}});
    output.push_back(item);
  }
  ordered_json resp;
  resp["id"] = rid;
  resp["object"] = "response";
  resp["created_at"] = (long)time(NULL);
  resp["status"] = res.finish_reason == "length" ? "incomplete" : "completed";
  if (res.finish_reason == "length") {
    resp["incomplete_details"] = ordered_json{{"reason", "max_output_tokens"}};
  }
  resp["model"] = g_cfg.model_alias;
  resp["output"] = output;
  resp["system_fingerprint"] = g_fingerprint;
  resp["usage"] = responses_usage(res);
  w.write_full(200, safe_dump(resp), "application/json");
}

// ------------------------------------------------------------------ /v1/embeddings
static void do_embeddings(const HttpRequest& rq, ResponseWriter& w) {
  if (!g_cfg.embeddings) {
    err_json(w, 404, "/v1/embeddings disabled", "invalid_request_error", "");
    return;
  }
  json body;
  try {
    body = json::parse(rq.body);
  } catch (...) {
    err_json(w, 400, "invalid json", "invalid_request_error", "");
    return;
  }
  if (!body.contains("input")) {
    err_json(w, 400, "'input' is required", "invalid_request_error", "");
    return;
  }
  std::vector<std::string> texts;
  if (body["input"].is_string()) {
    texts.push_back(body["input"].get<std::string>());
  } else if (body["input"].is_array()) {
    for (size_t i = 0; i < body["input"].size(); ++i) {
      if (body["input"][i].is_string()) {
        texts.push_back(body["input"][i].get<std::string>());
      }
    }
  }
  if (texts.empty()) {
    err_json(w, 400, "'input' must be a string or an array of strings", "invalid_request_error", "");
    return;
  }
  ordered_json data = ordered_json::array();
  long total_tokens = 0;
  for (size_t i = 0; i < texts.size(); ++i) {
    std::vector<float> vec;
    int ntok = 0;
    {
      FifoGuard guard(g_engine_gate);  // 词表在引擎里，读取时避免与生成/重载竞争
      ntok = g_engine.count_tokens(texts[i]);
      if (!g_engine.embed_mean(texts[i], vec)) {
        vec.clear();
      }
    }
    if (vec.empty()) {
      err_json(w, 400, "cannot embed input (empty text or tokenizer failure)", "invalid_request_error", "");
      return;
    }
    total_tokens += ntok;
    ordered_json item;
    item["object"] = "embedding";
    item["index"] = (int)i;
    ordered_json e = ordered_json::array();
    for (size_t k = 0; k < vec.size(); ++k) {
      e.push_back(vec[k]);
    }
    item["embedding"] = e;
    data.push_back(item);
  }
  ordered_json resp;
  resp["object"] = "list";
  resp["data"] = data;
  resp["model"] = g_cfg.model_alias + "-embed-mean";
  resp["usage"] = ordered_json{{"prompt_tokens", total_tokens}, {"total_tokens", total_tokens}};
  w.write_full(200, safe_dump(resp), "application/json");
}

// 常量时间比较：避免按字节提前返回泄露 key 前缀（长度不同仍会立刻返回，可接受）
static bool ct_equal(const std::string& a, const std::string& b) {
  if (a.size() != b.size()) {
    return false;
  }
  unsigned char diff = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    diff |= (unsigned char)(a[i] ^ b[i]);
  }
  return diff == 0;
}

// Authorization: Bearer <key>（scheme 不区分大小写）或 X-API-Key: <key>；key 必须完全一致
static bool api_key_ok(const HttpRequest& rq, const std::string& key) {
  if (ct_equal(hdr_of(rq, "x-api-key"), key)) {
    return true;
  }
  std::string auth = hdr_of(rq, "authorization");
  size_t sp = auth.find(' ');
  if (sp == std::string::npos) {
    return false;
  }
  std::string scheme = auth.substr(0, sp);
  std::transform(scheme.begin(), scheme.end(), scheme.begin(), ::tolower);
  return scheme == "bearer" && ct_equal(trim_copy(auth.substr(sp + 1)), key);
}

// ------------------------------------------------------------------ 路由
void dispatch(const HttpRequest& rq, ResponseWriter& w) {
  if (rq.method == "OPTIONS") {
    w.write_full(204, "", "text/plain");
    return;
  }
  // 健康检查不要求 api-key：外部的服务守护/容器探针拿不到密钥
  if ((rq.method == "GET" || rq.method == "HEAD") && (rq.path == "/health" || rq.path == "/healthz")) {
    do_health(w, rq.method == "HEAD");
    return;
  }
  if (!g_cfg.http.api_key.empty() && !api_key_ok(rq, g_cfg.http.api_key)) {
    err_json(w, 401, "missing or invalid api key", "authentication_error", "");
    return;
  }
  if (rq.method == "GET" && rq.path.compare(0, 11, "/v1/models/") == 0) {
    std::string mid = rq.path.size() > 11 ? rq.path.substr(11) : std::string();
    do_models(w, mid);
    return;
  }
  if (rq.method == "GET" && rq.path == "/metrics") {
    do_metrics_text(w);
    return;
  }
  if (rq.method == "GET" && rq.path == "/stats") {
    do_metrics(w);
    return;
  }
  if (rq.method == "GET" && (rq.path == "/v1/models" || rq.path == "/models")) {
    do_models(w, "");
    return;
  }
  if (rq.method == "GET" && (rq.path == "/v1/slots" || rq.path == "/slots")) {
    do_slots(w);
    return;
  }
  if (rq.method == "POST" && (rq.path == "/v1/chat/completions" || rq.path == "/chat/completions" ||
                              rq.path == "/v1/chat")) {
    do_chat(rq, w);
    return;
  }
  if (rq.method == "POST" && (rq.path == "/v1/responses" || rq.path == "/responses")) {
    do_responses(rq, w);
    return;
  }
  if (rq.method == "POST" && (rq.path == "/v1/embeddings" || rq.path == "/embeddings")) {
    do_embeddings(rq, w);
    return;
  }
  if (rq.method == "POST" && (rq.path == "/v1/completions" || rq.path == "/completions")) {
    if (!g_cfg.allow_raw_prompt) {
      err_json(w, 404, "/v1/completions disabled", "invalid_request_error", "");
      return;
    }
    do_completions(rq, w);
    return;
  }
  if (rq.method == "POST" && rq.path.size() > 6 && rq.path.compare(0, 6, "/slots") == 0) {
    do_slot_action(rq, w);
    return;
  }
  if (rq.method == "POST" && (rq.path == "/v1/kvcache/save" || rq.path == "/v1/kvcache/load")) {
    json body = json::object();
    if (!rq.body.empty()) {
      try {
        body = json::parse(rq.body);
      } catch (...) {
      }
    }
    std::string path = body.value("path", g_cfg.kvcache_dir + "/slot0.kv");
    int ret;
    {
      FifoGuard guard(g_engine_gate);
      ret = (rq.path == "/v1/kvcache/save") ? g_engine.save_kvcache(path) : g_engine.load_kvcache(path);
      g_slots.reset(g_engine);
    }
    if (ret != 0) {
      err_json(w, 500, "kvcache op failed ret=" + std::to_string(ret), "server_error", "");
      return;
    }
    json out;
    out["success"] = true;
    out["path"] = path;
    w.write_full(200, safe_dump(out), "application/json");
    return;
  }
  err_json(w, 404, "no route: " + rq.method + " " + rq.path, "not_found_error", "");
}
