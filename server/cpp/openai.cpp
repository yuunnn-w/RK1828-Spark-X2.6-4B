// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// openai.cpp — 请求参数解析与 runtime grammar 构造
#include "openai.h"

#include <cstring>

std::string tools_runtime_json(const json& tools) {
  // runtime 期望的就是 OpenAI 的 tools 数组本身
  return safe_dump(tools);
}

ordered_json tool_call_json(const std::string& id, const std::string& name, const std::string& args) {
  ordered_json call;
  call["id"] = id;
  call["type"] = "function";
  call["function"]["name"] = name;
  call["function"]["arguments"] = args.empty() ? std::string("{}") : args;
  return call;
}


// ------------------------------------------------------------------ response_format / tool_choice -> runtime grammar
static bool grammar_from_response_format(const json& rf, std::string& str, int& type) {
  if (!rf.is_object() || !rf.contains("type") || !rf["type"].is_string()) {
    return false;
  }
  std::string t = rf["type"].get<std::string>();
  if (t == "text") {
    return false;
  }
  if (t == "json_object") {
    type = RKNN3_GRAMMAR_BUILTIN_JSON;
    str = "";
    return true;
  }
  if (t == "json_schema") {
    type = RKNN3_GRAMMAR_JSON_SCHEMA;
    json schema;
    if (rf.contains("json_schema") && rf["json_schema"].is_object()) {
      const json& js = rf["json_schema"];
      schema = js.contains("schema") ? js["schema"] : js;
    } else if (rf.contains("schema")) {
      schema = rf["schema"];
    } else {
      return false;
    }
    str = safe_dump(schema);
    return true;
  }
  return false;
}

// 把 response_format / 强制的 tool_choice / 思考段的结束要求合成一份 runtime grammar。
// 官方优先级：json_schema > structural_tag > grammar；thinking 下必须把思考段包进同一份 grammar。
bool build_runtime_grammar(const json& rf, bool has_rf, bool thinking, const std::string& forced_tool,
                           bool require_tool, bool allow_tool_grammar, bool allow_thinking_grammar,
                           std::string& out, int& type) {
  out.clear();
  type = 0;
  bool want_json = has_rf;
  bool want_tool = allow_tool_grammar && !want_json && require_tool;
  // 没有别的约束但开着思考：可选地只要求"思考段 -> 结束标记 -> 自由回答"。
  // 1.2.0 实测这条不生效（运行时不匹配思考前缀），所以默认关着；留着开关是为了在新固件上重试。
  // 现象本身见 docs/server.md 4.3。
  bool want_think = thinking && !want_json && !want_tool && allow_thinking_grammar;
  if (!want_json && !want_tool && !want_think) {
    return false;
  }

  std::string json_str;
  int json_type = 0;
  if (want_json) {
    if (!grammar_from_response_format(rf, json_str, json_type)) {
      return false;
    }
    if (!thinking && json_type == RKNN3_GRAMMAR_JSON_SCHEMA) {
      // 不走 thinking：直接下 JSON Schema，简单可靠
      out = json_str;
      type = json_type;
      return true;
    }
  }

  ordered_json seq;
  seq["type"] = "sequence";
  ordered_json els = ordered_json::array();
  if (thinking) {
    ordered_json t;
    t["type"] = "tag";
    t["begin"] = "";
    t["content"] = ordered_json{{"type", "any_text"}};
    t["end"] = TAG_THINK_CLOSE;
    els.push_back(t);
    ordered_json ws;
    ws["type"] = "regex";
    ws["pattern"] = "[ \\n\\r\\t]*";
    els.push_back(ws);
  }
  if (want_json) {
    ordered_json e;
    e["type"] = "json_schema";
    if (json_type == RKNN3_GRAMMAR_BUILTIN_JSON || json_str.empty()) {
      e["json_schema"] = ordered_json{{"type", "object"}};
    } else {
      json schema;
      try {
        schema = json::parse(json_str);
      } catch (...) {
        return false;
      }
      e["json_schema"] = schema;
    }
    els.push_back(e);
  } else if (want_tool) {
    ordered_json e;
    e["type"] = "const_string";
    e["value"] = TAG_TOOL_OPEN;
    els.push_back(e);
    if (!forced_tool.empty()) {
      ordered_json n;
      n["type"] = "const_string";
      n["value"] = forced_tool;
      els.push_back(n);
    } else {
      ordered_json n;
      n["type"] = "regex";
      n["pattern"] = "[A-Za-z0-9_.:-]{1,64}";
      els.push_back(n);
    }
    ordered_json t;
    t["type"] = "tag";
    t["begin"] = "";
    t["content"] = ordered_json{{"type", "any_text"}};
    t["end"] = TAG_TOOL_CLOSE;
    els.push_back(t);
  } else {
    // 自由回答段：内容不设约束（允许普通正文，也允许工具调用的标记文本），
    // 但至少要有一个字符，所以 EOS 只在思考段闭合、并且回答已经开始之后才允许。
    ordered_json e;
    e["type"] = "regex";
    e["pattern"] = "[\\s\\S]+";
    els.push_back(e);
  }
  seq["elements"] = els;
  ordered_json root;
  root["type"] = "structural_tag";
  root["format"] = seq;
  out = safe_dump(root);
  type = RKNN3_GRAMMAR_STRUCTURAL_TAG;
  return true;
}


ChatParams parse_params(const json& body) {
  ChatParams c;
  c.temperature = g_cfg.temperature;
  c.top_p = g_cfg.top_p;
  c.top_k = g_cfg.top_k;
  c.repeat_penalty = g_cfg.repeat_penalty;
  c.frequency_penalty = g_cfg.frequency_penalty;
  c.presence_penalty = g_cfg.presence_penalty;
  c.max_new = g_cfg.max_new_tokens;
  c.seed = -1;
  c.thinking = g_cfg.default_thinking;
  c.stream = false;
  c.include_usage = false;
  c.want_logprobs = false;
  c.use_grammar = false;
  c.grammar_type = 0;
  c.use_tools = false;
  c.tool_choice = "auto";

  auto num = [&](const json& o, const char* k, float& dst) {
    if (o.contains(k) && o[k].is_number()) {
      dst = o[k].get<float>();
    }
  };
  num(body, "temperature", c.temperature);
  num(body, "top_p", c.top_p);
  num(body, "frequency_penalty", c.frequency_penalty);
  num(body, "presence_penalty", c.presence_penalty);

  json extra = (body.contains("extra_body") && body["extra_body"].is_object()) ? body["extra_body"] : json::object();
  if (extra.is_object()) {
    num(extra, "top_p", c.top_p);
    num(extra, "temperature", c.temperature);
    num(extra, "repeat_penalty", c.repeat_penalty);
    num(extra, "presence_penalty", c.presence_penalty);
    num(extra, "frequency_penalty", c.frequency_penalty);
    if (extra.contains("top_k") && extra["top_k"].is_number()) {
      c.top_k = extra["top_k"].get<int>();
    }
  }
  if (body.contains("top_k") && body["top_k"].is_number()) {
    c.top_k = body["top_k"].get<int>();
  }
  // OpenAI 的 repeat_penalty 也常放在 penalty 别名下
  num(body, "repeat_penalty", c.repeat_penalty);
  num(body, "repetition_penalty", c.repeat_penalty);

  if (c.temperature < 0.0f) {
    c.temperature = 0.0f;
  }
  if (c.temperature > 2.0f) {
    c.temperature = 2.0f;
  }
  if (c.top_p <= 0.0f || c.top_p > 1.0f) {
    c.top_p = 1.0f;
  }
  if (c.top_k < 0) {
    c.top_k = 0;
  }
  if (c.repeat_penalty < 1.0f) {
    c.repeat_penalty = 1.0f;
  }

  const char* keys[] = {"max_tokens", "max_completion_tokens", "n_predict", "max_new_tokens", NULL};
  for (int i = 0; keys[i] && c.max_new == g_cfg.max_new_tokens; ++i) {
    if (body.contains(keys[i]) && body[keys[i]].is_number()) {
      c.max_new = body[keys[i]].get<int>();
    }
  }
  if (extra.is_object()) {
    if (extra.contains("n_predict") && extra["n_predict"].is_number()) {
      c.max_new = extra["n_predict"].get<int>();
    }
  }
  if (c.max_new < 0) {
    c.max_new = 0;  // 客户端传 -1 也按"不限制"处理（与官方 -n -1 一致）
  }
  if (g_cfg.max_new_tokens_cap > 0 && c.max_new > g_cfg.max_new_tokens_cap) {
    c.max_new = g_cfg.max_new_tokens_cap;
  }

  if (body.contains("seed") && body["seed"].is_number()) {
    c.seed = body["seed"].get<long>();
  }
  if (body.contains("stream")) {
    c.stream = truthy(body["stream"], false);
  }
  if (body.contains("stream_options") && body["stream_options"].is_object()) {
    c.include_usage = truthy(body["stream_options"].value("include_usage", json(false)), false);
  }
  if (body.contains("logprobs")) {
    c.want_logprobs = truthy(body["logprobs"], false);
  }
  if (body.contains("top_logprobs") && body["top_logprobs"].is_number()) {
    c.top_logprobs = body["top_logprobs"].get<int>();
    if (c.top_logprobs < 0) {
      c.top_logprobs = 0;
    }
    if (c.top_logprobs > 8) {
      c.top_logprobs = 8;
    }
    if (c.top_logprobs > 0) {
      c.want_logprobs = true;
    }
  }
  if (body.contains("logit_bias") && body["logit_bias"].is_object()) {
    for (json::const_iterator it = body["logit_bias"].begin(); it != body["logit_bias"].end(); ++it) {
      if (!it.value().is_number()) {
        continue;
      }
      long id = strtol(it.key().c_str(), NULL, 10);
      if (id < 0 || id > 1000000) {
        continue;
      }
      float b = it.value().get<float>();
      if (b > 5.0f) {
        b = 5.0f;
      }
      if (b < -100.0f) {
        b = -100.0f;
      }
      c.lb_ids.push_back((int32_t)id);
      c.lb_vals.push_back(b);
    }
  }
  // reasoning_effort：本模型只有"想/不想"两档，none/minimal 关掉，其余打开
  {
    std::string effort = g_cfg.reasoning_effort;   // 服务端默认档位（空 = 跟官方模板一样默认 high）
    if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string()) {
      effort = body["reasoning_effort"].get<std::string>();
    }
    if (effort.empty() && extra.is_object() && extra.contains("reasoning_effort") &&
        extra["reasoning_effort"].is_string()) {
      effort = extra["reasoning_effort"].get<std::string>();
    }
    if (!effort.empty()) {
      c.thinking = !(effort == "none" || effort == "minimal");
      c.effort = effort;
    }
  }
  if (body.contains("thinking")) {
    c.thinking = truthy(body["thinking"], c.thinking);
  }
  if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
    const json& k = body["chat_template_kwargs"];
    if (k.contains("thinking")) {
      c.thinking = truthy(k["thinking"], c.thinking);
    }
    if (k.contains("enable_thinking")) {
      c.thinking = truthy(k["enable_thinking"], c.thinking);
    }
    // 关掉官方那句 "Now the system is invoked with `thinking_effort=…`" 的注入（实测它会被某些 prompt 复述）
    if (k.contains("thinking_effort_inject")) {
      c.inject_effort = truthy(k["thinking_effort_inject"], true) ? 1 : 0;
    }
    // 关掉"思考段必须闭合"的 grammar（只想对照旧行为时用）
    if (k.contains("thinking_grammar")) {
      c.thinking_grammar = truthy(k["thinking_grammar"], true) ? 1 : 0;
    }
  }
  if (extra.is_object() && extra.contains("enable_thinking")) {
    c.thinking = truthy(extra["enable_thinking"], c.thinking);
  }

  if (body.contains("stop")) {
    if (body["stop"].is_string()) {
      c.stops.push_back(body["stop"].get<std::string>());
    } else if (body["stop"].is_array()) {
      for (size_t i = 0; i < body["stop"].size(); ++i) {
        if (body["stop"][i].is_string()) {
          c.stops.push_back(body["stop"][i].get<std::string>());
        }
      }
    }
  }

  if (body.contains("response_format")) {
    const json& rf = body["response_format"];
    std::string t = (rf.is_object() && rf.contains("type") && rf["type"].is_string())
                        ? rf["type"].get<std::string>()
                        : std::string();
    if (t != "text" && t != "json_object" && t != "json_schema") {
      c.err = true;
      c.err_msg = "unsupported response_format.type: '" + t +
                  "' (this server supports text, json_object, json_schema)";
      c.err_code = "invalid_request_error";
      return c;
    }
    c.use_grammar = grammar_from_response_format(rf, c.grammar_str, c.grammar_type);
    if (t == "json_schema" && !c.use_grammar) {
      c.err = true;
      c.err_msg = "response_format.json_schema.schema is required for type=json_schema";
      c.err_code = "invalid_request_error";
      return c;
    }
  }
  if (body.contains("tools") && body["tools"].is_array() && !body["tools"].empty()) {
    c.use_tools = true;
    c.tools_json = tools_runtime_json(body["tools"]);
    if (body.contains("tool_choice")) {
      const json& tc = body["tool_choice"];
      if (tc.is_string()) {
        c.tool_choice = tc.get<std::string>();
      } else if (tc.is_object()) {
        if (tc.contains("function") && tc["function"].is_object() && tc["function"].contains("name") &&
            tc["function"]["name"].is_string()) {
          c.tool_choice = tc["function"]["name"].get<std::string>();
        } else if (tc.value("type", std::string("")) == "function" && tc.contains("name") && tc["name"].is_string()) {
          c.tool_choice = tc["name"].get<std::string>();
        }
      }
    }
    if (c.tool_choice == "required") {
      c.require_tool = true;
    } else if (c.tool_choice != "auto" && c.tool_choice != "none") {
      c.require_tool = true;
      c.forced_tool = c.tool_choice;
    }
    if (body.contains("parallel_tool_calls") && !truthy(body["parallel_tool_calls"], true)) {
      c.single_tool = true;
    }
  }
  if (c.use_tools && c.tool_choice == "none") {
    c.use_tools = false;  // 客户端显式禁用工具
  }
  return c;
}
