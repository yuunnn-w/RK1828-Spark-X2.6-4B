// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// main.cpp — rknn_spark_server：RK1828 板端生产级 OpenAI 兼容推理服务
//
// 端点：
//   POST /v1/chat/completions   （stream / 非 stream、tools、response_format、stop、penalties、logprobs）
//   POST /v1/completions        （OpenAI 传统补全，prompt 原样送入）
//   GET  /v1/models, /v1/models/{id}
//   GET  /health, /healthz, /metrics
//   POST /slots/{id}/pause|resume|interrupt, POST /slots/{id}/kvcache/save|load, GET /slots
//   GET/POST /v1/models/info（模型与运行时信息）
//
// 采样：默认贪心（temperature=0）+ 重复惩罚 + token 级复读熔断（周期块检测）+
//       片段级复读检测，保证交付文本不出现失控复读。

#include "engine.h"
#include "httpd.h"
#include "time_utils.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cctype>
#include <csignal>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "rknn3_api.h"

using nlohmann::json;
using nlohmann::ordered_json;

extern const char* now_str();

#define LOGF(fmt, ...)                                                                                               \
  do {                                                                                                               \
    fprintf(stderr, "[%s] " fmt "\n", now_str(), ##__VA_ARGS__);                                                    \
    fflush(stderr);                                                                                                  \
  } while (0)

static volatile bool g_requested_shutdown = false;
static HttpServer* g_srv = nullptr;
static void on_signal(int sig) { (void)sig; g_requested_shutdown = true; }

// ------------------------------------------------------------------ Spark 提示词标记
// 与 rknn3-model-zoo/examples/Spark_X2_5/cpp 里已验证可用的写法逐字一致
static const char* SPK_SOS = "<\uff5cstart\u2581of\u2581sentence\uff5c>";  // 词表 id 0
static const char* SPK_EOS = "<\uff5cend\u2581of\u2581sentence\uff5c>";    // 词表 id 1
static const char* SPK_SYS = "<|System|>\n";
static const char* SPK_USER = "<|User|>";
static const char* SPK_BOT = "<|Bot|>";

// ------------------------------------------------------------------ 配置
struct AppConfig {
  EngineConfig eng;
  HttpServerOptions http;

  // 默认采样参数对齐模型厂商官方推荐（ModelScope XHToken/Spark-X2.6-4B 的
  // generation_config.json 与 README 服务示例）：
  //   temperature=1.0  top_p=0.95  top_k=-1(=不截断，这里用 0 表示关闭)  repetition_penalty=1.0
  //   presence_penalty=0  frequency_penalty=0  do_sample=true  max_tokens=1048576（=不限制）
  // 注：rkllm3-server 1.2.0 的 --help 默认是 temp 0.8 / top_k 1 / top_p 0.9，那是 Rockchip 的默认，
  //     不是模型厂商的推荐；这里以模型厂商为准。
  float temperature = 1.0f;
  int top_k = 0;                 // 0/负数 = 不截断（官方 top_k=-1）
  float top_p = 0.95f;
  float repeat_penalty = 1.0f;   // 官方不惩罚重复；复读防线靠 sampling 熔断
  float frequency_penalty = 0.0f;
  float presence_penalty = 0.0f;
  int max_new_tokens = 0;        // 0 = 不限制（生成到 EOS 或上下文用完）
  int max_new_tokens_cap = 0;    // 0 = 不对客户端给的 max_tokens 设上限
  int max_queue = 64;
  int request_timeout_sec = 0;  // 单次生成墙钟上限（0=不限）
  bool default_thinking = true;  // 官方 --reasoning auto：Spark 模板默认 enable_thinking=true
  bool use_runtime_sampler = false;
  bool allow_raw_prompt = true;  // /v1/completions
  std::string model_alias = "Spark-X2.6-4B";
  std::string system_prompt = "you are a helpful assistant.";
  std::string reasoning_format = "none";  // none | deepseek
  std::string reasoning_effort;          // 默认思考档位：空 = 官方模板默认 high；可填 high|low|none|minimal
  bool thinking_effort_inject = true;    // 官方模板会把 "Now the system is invoked with thinking_effort=..." 写进 system
  std::string kvcache_dir = "/run/rknn3/kvcache";
  std::string params_file;
};

static AppConfig g_cfg;
static Engine g_engine;
static std::mutex g_engine_mu;  // 单 session：串行化生成
static std::atomic<bool> g_ready(false);
static std::atomic<long> g_req_id(0);
static std::atomic<long> g_waiting(0);
static int g_max_queue = 64;
static long g_start_time = 0;

// ------------------------------------------------------------------ 指标
struct Metrics {
  std::atomic<long> req_total{0}, req_ok{0}, req_err{0}, req_stream{0}, req_tool{0}, req_guard{0}, req_veto{0};
  std::atomic<long> tok_in{0}, tok_out{0}, conns{0}, active{0};
  std::atomic<long> lat_us_sum{0}, lat_us_max{0};
  std::mutex mu;
  std::vector<double> lat;  // 最近若干条端到端延迟（ms）

  void record(double ms, long tin, long tout, bool ok, bool guard, int veto, bool stream, bool tool) {
    req_total++;
    if (ok) {
      req_ok++;
    } else {
      req_err++;
    }
    if (stream) {
      req_stream++;
    }
    if (tool) {
      req_tool++;
    }
    if (guard) {
      req_guard++;
    }
    if (veto > 0) {
      req_veto += veto;
    }
    tok_in += tin;
    tok_out += tout;
    long us = (long)(ms * 1000.0);
    lat_us_sum += us;
    long prev = lat_us_max.load();
    while (us > prev && !lat_us_max.compare_exchange_weak(prev, us)) {
    }
    std::lock_guard<std::mutex> lk(mu);
    if (lat.size() >= 4096) {
      lat.erase(lat.begin(), lat.begin() + 2048);
    }
    lat.push_back(ms);
  }

  static double pct(std::vector<double> v, double p) {
    if (v.empty()) {
      return 0.0;
    }
    std::sort(v.begin(), v.end());
    size_t i = (size_t)(p * (double)(v.size() - 1));
    return v[i];
  }

  json snapshot() {
    std::vector<double> copy;
    {
      std::lock_guard<std::mutex> lk(mu);
      copy = lat;
    }
    json j;
    j["uptime_sec"] = (long)(time(NULL) - g_start_time);
    j["ready"] = g_ready.load();
    j["requests_total"] = req_total.load();
    j["requests_ok"] = req_ok.load();
    j["requests_error"] = req_err.load();
    j["requests_stream"] = req_stream.load();
    j["requests_tool_calls"] = req_tool.load();
    j["requests_degeneration_guard"] = req_guard.load();
    j["veto_hits_total"] = req_veto.load();
    j["prompt_tokens_total"] = tok_in.load();
    j["completion_tokens_total"] = tok_out.load();
    j["connections_total"] = conns.load();
    j["connections_active"] = conns_active();
    j["generating_now"] = active.load();
    j["queued_now"] = g_waiting.load();
    j["queue_limit"] = g_max_queue;
    j["latency_ms_avg"] = req_total.load() ? (double)lat_us_sum.load() / (double)req_total.load() / 1000.0 : 0.0;
    j["latency_ms_max"] = (double)lat_us_max.load() / 1000.0;
    j["latency_ms_p50"] = pct(copy, 0.50);
    j["latency_ms_p95"] = pct(copy, 0.95);
    j["latency_ms_p99"] = pct(copy, 0.99);
    j["model"] = g_cfg.model_alias;
    j["model_type"] = g_engine.model_type();
    j["max_context_len"] = g_engine.model_max_ctx();
    return j;
  }

  long conns_active();
};

static Metrics g_m;

// ------------------------------------------------------------------ 小工具
static std::string json_escape(const std::string& in) {
  std::string o;
  o.reserve(in.size() + 8);
  for (size_t i = 0; i < in.size(); ++i) {
    unsigned char c = (unsigned char)in[i];
    if (c == '"' || c == '\\') {
      o += '\\';
      o += (char)c;
    } else if (c == '\n') {
      o += "\\n";
    } else if (c == '\r') {
      o += "\\r";
    } else if (c == '\t') {
      o += "\\t";
    } else if (c < 0x20) {
      char buf[8];
      snprintf(buf, sizeof(buf), "\\u%04x", c);
      o += buf;
    } else {
      o += (char)c;
    }
  }
  return o;
}

static std::string hdr_of(const HttpRequest& rq, const char* key) {
  std::map<std::string, std::string>::const_iterator it = rq.headers.find(key);
  return it == rq.headers.end() ? std::string() : it->second;
}

static bool truthy(const json& v, bool dflt) {
  if (v.is_boolean()) {
    return v.get<bool>();
  }
  if (v.is_number()) {
    return v.get<int>() != 0;
  }
  if (v.is_string()) {
    std::string s = v.get<std::string>();
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s == "true" || s == "on" || s == "1" || s == "yes";
  }
  return dflt;
}

static std::string safe_dump(const json& j) {
  // 模型输出可能带非法 UTF-8（BPE 片段拼接），默认 dump 是 strict 会抛异常 -> 整条请求 500
  return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

static std::string json_content_to_text(const json& c) {
  if (c.is_string()) {
    return c.get<std::string>();
  }
  if (c.is_array()) {
    std::string out;
    for (size_t i = 0; i < c.size(); ++i) {
      const json& it = c[i];
      if (it.is_string()) {
        out += it.get<std::string>();
      } else if (it.is_object() && it.contains("text") && it["text"].is_string()) {
        out += it["text"].get<std::string>();
      }
    }
    return out;
  }
  if (c.is_null()) {
    return "";
  }
  return safe_dump(c);
}

static std::string rand_id() {
  static const char* hexd = "0123456789abcdef";
  std::string s = "chatcmpl-";
  for (int i = 0; i < 24; ++i) {
    s += hexd[rand() & 15];
  }
  return s;
}

static void endswith_trim_special(std::string& s) {
  bool changed = true;
  while (changed) {
    changed = false;
    const char* tails[] = {SPK_EOS, "<|im_end|>", "<|end|>", "</tool_call>", NULL};
    for (int i = 0; tails[i]; ++i) {
      size_t L = strlen(tails[i]);
      if (s.size() >= L && s.compare(s.size() - L, L, tails[i]) == 0) {
        s.erase(s.size() - L);
        changed = true;
      }
    }
  }
  while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r')) {
    s.pop_back();
  }
}

// ------------------------------------------------------------------ 提示词组装
struct PromptParts {
  std::string system, prefix, postfix, text;
  std::string last_role;   // 最后一轮的角色（user/assistant/tool）
  bool ok = true;
  std::string err;
};

static std::string spark_tools_block(const json& tools) {
  // 官方 chat_template：工具清单直接写进 system 块，模型才知道有工具可用
  if (!tools.is_array() || tools.empty()) {
    return std::string();
  }
  std::string out = "\n\n## Tools\nYou have access to the following functions:\n<tools>";
  for (size_t i = 0; i < tools.size(); ++i) {
    const json& t = tools[i];
    if (t.is_object() && t.contains("function") && t["function"].is_object()) {
      out += "\n" + safe_dump(t["function"]);
    }
  }
  out += "\n</tools>";
  return out;
}

static std::string spark_tool_calls_text(const json& m) {
  // 官方格式： <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value> ...
  std::string out;
  if (!m.contains("tool_calls") || !m["tool_calls"].is_array()) {
    return out;
  }
  for (size_t k = 0; k < m["tool_calls"].size(); ++k) {
    const json& tc = m["tool_calls"][k];
    if (!tc.is_object() || !tc.contains("function") || !tc["function"].is_object()) {
      continue;
    }
    const json& f = tc["function"];
    std::string name = (f.contains("name") && f["name"].is_string()) ? f["name"].get<std::string>() : "";
    out += std::string("<tool_call>") + name;
    if (f.contains("arguments")) {
      json args = f["arguments"];
      if (args.is_string()) {
        try {
          args = json::parse(args.get<std::string>());
        } catch (...) {
        }
      }
      if (args.is_object()) {
        for (json::const_iterator it = args.begin(); it != args.end(); ++it) {
          std::string val = it.value().is_string() ? it.value().get<std::string>() : safe_dump(it.value());
          out += "<arg_key>" + it.key() + "</arg_key><arg_value>" + val + "</arg_value>";
        }
      }
    }
    out += "</tool_call>";
  }
  return out;
}

static PromptParts compose_spark(const json& messages, bool thinking, const std::string& default_system,
                                 const json& tools, const std::string& reasoning_effort, bool inject_effort) {
  PromptParts p;

  // 照抄官方 chat_template 的 system 组装：默认 system + thinking_effort 说明 + tools 段 + 首条 system
  std::string effort = (reasoning_effort == "low") ? "low" : "high";
  std::string reasoning_instructions;
  if (thinking && inject_effort) {
    reasoning_instructions = "\n`thinking_effort` guides on how much to think in your thinking channel "
                             "(not including the response channel), supported values include `low` and `high`.\n"
                             "Now the system is invoked with `thinking_effort=" + effort + "`.";
  }
  std::string sys_body = default_system + reasoning_instructions;

  std::string initial_system;
  if (!messages.empty() && messages[0].is_object() && messages[0].contains("role") &&
      messages[0]["role"].is_string() && messages[0]["role"].get<std::string>() == "system" &&
      messages[0].contains("content")) {
    initial_system = json_content_to_text(messages[0]["content"]);
  }

  std::string tools_block = spark_tools_block(tools);
  p.system = std::string(SPK_SOS) + SPK_SYS + sys_body + tools_block +
             (initial_system.empty() ? std::string() : std::string("\n\n") + initial_system) + SPK_EOS;

  struct Turn {
    std::string marker;
    std::string body;
  };
  std::vector<Turn> turns;
  static const char* SPK_TOOL = "<|Tool|>";

  for (size_t i = 0; i < messages.size(); ++i) {
    const json& m = messages[i];
    if (!m.is_object()) {
      p.ok = false;
      p.err = "each message must be an object";
      return p;
    }
    std::string role = (m.contains("role") && m["role"].is_string()) ? m["role"].get<std::string>() : "user";
    std::string content = m.contains("content") ? json_content_to_text(m["content"]) : "";
    if (role == "system" && i == 0) {
      continue;  // 已并进首个 system 块
    }
    if (role == "system") {
      Turn t;
      t.marker = SPK_SYS;
      t.body = content;
      turns.push_back(t);
    } else if (role == "assistant") {
      std::string rc;
      if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
        rc = m["reasoning_content"].get<std::string>();
      }
      Turn t;
      t.marker = SPK_BOT;
      if (!rc.empty()) {
        t.body = "<think>" + rc + "</think>" + content + spark_tool_calls_text(m);
      } else {
        t.body = "</think>" + content + spark_tool_calls_text(m);
      }
      turns.push_back(t);
    } else if (role == "tool") {
      std::string wrapped = std::string("<tool_response>") + content + std::string("</tool_response>");
      if (!turns.empty() && turns.back().marker == SPK_TOOL) {
        turns.back().body += wrapped;  // 连续多个 tool 结果共用一个 <|Tool|> 轮
      } else {
        Turn t;
        t.marker = SPK_TOOL;
        t.body = wrapped;
        turns.push_back(t);
      }
    } else {
      Turn t;
      t.marker = SPK_USER;
      t.body = content;
      turns.push_back(t);
    }
  }

  if (turns.empty()) {
    p.ok = false;
    p.err = "no user/assistant turn in messages";
    return p;
  }

  std::string text;
  for (size_t i = 0; i < turns.size(); ++i) {
    if (i == 0) {
      text = turns[i].body;  // 第一轮的标记由 runtime 的 prompt_prefix 提供
    } else {
      text += SPK_EOS;
      text += SPK_SOS;
      text += turns[i].marker;
      text += turns[i].body;
    }
  }
  p.prefix = std::string(SPK_SOS) + turns[0].marker;

  p.last_role = (turns.back().marker == SPK_TOOL) ? "tool" : "user";
  if (turns.back().marker == SPK_BOT) {
    // 续写 assistant 轮：不追加关闭标记
    p.postfix = "";
  } else {
    p.postfix = std::string(SPK_EOS) + SPK_SOS + SPK_BOT + (thinking ? "<think>" : "</think>");
  }
  p.text = text;
  return p;
}

// ------------------------------------------------------------------ tools
static std::string tools_runtime_json(const json& tools) {
  // runtime 期望的就是 OpenAI 的 tools 数组本身
  return safe_dump(tools);
}

static void extract_tool_calls(const std::string& text, std::string& clean, ordered_json& calls) {
  clean = text;
  std::regex re("<tool_call>([\\s\\S]*?)</tool_call>");
  std::sregex_iterator it(text.begin(), text.end(), re), end_it;
  int idx = 0;
  for (; it != end_it; ++it) {
    std::string block = (*it)[1];
    while (!block.empty() && isspace((unsigned char)block.front())) {
      block.erase(block.begin());
    }
    while (!block.empty() && isspace((unsigned char)block.back())) {
      block.pop_back();
    }
    ordered_json call;
    call["id"] = "call_" + std::to_string(idx + 1) + "_" + std::to_string(time(NULL) % 100000);
    call["type"] = "function";
    std::string name;
    std::string args_json;

    // 首选官方 Spark 形态：<tool_call> 函数名 <arg_key>k</arg_key><arg_value>v</arg_value> ...
    std::regex arg_re("<arg_key>([\\s\\S]*?)</arg_key><arg_value>([\\s\\S]*?)</arg_value>");
    std::sregex_iterator ait(block.begin(), block.end(), arg_re), aend;
    if (ait != aend) {
      std::string head = std::regex_replace(block, arg_re, "");
      size_t pos = head.find_first_of(" \t\r\n");
      name = head.substr(0, pos == std::string::npos ? head.size() : pos);
      while (!name.empty() && isspace((unsigned char)name.back())) {
        name.pop_back();
      }
      json args = json::object();
      for (; ait != aend; ++ait) {
        std::string k = (*ait)[1];
        std::string v = (*ait)[2];
        while (!k.empty() && isspace((unsigned char)k.front())) {
          k.erase(k.begin());
        }
        while (!k.empty() && isspace((unsigned char)k.back())) {
          k.pop_back();
        }
        while (!v.empty() && isspace((unsigned char)v.front())) {
          v.erase(v.begin());
        }
        while (!v.empty() && isspace((unsigned char)v.back())) {
          v.pop_back();
        }
        json jv;
        try {
          jv = json::parse(v);
        } catch (...) {
          jv = json(v);
        }
        if (!k.empty()) {
          args[k] = jv;
        }
      }
      args_json = safe_dump(args);
    } else {
      // 兼容形态：整块是一个 JSON 对象（模型/内置工具链有时会走这条路）
      json parsed;
      bool ok = false;
      try {
        parsed = json::parse(block);
        ok = parsed.is_object();
      } catch (...) {
        ok = false;
      }
      if (ok) {
        name = parsed.value("name", parsed.value("tool_name", parsed.value("function_name", "")));
        json args = parsed.contains("arguments")      ? parsed["arguments"]
                    : (parsed.contains("parameters") ? parsed["parameters"] : json());
        args_json = args.is_string() ? args.get<std::string>() : safe_dump(args);
      } else {
        // 什么标记都没有：整块当 arguments 原样交给上层，至少不丢信息
        name = "";
        args_json = block;
      }
    }
    call["function"]["name"] = name;
    call["function"]["arguments"] = args_json;
    calls.push_back(call);
    idx++;
  }
  if (idx > 0) {
    clean = std::regex_replace(text, re, "");
    clean = std::regex_replace(clean, std::regex("<arg_key>[\\s\\S]*?</arg_value>"), "");
    // 去掉残留的半截标记
    static const char* leftovers[] = {"<tool_call>", "<arg_key>", "</arg_value>", NULL};
    for (int i = 0; leftovers[i]; ++i) {
      size_t pos;
      while ((pos = clean.find(leftovers[i])) != std::string::npos) {
        clean.erase(pos, strlen(leftovers[i]));
      }
    }
    endswith_trim_special(clean);
  }
}

// ------------------------------------------------------------------ response_format -> grammar
static bool grammar_from_response_format(const json& rf, std::string& str, int& type) {
  if (!rf.is_object() || !rf.contains("type")) {
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

// ------------------------------------------------------------------ 流式出口
class SseSink : public TextSink {
 public:
  // split=true：
  static const char* const THINK_CLOSE;

  SseSink(ResponseWriter& w, const std::string& id, const std::string& model, bool split)
      : m_w(w), m_id(id), m_model(model), m_split(split) {}

  bool on_text(const std::string& piece) override {
    if (!m_split) {
      return emit("content", piece);
    }
    m_buf += piece;
    if (!m_split_done) {
      size_t pos = m_buf.find(THINK_CLOSE);
      if (pos != std::string::npos) {
        std::string head = m_buf.substr(0, pos);
        m_buf = m_buf.substr(pos + strlen(THINK_CLOSE));
        m_split_done = true;
        if (!head.empty() && !emit("reasoning_content", head)) {
          return false;
        }
        if (!m_buf.empty() && !emit("content", m_buf)) {
          return false;
        }
        m_buf.clear();
        return true;
      }
      size_t keep = m_buf.size() > 8 ? 8 : m_buf.size();
      std::string out = m_buf.substr(0, m_buf.size() - keep);
      m_buf = m_buf.substr(m_buf.size() - keep);
      if (!out.empty() && !emit("reasoning_content", out)) {
        return false;
      }
      return true;
    }
    if (!emit("content", m_buf)) {
      return false;
    }
    m_buf.clear();
    return true;
  }

  // 生成结束后把残留缓冲冲出去（分隔串被切开时靠这里补齐）
  void flush_tail() {
    if (m_buf.empty()) {
      return;
    }
    emit(m_split_done ? "content" : "reasoning_content", m_buf);
    m_buf.clear();
  }

  bool split_done() const { return m_split_done; }

  void emit_role() {
    ordered_json j;
    j["id"] = m_id;
    j["object"] = "chat.completion.chunk";
    j["created"] = (long)time(NULL);
    j["model"] = m_model;
    ordered_json ch;
    ch["index"] = 0;
    ch["delta"] = ordered_json{{"role", "assistant"}, {"content", ""}};
    ch["finish_reason"] = nullptr;
    j["choices"] = ordered_json::array({ch});
    m_w.write_sse_data(safe_dump(j));
  }

  void emit_final(const std::string& finish, const json* usage) {
    ordered_json j;
    j["id"] = m_id;
    j["object"] = "chat.completion.chunk";
    j["created"] = (long)time(NULL);
    j["model"] = m_model;
    ordered_json ch;
    ch["index"] = 0;
    ch["delta"] = ordered_json::object();
    ch["finish_reason"] = finish;
    j["choices"] = ordered_json::array({ch});
    if (usage) {
      j["usage"] = *usage;
    }
    m_w.write_sse_data(safe_dump(j));
  }

 private:
  bool emit(const char* field, const std::string& text) {
    ordered_json j;
    j["id"] = m_id;
    j["object"] = "chat.completion.chunk";
    j["created"] = (long)time(NULL);
    j["model"] = m_model;
    ordered_json ch;
    ch["index"] = 0;
    ch["delta"] = ordered_json{{field, text}};
    ch["finish_reason"] = nullptr;
    j["choices"] = ordered_json::array({ch});
    m_first = false;
    return m_w.write_sse_data(safe_dump(j));
  }

  ResponseWriter& m_w;
  std::string m_id, m_model;
  bool m_first = true;
  bool m_split;
  bool m_split_done = false;
  std::string m_buf;
};

const char* const SseSink::THINK_CLOSE = "</think>";

// ------------------------------------------------------------------ 请求解析
struct ChatParams {
  float temperature, top_p;
  int top_k;
  float repeat_penalty, frequency_penalty, presence_penalty;
  int max_new;
  long seed;
  bool thinking;
  bool stream;
  bool include_usage;
  bool want_logprobs;
  std::string effort;
  int inject_effort = -1;  // -1 = 用服务端配置；0/1 = 请求级覆盖官方 thinking_effort 注入
  int top_logprobs = 0;
  std::vector<int32_t> lb_ids;
  std::vector<float> lb_vals;
  std::vector<std::string> stops;
  std::string grammar_str;
  int grammar_type;
  bool use_grammar;
  bool use_tools;
  std::string tools_json;
  std::string tool_choice;
  bool err = false;
  std::string err_msg;
};

static ChatParams parse_params(const json& body) {
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
    c.use_grammar = grammar_from_response_format(body["response_format"], c.grammar_str, c.grammar_type);
  }
  if (body.contains("tools") && body["tools"].is_array() && !body["tools"].empty()) {
    c.use_tools = true;
    c.tools_json = tools_runtime_json(body["tools"]);
    if (body.contains("tool_choice")) {
      if (body["tool_choice"].is_string()) {
        c.tool_choice = body["tool_choice"].get<std::string>();
      } else if (body["tool_choice"].is_object() && body["tool_choice"].contains("function")) {
        c.tool_choice = body["tool_choice"]["function"].value("name", "auto");
      }
    }
    if (body.contains("parallel_tool_calls") && !truthy(body["parallel_tool_calls"], true)) {
      c.tool_choice = "single";
    }
  }
  if (c.use_tools && c.tool_choice == "none") {
    c.use_tools = false;  // 客户端显式禁用工具
  }
  return c;
}

// ------------------------------------------------------------------ 生成主流程
static void do_chat(const HttpRequest& rq, ResponseWriter& w) {
  json body;
  try {
    body = json::parse(rq.body);
  } catch (const std::exception& e) {
    w.write_full(400, "{\"error\":{\"code\":400,\"message\":\"" + json_escape(e.what()) + 
                          "\",\"type\":\"invalid_request_error\"}}",
                 "application/json");
    return;
  }
  if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
    w.write_full(400, "{\"error\":{\"code\":400,\"message\":\"'messages' must be a non-empty array\","
                      "\"type\":\"invalid_request_error\"}}",
                 "application/json");
    return;
  }
  ChatParams cp = parse_params(body);
  json tools_arg = (body.contains("tools") && body["tools"].is_array()) ? body["tools"] : json::array();
  bool inject_effort = (cp.inject_effort >= 0) ? (cp.inject_effort == 1) : g_cfg.thinking_effort_inject;
  PromptParts pp = compose_spark(body["messages"], cp.thinking, g_cfg.system_prompt, tools_arg, cp.effort,
                                 inject_effort);
  if (!pp.ok) {
    w.write_full(400, "{\"error\":{\"code\":400,\"message\":\"" + pp.err + "\",\"type\":\"invalid_request_error\"}}",
                 "application/json");
    return;
  }

  if (body.contains("n") && body["n"].is_number() && body["n"].get<int>() > 1) {
    w.write_full(400, "{\"error\":{\"code\":400,\"message\":\"n>1 is not supported by this server (single NPU "
                      "session per request); use n=1\",\"type\":\"invalid_request_error\"}}",
                 "application/json");
    return;
  }

  // 粗长的 prompt 先按字符数拦一道，避免把 KV 打满（4 字符/token 是保守估计）
  long est_chars = (long)pp.text.size() + (long)pp.system.size() + (long)pp.postfix.size();
  int max_ctx = g_engine.model_max_ctx();
  if (max_ctx > 0 && est_chars > (long)max_ctx * 6) {
    w.write_full(400, "{\"error\":{\"code\":400,\"message\":\"prompt too long for the loaded context window\","
                      "\"type\":\"invalid_request_error\"}}",
                 "application/json");
    return;
  }

  GenRequest gr;
  gr.system_prompt = pp.system;
  gr.prompt_prefix = pp.prefix;
  gr.prompt_postfix = pp.postfix;
  gr.text = pp.text;
  gr.role = (pp.last_role == "tool") ? "tool" : "user";
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
  gr.want_logprobs = cp.want_logprobs;
  gr.top_logprobs = cp.top_logprobs;
  gr.logit_bias_ids = cp.lb_ids;
  gr.logit_bias_vals = cp.lb_vals;
  gr.use_grammar = cp.use_grammar;
  gr.grammar_str = cp.grammar_str;
  gr.grammar_type = cp.grammar_type;
  gr.use_tools = cp.use_tools;
  gr.tools_json = cp.tools_json;

  std::string id = rand_id();
  long rid = ++g_req_id;
  bool streaming = cp.stream;
  bool client_gone = false;

  if (streaming) {
    if (!w.write_sse_start()) {
      return;
    }
  }

  bool split_reason = (g_cfg.reasoning_format == "deepseek");
  if (body.contains("reasoning_format") && body["reasoning_format"].is_string()) {
    split_reason = (body["reasoning_format"].get<std::string>() == "deepseek");
  }

  GenResult res;
  SseSink* sink = NULL;
  if (streaming) {
    sink = new SseSink(w, id, g_cfg.model_alias, split_reason);
    sink->emit_role();
  }

  long t0 = (long)getCurrentTimeUs();
  if (g_waiting.load() >= g_max_queue) {
    w.write_full(429, "{\"error\":{\"code\":429,\"message\":\"inference queue is full, retry later\","
                       "\"type\":\"rate_limit_error\"}}",
                 "application/json");
    return;
  }
  int ret;
  {
    g_waiting++;
    std::unique_lock<std::mutex> lk(g_engine_mu);
    g_waiting--;
    g_m.active++;
    ret = g_engine.generate(gr, res, sink);
    g_m.active--;
  }
  if (sink) {
    sink->flush_tail();
  }
  delete sink;
  double e2e_ms = (double)((long)getCurrentTimeUs() - t0) / 1000.0;

  // 工具调用解析
  std::string content = res.text;
  ordered_json tool_calls;
  bool has_tools = false;
  if (cp.use_tools) {
    extract_tool_calls(content, content, tool_calls);
    has_tools = !tool_calls.empty();
  }
  endswith_trim_special(content);

  // 思考通道分离：模型输出里 </think> 之前是推理内容，之后是正式回答
  std::string reasoning;
  if (split_reason) {
    size_t pos = content.find("</think>");
    if (pos != std::string::npos) {
      reasoning = content.substr(0, pos);
      content = content.substr(pos + 8);
    } else {
      // 模型没写出 </think>：不硬拆，整段留在 content 里，避免客户端拿到空回答
      reasoning.clear();
    }
    while (!reasoning.empty() && (reasoning.front() == ' ' || reasoning.front() == '\n' || reasoning.front() == '\r')) {
      reasoning.erase(reasoning.begin());
    }
    if (reasoning.compare(0, 7, "<think>") == 0) {
      reasoning.erase(0, 7);
    }
    while (!content.empty() && (content.front() == ' ' || content.front() == '\n' || content.front() == '\r')) {
      content.erase(content.begin());
    }
  }

  std::string finish = res.finish_reason;
  if (has_tools) {
    finish = "tool_calls";
  }

  json usage;
  usage["prompt_tokens"] = res.prompt_tokens;
  usage["completion_tokens"] = res.completion_tokens;
  usage["total_tokens"] = res.prompt_tokens + res.completion_tokens;
  usage["prefill_ms"] = (long)(res.prefill_ms + 0.5);
  usage["decode_ms"] = (long)(res.decode_ms + 0.5);
  usage["e2e_ms"] = (long)(e2e_ms + 0.5);
  if (res.decode_ms > 0 && res.completion_tokens > 0) {
    usage["tokens_per_second"] = (double)res.completion_tokens / (res.decode_ms / 1000.0);
  } else {
    usage["tokens_per_second"] = 0.0;
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

  bool ok = (ret == RKNN3_SUCCESS);
  LOGF("REQ %ld %s prompt=%ld completion=%ld finish=%s prefill=%.1fms decode=%.1fms tps=%.2f veto=%d guard=%d loop=%d "
       "client_gone=%d ret=%d",
       rid, streaming ? "stream" : "json", res.prompt_tokens, res.completion_tokens, finish.c_str(), res.prefill_ms,
       res.decode_ms, res.completion_tokens > 0 && res.decode_ms > 0 ? (double)res.completion_tokens /
                                                                           (res.decode_ms / 1000.0)
                                                                     : 0.0,
       res.veto_hits, (int)res.degeneration_guard, res.loop_hits, (int)res.client_gone, ret);
  g_m.record(e2e_ms, res.prompt_tokens, res.completion_tokens, ok, res.degeneration_guard, res.veto_hits, streaming,
             has_tools);
  (void)client_gone;

  if (!ok) {
    if (streaming) {
      ordered_json j;
      j["error"] = {{"code", 500}, {"message", "inference failed ret=" + std::to_string(ret)}};
      w.write_sse_data(safe_dump(j));
      w.write_sse_end();
      return;
    }
    w.write_full(500, "{\"error\":{\"code\":500,\"message\":\"inference failed ret=" + std::to_string(ret) +
                          "\",\"type\":\"server_error\"}}",
                 "application/json");
    return;
  }

  if (streaming) {
    if (has_tools) {
      ordered_json j;
      j["id"] = id;
      j["object"] = "chat.completion.chunk";
      j["created"] = (long)time(NULL);
      j["model"] = g_cfg.model_alias;
      ordered_json ch;
      ch["index"] = 0;
      ordered_json delta;
      delta["tool_calls"] = tool_calls;
      ch["delta"] = delta;
      ch["finish_reason"] = nullptr;
      j["choices"] = ordered_json::array({ch});
      w.write_sse_data(safe_dump(j));
    }
    json* uptr = cp.include_usage ? &usage : NULL;
    SseSink s2(w, id, g_cfg.model_alias, false);
    s2.emit_final(finish, uptr);
    w.write_sse_end();
    return;
  }

  ordered_json resp;
  resp["id"] = id;
  resp["object"] = "chat.completion";
  resp["created"] = (long)time(NULL);
  resp["model"] = body.contains("model") && body["model"].is_string() ? body["model"].get<std::string>()
                                                                      : g_cfg.model_alias;
  ordered_json msg;
  msg["role"] = "assistant";
  msg["content"] = content;
  if (split_reason && !reasoning.empty()) {
    msg["reasoning_content"] = reasoning;
  }
  if (has_tools) {
    msg["tool_calls"] = tool_calls;
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
  } catch (const std::exception& e) {
    w.write_full(400, "{\"error\":{\"code\":400,\"message\":\"invalid json\"}}", "application/json");
    return;
  }
  if (!body.contains("prompt")) {
    w.write_full(400, "{\"error\":{\"code\":400,\"message\":\"'prompt' is required\"}}", "application/json");
    return;
  }
  std::string prompt;
  if (body["prompt"].is_string()) {
    prompt = body["prompt"].get<std::string>();
  } else if (body["prompt"].is_array() && body["prompt"].size() == 1 && body["prompt"][0].is_string()) {
    prompt = body["prompt"][0].get<std::string>();
  } else {
    w.write_full(400, "{\"error\":{\"code\":400,\"message\":\"'prompt' must be a string (or 1-element array)\"}}",
                 "application/json");
    return;
  }

  ChatParams cp = parse_params(body);
  GenRequest gr;
  gr.system_prompt = std::string();  // 空 system：prompt 完全由客户端控制
  gr.prompt_prefix = std::string();
  gr.prompt_postfix = std::string();
  gr.text = prompt;
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
  gr.use_grammar = cp.use_grammar;
  gr.grammar_str = cp.grammar_str;
  gr.grammar_type = cp.grammar_type;

  GenResult res;
  int ret;
  long t0 = (long)getCurrentTimeUs();
  if (g_waiting.load() >= g_max_queue) {
    w.write_full(429, "{\"error\":{\"code\":429,\"message\":\"inference queue is full, retry later\"}}",
                 "application/json");
    return;
  }
  {
    g_waiting++;
    std::unique_lock<std::mutex> lk(g_engine_mu);
    g_waiting--;
    g_m.active++;
    ret = g_engine.generate(gr, res, NULL);
    g_m.active--;
  }
  double e2e_ms = (double)((long)getCurrentTimeUs() - t0) / 1000.0;
  g_m.record(e2e_ms, res.prompt_tokens, res.completion_tokens, ret == RKNN3_SUCCESS, res.degeneration_guard,
             res.veto_hits, false, false);
  if (ret != RKNN3_SUCCESS) {
    w.write_full(500, "{\"error\":{\"code\":500,\"message\":\"inference failed ret=" + std::to_string(ret) + "\"}}",
                 "application/json");
    return;
  }
  std::string text = res.text;
  endswith_trim_special(text);
  ordered_json resp;
  resp["id"] = "cmpl-" + std::to_string((long)time(NULL));
  resp["object"] = "text_completion";
  resp["created"] = (long)time(NULL);
  resp["model"] = g_cfg.model_alias;
  ordered_json item;
  item["text"] = text;
  item["index"] = 0;
  item["finish_reason"] = res.finish_reason;
  resp["choices"] = ordered_json::array({item});
  resp["usage"] = {{"prompt_tokens", res.prompt_tokens}, {"completion_tokens", res.completion_tokens},
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
  ordered_json meta;
  meta["model_type"] = g_engine.model_type();
  meta["max_context_len"] = g_engine.model_max_ctx();
  meta["vocab_size"] = g_engine.vocab_size();
  meta["quant"] = "w4a16 + GRQ + group32 (weights), Int4_to_F16/GroupQuant/group16/resid64 (KV cache)";
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
                               {"thinking_effort_inject", g_cfg.thinking_effort_inject}};
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
      w.write_full(404, "{\"error\":{\"code\":404,\"message\":\"model not found: " + json_escape(req_id) +
                            "\",\"type\":\"invalid_request_error\"}}",
                   "application/json");
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

long Metrics::conns_active() { return g_srv ? (long)g_srv->active_conns() : 0; }

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

static void do_metrics(ResponseWriter& w) {
  w.write_full(200, g_m.snapshot().dump(2), "application/json");
}

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
  ordered_json slot;
  slot["id"] = 0;
  slot["n_ctx"] = g_engine.model_max_ctx();
  slot["state"] = g_m.active.load() > 0 ? "running" : "idle";
  ordered_json resp;
  resp["object"] = "list";
  resp["data"] = ordered_json::array({slot});
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
    w.write_full(404, "{\"error\":{\"code\":404,\"message\":\"unknown slot action\"}}", "application/json");
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
  std::lock_guard<std::mutex> lk(g_engine_mu);
  int ret = -1;
  json out;
  out["id_slot"] = 0;
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
  } else {
    std::string path = body.value("path", std::string(""));
    if (path.empty()) {
      w.write_full(400, "{\"error\":{\"code\":400,\"message\":\"'path' is required\"}}", "application/json");
      return;
    }
    ret = g_engine.load_kvcache(path);
    out["path"] = path;
  }
  if (ret != 0) {
    w.write_full(500, "{\"error\":{\"code\":500,\"message\":\"" + action + " failed ret=" + std::to_string(ret) +
                          "\"}}",
                 "application/json");
    return;
  }
  out["success"] = true;
  w.write_full(200, safe_dump(out), "application/json");
}

static void dispatch(const HttpRequest& rq, ResponseWriter& w) {
  if (rq.method == "OPTIONS") {
    w.write_full(204, "", "text/plain");
    return;
  }
  if (!g_cfg.http.api_key.empty()) {
    std::string auth = hdr_of(rq, "authorization");
    if (auth.find(g_cfg.http.api_key) == std::string::npos) {
      w.write_full(401, "{\"error\":{\"code\":401,\"message\":\"missing or invalid api key\","
                        "\"type\":\"authentication_error\"}}",
                   "application/json");
      return;
    }
  }
  if ((rq.method == "GET" || rq.method == "HEAD") && (rq.path == "/health" || rq.path == "/healthz")) {
    do_health(w, rq.method == "HEAD");
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
  if (rq.method == "POST" && (rq.path == "/v1/completions" || rq.path == "/completions")) {
    if (!g_cfg.allow_raw_prompt) {
      w.write_full(404, "{\"error\":{\"code\":404,\"message\":\"/v1/completions disabled\"}}", "application/json");
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
    std::lock_guard<std::mutex> lk(g_engine_mu);
    int ret = (rq.path == "/v1/kvcache/save") ? g_engine.save_kvcache(path) : g_engine.load_kvcache(path);
    if (ret != 0) {
      w.write_full(500, "{\"error\":{\"code\":500,\"message\":\"kvcache op failed ret=" + std::to_string(ret) +
                            "\"}}",
                   "application/json");
      return;
    }
    json out;
    out["success"] = true;
    out["path"] = path;
    w.write_full(200, safe_dump(out), "application/json");
    return;
  }
  w.write_full(404, "{\"error\":{\"code\":404,\"message\":\"no route: " + json_escape(rq.method + " " + rq.path) + 
                        "\",\"type\":\"not_found_error\"}}",
               "application/json");
}

// ------------------------------------------------------------------ 参数解析
static void print_usage(const char* prog) {
  printf(
      "rknn_spark_server — RK1828 OpenAI-compatible LLM server (Spark-X2.5/X2.6)\n"
      "Usage: %s <llm.rknn> <llm.weight> <core_mask> <tokenizer.gguf> <embed.bin> <max_context_len> "
      "<safetensors> [options]\n\n"
      "  --host H                监听地址（默认 0.0.0.0）\n"
      "  --port P                端口（默认 8080）\n"
      "  --api-key K             需要 Authorization: Bearer K\n"
      "  --model-alias NAME      OpenAI model id（默认 Spark-X2.6-4B）\n"
      "  --system-prompt TEXT    默认 system（默认 'you are a helpful assistant.'）\n"
      "  --thinking on|off       默认思考模式（默认 on）\n"
      "  --temp F --top-k K --top-p F           采样（默认模型厂商推荐：temp=1.0 top_k=0(不截断) top_p=0.95）\n"
      "  --repeat-penalty F      重复惩罚（默认 1.0=关闭；本服务刻意不加惩罚）\n"
      "  --frequency-penalty F   频率惩罚（默认 0）\n"
      "  --presence-penalty F    存在惩罚（默认 0）\n"
      "  --max-new-tokens N      默认生成长度（默认 0=不限制，生成到 EOS 或上下文用完）\n"
      "  --max-new-tokens-cap N  单请求生成上限（默认 0=不给客户端设上限）\n"
      "  --request-timeout-sec N 单次生成墙钟上限（默认 0=不限）\n"
      "  --sampler custom|runtime  采样器（默认 custom：带复读熔断）\n"
      "  --max-connections N     HTTP 并发连接上限（默认 128）\n"
      "  --max-queue N           排队上限，超出返回 429（默认 64）\n"
      "  --veto-limit N          复读熔断触发上限（默认 6）\n"
      "  --veto-period-max N     token 级周期熔断的最大周期（默认 400，0=关闭）\n"
      "  --ngram-len N --ngram-ban-at N   token 级 n-gram 熔断（默认 8 / 出现 3 次即禁止）\n"
      "  --loop-max-occurrences N         文本级长块熔断（默认 3，0=关闭）\n"
      "  --fuse-off              关掉全部三层熔断（只靠 EOS / 上下文窗口 / 墙钟超时收尾）\n"
      "  --thinking-effort-inject on|off  是否把官方那句 'Now the system is invoked with `thinking_effort=…`'\n"
      "                          写进 system（默认 on=与官方模板一致；off 可避免模型复述它）\n"
      "  --reasoning-effort high|low|none|minimal  默认思考档位（默认为空 = 跟官方模板一致用 high；\n"
      "                          none/minimal 等同关闭思考；客户端传 reasoning_effort 会覆盖本默认）\n"
      "  --device-id ID          指定卡（默认第一张）\n"
      "  --log-dir DIR           日志目录（保留参数）\n",
      prog);
}

int main(int argc, char** argv) {
  srand((unsigned)time(NULL) ^ (unsigned)getpid());
  g_start_time = (long)time(NULL);

  if (argc < 8) {
    print_usage(argv[0]);
    return 1;
  }
  g_cfg.eng.model_path = argv[1];
  g_cfg.eng.weight_path = argv[2];
  g_cfg.eng.core_mask = (uint32_t)strtoul(argv[3], NULL, 16);
  g_cfg.eng.tokenizer_path = argv[4];
  g_cfg.eng.embed_path = argv[5];
  g_cfg.eng.max_context_len = atoi(argv[6]);
  g_cfg.eng.safetensors_path = argv[7];

  for (int i = 8; i < argc; ++i) {
    std::string a = argv[i];
    std::string v = (i + 1 < argc) ? argv[i + 1] : "";
    if (a == "--host") { g_cfg.http.host = v; ++i; }
    else if (a == "--port") { g_cfg.http.port = atoi(v.c_str()); ++i; }
    else if (a == "--api-key") { g_cfg.http.api_key = v; ++i; }
    else if (a == "--model-alias") { g_cfg.model_alias = v; ++i; }
    else if (a == "--system-prompt") { g_cfg.system_prompt = v; ++i; }
    else if (a == "--thinking") { g_cfg.default_thinking = (v == "on" || v == "1" || v == "true"); ++i; }
    else if (a == "--temp") { g_cfg.temperature = (float)atof(v.c_str()); ++i; }
    else if (a == "--top-k") { g_cfg.top_k = atoi(v.c_str()); ++i; }
    else if (a == "--top-p") { g_cfg.top_p = (float)atof(v.c_str()); ++i; }
    else if (a == "--repeat-penalty") { g_cfg.repeat_penalty = (float)atof(v.c_str()); ++i; }
    else if (a == "--frequency-penalty") { g_cfg.frequency_penalty = (float)atof(v.c_str()); ++i; }
    else if (a == "--presence-penalty") { g_cfg.presence_penalty = (float)atof(v.c_str()); ++i; }
    else if (a == "--max-new-tokens") { g_cfg.max_new_tokens = atoi(v.c_str()); ++i; }
    else if (a == "--max-new-tokens-cap") { g_cfg.max_new_tokens_cap = atoi(v.c_str()); ++i; }
    else if (a == "--request-timeout-sec") { g_cfg.request_timeout_sec = atoi(v.c_str()); ++i; }
    else if (a == "--sampler") { g_cfg.use_runtime_sampler = (v == "runtime"); ++i; }
    else if (a == "--max-connections") { g_cfg.http.max_connections = atoi(v.c_str()); ++i; }
    else if (a == "--max-queue") { g_cfg.max_queue = atoi(v.c_str()); ++i; }
    else if (a == "--veto-limit") { g_cfg.eng.veto_limit = atoi(v.c_str()); ++i; }
    else if (a == "--device-id") { g_cfg.eng.device_id = v; ++i; }
    else if (a == "--kvcache-policy") { g_cfg.eng.kvcache_policy = atoi(v.c_str()); ++i; }
    else if (a == "--kv-n-keep") { g_cfg.eng.kv_n_keep = atol(v.c_str()); ++i; }
    else if (a == "--kv-n-keep-aligned") { g_cfg.eng.kv_n_keep_aligned = atol(v.c_str()); ++i; }
    else if (a == "--veto-period-max") { g_cfg.eng.veto_period_max = atoi(v.c_str()); ++i; }
    else if (a == "--ngram-len") { g_cfg.eng.ngram_len = atoi(v.c_str()); ++i; }
    else if (a == "--ngram-ban-at") { g_cfg.eng.ngram_ban_at = atoi(v.c_str()); ++i; }
    else if (a == "--loop-max-occurrences") { g_cfg.eng.loop_max_occurrences = atoi(v.c_str()); ++i; }
    else if (a == "--fuse-off") { g_cfg.eng.veto_period_max = 0; g_cfg.eng.ngram_len = 0;
                                  g_cfg.eng.loop_max_occurrences = 0; }
    else if (a == "--kvcache-dir") { g_cfg.kvcache_dir = v; ++i; }
    else if (a == "--reasoning-format") { g_cfg.reasoning_format = v; ++i; }
    else if (a == "--thinking-effort-inject") { g_cfg.thinking_effort_inject = (v == "on" || v == "1" || v == "true"); ++i; }
    else if (a == "--reasoning-effort") { g_cfg.reasoning_effort = v; ++i; }
    else if (a == "--params-file") { g_cfg.params_file = v; ++i; }
    else if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
    else { LOGF("unknown option: %s", a.c_str()); }
  }

  // params-file（与官方 rkllm3-server 的 JSON 配置风格对齐）
  if (!g_cfg.params_file.empty()) {
    std::ifstream f(g_cfg.params_file.c_str());
    if (!f.is_open()) {
      LOGF("cannot open params-file: %s", g_cfg.params_file.c_str());
      return 1;
    }
    json p;
    try {
      f >> p;
    } catch (const std::exception& e) {
      LOGF("params-file parse failed: %s", e.what());
      return 1;
    }
    if (p.contains("host")) g_cfg.http.host = p["host"].get<std::string>();
    if (p.contains("port")) g_cfg.http.port = p["port"].get<int>();
    if (p.contains("api-key") && p["api-key"].is_string()) g_cfg.http.api_key = p["api-key"].get<std::string>();
    if (p.contains("timeout") && p["timeout"].is_number()) g_cfg.request_timeout_sec = p["timeout"].get<int>();
    if (p.contains("max-connections")) g_cfg.http.max_connections = p["max-connections"].get<int>();
    if (p.contains("models") && p["models"].is_object() && !p["models"].empty()) {
      json m = p["models"].begin().value();
      if (m.contains("model")) g_cfg.eng.model_path = m["model"].get<std::string>();
      if (m.contains("weight")) g_cfg.eng.weight_path = m["weight"].get<std::string>();
      if (m.contains("vocab")) g_cfg.eng.tokenizer_path = m["vocab"].get<std::string>();
      if (m.contains("embed")) g_cfg.eng.embed_path = m["embed"].get<std::string>();
      if (m.contains("rope-tensor")) g_cfg.eng.safetensors_path = m["rope-tensor"].get<std::string>();
      if (m.contains("alias")) g_cfg.model_alias = m["alias"].get<std::string>();
      if (m.contains("ctx-size") && m["ctx-size"].is_number()) g_cfg.eng.max_context_len = m["ctx-size"].get<int>();
      if (m.contains("predict") && m["predict"].is_number()) g_cfg.max_new_tokens = m["predict"].get<int>();
      if (m.contains("temp")) g_cfg.temperature = m["temp"].get<float>();
      if (m.contains("top-k")) g_cfg.top_k = m["top-k"].get<int>();
      if (m.contains("top-p")) g_cfg.top_p = m["top-p"].get<float>();
      if (m.contains("repeat-penalty")) g_cfg.repeat_penalty = m["repeat-penalty"].get<float>();
      if (m.contains("frequency-penalty")) g_cfg.frequency_penalty = m["frequency-penalty"].get<float>();
      if (m.contains("presence-penalty")) g_cfg.presence_penalty = m["presence-penalty"].get<float>();
      if (m.contains("core-mask")) {
        if (m["core-mask"].is_string()) {
          g_cfg.eng.core_mask = (uint32_t)strtoul(m["core-mask"].get<std::string>().c_str(), NULL, 16);
        } else {
          g_cfg.eng.core_mask = (uint32_t)m["core-mask"].get<long>();
        }
      }
      if (m.contains("thinking")) g_cfg.default_thinking = truthy(m["thinking"], g_cfg.default_thinking);
    }
  }

  if (!g_cfg.kvcache_dir.empty()) {
    std::string cmd = "mkdir -p " + g_cfg.kvcache_dir;
    if (system(cmd.c_str()) != 0) {
      LOGF("mkdir %s failed", g_cfg.kvcache_dir.c_str());
    }
  }

  g_max_queue = g_cfg.max_queue > 0 ? g_cfg.max_queue : 64;

  signal(SIGTERM, on_signal);
  signal(SIGINT, on_signal);

  LOGF("=== rknn_spark_server starting ===");
  LOGF("model=%s core_mask=%02x ctx=%d sampler=%s", g_cfg.eng.model_path.c_str(), g_cfg.eng.core_mask,
       g_cfg.eng.max_context_len, g_cfg.use_runtime_sampler ? "runtime" : "custom(veto)");

  int ret = g_engine.init(g_cfg.eng);
  if (ret != 0) {
    LOGF("engine init failed ret=%d", ret);
    return 2;
  }
  g_ready = true;

  HttpServer srv;
  g_srv = &srv;
  if (srv.start(g_cfg.http, dispatch) != 0) {
    LOGF("http start failed");
    return 3;
  }
  LOGF("listening on http://%s:%d alias=%s ctx=%d thinking=%d temp=%.2f top_k=%d top_p=%.2f rep_pen=%.2f "
       "max_new=%d cap=%d conns=%d",
       g_cfg.http.host.c_str(), g_cfg.http.port, g_cfg.model_alias.c_str(), g_engine.model_max_ctx(),
       (int)g_cfg.default_thinking, g_cfg.temperature, g_cfg.top_k, g_cfg.top_p, g_cfg.repeat_penalty,
       g_cfg.max_new_tokens, g_cfg.max_new_tokens_cap, g_cfg.http.max_connections);

  while (!g_requested_shutdown) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  LOGF("shutdown requested, draining connections...");
  srv.stop();
  g_ready = false;
  g_engine.shutdown();
  LOGF("clean shutdown complete");
  return 0;
}
