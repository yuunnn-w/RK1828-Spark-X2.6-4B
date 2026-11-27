// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// main.cpp — rknn_spark_server：RK1828 板端生产级 OpenAI 兼容推理服务
//
// 端点：
//   POST /v1/chat/completions   （stream / 非 stream、tools、response_format、stop、penalties、logprobs）
//   POST /v1/completions        （OpenAI 传统补全，prompt 原样送入）
//   POST /v1/responses          （OpenAI Responses API，DeepSeek Harness 的 openai-responses 路由）
//   POST /v1/embeddings         （词嵌入表均值池化，近似嵌入，非训练过的 embedding 模型）
//   GET  /v1/models, /v1/models/{id}
//   GET  /health, /healthz, /metrics
//   POST /slots/{id}/pause|resume|interrupt, POST /slots/{id}/kvcache/save|load, GET /v1/slots
//   GET/POST /v1/models/info（模型与运行时信息）
//
// 采样：默认贪心（temperature=0）+ 重复惩罚 + token 级复读熔断（周期块检测）+
//       片段级复读检测，保证交付文本不出现失控复读。
//
// 并发模型：卡上只有一个 session，生成严格串行；HTTP 层并发接收，按到达顺序（FIFO 票据）排队。
// KV 槽位（--kv-slots N）：每个会话的 KV 存成主机侧镜像，卡上只放活跃槽，换入时按文本前缀续接。

#include "engine.h"
#include "httpd.h"
#include "kvslots.h"
#include "time_utils.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
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

// 模板标签（用转义拼接，源码里不出现可被外部工具误认的字面量）
static const char* TAG_THINK_OPEN = "\x3C" "think" "\x3E";
static const char* TAG_THINK_CLOSE = "\x3C/" "think" "\x3E";
static const char* TAG_TOOL_OPEN = "\x3C" "tool_call" "\x3E";
static const char* TAG_TOOL_CLOSE = "\x3C/" "tool_call" "\x3E";
static const char* TAG_ARGK_OPEN = "\x3C" "arg_key" "\x3E";
static const char* TAG_ARGK_CLOSE = "\x3C/" "arg_key" "\x3E";
static const char* TAG_ARGV_OPEN = "\x3C" "arg_value" "\x3E";
static const char* TAG_ARGV_CLOSE = "\x3C/" "arg_value" "\x3E";

static const char* g_fingerprint = "fp_rk1828_spark_20261126";

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
  bool system_in_prompt = true;          // 整段 ChatML（含 system 块）拼进 prompt 文本，而不是交给 set_chat_template
  std::string kvcache_dir = "/run/rknn3/kvcache";
  std::string params_file;

  // KV 槽位分页：主机内存里存多个会话的 KV 镜像，卡上只放活跃的那一个（0 = 关闭，旧行为）
  int kv_slots = 0;
  std::string kv_slot_dir = "/dev/shm/rk1828-kv";
  int kv_slot_min_tokens = 1024;  // 前缀短于此就不如直接重算，别浪费拷贝时间
  int kv_slot_max_image_mb = 1536;  // 单槽镜像上限（64K 上下文实测约 2.4 GB，超出就丢弃该槽镜像）
  int kv_slot_max_total_mb = 4096;  // 所有镜像总预算（超出按 LRU 丢弃）

  // 流式工具调用：true = 按 OpenAI 规范增量下发 delta.tool_calls 并从正文里隐藏模板标记
  bool stream_tool_parse = true;
  // tool_choice=required/指定函数时用 structural_tag 语法强制结构（失败会退化为提示词约束）
  bool tool_grammar = true;
  // thinking 开启时用 structural_tag 强制"思考段 + 结束标记 + 自由回答"这个形状。
  // 默认 off：1.2.0 的运行时不匹配 grammar 里的思考前缀（把结束标记换成无意义字符串模型也不认；
  // 加硬性 JSON 约束也照样在思考通道里选 EOS）。实测不能靠它消除空回答，见 docs/server.md 4.3。
  bool thinking_grammar = false;
  // /v1/embeddings：词嵌入均值池化（近似嵌入，非训练过的 embedding 模型）
  bool embeddings = true;
};

static AppConfig g_cfg;
static Engine g_engine;
static KvSlots g_slots;
static std::atomic<bool> g_ready(false);
static std::string g_kv_probe = "off";  // off | ok | unsupported
static std::atomic<long> g_req_id(0);
static std::atomic<long> g_waiting(0);
static int g_max_queue = 64;
static long g_start_time = 0;

// ------------------------------------------------------------------ FIFO 排队闸门
// std::mutex 不保证先来后到；这里用票据 + 条件变量让请求严格按到达顺序拿到卡。
class FifoGate {
 public:
  void lock() {
    std::unique_lock<std::mutex> lk(m_mu);
    long t = m_next++;
    while (t != m_serving) {
      m_cv.wait(lk);
    }
  }
  void unlock() {
    {
      std::lock_guard<std::mutex> lk(m_mu);
      ++m_serving;
    }
    m_cv.notify_all();
  }

 private:
  std::mutex m_mu;
  std::condition_variable m_cv;
  long m_next = 0;
  long m_serving = 0;
};

class FifoGuard {
 public:
  explicit FifoGuard(FifoGate& g) : m_g(g) { m_g.lock(); }
  ~FifoGuard() { m_g.unlock(); }

 private:
  FifoGate& m_g;
};

static FifoGate g_engine_gate;  // 单 session：串行化生成（请求按到达顺序排队）

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
    j["kv_slots"] = g_cfg.kv_slots;
    j["kv_slot_resident"] = g_slots.resident_id();
    j["kv_image_bytes"] = g_slots.image_bytes();
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

static std::string trim_copy(const std::string& s) {
  size_t a = 0;
  size_t b = s.size();
  while (a < b && isspace((unsigned char)s[a])) {
    ++a;
  }
  while (b > a && isspace((unsigned char)s[b - 1])) {
    --b;
  }
  return s.substr(a, b - a);
}

static void trim_front(std::string& s) {
  size_t a = 0;
  while (a < s.size() && (s[a] == ' ' || s[a] == '\n' || s[a] == '\r' || s[a] == '\t')) {
    ++a;
  }
  if (a) {
    s.erase(0, a);
  }
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
      } else if (it.is_object() && it.contains("content") && it["content"].is_string()) {
        out += it["content"].get<std::string>();  // Responses 的 input_text/output_text
      }
    }
    return out;
  }
  if (c.is_null()) {
    return "";
  }
  return safe_dump(c);
}

static std::string rand_hex(int n) {
  static const char* hexd = "0123456789abcdef";
  std::string s;
  s.reserve((size_t)n);
  for (int i = 0; i < n; ++i) {
    s += hexd[rand() & 15];
  }
  return s;
}

static std::string rand_id() { return "chatcmpl-" + rand_hex(24); }

// tool_call 的 id 必须跨请求、跨并发都不撞车（客户端用它把工具结果对应回去）
static std::string new_tool_call_id() {
  static std::atomic<long> seq(0);
  return "call_" + std::to_string(++seq) + "_" + rand_hex(12);
}

static void endswith_trim_special(std::string& s) {
  bool changed = true;
  while (changed) {
    changed = false;
    const char* tails[] = {SPK_EOS, "<|im_end|>", "<|end|>", TAG_TOOL_CLOSE, NULL};
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

static std::string spark_tools_block(const json& tools, const std::string& forced_tool, bool require_tool,
                                     bool single_tool) {
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
  if (require_tool) {
    if (!forced_tool.empty()) {
      out += "\nYou must respond by calling the function `" + forced_tool +
             "` now. Do not answer in plain text.";
    } else {
      out += "\nYou must respond by calling one of the functions above now. Do not answer in plain text.";
    }
    if (single_tool) {
      out += " Emit exactly one tool call.";
    }
  }
  return out;
}

static std::string spark_tool_calls_text(const json& m) {
  // 官方格式： 工具调用标签 + 函数名 + arg_key/arg_value 序列
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
    out += std::string(TAG_TOOL_OPEN) + name;
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
          out += std::string(TAG_ARGK_OPEN) + it.key() + TAG_ARGK_CLOSE + TAG_ARGV_OPEN + val + TAG_ARGV_CLOSE;
        }
      }
    }
    out += TAG_TOOL_CLOSE;
  }
  return out;
}

static bool is_system_role(const std::string& r) {  // OpenAI 新版用 developer 取代 system
  return r == "system" || r == "developer";
}

static bool is_tool_role(const std::string& r) {  // 遗留形态 role:"function" 等同 role:"tool"
  return r == "tool" || r == "function";
}

static PromptParts compose_spark(const json& messages, bool thinking, const std::string& default_system,
                                 const json& tools, const std::string& reasoning_effort, bool inject_effort,
                                 bool system_in_prompt, const std::string& forced_tool, bool require_tool,
                                 bool single_tool) {
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
      messages[0]["role"].is_string() && is_system_role(messages[0]["role"].get<std::string>()) &&
      messages[0].contains("content")) {
    initial_system = json_content_to_text(messages[0]["content"]);
  }

  std::string tools_block = spark_tools_block(tools, forced_tool, require_tool, single_tool);
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
    if (is_system_role(role) && i == 0) {
      continue;  // 已并进首个 system 块
    }
    if (is_system_role(role)) {
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
        t.body = std::string(TAG_THINK_OPEN) + rc + TAG_THINK_CLOSE + content + spark_tool_calls_text(m);
      } else {
        t.body = std::string(TAG_THINK_CLOSE) + content + spark_tool_calls_text(m);
      }
      turns.push_back(t);
    } else if (is_tool_role(role)) {
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
  std::string gen_prompt;  // 生成引导；最后一轮已是 assistant 续写时为空
  if (turns.back().marker != SPK_BOT) {
    gen_prompt = std::string(SPK_EOS) + SPK_SOS + SPK_BOT + (thinking ? std::string(TAG_THINK_OPEN) : std::string(TAG_THINK_CLOSE));
  }

  p.last_role = (turns.back().marker == SPK_TOOL) ? "tool" : "user";
  if (system_in_prompt) {
    // 实测（RKNN3 1.2.0 + Spark-X2.6）：set_chat_template 的 system_prompt 根本不进模型上下文
    // （往 system 里放独有信息模型读不到，prompt_tokens 也不计它），官方 multicard 示例同样
    // 建议"自己把模板拼进 prompt"。所以这里把 system 块、各轮标记与生成引导全写进 prompt 文本。
    p.text = p.system + std::string(SPK_SOS) + turns[0].marker + text + gen_prompt;
    p.system.clear();
    p.prefix.clear();
    p.postfix.clear();
  } else {
    p.prefix = std::string(SPK_SOS) + turns[0].marker;
    p.postfix = gen_prompt;
    p.text = text;
  }
  return p;
}

// ------------------------------------------------------------------ tools
static std::string tools_runtime_json(const json& tools) {
  // runtime 期望的就是 OpenAI 的 tools 数组本身
  return safe_dump(tools);
}

static ordered_json tool_call_json(const std::string& id, const std::string& name, const std::string& args) {
  ordered_json call;
  call["id"] = id;
  call["type"] = "function";
  call["function"]["name"] = name;
  call["function"]["arguments"] = args.empty() ? std::string("{}") : args;
  return call;
}

// 非流式/兜底解析：把正文里的工具调用块抽出来（只作用于答案通道）
static void extract_tool_calls(const std::string& text, std::string& clean, ordered_json& calls) {
  clean = text;
  std::regex re(std::string(TAG_TOOL_OPEN) + "([\\s\\S]*?)" + TAG_TOOL_CLOSE);
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
    std::string name;
    std::string args_json;

    // 首选官方 Spark 形态：标签后是函数名，随后是 arg_key/arg_value 序列
    std::regex arg_re(std::string(TAG_ARGK_OPEN) + "([\\s\\S]*?)" + TAG_ARGK_CLOSE + TAG_ARGV_OPEN +
                      "([\\s\\S]*?)" + TAG_ARGV_CLOSE);
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
    if (name.empty()) {
      continue;  // 空函数名不是可执行的工具调用
    }
    calls.push_back(tool_call_json(new_tool_call_id(), name, args_json));
    idx++;
  }
  if (idx > 0) {
    clean = std::regex_replace(text, re, "");
    clean = std::regex_replace(clean, std::regex(std::string(TAG_ARGK_OPEN) + "[\\s\\S]*?" + TAG_ARGV_CLOSE), "");
    // 去掉残留的半截标记
    const char* leftovers[] = {TAG_TOOL_OPEN, TAG_ARGK_OPEN, TAG_ARGV_CLOSE, NULL};
    for (int i = 0; leftovers[i]; ++i) {
      size_t pos;
      while ((pos = clean.find(leftovers[i])) != std::string::npos) {
        clean.erase(pos, strlen(leftovers[i]));
      }
    }
    endswith_trim_special(clean);
  }
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
static bool build_runtime_grammar(const json& rf, bool has_rf, bool thinking, const std::string& forced_tool,
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

// ------------------------------------------------------------------ 流式解析（通道分离 + 工具调用增量）
// 切点回退到 UTF-8 字符边界：每个 SSE data 行必须是独立合法的 UTF-8，
// 不能把一个多字节字符拆到两个事件里（否则 safe_dump 会把它替换成 U+FFFD）。
static size_t utf8_cut_back(const std::string& s, size_t pos) {
  while (pos > 0 && pos < s.size() && ((unsigned char)s[pos] & 0xC0) == 0x80) {
    --pos;
  }
  return pos;
}

class StreamHandler {
 public:
  virtual ~StreamHandler() {}
  virtual bool on_text(bool reasoning, const std::string& text) = 0;
  virtual bool on_tool_start(int index, const std::string& id, const std::string& name) = 0;
  virtual bool on_tool_args(int index, const std::string& frag) = 0;
  virtual bool on_tool_end(int index, const std::string& args) = 0;
};

// 把模型的原始文本流切成 reasoning / content / tool_calls 三个通道，
// 工具调用按 OpenAI 规范增量产出（首片带 id/type/name，后续片只带 index+arguments）。
class StreamParser : public TextSink {
 public:
  StreamParser(StreamHandler& h, bool split_reason, bool tools)
      : m_h(h), m_split(split_reason), m_tools(tools) {}

  bool on_text(const std::string& piece) override {
    if (m_gone) {
      return false;
    }
    m_buf += piece;
    return drain(false);
  }

  void finish() {
    if (m_gone) {
      return;
    }
    drain(true);
    if (m_chan == CH_REASON && !m_buf.empty()) {
      m_h.on_text(true, m_buf);
      m_buf.clear();
    }
    if ((m_chan == CH_CONTENT) && !m_buf.empty()) {
      m_h.on_text(false, m_buf);
      m_buf.clear();
    }
  }

  int tool_count() const { return m_tool_calls; }
  bool split_done() const { return m_split_done; }
  bool client_gone() const { return m_gone; }

 private:
  enum Chan { CH_REASON, CH_CONTENT, CH_TOOL, CH_TAIL };
  enum Step { STEP_PROGRESS, STEP_WAIT, STEP_ERROR };

  bool emit_text(bool reasoning, const std::string& s) {
    if (s.empty()) {
      return true;
    }
    if (!m_h.on_text(reasoning, s)) {
      m_gone = true;
      return false;
    }
    return true;
  }

  bool start_tool(const std::string& name) {
    m_tool_index = m_tool_calls++;
    m_args_open = false;
    m_args_text.clear();
    if (!m_h.on_tool_start(m_tool_index, new_tool_call_id(), name)) {
      m_gone = true;
      return false;
    }
    return true;
  }

  bool emit_args(const std::string& frag) {
    if (frag.empty()) {
      return true;
    }
    m_args_text += frag;
    if (!m_h.on_tool_args(m_tool_index, frag)) {
      m_gone = true;
      return false;
    }
    return true;
  }

  bool end_tool() {
    if (!m_h.on_tool_end(m_tool_index, m_args_text)) {
      m_gone = true;
      return false;
    }
    return true;
  }

  Step advance_tool(bool eof) {
    if (m_name_phase) {
      size_t first = m_buf.find_first_not_of(" \t\r\n");
      if (first != std::string::npos && m_buf[first] == '{') {
        // 兼容 JSON-in-XML 形态：等整块结束再按 JSON 解析
        size_t tc = m_buf.find(TAG_TOOL_CLOSE);
        if (tc == std::string::npos) {
          return eof ? STEP_ERROR : STEP_WAIT;
        }
        std::string block = m_buf.substr(0, tc);
        m_buf.erase(0, tc + strlen(TAG_TOOL_CLOSE));
        json parsed;
        bool ok = false;
        try {
          parsed = json::parse(block);
          ok = parsed.is_object();
        } catch (...) {
          ok = false;
        }
        std::string name = ok ? parsed.value("name", parsed.value("tool_name", std::string())) : std::string();
        std::string args = "{}";
        if (ok && parsed.contains("arguments")) {
          args = parsed["arguments"].is_string() ? parsed["arguments"].get<std::string>() : safe_dump(parsed["arguments"]);
        }
        if (!name.empty()) {
          if (!start_tool(name)) {
            return STEP_ERROR;
          }
          if (!emit_args(args)) {
            return STEP_ERROR;
          }
          if (!end_tool()) {
            return STEP_ERROR;
          }
        }
        m_chan = CH_TAIL;
        return STEP_PROGRESS;
      }
      size_t a = m_buf.find(TAG_ARGK_OPEN);
      size_t b = m_buf.find(TAG_TOOL_CLOSE);
      if (a != std::string::npos && (b == std::string::npos || a < b)) {
        std::string name = trim_copy(m_buf.substr(0, a));
        m_buf.erase(0, a);
        m_name_phase = false;
        if (name.empty()) {
          m_chan = CH_TAIL;  // 空函数名：丢弃这个块
          return STEP_ERROR;
        }
        return start_tool(name) ? STEP_PROGRESS : STEP_ERROR;
      }
      if (b != std::string::npos) {
        std::string name = trim_copy(m_buf.substr(0, b));
        m_buf.erase(0, b + strlen(TAG_TOOL_CLOSE));
        m_name_phase = false;
        if (!name.empty()) {
          if (!start_tool(name)) {
            return STEP_ERROR;
          }
          if (!emit_args("{}")) {
            return STEP_ERROR;
          }
          if (!end_tool()) {
            return STEP_ERROR;
          }
        }
        m_chan = CH_TAIL;
        return STEP_PROGRESS;
      }
      if (eof) {
        m_buf.clear();
        m_chan = CH_TAIL;
        return STEP_PROGRESS;
      }
      return STEP_WAIT;
    }

    size_t tc = m_buf.find(TAG_TOOL_CLOSE);
    size_t ks = m_buf.find(TAG_ARGK_OPEN);
    if (ks != std::string::npos && (tc == std::string::npos || ks < tc)) {
      size_t klen = strlen(TAG_ARGK_OPEN);
      size_t ke = m_buf.find(TAG_ARGK_CLOSE, ks + klen);
      if (ke == std::string::npos) {
        return eof ? STEP_ERROR : STEP_WAIT;
      }
      size_t vs = m_buf.find(TAG_ARGV_OPEN, ke);
      if (vs == std::string::npos) {
        return eof ? STEP_ERROR : STEP_WAIT;
      }
      size_t ve = m_buf.find(TAG_ARGV_CLOSE, vs + strlen(TAG_ARGV_OPEN));
      if (ve == std::string::npos) {
        return eof ? STEP_ERROR : STEP_WAIT;
      }
      std::string k = trim_copy(m_buf.substr(ks + klen, ke - (ks + klen)));
      std::string v = m_buf.substr(vs + strlen(TAG_ARGV_OPEN), ve - (vs + strlen(TAG_ARGV_OPEN)));
      trim_front(v);
      while (!v.empty() && isspace((unsigned char)v.back())) {
        v.pop_back();
      }
      json jv;
      try {
        jv = json::parse(v);
      } catch (...) {
        jv = json(v);
      }
      std::string frag = m_args_open ? "," : "{";
      m_args_open = true;
      frag += safe_dump(k);
      frag += ":";
      frag += safe_dump(jv);
      if (!emit_args(frag)) {
        return STEP_ERROR;
      }
      m_buf.erase(0, ve + strlen(TAG_ARGV_CLOSE));
      return STEP_PROGRESS;
    }
    if (tc != std::string::npos) {
      std::string rest = trim_copy(m_buf.substr(0, tc));
      if (!m_args_open && !rest.empty()) {
        // 兼容：整块是裸 JSON 或裸参数
        json parsed;
        bool ok = false;
        try {
          parsed = json::parse(rest);
          ok = parsed.is_object();
        } catch (...) {
          ok = false;
        }
        if (!emit_args(ok ? safe_dump(parsed) : rest)) {
          return STEP_ERROR;
        }
      } else if (m_args_open) {
        if (!emit_args("}")) {
          return STEP_ERROR;
        }
      } else {
        if (!emit_args("{}")) {
          return STEP_ERROR;
        }
      }
      m_buf.erase(0, tc + strlen(TAG_TOOL_CLOSE));
      if (!end_tool()) {
        return STEP_ERROR;
      }
      m_chan = CH_TAIL;
      return STEP_PROGRESS;
    }
    if (eof) {
      if (m_args_open) {
        emit_args("}");
        end_tool();
      } else {
        emit_args("{}");
        end_tool();
      }
      m_buf.clear();
      m_chan = CH_TAIL;
      return STEP_PROGRESS;
    }
    return STEP_WAIT;
  }

  bool drain(bool eof) {
    for (;;) {
      if (m_gone) {
        return false;
      }
      if (m_chan == CH_REASON) {
        if (!m_split) {
          m_chan = CH_CONTENT;
          continue;
        }
        size_t p = m_buf.find(TAG_THINK_CLOSE);
        size_t q = m_tools ? m_buf.find(TAG_TOOL_OPEN) : std::string::npos;
        if (p != std::string::npos && (q == std::string::npos || p < q)) {
          if (p > 0 && !emit_text(true, m_buf.substr(0, p))) {
            return false;
          }
          m_buf.erase(0, p + strlen(TAG_THINK_CLOSE));
          m_split_done = true;
          m_chan = CH_CONTENT;
          continue;
        }
        if (q != std::string::npos) {
          if (q > 0 && !emit_text(true, m_buf.substr(0, q))) {
            return false;
          }
          m_buf.erase(0, q + strlen(TAG_TOOL_OPEN));
          m_chan = CH_TOOL;
          m_name_phase = true;
          continue;
        }
        size_t keep = strlen(TAG_THINK_CLOSE);
        if (m_tools && strlen(TAG_TOOL_OPEN) > keep) {
          keep = strlen(TAG_TOOL_OPEN);
        }
        keep = keep ? keep - 1 : 0;
        if (eof || m_buf.size() > keep) {
          size_t cut = eof ? m_buf.size() : m_buf.size() - keep;
          cut = utf8_cut_back(m_buf, cut);
          if (cut > 0) {
            if (!emit_text(true, m_buf.substr(0, cut))) {
              return false;
            }
            m_buf.erase(0, cut);
          }
        }
        return true;
      }
      if (m_chan == CH_CONTENT || m_chan == CH_TAIL) {
        size_t q = m_tools ? m_buf.find(TAG_TOOL_OPEN) : std::string::npos;
        if (q != std::string::npos) {
          if (m_chan == CH_CONTENT && q > 0 && !emit_text(false, m_buf.substr(0, q))) {
            return false;
          }
          m_buf.erase(0, q + strlen(TAG_TOOL_OPEN));
          m_chan = CH_TOOL;
          m_name_phase = true;
          continue;
        }
        size_t keep = m_tools ? strlen(TAG_TOOL_OPEN) - 1 : 0;
        if (eof || m_buf.size() > keep) {
          size_t cut = eof ? m_buf.size() : m_buf.size() - keep;
          if (m_chan == CH_CONTENT) {
            cut = utf8_cut_back(m_buf, cut);
          }
          if (cut > 0) {
            if (m_chan == CH_CONTENT && !emit_text(false, m_buf.substr(0, cut))) {
              return false;
            }
            m_buf.erase(0, cut);
          }
        }
        return true;
      }
      // CH_TOOL
      Step st = advance_tool(eof);
      if (st == STEP_WAIT) {
        return true;
      }
      if (st == STEP_ERROR) {
        if (m_chan == CH_TOOL) {
          m_chan = CH_TAIL;
        }
        continue;
      }
      continue;
    }
  }

  StreamHandler& m_h;
  bool m_split;
  bool m_tools;
  bool m_split_done = false;
  bool m_gone = false;
  Chan m_chan = CH_REASON;
  std::string m_buf;
  bool m_name_phase = false;
  bool m_args_open = false;
  int m_tool_index = -1;
  int m_tool_calls = 0;
  std::string m_args_text;
};

// ------------------------------------------------------------------ chat.completion.chunk 出口
class ChatSse : public StreamHandler {
 public:
  ChatSse(ResponseWriter& w, const std::string& id, const std::string& model, bool stream_tools = true)
      : m_w(w), m_id(id), m_model(model), m_stream_tools(stream_tools) {}

  ordered_json base() const {
    ordered_json j;
    j["id"] = m_id;
    j["object"] = "chat.completion.chunk";
    j["created"] = (long)time(NULL);
    j["model"] = m_model;
    j["system_fingerprint"] = g_fingerprint;
    return j;
  }

  void emit_role(bool tools) {
    ordered_json delta;
    delta["role"] = "assistant";
    if (!tools) {
      delta["content"] = "";
    }
    emit_delta(delta);
  }

  bool on_text(bool reasoning, const std::string& text) override {
    ordered_json delta;
    delta[reasoning ? "reasoning_content" : "content"] = text;
    return emit_delta(delta);
  }

  bool on_tool_start(int index, const std::string& id, const std::string& name) override {
    ordered_json tc;
    tc["index"] = index;
    if (!m_stream_tools) {
      tc["id"] = id;
      tc["type"] = "function";
      tc["function"]["name"] = name;
      tc["function"]["arguments"] = "";
      m_cur = tc;
      return true;
    }
    tc["id"] = id;
    tc["type"] = "function";
    tc["function"]["name"] = name;
    tc["function"]["arguments"] = "";
    ordered_json delta;
    delta["tool_calls"] = ordered_json::array({tc});
    return emit_delta(delta);
  }

  bool on_tool_args(int index, const std::string& frag) override {
    if (!m_stream_tools) {
      if (m_cur.is_object() && m_cur.contains("function")) {
        std::string acc = m_cur["function"].value("arguments", std::string(""));
        m_cur["function"]["arguments"] = acc + frag;
      }
      return true;
    }
    ordered_json tc;
    tc["index"] = index;
    tc["function"]["arguments"] = frag;
    ordered_json delta;
    delta["tool_calls"] = ordered_json::array({tc});
    return emit_delta(delta);
  }

  bool on_tool_end(int index, const std::string& args) override {
    (void)index;
    (void)args;
    if (!m_stream_tools && m_cur.is_object()) {
      m_buf_calls.push_back(m_cur);
      m_cur = ordered_json::object();
    }
    return true;
  }

  bool emit_delta(const ordered_json& delta) {
    ordered_json j = base();
    ordered_json ch;
    ch["index"] = 0;
    ch["delta"] = delta;
    ch["finish_reason"] = nullptr;
    j["choices"] = ordered_json::array({ch});
    return m_w.write_sse_data(safe_dump(j));
  }

  bool emit_final(const std::string& finish, const json* usage) {
    if (!m_stream_tools && !m_buf_calls.empty()) {
      ordered_json delta;
      delta["tool_calls"] = m_buf_calls;
      if (!emit_delta(delta)) {
        return false;
      }
    }
    ordered_json j = base();
    ordered_json ch;
    ch["index"] = 0;
    ch["delta"] = ordered_json::object();
    ch["finish_reason"] = finish;
    j["choices"] = ordered_json::array({ch});
    if (!m_w.write_sse_data(safe_dump(j))) {
      return false;
    }
    if (usage) {
      ordered_json u = base();
      u["choices"] = ordered_json::array();
      u["usage"] = *usage;
      return m_w.write_sse_data(safe_dump(u));
    }
    return true;
  }

  bool alive() const { return m_w.alive(); }

 private:
  ResponseWriter& m_w;
  std::string m_id, m_model;
  bool m_stream_tools = true;
  ordered_json m_cur = ordered_json::object();
  ordered_json m_buf_calls = ordered_json::array();
};

// ------------------------------------------------------------------ OpenAI Responses API 出口
class ResponsesSse : public StreamHandler {
 public:
  ResponsesSse(ResponseWriter& w, const std::string& id, const std::string& model)
      : m_w(w), m_id(id), m_model(model) {}

  void emit_created() {
    ordered_json r;
    r["id"] = m_id;
    r["object"] = "response";
    r["created_at"] = (long)time(NULL);
    r["status"] = "in_progress";
    r["model"] = m_model;
    r["output"] = ordered_json::array();
    ordered_json j;
    j["type"] = "response.created";
    j["response"] = r;
    send(j);
  }

  bool on_text(bool reasoning, const std::string& text) override {
    if (reasoning) {
      return true;  // Responses 路由暂不推送 reasoning 事件（正文与工具调用不受影响）
    }
    if (!ensure_message()) {
      return false;
    }
    if (!send_text_delta(text)) {
      return false;
    }
    m_text += text;
    return true;
  }

  bool on_tool_start(int index, const std::string& id, const std::string& name) override {
    (void)index;
    m_cur = Call();
    m_cur.id = "fc_" + rand_hex(16);
    m_cur.call_id = id;
    m_cur.name = name;
    m_cur.output_index = m_next_output++;
    ordered_json item;
    item["type"] = "function_call";
    item["id"] = m_cur.id;
    item["call_id"] = m_cur.call_id;
    item["name"] = m_cur.name;
    item["arguments"] = "";
    item["status"] = "in_progress";
    ordered_json j;
    j["type"] = "response.output_item.added";
    j["output_index"] = m_cur.output_index;
    j["item"] = item;
    return send(j);
  }

  bool on_tool_args(int index, const std::string& frag) override {
    (void)index;
    m_cur.args += frag;
    ordered_json j;
    j["type"] = "response.function_call_arguments.delta";
    j["item_id"] = m_cur.id;
    j["output_index"] = m_cur.output_index;
    j["delta"] = frag;
    return send(j);
  }

  bool on_tool_end(int index, const std::string& args) override {
    (void)index;
    if (m_cur.args.empty()) {
      m_cur.args = args.empty() ? "{}" : args;
    }
    ordered_json d;
    d["type"] = "response.function_call_arguments.done";
    d["item_id"] = m_cur.id;
    d["output_index"] = m_cur.output_index;
    d["arguments"] = m_cur.args;
    if (!send(d)) {
      return false;
    }
    ordered_json item;
    item["type"] = "function_call";
    item["id"] = m_cur.id;
    item["call_id"] = m_cur.call_id;
    item["name"] = m_cur.name;
    item["arguments"] = m_cur.args;
    item["status"] = "completed";
    ordered_json j;
    j["type"] = "response.output_item.done";
    j["output_index"] = m_cur.output_index;
    j["item"] = item;
    if (!send(j)) {
      return false;
    }
    m_calls.push_back(m_cur);
    m_cur = Call();
    return true;
  }

  void finish(const json& usage, const std::string& status) {
    if (m_msg_open) {
      ordered_json d;
      d["type"] = "response.output_text.done";
      d["item_id"] = m_msg_id;
      d["output_index"] = m_msg_index;
      d["content_index"] = 0;
      d["text"] = m_text;
      send(d);
      ordered_json item;
      item["type"] = "message";
      item["id"] = m_msg_id;
      item["status"] = "completed";
      item["role"] = "assistant";
      item["content"] = ordered_json::array({ordered_json{
          {"type", "output_text"}, {"text", m_text}, {"annotations", ordered_json::array()}}});
      ordered_json j;
      j["type"] = "response.output_item.done";
      j["output_index"] = m_msg_index;
      j["item"] = item;
      send(j);
    }
    ordered_json output = ordered_json::array();
    for (size_t i = 0; i < m_calls.size(); ++i) {
      ordered_json item;
      item["type"] = "function_call";
      item["id"] = m_calls[i].id;
      item["call_id"] = m_calls[i].call_id;
      item["name"] = m_calls[i].name;
      item["arguments"] = m_calls[i].args;
      item["status"] = "completed";
      output.push_back(item);
    }
    if (m_msg_open) {
      ordered_json item;
      item["type"] = "message";
      item["id"] = m_msg_id;
      item["status"] = "completed";
      item["role"] = "assistant";
      item["content"] = ordered_json::array({ordered_json{
          {"type", "output_text"}, {"text", m_text}, {"annotations", ordered_json::array()}}});
      output.push_back(item);
    }
    ordered_json r;
    r["id"] = m_id;
    r["object"] = "response";
    r["created_at"] = (long)time(NULL);
    r["status"] = status;
    r["model"] = m_model;
    r["output"] = output;
    r["usage"] = usage;
    ordered_json j;
    j["type"] = "response.completed";
    j["response"] = r;
    send(j);
  }

  int tool_count() const { return (int)m_calls.size(); }
  bool has_message() const { return m_msg_open; }
  const std::string& text() const { return m_text; }

 private:
  struct Call {
    std::string id, call_id, name, args;
    int output_index = 0;
  };

  bool send(const ordered_json& j) { return m_w.write_sse_data(safe_dump(j)); }

  bool ensure_message() {
    if (m_msg_open) {
      return true;
    }
    m_msg_open = true;
    m_msg_id = "msg_" + rand_hex(16);
    m_msg_index = m_next_output++;
    ordered_json item;
    item["type"] = "message";
    item["id"] = m_msg_id;
    item["status"] = "in_progress";
    item["role"] = "assistant";
    item["content"] = ordered_json::array();
    ordered_json j;
    j["type"] = "response.output_item.added";
    j["output_index"] = m_msg_index;
    j["item"] = item;
    if (!send(j)) {
      return false;
    }
    ordered_json part;
    part["type"] = "output_text";
    part["text"] = "";
    part["annotations"] = ordered_json::array();
    ordered_json c;
    c["type"] = "response.content_part.added";
    c["item_id"] = m_msg_id;
    c["output_index"] = m_msg_index;
    c["content_index"] = 0;
    c["part"] = part;
    return send(c);
  }

  bool send_text_delta(const std::string& text) {
    ordered_json j;
    j["type"] = "response.output_text.delta";
    j["item_id"] = m_msg_id;
    j["output_index"] = m_msg_index;
    j["content_index"] = 0;
    j["delta"] = text;
    return send(j);
  }

  ResponseWriter& m_w;
  std::string m_id, m_model;
  bool m_msg_open = false;
  std::string m_msg_id;
  int m_msg_index = 0;
  std::string m_text;
  Call m_cur;
  std::vector<Call> m_calls;
  int m_next_output = 0;
};

// 非流式累积器：与流式走同一套解析，只是把片段攒起来
class AccHandler : public StreamHandler {
 public:
  struct Call {
    std::string id, name, args;
  };
  std::string text, reasoning;
  std::vector<Call> calls;

  bool on_text(bool r, const std::string& t) override {
    if (r) {
      reasoning += t;
    } else {
      text += t;
    }
    return true;
  }
  bool on_tool_start(int index, const std::string& id, const std::string& name) override {
    (void)index;
    Call c;
    c.id = id;
    c.name = name;
    calls.push_back(c);
    return true;
  }
  bool on_tool_args(int index, const std::string& frag) override {
    (void)index;
    if (!calls.empty()) {
      calls.back().args += frag;
    }
    return true;
  }
  bool on_tool_end(int index, const std::string& args) override {
    (void)index;
    if (!calls.empty() && calls.back().args.empty()) {
      calls.back().args = args.empty() ? "{}" : args;
    }
    return true;
  }
};

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
  int thinking_grammar = -1;  // -1 = 用服务端配置；0/1 = 请求级覆盖"思考段必须闭合"的 grammar
  int top_logprobs = 0;
  std::vector<int32_t> lb_ids;
  std::vector<float> lb_vals;
  std::vector<std::string> stops;
  std::string grammar_str;
  int grammar_type = 0;
  bool use_grammar = false;
  bool use_tools = false;
  std::string tools_json;
  std::string tool_choice = "auto";
  bool require_tool = false;   // tool_choice=required 或指定了函数
  std::string forced_tool;     // 指定函数名（tool_choice={"type":"function",...}）
  bool single_tool = false;    // parallel_tool_calls=false
  bool err = false;
  std::string err_msg;
  std::string err_code;
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

// ------------------------------------------------------------------ 生成主流程
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
        g_slots.commit(plan.slot, pp.text, res.prompt_text_tokens, res.text, res.completion_tokens,
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
  std::vector<int32_t> ptoks;
  int ntok = g_engine.encode_tokens(pp.text, ptoks);
  int max_ctx = g_engine.model_max_ctx();
  if (max_ctx > 0 && ntok > 0) {
    if (ntok >= max_ctx - 8) {
      err_json(w, 400,
               "prompt is " + std::to_string(ntok) + " tokens, exceeding the loaded context window (" +
                   std::to_string(max_ctx) + " tokens)",
               "invalid_request_error", "context_length_exceeded");
      return;
    }
    if (cp.max_new > 0 && (long)ntok + cp.max_new > (long)max_ctx - 8) {
      cp.max_new = (int)((long)max_ctx - 8 - ntok);
      if (cp.max_new < 1) {
        cp.max_new = 1;
      }
    }
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
  gr.use_tools = cp.use_tools;
  gr.tools_json = cp.tools_json;
  gr.relax_guard = cp.use_tools;
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

  if (g_waiting.load() >= g_max_queue) {
    err_json(w, 429, "inference queue is full, retry later", "rate_limit_error", "");
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

  if (g_waiting.load() >= g_max_queue) {
    err_json(w, 429, "inference queue is full, retry later", "rate_limit_error", "");
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

  std::vector<int32_t> ptoks;
  int ntok = g_engine.encode_tokens(pp.text, ptoks);
  int max_ctx = g_engine.model_max_ctx();
  if (max_ctx > 0 && ntok > 0) {
    if (ntok >= max_ctx - 8) {
      err_json(w, 400, "prompt is " + std::to_string(ntok) + " tokens, exceeding the loaded context window (" +
                            std::to_string(max_ctx) + " tokens)",
               "invalid_request_error", "context_length_exceeded");
      return;
    }
    if (cp.max_new > 0 && (long)ntok + cp.max_new > (long)max_ctx - 8) {
      cp.max_new = (int)((long)max_ctx - 8 - ntok);
      if (cp.max_new < 1) {
        cp.max_new = 1;
      }
    }
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

  if (g_waiting.load() >= g_max_queue) {
    err_json(w, 429, "inference queue is full, retry later", "rate_limit_error", "");
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
    json usage;
    usage["input_tokens"] = res.prompt_tokens;
    usage["output_tokens"] = res.completion_tokens;
    usage["total_tokens"] = res.prompt_tokens + res.completion_tokens;
    usage["input_tokens_details"]["cached_tokens"] = res.reused_tokens;
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
  ordered_json u;
  u["input_tokens"] = res.prompt_tokens;
  u["output_tokens"] = res.completion_tokens;
  u["total_tokens"] = res.prompt_tokens + res.completion_tokens;
  u["input_tokens_details"]["cached_tokens"] = res.reused_tokens;
  resp["usage"] = u;
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

// ------------------------------------------------------------------ 路由
static void dispatch(const HttpRequest& rq, ResponseWriter& w) {
  if (rq.method == "OPTIONS") {
    w.write_full(204, "", "text/plain");
    return;
  }
  // 健康检查不要求 api-key：外部的服务守护/容器探针拿不到密钥
  if ((rq.method == "GET" || rq.method == "HEAD") && (rq.path == "/health" || rq.path == "/healthz")) {
    do_health(w, rq.method == "HEAD");
    return;
  }
  if (!g_cfg.http.api_key.empty()) {
    std::string auth = hdr_of(rq, "authorization");
    std::string xkey = hdr_of(rq, "x-api-key");
    if (auth.find(g_cfg.http.api_key) == std::string::npos && xkey != g_cfg.http.api_key) {
      err_json(w, 401, "missing or invalid api key", "authentication_error", "");
      return;
    }
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

// ------------------------------------------------------------------ 参数解析
static void print_usage(const char* prog) {
  printf(
      "rknn_spark_server — RK1828 OpenAI-compatible LLM server (Spark-X2.5/X2.6)\n"
      "Usage: %s <llm.rknn> <llm.weight> <core_mask> <tokenizer.gguf> <embed.bin> <max_context_len> "
      "<safetensors> [options]\n\n"
      "  --host H                监听地址（默认 0.0.0.0）\n"
      "  --port P                端口（默认 8080）\n"
      "  --api-key K             需要 Authorization: Bearer K（/health 免鉴权）\n"
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
      "  --system-in-prompt on|off  整段 ChatML（含 system 块）自己拼进 prompt 文本（默认 on）。\n"
      "                          off = 改走 rknn3_session_set_chat_template 的 system_prompt；实测该参数\n"
      "                          对 Spark-X2.6 不生效（客户端 system 内容进不了模型），仅作对照用\n"
      "  --kv-slots N            KV 槽位数（0=关闭；最多 8，默认 0）。主机内存保存每个会话的 KV 镜像，\n"
      "                          卡上只放活跃槽；未命中且槽满时按 LRU 清理最不活跃的槽\n"
      "  --kv-slot-dir DIR       KV 镜像目录（默认 /dev/shm/rk1828-kv）\n"
      "  --kv-slot-min-tokens N  前缀短于此 token 数不建镜像（默认 1024）\n"
      "  --stream-tools on|off   流式 tool_calls 增量下发（默认 on；off = 末尾整包下发，仅排障用）\n"
      "  --tool-grammar on|off   tool_choice=required/指定函数时用 structural_tag 强制结构（默认 on）\n"
      "  --thinking-grammar on|off  思考开启时用 structural_tag 强制'思考段 -> 结束标记 -> 自由回答'\n"
      "                          （默认 off：1.2.0 运行时不匹配思考前缀，实测无效，见 docs/server.md 4.3）。\n"
      "                          请求级可用 chat_template_kwargs.thinking_grammar 覆盖\n"
      "  --embeddings on|off     /v1/embeddings（词嵌入均值池化；默认 on）\n"
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
    else if (a == "--kv-ckpt-start") { g_cfg.eng.kv_ckpt_start = atoi(v.c_str()); ++i; }
    else if (a == "--kv-ckpt-interval") { g_cfg.eng.kv_ckpt_interval = atoi(v.c_str()); ++i; }
    else if (a == "--kv-ckpt-count") { g_cfg.eng.kv_ckpt_count = atoi(v.c_str()); ++i; }
    else if (a == "--kv-align") { g_cfg.eng.kv_align_cfg = atoi(v.c_str()); ++i; }
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
    else if (a == "--system-in-prompt") { g_cfg.system_in_prompt = (v == "on" || v == "1" || v == "true"); ++i; }
    else if (a == "--kv-slots") { g_cfg.kv_slots = atoi(v.c_str()); ++i; }
    else if (a == "--kv-slot-dir") { g_cfg.kv_slot_dir = v; ++i; }
    else if (a == "--kv-slot-min-tokens") { g_cfg.kv_slot_min_tokens = atoi(v.c_str()); ++i; }
    else if (a == "--kv-slot-max-image-mb") { g_cfg.kv_slot_max_image_mb = atoi(v.c_str()); ++i; }
    else if (a == "--kv-slot-max-total-mb") { g_cfg.kv_slot_max_total_mb = atoi(v.c_str()); ++i; }
    else if (a == "--stream-tools") { g_cfg.stream_tool_parse = (v == "on" || v == "1" || v == "true"); ++i; }
    else if (a == "--tool-grammar") { g_cfg.tool_grammar = (v == "on" || v == "1" || v == "true"); ++i; }
    else if (a == "--thinking-grammar") { g_cfg.thinking_grammar = (v == "on" || v == "1" || v == "true"); ++i; }
    else if (a == "--embeddings") { g_cfg.embeddings = (v == "on" || v == "1" || v == "true"); ++i; }
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
    if (p.contains("kv-slots") && p["kv-slots"].is_number()) g_cfg.kv_slots = p["kv-slots"].get<int>();
    if (p.contains("kv-slot-dir") && p["kv-slot-dir"].is_string()) g_cfg.kv_slot_dir = p["kv-slot-dir"].get<std::string>();
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
  LOGF("model=%s core_mask=%02x ctx=%d sampler=%s kv_slots=%d", g_cfg.eng.model_path.c_str(), g_cfg.eng.core_mask,
       g_cfg.eng.max_context_len, g_cfg.use_runtime_sampler ? "runtime" : "custom(veto)", g_cfg.kv_slots);

  int ret = g_engine.init(g_cfg.eng);
  if (ret != 0) {
    LOGF("engine init failed ret=%d", ret);
    return 2;
  }
  g_slots.configure(g_cfg.kv_slots, g_cfg.kv_slot_dir, g_cfg.kv_slot_min_tokens,
                    (long)g_cfg.kv_slot_max_image_mb * 1024 * 1024, (long)g_cfg.kv_slot_max_total_mb * 1024 * 1024);
  if (g_cfg.kv_slots > 0) {
    // 启动自检：确认这个 runtime/模型是否真的支持 load_kvcache 后的前缀复用。
    // 实测 RKNN3 1.2.0 + Spark-X2.6：load 返回 0，但下一轮 reuse=0（KV 被重置），
    // 若继续按"只送新增文本"续接就会丢上下文，所以这里检测失败就自动关闭槽位。
    std::string probe;
    for (int i = 0; i < 60; ++i) {
      probe += "The quick brown fox jumps over the lazy dog. ";
    }
    GenRequest r1;
    r1.text = probe;
    r1.max_new_tokens = 1;
    r1.temperature = 0.0f;
    r1.thinking = false;
    r1.keep_kv_after = true;
    r1.timeout_ms = 30000;
    GenResult g1;
    int pr1 = g_engine.generate(r1, g1, NULL);
    int pr2 = -1;
    bool reuse_ok = false;
    if (pr1 == RKNN3_SUCCESS) {
      const char* tmp = "/tmp/rk1828-kv-selftest.kv";
      if (g_engine.save_kvcache(tmp) == 0) {
        g_engine.clear_kvcache();
        if (g_engine.load_kvcache(tmp) == 0) {
          GenRequest r2 = r1;
          r2.cached_prefix_tokens = (int)g1.kv_total_tokens;
          r2.keep_kv_after = false;
          GenResult g2;
          pr2 = g_engine.generate(r2, g2, NULL);
          reuse_ok = (pr2 == RKNN3_SUCCESS && g2.runtime_reused_tokens > 0);
          LOGF("kv self-test: ret=%d runtime_reuse=%ld prompt=%ld total=%ld", pr2, g2.runtime_reused_tokens,
               (long)g2.prompt_tokens, g2.kv_total_tokens);
        }
        unlink(tmp);
      }
      g_engine.clear_kvcache();
    }
    if (reuse_ok) {
      g_kv_probe = "ok";
      LOGF("kv self-test passed: runtime supports KV prefix reuse");
    } else {
      g_kv_probe = "unsupported";
      LOGF("kv self-test FAILED (ret=%d/%d): this runtime does not reuse a loaded KV across runs; "
           "KV slot paging disabled -- requests keep full history and stay correct", pr1, pr2);
      g_slots.configure(0, g_cfg.kv_slot_dir, g_cfg.kv_slot_min_tokens);
    }
  }
  g_ready = true;

  HttpServer srv;
  g_srv = &srv;
  if (srv.start(g_cfg.http, dispatch) != 0) {
    LOGF("http start failed");
    return 3;
  }
  LOGF("listening on http://%s:%d alias=%s ctx=%d thinking=%d temp=%.2f top_k=%d top_p=%.2f rep_pen=%.2f "
       "max_new=%d cap=%d conns=%d kv_slots=%d",
       g_cfg.http.host.c_str(), g_cfg.http.port, g_cfg.model_alias.c_str(), g_engine.model_max_ctx(),
       (int)g_cfg.default_thinking, g_cfg.temperature, g_cfg.top_k, g_cfg.top_p, g_cfg.repeat_penalty,
       g_cfg.max_new_tokens, g_cfg.max_new_tokens_cap, g_cfg.http.max_connections, g_cfg.kv_slots);

  while (!g_requested_shutdown) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  LOGF("shutdown requested, draining connections...");
  srv.stop();
  g_ready = false;
  {
    FifoGuard guard(g_engine_gate);
    g_slots.reset(g_engine);
  }
  g_engine.shutdown();
  LOGF("clean shutdown complete");
  return 0;
}
