// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// app.cpp — 全局对象、进程级指标与通用小工具
#include "app.h"

#include <cctype>

const char* const SPK_SOS = "<\uff5cstart\u2581of\u2581sentence\uff5c>";  // 词表 id 0
const char* const SPK_EOS = "<\uff5cend\u2581of\u2581sentence\uff5c>";    // 词表 id 1
const char* const SPK_SYS = "<|System|>\n";
const char* const SPK_USER = "<|User|>";
const char* const SPK_BOT = "<|Bot|>";

const char* const TAG_THINK_OPEN = " thinking";
const char* const TAG_THINK_CLOSE = " response";
const char* const TAG_TOOL_OPEN = "<tool_call>";
const char* const TAG_TOOL_CLOSE = "</tool_call>";
const char* const TAG_ARGK_OPEN = "<arg_key>";
const char* const TAG_ARGK_CLOSE = "</arg_key>";
const char* const TAG_ARGV_OPEN = "<arg_value>";
const char* const TAG_ARGV_CLOSE = "</arg_value>";

const char* const g_fingerprint = "fp_rk1828_spark_20261126";

AppConfig g_cfg;
Engine g_engine;
KvSlots g_slots;
std::atomic<bool> g_ready(false);
std::string g_kv_probe = "off";
std::atomic<long> g_req_id(0);
std::atomic<long> g_waiting(0);
int g_max_queue = 64;
long g_start_time = 0;
HttpServer* g_srv = nullptr;
volatile bool g_requested_shutdown = false;
FifoGate g_engine_gate;
Metrics g_m;

// ------------------------------------------------------------------ 指标
void Metrics::record(double ms, long tin, long tout, bool ok, bool guard, int veto, bool stream, bool tool) {
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

double Metrics::pct(std::vector<double> v, double p) {
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  size_t i = (size_t)(p * (double)(v.size() - 1));
  return v[i];
}

json Metrics::snapshot() {
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

long Metrics::conns_active() { return g_srv ? (long)g_srv->active_conns() : 0; }

// ------------------------------------------------------------------ 小工具
std::string hdr_of(const HttpRequest& rq, const char* key) {
  std::map<std::string, std::string>::const_iterator it = rq.headers.find(key);
  return it == rq.headers.end() ? std::string() : it->second;
}

bool truthy(const json& v, bool dflt) {
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

std::string safe_dump(const json& j) {
  // 模型输出可能带非法 UTF-8（BPE 片段拼接），默认 dump 是 strict 会抛异常 -> 整条请求 500
  return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string trim_copy(const std::string& s) {
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

void trim_front(std::string& s) {
  size_t a = 0;
  while (a < s.size() && (s[a] == ' ' || s[a] == '\n' || s[a] == '\r' || s[a] == '\t')) {
    ++a;
  }
  if (a) {
    s.erase(0, a);
  }
}

std::string rand_hex(int n) {
  static const char* hexd = "0123456789abcdef";
  std::string s;
  s.reserve((size_t)n);
  for (int i = 0; i < n; ++i) {
    s += hexd[rand() & 15];
  }
  return s;
}

std::string rand_id() { return "chatcmpl-" + rand_hex(24); }

// tool_call 的 id 必须跨请求、跨并发都不撞车（客户端用它把工具结果对应回去）
std::string new_tool_call_id() {
  static std::atomic<long> seq(0);
  return "call_" + std::to_string(++seq) + "_" + rand_hex(12);
}

void endswith_trim_special(std::string& s) {
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
