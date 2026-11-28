// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// app.h — 进程级共享状态：配置、全局对象、指标、FIFO 闸门、通用小工具、Spark 标记
#ifndef _RKNN_SPARK_APP_H_
#define _RKNN_SPARK_APP_H_

#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine.h"
#include "httpd.h"
#include "kvslots.h"
#include "time_utils.h"

using nlohmann::json;
using nlohmann::ordered_json;

extern const char* now_str();

#define LOGF(fmt, ...)                                                                                               \
  do {                                                                                                               \
    fprintf(stderr, "[%s] " fmt "\n", now_str(), ##__VA_ARGS__);                                                    \
    fflush(stderr);                                                                                                  \
  } while (0)

// ------------------------------------------------------------------ Spark 提示词标记
// 与 rknn3-model-zoo/examples/Spark_X2_5/cpp 里已验证可用的写法逐字一致
extern const char* const SPK_SOS;  // 词表 id 0
extern const char* const SPK_EOS;  // 词表 id 1
extern const char* const SPK_SYS;
extern const char* const SPK_USER;
extern const char* const SPK_BOT;

// 模板标签
extern const char* const TAG_THINK_OPEN;
extern const char* const TAG_THINK_CLOSE;
extern const char* const TAG_TOOL_OPEN;
extern const char* const TAG_TOOL_CLOSE;
extern const char* const TAG_ARGK_OPEN;
extern const char* const TAG_ARGK_CLOSE;
extern const char* const TAG_ARGV_OPEN;
extern const char* const TAG_ARGV_CLOSE;

extern const char* const g_fingerprint;

// ------------------------------------------------------------------ 配置
struct AppConfig {
  EngineConfig eng;
  HttpServerOptions http;

  // 默认采样参数对齐模型厂商官方推荐（ModelScope XHToken/Spark-X2.6-4B 的
  // generation_config.json 与 README 服务示例）：
  //   temperature=1.0  top_p=0.95  top_k=-1(=不截断，这里用 0 表示关闭)  repetition_penalty=1.0
  //   presence_penalty=0  frequency_penalty=0  do_sample=true  max_tokens=1048576（=不限制）
  // 注：rkllm3-server 1.2.0 的 --help 默认是 temp 0.8 / top_k 40 / top_p 0.9，那是 Rockchip 的默认，
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
  // 默认 off：1.2.0 的运行时不匹配 grammar 里的思考前缀，实测不能靠它消除空回答，见 docs/server.md 4.3。
  bool thinking_grammar = false;
  // /v1/embeddings：词嵌入均值池化（近似嵌入，非训练过的 embedding 模型）
  bool embeddings = true;
};

// ------------------------------------------------------------------ 全局对象
extern AppConfig g_cfg;
extern Engine g_engine;
extern KvSlots g_slots;
extern std::atomic<bool> g_ready;
extern std::string g_kv_probe;  // off | ok | unsupported
extern std::atomic<long> g_req_id;
extern std::atomic<long> g_waiting;
extern int g_max_queue;
extern long g_start_time;
extern HttpServer* g_srv;
extern volatile bool g_requested_shutdown;

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

extern FifoGate g_engine_gate;  // 单 session：串行化生成（请求按到达顺序排队）

// ------------------------------------------------------------------ 指标
struct Metrics {
  std::atomic<long> req_total{0}, req_ok{0}, req_err{0}, req_stream{0}, req_tool{0}, req_guard{0}, req_veto{0};
  std::atomic<long> tok_in{0}, tok_out{0}, conns{0}, active{0};
  std::atomic<long> lat_us_sum{0}, lat_us_max{0};
  std::mutex mu;
  std::vector<double> lat;  // 最近若干条端到端延迟（ms）

  void record(double ms, long tin, long tout, bool ok, bool guard, int veto, bool stream, bool tool);
  static double pct(std::vector<double> v, double p);
  json snapshot();
  long conns_active();
};

extern Metrics g_m;

// ------------------------------------------------------------------ 小工具
std::string hdr_of(const HttpRequest& rq, const char* key);
bool truthy(const json& v, bool dflt);
std::string safe_dump(const json& j);
std::string trim_copy(const std::string& s);
void trim_front(std::string& s);
std::string rand_hex(int n);
std::string rand_id();
std::string new_tool_call_id();
void endswith_trim_special(std::string& s);

#endif
