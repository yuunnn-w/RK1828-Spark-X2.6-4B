// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// main.cpp — rknn_spark_server：命令行解析、引擎/槽位初始化、启动与优雅退出
//
// 源码分层：
//   app.*        进程级配置、全局对象、指标、FIFO 闸门、通用小工具
//   prompt.*     Spark-X2.6 对话模板组装
//   openai.*     OpenAI 请求参数与 runtime grammar
//   stream.h     模型文本流的通道分离与工具调用增量
//   sse.h        SSE 出口（chat chunk / Responses 事件）
//   endpoints.*  HTTP 端点与路由
//   engine.*     RKNN3 session 封装（加载、采样、KV 控制）
//   httpd.*      最小 HTTP/1.1 服务层
//   kvslots.*    多会话 KV 镜像分页
#include <csignal>
#include <fstream>
#include <thread>
#include <unistd.h>

#include "app.h"
#include "endpoints.h"

static void on_signal(int sig) {
  (void)sig;
  g_requested_shutdown = true;
}

static bool has_arg(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], name) == 0) {
      return true;
    }
  }
  return false;
}

// --print-config：打印最终生效的配置后退出，不加载模型、不碰卡
static void print_effective_config() {
  printf("model=%s\n", g_cfg.eng.model_path.c_str());
  printf("weight=%s\n", g_cfg.eng.weight_path.c_str());
  printf("tokenizer=%s\n", g_cfg.eng.tokenizer_path.c_str());
  printf("embed=%s\n", g_cfg.eng.embed_path.c_str());
  printf("safetensors=%s\n", g_cfg.eng.safetensors_path.c_str());
  printf("core_mask=0x%02x ctx=%d\n", g_cfg.eng.core_mask, g_cfg.eng.max_context_len);
  printf("host=%s port=%d alias=%s\n", g_cfg.http.host.c_str(), g_cfg.http.port, g_cfg.model_alias.c_str());
  printf("api_key=%s\n", g_cfg.http.api_key.empty() ? "(disabled)" : "***");
  printf("temp=%.3f top_k=%d top_p=%.3f repeat_penalty=%.3f\n", g_cfg.temperature, g_cfg.top_k, g_cfg.top_p,
         g_cfg.repeat_penalty);
  printf("max_new=%d cap=%d thinking=%d reasoning_format=%s reasoning_effort=%s\n", g_cfg.max_new_tokens,
         g_cfg.max_new_tokens_cap, (int)g_cfg.default_thinking, g_cfg.reasoning_format.c_str(),
         g_cfg.reasoning_effort.empty() ? "(template default)" : g_cfg.reasoning_effort.c_str());
  printf("kv_slots=%d kv_slot_dir=%s\n", g_cfg.kv_slots, g_cfg.kv_slot_dir.c_str());
  printf("params_file=%s\n", g_cfg.params_file.empty() ? "(none)" : g_cfg.params_file.c_str());
}

// ------------------------------------------------------------------ 参数解析
static void print_usage(const char* prog) {
  printf(
      "rknn_spark_server — RK1828 OpenAI-compatible LLM server (Spark-X2.5/X2.6)\n"
      "Usage: %s <llm.rknn> <llm.weight> <core_mask> <tokenizer.gguf> <embed.bin> <max_context_len> "
      "<safetensors> [options]\n"
      "       %s --params-file FILE [options]\n\n"
      "  --params-file FILE      JSON 配置；可单独使用，同名字段会覆盖命令行\n"
      "  --print-config          打印最终生效的配置后退出（不加载模型、不碰卡）\n"
      "  --host H                监听地址（默认 0.0.0.0）\n"
      "  --port P                端口（默认 8080）\n"
      "  --api-key K             需要 Authorization: Bearer K 或 X-API-Key: K（/health 免鉴权）\n"
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
      "  --kvcache-dir DIR       /v1/kvcache/* 的默认目录（默认 /run/rknn3/kvcache）\n"
      "  --kvcache-policy N      KV 策略：0=RECURRENT 1=NORMAL 256=SAVE_CHECKPOINT（默认不设置）\n"
      "  --kv-n-keep N --kv-align N   上面两项的保留长度与对齐（一般不用改）\n",
      prog, prog);
}

int main(int argc, char** argv) {
  srand((unsigned)time(NULL) ^ (unsigned)getpid());
  g_start_time = (long)time(NULL);

  // 两种启动方式：7 个位置参数，或只用 --params-file（可再叠加命令行选项）
  const bool positional = (argc >= 8 && argv[1][0] != '-');
  if (!positional && !has_arg(argc, argv, "--params-file")) {
    print_usage(argv[0]);
    return 1;
  }
  if (positional) {
    g_cfg.eng.model_path = argv[1];
    g_cfg.eng.weight_path = argv[2];
    g_cfg.eng.core_mask = (uint32_t)strtoul(argv[3], NULL, 16);
    g_cfg.eng.tokenizer_path = argv[4];
    g_cfg.eng.embed_path = argv[5];
    g_cfg.eng.max_context_len = atoi(argv[6]);
    g_cfg.eng.safetensors_path = argv[7];
  }

  bool print_config_only = false;
  for (int i = positional ? 8 : 1; i < argc; ++i) {
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
    else if (a == "--params-file") {
      if (v.empty()) {
        LOGF("--params-file requires a path");
        return 1;
      }
      g_cfg.params_file = v;
      ++i;
    }
    else if (a == "--print-config") { print_config_only = true; }
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

  g_cfg.http.api_key = trim_copy(g_cfg.http.api_key);
  if (print_config_only) {
    print_effective_config();
  }
  if (g_cfg.eng.model_path.empty() || g_cfg.eng.weight_path.empty() || g_cfg.eng.tokenizer_path.empty() ||
      g_cfg.eng.embed_path.empty()) {
    LOGF("缺少模型文件：位置参数或 --params-file 的 models.<name> 至少要有 model/weight/vocab/embed");
    return 1;
  }
  if (print_config_only) {
    return 0;
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
