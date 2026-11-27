// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// engine.h — RKNN3 Spark 推理引擎封装（常驻 session + 自定义采样 + 复读熔断）
#ifndef _RKNN_SPARK_ENGINE_H_
#define _RKNN_SPARK_ENGINE_H_

#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "Tokenizer.h"
#include "rknn3_api.h"

// 流式文本出口（HTTP 层实现，用于逐 token 推送 SSE）
struct TextSink {
  virtual ~TextSink() {}
  // 返回 false 表示对端已断开，需要尽快停止生成
  virtual bool on_text(const std::string& piece) = 0;
};

struct GenRequest {
  std::string system_prompt;   // 完整的 system 块（含 Spark 控制标记）
  std::string prompt_prefix;   // 用户轮起始标记
  std::string prompt_postfix;  // 生成引导（<think> 或 ）
  std::string text;            // 本轮文本（多轮历史已串在里面）

  int   max_new_tokens = 512;
  float temperature = 0.0f;
  int   top_k = 1;
  float top_p = 1.0f;
  float repeat_penalty = 1.05f;
  float frequency_penalty = 0.0f;
  float presence_penalty = 0.0f;
  long  seed = -1;

  bool thinking = false;
  std::string role;   // "user" 或 "tool"（runtime 需要它来判断这一轮是什么）
  bool use_runtime_sampler = false;  // true = 用 runtime 内置采样（不做熔断）

  std::vector<std::string> stop_strings;
  int  timeout_ms = 0;               // 0 = 不限
  bool want_logprobs = false;
  int  top_logprobs = 0;                  // >0：每个生成 token 额外给 top_logprobs 条候选
  std::vector<int32_t> logit_bias_ids;    // OpenAI logit_bias（key 是 token id 的十进制字符串）
  std::vector<float> logit_bias_vals;

  bool          use_grammar = false;
  std::string   grammar_str;
  int           grammar_type = 0;  // rknn3_grammar_type
  bool          use_tools = false;
  std::string   tools_json;
  // tool_choice 需要语法约束时为真："required" 或指定函数名 → 下发 Spark 工具调用 structural_tag
  bool          force_tool_grammar = false;
  std::vector<std::string> allowed_tools;  // 约束里允许出现的工具名（空 = 任意合法名）

  // KV 槽位续接：cached_prefix_tokens > 0 表示卡上已缓存该 prompt 的前缀，不清 KV、不重算
  int cached_prefix_tokens = 0;
  bool keep_kv_after = false;  // 生成结束后不清 KV（槽位模式：由调用方保存/换入下一个会话）
  // 带工具时放宽复读熔断：工具参数里重复的键/括号会触发 n-gram 与长块检测
  bool relax_guard = false;
};

struct TokenProb {
  int   token_id;
  float logprob;
};

struct GenResult {
  std::string text;
  long prompt_tokens = 0;        // runtime 报告的本次 prefill token 数（不含复用部分）
  long prompt_text_tokens = 0;   // 整段 prompt 的真实 token 数（含复用部分）
  long reused_tokens = 0;        // runtime 从卡上 KV 复用的 token 数
  long completion_tokens = 0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  std::string finish_reason;   // stop | length | tool_calls | error
  int  veto_hits = 0;
  int  loop_hits = 0;      // 文本级长块复读检测命中次数
  bool degeneration_guard = false;  // 触发了复读熔断并强行收尾
  bool client_gone = false;
  int  ret = 0;
  std::vector<std::vector<TokenProb> > logprobs;  // 每个生成 token 的 top-N
  std::vector<int32_t> gen_tokens;               // 实际生成的 token id 序列
  long kv_total_tokens = 0;      // runtime 报告的当前 KV 总 token 数（镜像真实长度）
  long runtime_reused_tokens = 0;  // runtime 自己报告的 n_reuse_tokens（未加工）
  std::vector<int32_t> prompt_token_ids;  // runtime 自己的 prompt token id（st.input_tokens 的副本）
  bool prompt_ids_suffix = false;         // true = 该 ids 只是本次新增的后缀（KV 复用时）
};

struct EngineConfig {
  std::string model_path;
  std::string weight_path;
  std::string tokenizer_path;
  std::string embed_path;
  std::string safetensors_path;
  std::string device_id;      // 空 = 第一个设备
  uint32_t    core_mask = 0xff;
  int         max_context_len = 0;  // 0 = 用模型自带
  std::string logits_name = "logits_gathered";
  int         veto_period_max = 400;
  int         veto_limit = 6;
  int         penalty_window = 512;
  int         loop_window_bytes = 120;   // 文本级长块复读检测窗口
  int         loop_check_min_bytes = 420;
  int         loop_check_stride_bytes = 64;
  int         loop_max_occurrences = 3;
  int         ngram_len = 8;         // token 级 n-gram 复读检测窗口
  int         ngram_ban_at = 3;      // 同一 n-gram 已出现这么多次就禁止续写
  int  kvcache_policy = -1;   // -1=不设置(用 runtime 默认)；0=RECURRENT 1=NORMAL 256=SAVE_CHECKPOINT
  long kv_n_keep = 0;         // RECURRENT 下保留的 cache 数（有 system prompt 时 runtime 会自行取用）
  long kv_n_keep_aligned = 0;
  // SAVE_CHECKPOINT 策略（滑窗/线性注意力模型）：复用点必须落在 checkpoint 网格上
  int  kv_ckpt_start = 0;
  int  kv_ckpt_interval = 128;
  int  kv_ckpt_count = 0;     // 0 = 按 max_context_len/interval 自动算
  int  kv_align_cfg = 0;      // 复用/截断对齐：0=自动(32；checkpoint 策略用 interval)，可强制（滑窗模型常用 128）
};

class Engine {
public:
  Engine();
  ~Engine();

  int init(const EngineConfig& cfg);
  void shutdown();

  bool ready() const { return m_ready; }
  int  kv_align() const { return m_kv_align; }
  int  model_max_ctx() const { return m_max_ctx; }
  const std::string& model_type() const { return m_model_type; }
  int  vocab_size() const { return m_vocab_size; }
  std::string token_piece(int id) const { return m_tokenizer ? m_tokenizer->TokenToPiece(id) : std::string(); }

  // 用真实词表统计一段文本的 token 数（主机侧，不上卡）；用于 prompt 预算与 KV 槽位前缀匹配
  int count_tokens(const std::string& text) const;
  // 把文本编码成 token 序列（用于槽位前缀指纹）；n_max 为上限
  int encode_tokens(const std::string& text, std::vector<int32_t>& out) const;
  // 把 token 序列的 [from,to) 段还原成文本（用于 KV 截断后续写）
  std::string decode_range(const std::vector<int32_t>& ids, int from, int to) const;
  // 词嵌入表的均值池化 + L2 归一化（近似嵌入，不是训练过的 embedding 模型）
  bool embed_mean(const std::string& text, std::vector<float>& out) const;

  // 生成一次；线程安全由调用方保证（同一时刻只允许一个 generate）
  int generate(const GenRequest& req, GenResult& out, TextSink* sink);

  // KV cache / slot 控制（与官方 rkllm3-server 的 API 面对齐）
  int clear_kvcache();
  int save_kvcache(const std::string& path);
  int load_kvcache(const std::string& path);
  // 把卡上的 KV 截断到前 n_tokens 个 token（RKNN3_KVCACHE_KEEP_N_TOKENS）
  int truncate_kvcache(uint32_t n_tokens);
  int stop_generation();
  int pause_session();
  int resume_session();

 private:
  static int cb_tokenizer(void* ud, const char* text, int32_t len, int32_t* tokens, int32_t n_max);
  static int cb_embed(void* ud, int32_t* tokens, uint64_t n, void* embed, uint64_t len);
  static int cb_sampling(void* ud, float16* logits, char* logits_name);
  static int cb_result(void* ud, RKLLMResult* result, LLMCallState state);
  static int cb_input(void* ud, rknn3_tensor* tensors, uint32_t n, LLMInputCallbackParam param);
  static int cb_output(void* ud, rknn3_tensor* tensors, uint32_t n, LLMOutputCallbackState state);

  int sample_token(const float16* logits, int V);
  void update_bans();
  uint64_t gram_hash(int candidate) const;
  void note_text(const std::string& piece);

  EngineConfig m_cfg;
  rknn3_context m_ctx = 0;
  rknn3_session* m_sess = NULL;
  rknn3_llm_config m_llm_config;
  Tokenizer* m_tokenizer = NULL;
  float16* m_embed = NULL;
  size_t m_embed_size = 0;
  int m_embed_fd = -1;
  int m_embedding_dim = 0;
  int m_vocab_size = 0;
  void* m_rope_map = NULL;
  size_t m_rope_size = 0;
  int m_rope_fd = -1;
  struct RopeCache {
    const uint8_t* data;
    int shape[5];
    int dtype;
  };
  RopeCache m_rope[4];
  std::string m_rope_names[4];
  rknn3_tensor* m_input_tensors = NULL;
  int m_n_input_tensors = 0;
  std::vector<int32_t> m_ext_input_idx;
  rknn3_tensor m_output_tensors[1];
  int m_n_output_tensors = 0;
  VocabInfo m_vocab;
  bool m_grammar_ready = false;
  int m_max_ctx = 0;
  int m_kv_align = 32;
  int m_eos_id = 1;
  int m_linefeed_id = -1;
  bool m_ready = false;
  std::string m_model_type;

  // 每请求状态
  GenRequest m_req;
  GenResult* m_res = NULL;
  TextSink* m_sink = NULL;
  std::vector<int32_t> m_hist;   // prompt + 已生成
  std::vector<int32_t> m_gen;    // 仅已生成
  std::vector<int32_t> m_banned;
  std::unordered_map<uint64_t, int> m_ng;   // 本次生成里 n-gram 出现次数
  std::vector<int16_t> m_cnt;
  std::vector<int32_t> m_touched;
  std::vector<float> m_lb;              // logit_bias 增量表（按 vocab 大小，只清 touched）
  std::vector<int32_t> m_lb_touched;
  std::string m_pending_utf8;     // 半截 UTF-8 续接缓冲
  bool m_client_gone = false;
  bool m_force_stop = false;
  bool m_deadline_hit = false;
  bool m_stop_hit = false;
  bool m_eos_emitted = false;  // 采样器自己选中了 EOS（区分"自然收尾"与"被 max_new_tokens 截断"）
  int m_veto_hits = 0;
  int m_loop_hits = 0;
  size_t m_next_loop_check = 0;
  long m_t0 = 0;
  long m_first_token_t = 0;
  long m_deadline = 0;
  std::string m_out;
  size_t m_sent_len = 0;    // 已推送给 sink 的字节数（stop 截断后不再补发）
  size_t m_stop_scan = 0;   // stop 串扫描水位
  int m_prompt_tok = 0;     // 本次 prompt 的真实 token 数（tokenizer 回调记录）
  std::string m_tmpl_sys, m_tmpl_pre, m_tmpl_post;  // 已下发的 chat 模板（变化才重设，避免重置 KV）
  bool m_tmpl_valid = false;
  std::string m_tools_sent;      // 已下发的工具定义
  std::string m_grammar_sent;    // 已下发的 grammar
  int m_grammar_type_sent = -1;
  std::string m_last_piece;       // 用于重复行检测
  int m_repeat_line_run = 0;
};

#endif
