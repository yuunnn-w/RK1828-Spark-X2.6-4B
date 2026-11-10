// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// engine.cpp — RKNN3 Spark 引擎实现
#include "engine.h"

#include <nlohmann/json.hpp>

#include "Tokenizer.h"
#include "float16.h"
#include "time_utils.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#define LOGE(fmt, ...)                                                                                               \
  do {                                                                                                               \
    fprintf(stderr, "[%s] E " fmt "\n", now_str(), ##__VA_ARGS__);                                                   \
    fflush(stderr);                                                                                                  \
  } while (0)
#define LOGN(fmt, ...)                                                                                               \
  do {                                                                                                               \
    fprintf(stderr, "[%s] " fmt "\n", now_str(), ##__VA_ARGS__);                                                     \
    fflush(stderr);                                                                                                  \
  } while (0)

static const char* ROPE_NAMES[4] = {"rope_cos_cache_0", "rope_sin_cache_0", "rope_cos_cache_1", "rope_sin_cache_1"};

const char* now_str() {
  static char buf[24];
  time_t t = time(NULL);
  struct tm tmv;
  localtime_r(&t, &tmv);
  strftime(buf, sizeof(buf), "%H:%M:%S", &tmv);
  return buf;
}

static long now_us() { return getCurrentTimeUs(); }

static size_t elem_size(int dtype) {
  switch (dtype) {
    case RKNN3_TENSOR_FLOAT32: return 4;
    case RKNN3_TENSOR_FLOAT16: return 2;
    case RKNN3_TENSOR_BFLOAT16: return 2;
    default: return 0;
  }
}

Engine::Engine() { memset(&m_llm_config, 0, sizeof(m_llm_config)); }

Engine::~Engine() { shutdown(); }

void Engine::shutdown() {
  // 顺序很重要：先释放用户创建的输出内存，再销毁 session，最后销毁 context，
  // 否则 runtime 会报 "force free user-created mem" / "context not found"
  if (m_n_output_tensors && m_ctx) {
    for (int i = 0; i < m_n_output_tensors; ++i) {
      if (m_output_tensors[i].mem) {
        rknn3_destroy_mem(m_ctx, m_output_tensors[i].mem);
        m_output_tensors[i].mem = NULL;
      }
    }
  }
  m_n_output_tensors = 0;
  if (m_sess) {
    rknn3_session_destroy(m_sess);
    m_sess = NULL;
  }
  if (m_ctx) {
    rknn3_destroy(m_ctx);
    m_ctx = 0;
  }
  if (m_tokenizer) {
    delete m_tokenizer;
    m_tokenizer = NULL;
  }
  if (m_embed && m_embed != MAP_FAILED) {
    munmap(m_embed, m_embed_size);
    m_embed = NULL;
  }
  if (m_embed_fd >= 0) {
    close(m_embed_fd);
    m_embed_fd = -1;
  }
  if (m_rope_map && m_rope_map != MAP_FAILED) {
    munmap(m_rope_map, m_rope_size);
    m_rope_map = NULL;
  }
  if (m_rope_fd >= 0) {
    close(m_rope_fd);
    m_rope_fd = -1;
  }
  if (m_input_tensors) {
    for (int i = 0; i < m_n_input_tensors; ++i) {
      if (m_input_tensors[i].attr) {
        free(m_input_tensors[i].attr);
      }
    }
    free(m_input_tensors);
    m_input_tensors = NULL;
  }
  if (m_n_output_tensors) {
    for (int i = 0; i < m_n_output_tensors; ++i) {
      if (m_output_tensors[i].attr) {
        free(m_output_tensors[i].attr);
        m_output_tensors[i].attr = NULL;
      }
      if (m_output_tensors[i].mem) {
        rknn3_destroy_mem(m_ctx, m_output_tensors[i].mem);
        m_output_tensors[i].mem = NULL;
      }
    }
    m_n_output_tensors = 0;
  }
  m_ready = false;
}

int Engine::init(const EngineConfig& cfg) {
  m_cfg = cfg;

  // ---- 设备 + 上下文 ----
  rknn3_devices devs;
  memset(&devs, 0, sizeof(devs));
  int ret = rknn3_find_devices(&devs);
  if (ret != RKNN3_SUCCESS || devs.n_devices == 0) {
    LOGE("rknn3_find_devices fail ret=%d n=%d", ret, devs.n_devices);
    return -1;
  }
  static char dev_buf[128];
  const char* dev_id = devs.devices[0].id;
  for (uint32_t i = 0; i < devs.n_devices; ++i) {
    if (!cfg.device_id.empty() && cfg.device_id == devs.devices[i].id) {
      dev_id = devs.devices[i].id;
    }
  }
  LOGN("device %s type %s", dev_id, devs.devices[0].type);

  rknn3_init_extend ext;
  memset(&ext, 0, sizeof(ext));
  strncpy(dev_buf, dev_id, sizeof(dev_buf) - 1);
  dev_buf[sizeof(dev_buf) - 1] = 0;
  ext.device_id = dev_buf;
  ret = rknn3_init(&m_ctx, &ext);
  if (ret != RKNN3_SUCCESS) {
    LOGE("rknn3_init fail ret=%d", ret);
    return ret;
  }

  ret = rknn3_load_model_from_path(m_ctx, cfg.model_path.c_str(), cfg.weight_path.c_str());
  if (ret != RKNN3_SUCCESS) {
    LOGE("rknn3_load_model_from_path fail ret=%d", ret);
    return ret;
  }

  rknn3_config rc;
  memset(&rc, 0, sizeof(rc));
  rc.run_core_mask = cfg.core_mask;
  ret = rknn3_model_init(m_ctx, &rc);
  if (ret != RKNN3_SUCCESS) {
    LOGE("rknn3_model_init fail ret=%d", ret);
    return ret;
  }

  ret = rknn3_query(m_ctx, RKNN3_QUERY_LLM_CONFIG, &m_llm_config, sizeof(m_llm_config));
  if (ret != RKNN3_SUCCESS) {
    LOGE("query LLM config fail ret=%d", ret);
    return ret;
  }
  m_model_type = m_llm_config.model_type ? m_llm_config.model_type : "?";
  m_max_ctx = m_llm_config.max_ctx_len;
  LOGN("model type=%s max_ctx=%d max_pos=%d rope_host=%d", m_model_type.c_str(), m_llm_config.max_ctx_len,
       m_llm_config.max_position_embeddings, m_llm_config.rope_cache_host_storage);

  // ---- 输入张量表 ----
  rknn3_input_output_num io_num;
  memset(&io_num, 0, sizeof(io_num));
  ret = rknn3_query(m_ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN3_SUCCESS) {
    LOGE("query IN_OUT_NUM fail ret=%d", ret);
    return ret;
  }
  m_n_input_tensors = (int)io_num.n_input;
  m_input_tensors = (rknn3_tensor*)calloc((size_t)m_n_input_tensors, sizeof(rknn3_tensor));
  for (int i = 0; i < m_n_input_tensors; ++i) {
    m_input_tensors[i].attr = (rknn3_tensor_attr*)calloc(1, sizeof(rknn3_tensor_attr));
    m_input_tensors[i].attr->index = (uint32_t)i;
    ret = rknn3_query(m_ctx, RKNN3_QUERY_INPUT_ATTR, m_input_tensors[i].attr, sizeof(rknn3_tensor_attr));
    if (ret != RKNN3_SUCCESS) {
      LOGE("query INPUT_ATTR[%d] fail ret=%d", i, ret);
      return ret;
    }
    if (m_llm_config.rope_cache_host_storage) {
      for (int c = 0; c < 4; ++c) {
        if (strcmp(m_input_tensors[i].attr->name, ROPE_NAMES[c]) == 0) {
          m_ext_input_idx.push_back(i);
        }
      }
    }
  }
  if (m_llm_config.rope_cache_host_storage && m_ext_input_idx.empty()) {
    LOGE("model needs host rope cache but no rope tensors found");
    return -1;
  }

  // ---- tokenizer / embed ----
  m_tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, cfg.tokenizer_path.c_str());
  if (!m_tokenizer) {
    LOGE("load tokenizer fail: %s", cfg.tokenizer_path.c_str());
    return -1;
  }
  memset(&m_vocab, 0, sizeof(m_vocab));
  m_tokenizer->GetVocabInfo(&m_vocab);
  m_vocab_size = m_vocab.vocab_size;
  if (m_vocab.n_special_eos_id > 0) {
    m_eos_id = m_vocab.special_eos_id[0];
  }
  m_linefeed_id = m_vocab.linefeed_id;
  LOGN("vocab=%d eos=%d linefeed=%d", m_vocab_size, m_eos_id, m_linefeed_id);

  {
    struct stat st;
    m_embed_fd = open(cfg.embed_path.c_str(), O_RDONLY);
    if (m_embed_fd < 0) {
      LOGE("open embed fail: %s", cfg.embed_path.c_str());
      return -1;
    }
    if (fstat(m_embed_fd, &st) < 0) {
      return -1;
    }
    m_embed = (float16*)mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, m_embed_fd, 0);
    if (m_embed == MAP_FAILED) {
      LOGE("mmap embed fail");
      return -1;
    }
    m_embed_size = (size_t)st.st_size;
    if (m_vocab_size <= 0 || st.st_size % ((size_t)m_vocab_size * sizeof(float16)) != 0) {
      LOGE("embed size mismatch: file=%lld vocab=%d", (long long)st.st_size, m_vocab_size);
      return -1;
    }
    m_embedding_dim = (int)((size_t)st.st_size / ((size_t)m_vocab_size * sizeof(float16)));
  }

  // ---- rope safetensors ----
  if (m_llm_config.rope_cache_host_storage) {
    struct stat st;
    m_rope_fd = open(cfg.safetensors_path.c_str(), O_RDONLY);
    if (m_rope_fd < 0) {
      LOGE("open safetensors fail: %s", cfg.safetensors_path.c_str());
      return -1;
    }
    if (fstat(m_rope_fd, &st) < 0) {
      return -1;
    }
    uint64_t hdr = 0;
    if (read(m_rope_fd, &hdr, 8) != 8 || hdr == 0 || hdr > (uint64_t)st.st_size - 8) {
      LOGE("bad safetensors header");
      return -1;
    }
    m_rope_map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, m_rope_fd, 0);
    if (m_rope_map == MAP_FAILED) {
      LOGE("mmap safetensors fail");
      return -1;
    }
    m_rope_size = (size_t)st.st_size;
    const char* base_json = (const char*)m_rope_map + 8;
    const uint8_t* data_base = (const uint8_t*)m_rope_map + 8 + hdr;
    try {
      nlohmann::json j = nlohmann::json::parse(base_json, base_json + hdr);
      nlohmann::json meta = nlohmann::json::parse(j.at("__metadata__").at("index").get<std::string>());
      for (int c = 0; c < 4; ++c) {
        std::vector<int> shape = meta.at(ROPE_NAMES[c]).at("shape").get<std::vector<int> >();
        std::vector<int64_t> off = j.at(ROPE_NAMES[c]).at("data_offsets").get<std::vector<int64_t> >();
        m_rope[c].dtype = meta.at(ROPE_NAMES[c]).at("dtype").get<int>();
        if (shape.size() != 5) {
          LOGE("rope %s unexpected dims", ROPE_NAMES[c]);
          return -1;
        }
        for (int d = 0; d < 5; ++d) {
          m_rope[c].shape[d] = shape[d];
        }
        m_rope[c].data = data_base + off[0];
        m_rope_names[c] = ROPE_NAMES[c];
      }
    } catch (const std::exception& e) {
      LOGE("parse safetensors json fail: %s", e.what());
      return -1;
    }
  }

  // ---- 输出张量（自定义采样需要 logits 落地） ----
  if (m_n_input_tensors >= 0) {
    // 只挂第一个输出（logits）
    m_n_output_tensors = 1;
    memset(m_output_tensors, 0, sizeof(m_output_tensors));
    m_output_tensors[0].attr = (rknn3_tensor_attr*)calloc(1, sizeof(rknn3_tensor_attr));
    m_output_tensors[0].attr->index = 0;
    ret = rknn3_query(m_ctx, RKNN3_QUERY_OUTPUT_ATTR, m_output_tensors[0].attr, sizeof(rknn3_tensor_attr));
    if (ret != RKNN3_SUCCESS) {
      LOGE("query OUTPUT_ATTR fail ret=%d", ret);
      return ret;
    }
    m_output_tensors[0].mem =
        rknn3_create_mem(m_ctx, m_output_tensors[0].attr->aligned_size, m_output_tensors[0].attr->core_id,
                         RKNN3_FLAG_MEMORY_CACHEABLE);
    if (!m_output_tensors[0].mem) {
      LOGE("create output mem fail");
      return -1;
    }
    LOGN("logits tensor: %s size=%llu", m_output_tensors[0].attr->name,
         (unsigned long long)m_output_tensors[0].attr->aligned_size);
  }

  // ---- session ----
  rknn3_llm_param params;
  memset(&params, 0, sizeof(params));
  params.logits_name = (char*)cfg.logits_name.c_str();
  params.max_context_len = cfg.max_context_len > 0 ? cfg.max_context_len : m_llm_config.max_ctx_len;
  params.sampling_param.top_k = 1;
  params.sampling_param.top_p = 0.9f;
  params.sampling_param.temperature = 1.0f;
  params.sampling_param.repeat_penalty = 1.0f;
  params.vocab_info.vocab_size = m_vocab_size;
  params.vocab_info.n_special_bos_id = m_vocab.n_special_bos_id;
  params.vocab_info.n_special_eos_id = m_vocab.n_special_eos_id;
  params.vocab_info.linefeed_id = m_vocab.linefeed_id;
  params.vocab_info.ignore_eos_token = 0;
  memcpy(params.vocab_info.special_bos_id, m_vocab.special_bos_id, sizeof(m_vocab.special_bos_id));
  memcpy(params.vocab_info.special_eos_id, m_vocab.special_eos_id, sizeof(m_vocab.special_eos_id));

  m_sess = rknn3_session_init(m_ctx, &params, 1);
  if (!m_sess) {
    LOGE("rknn3_session_init fail");
    return -1;
  }

  RKLLMCallback cb;
  memset(&cb, 0, sizeof(cb));
  cb.result_callback = cb_result;
  cb.result_userdata = this;
  cb.tokenizer_callback = cb_tokenizer;
  cb.tokenizer_userdata = this;
  cb.embed_callback = cb_embed;
  cb.embed_userdata = this;
  cb.sampling_callback = cb_sampling;
  cb.sampling_userdata = this;
  cb.output_callback = cb_output;
  cb.output_userdata = this;
  cb.output_tensors = m_output_tensors;
  cb.n_output_tensors = (uint32_t)m_n_output_tensors;
  cb.input_callback = cb_input;
  cb.input_userdata = this;
  if (!m_ext_input_idx.empty()) {
    cb.input_tensors_index = m_ext_input_idx.data();
    cb.n_input_tensors = (uint32_t)m_ext_input_idx.size();
  }
  ret = rknn3_session_set_callback(m_sess, &cb);
  if (ret != RKNN3_SUCCESS) {
    LOGE("rknn3_session_set_callback fail ret=%d", ret);
    return ret;
  }

  if (m_cfg.kvcache_policy >= 0) {
    rknn3_kvcache_policy_param kvp;
    memset(&kvp, 0, sizeof(kvp));
    kvp.recurrent.n_keep = (int64_t)m_cfg.kv_n_keep;
    kvp.recurrent.n_keep_aligned = (int64_t)m_cfg.kv_n_keep_aligned;
    ret = rknn3_session_set_kvcache_policy(m_sess, (rknn3_kvcache_policy)m_cfg.kvcache_policy, &kvp);
    if (ret != RKNN3_SUCCESS) {
      LOGE("set kvcache policy %d fail ret=%d", m_cfg.kvcache_policy, ret);
      return ret;
    }
    LOGN("kvcache policy=%d n_keep=%ld aligned=%ld", m_cfg.kvcache_policy, (long)m_cfg.kv_n_keep,
         (long)m_cfg.kv_n_keep_aligned);
  }

  // 语法约束（response_format）需要 context 级 grammar 初始化
  {
    std::vector<uint8_t> blob;
    blob.reserve((size_t)m_vocab_size * 8);
    for (int i = 0; i < m_vocab_size; ++i) {
      std::string piece = m_tokenizer->TokenToPiece(i);
      int32_t len = (int32_t)piece.size();
      blob.insert(blob.end(), (uint8_t*)&len, (uint8_t*)&len + sizeof(int32_t));
      blob.insert(blob.end(), (uint8_t*)piece.data(), (uint8_t*)piece.data() + piece.size());
    }
    std::vector<int32_t> stops;
    for (int i = 0; i < m_vocab.n_special_eos_id; ++i) {
      stops.push_back(m_vocab.special_eos_id[i]);
    }
    rknn3_grammar_config gc;
    memset(&gc, 0, sizeof(gc));
    gc.vocab_size = m_vocab_size;
    gc.encoded_vocab = blob.data();
    gc.encoded_vocab_len = (int32_t)blob.size();
    gc.n_stop_tokens = (int32_t)stops.size();
    gc.stop_tokens = stops.data();
    ret = rknn3_init_grammar(m_ctx, &gc);
    if (ret != 0) {
      LOGN("rknn3_init_grammar ret=%d (response_format 将不可用)", ret);
      m_grammar_ready = false;
    } else {
      m_grammar_ready = true;
      LOGN("grammar engine ready (vocab=%d)", m_vocab_size);
    }
  }

  m_cnt.assign((size_t)m_vocab_size, 0);
  m_ready = true;
  return RKNN3_SUCCESS;
}

// ------------------------------------------------------------------ callbacks
int Engine::cb_tokenizer(void* ud, const char* text, int32_t len, int32_t* tokens, int32_t n_max) {
  Engine* e = (Engine*)ud;
  int n = e->m_tokenizer->Tokenize(text, len, tokens, n_max);
  if (n <= 0) {
    LOGE("tokenize failed (%d chars)", (int)len);
    return n;
  }
  e->m_hist.assign(tokens, tokens + n);
  return n;
}

int Engine::cb_embed(void* ud, int32_t* tokens, uint64_t n, void* embed, uint64_t len) {
  Engine* e = (Engine*)ud;
  uint64_t row = (uint64_t)e->m_embedding_dim * sizeof(float16);
  if (len != n * row) {
    LOGE("embed buffer size mismatch");
    return -1;
  }
  for (uint64_t i = 0; i < n; ++i) {
    int32_t id = tokens[i];
    if (id < 0 || id >= e->m_vocab_size) {
      LOGE("bad token id %d", id);
      return -1;
    }
    memcpy((unsigned char*)embed + i * row, (unsigned char*)e->m_embed + (uint64_t)id * row, row);
  }
  return 0;
}

int Engine::cb_output(void* ud, rknn3_tensor* t, uint32_t n, LLMOutputCallbackState st) {
  (void)ud;
  (void)t;
  (void)n;
  (void)st;
  return 0;
}

int Engine::cb_input(void* ud, rknn3_tensor* tensors, uint32_t n_tensors, LLMInputCallbackParam param) {
  Engine* e = (Engine*)ud;
  for (uint32_t i = 0; i < n_tensors; ++i) {
    for (int c = 0; c < 4; ++c) {
      if (strcmp(tensors[i].attr->name, ROPE_NAMES[c]) != 0) {
        continue;
      }
      const rknn3_tensor_attr* attr = tensors[i].attr;
      if (attr->n_dims < 5) {
        LOGE("rope input not 5-D");
        return -1;
      }
      const RopeCache& cache = e->m_rope[c];
      int input_w = (int)attr->shape[3];
      if (param.pos < 0 || input_w <= 0 || param.pos + input_w > cache.shape[3]) {
        LOGE("rope slice out of range pos=%d w=%d cacheW=%d", param.pos, input_w, cache.shape[3]);
        return -1;
      }
      size_t es = elem_size(cache.dtype);
      if (es == 0) {
        LOGE("rope dtype unsupported %d", cache.dtype);
        return -1;
      }
      int C1 = cache.shape[1];
      size_t c2 = (size_t)cache.shape[4] * es;
      size_t src_stride = (size_t)cache.shape[3] * c2;
      size_t dst_stride = (size_t)input_w * c2;
      if ((size_t)tensors[i].mem->size < dst_stride * (size_t)C1) {
        LOGE("rope dst too small");
        return -1;
      }
      const uint8_t* src = cache.data + (size_t)param.pos * c2;
      uint8_t* dst = (uint8_t*)tensors[i].mem->virt_addr;
      for (int c1 = 0; c1 < C1; ++c1, src += src_stride, dst += dst_stride) {
        memcpy(dst, src, dst_stride);
      }
      break;
    }
  }
  return 0;
}

void Engine::update_bans() {
  m_banned.clear();
  int n = (int)m_gen.size();
  for (int L = 1; L <= m_cfg.veto_period_max; ++L) {
    if (3 * L > n) {
      break;
    }
    bool ok = true;
    for (int i = 0; i < L && ok; ++i) {
      ok = (m_gen[n - 1 - i] == m_gen[n - 1 - L - i]) && (m_gen[n - 1 - i] == m_gen[n - 1 - 2 * L - i]);
    }
    if (ok) {
      m_banned.push_back(m_gen[n - L]);
      ++m_veto_hits;
      break;
    }
  }
  if (m_linefeed_id >= 0 && n >= 3) {
    int run = 0;
    for (int i = n - 1; i >= 0 && m_gen[i] == m_linefeed_id; --i) {
      ++run;
    }
    if (run >= 3) {
      m_banned.push_back(m_linefeed_id);
      ++m_veto_hits;
    }
  }
  if (m_veto_hits > m_cfg.veto_limit) {
    m_force_stop = true;
    if (m_res) {
      m_res->degeneration_guard = true;
    }
  }
}

int Engine::sample_token(const float16* logits, int V) {
  // 计数表（重复惩罚用），只清理 touched
  for (size_t i = 0; i < m_touched.size(); ++i) {
    m_cnt[m_touched[i]] = 0;
  }
  m_touched.clear();
  size_t start = m_hist.size() > (size_t)m_cfg.penalty_window ? m_hist.size() - (size_t)m_cfg.penalty_window : 0;
  for (size_t i = start; i < m_hist.size(); ++i) {
    int32_t t = m_hist[i];
    if (t < 0 || t >= V) {
      continue;
    }
    if (m_cnt[t] == 0) {
      m_touched.push_back(t);
    }
    if (m_cnt[t] < 3000) {
      m_cnt[t] += 1;
    }
  }

  std::vector<char> banned((size_t)V, 0);
  for (size_t i = 0; i < m_banned.size(); ++i) {
    if (m_banned[i] >= 0 && m_banned[i] < V) {
      banned[m_banned[i]] = 1;
    }
  }
  for (size_t i = 0; i < m_lb_touched.size(); ++i) {
    int id = m_lb_touched[i];
    if (id >= 0 && id < V && m_lb[id] <= -100.0f) {
      banned[id] = 1;
    }
  }

  const float rep = m_req.repeat_penalty;
  const float fp = m_req.frequency_penalty;
  const float pp = m_req.presence_penalty;

  auto pen = [&](float x, int c) {
    if (c > 0) {
      if (rep > 1.0f) {
        x = x > 0 ? x / rep : x * rep;
      }
      if (fp != 0.0f) {
        x -= fp * (float)c;
      }
      if (pp != 0.0f) {
        x -= pp;
      }
    }
    return x;
  };

  if (m_req.temperature <= 0.001f) {
    int best = -1;
    float bestv = -1e30f;
    int K = m_req.want_logprobs ? (m_req.top_logprobs > 0 ? m_req.top_logprobs : 1) : 0;
    if (K > 8) {
      K = 8;
    }
    int top_id[8];
    float top_v[8];
    for (int j = 0; j < 8; ++j) {
      top_id[j] = -1;
      top_v[j] = -1e30f;
    }
    for (int i = 0; i < V; ++i) {
      if (banned[i]) {
        continue;
      }
      float x = pen(fp16_to_fp32(logits[i]) + m_lb[i], m_cnt[i]);
      if (x > bestv) {
        bestv = x;
        best = i;
      }
      if (K > 1 && x > top_v[K - 1]) {
        int j = K - 1;
        while (j > 0 && top_v[j - 1] < x) {
          top_v[j] = top_v[j - 1];
          top_id[j] = top_id[j - 1];
          --j;
        }
        top_v[j] = x;
        top_id[j] = i;
      }
    }
    if (m_req.want_logprobs && best >= 0) {
      float sm = 0.0f;
      for (int i = 0; i < V; ++i) {
        if (!banned[i]) {
          sm += std::exp(pen(fp16_to_fp32(logits[i]) + m_lb[i], m_cnt[i]));
        }
      }
      float denom = sm > 0.0f ? sm : 1.0f;
      std::vector<TokenProb> v;
      if (K > 1) {
        for (int j = 0; j < K; ++j) {
          if (top_id[j] >= 0) {
            v.push_back(TokenProb{top_id[j], std::log(top_v[j] / denom)});
          }
        }
      } else {
        v.push_back(TokenProb{best, std::log(bestv / denom)});
      }
      m_res->logprobs.push_back(v);
    }
    return best;
  }

  std::vector<std::pair<float, int> > cand;
  cand.reserve(2048);
  for (int i = 0; i < V; ++i) {
    if (banned[i]) {
      continue;
    }
    cand.push_back(std::make_pair(pen(fp16_to_fp32(logits[i]) + m_lb[i], m_cnt[i]), i));
  }
  if (cand.empty()) {
    return -1;
  }
  int k = m_req.top_k > 0 ? std::min((int)cand.size(), m_req.top_k) : (int)cand.size();
  // top_k 关闭（官方 top_k=-1）时，如果还会做核采样截断，就把候选裁到 4096：
  // 温度 1.0 下 0.95 的质量几乎必然落在前几百个 token 内，裁掉的部分不影响抽样结果，
  // 但省掉整表 13 万次的 exp/排序，实测能把解码速度救回来。
  if (m_req.top_k <= 0 && m_req.top_p > 0.0f && m_req.top_p < 1.0f && k > 4096) {
    k = 4096;
  }
  std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                    [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
  float t = m_req.temperature;
  std::vector<float> p((size_t)k);
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    p[i] = std::exp((cand[i].first - cand[0].first) / t);
    sum += p[i];
  }
  int limit = k;
  if (m_req.top_p > 0.0f && m_req.top_p < 1.0f && sum > 0.0f) {
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
      acc += p[i] / sum;
      if (acc >= m_req.top_p) {
        limit = i + 1;
        break;
      }
    }
  }
  if (m_req.want_logprobs && sum > 0.0f) {
    std::vector<TokenProb> v;
    int take = std::min(limit, 5);
    for (int i = 0; i < take; ++i) {
      v.push_back(TokenProb{cand[i].second, std::log(p[i] / sum)});
    }
    m_res->logprobs.push_back(v);
  }
  float u = (float)rand() / ((float)RAND_MAX + 1.0f) * (limit == k ? sum : 0.0f);
  if (limit != k) {
    // 重新归一化到 limit 内
    float s2 = 0.0f;
    for (int i = 0; i < limit; ++i) {
      s2 += p[i];
    }
    u = (float)rand() / ((float)RAND_MAX + 1.0f) * s2;
    float cs = 0.0f;
    for (int i = 0; i < limit; ++i) {
      cs += p[i];
      if (cs >= u) {
        return cand[i].second;
      }
    }
    return cand[limit - 1].second;
  }
  float cs = 0.0f;
  for (int i = 0; i < k; ++i) {
    cs += p[i];
    if (cs >= u) {
      return cand[i].second;
    }
  }
  return cand[k - 1].second;
}

uint64_t Engine::gram_hash(int candidate) const {
  const int K = m_cfg.ngram_len;
  const int n = (int)m_gen.size();
  const int start = (n > K - 1) ? n - (K - 1) : 0;
  uint64_t h = 0xcbf29ce484222325ULL;
  h = (h ^ (uint64_t)(uint32_t)K) * 0x100000001b3ULL;
  for (int i = start; i < n; ++i) {
    h = (h ^ (uint64_t)(uint32_t)m_gen[i]) * 0x100000001b3ULL;
  }
  h = (h ^ (uint64_t)(uint32_t)candidate) * 0x100000001b3ULL;
  return h;
}

int Engine::cb_sampling(void* ud, float16* logits, char* logits_name) {
  Engine* e = (Engine*)ud;
  (void)logits_name;
  if (e->m_force_stop || e->m_stop_hit || e->m_client_gone) {
    e->m_force_stop = true;
    return e->m_eos_id;
  }
  if (e->m_deadline && now_us() > e->m_deadline) {
    e->m_force_stop = true;
    e->m_deadline_hit = true;
    return e->m_eos_id;
  }
  e->update_bans();
  int tok = e->sample_token(logits, e->m_vocab_size);
  // n-gram 复读检测：候选 token 若会让某个已经出现多次的 n-gram 再次闭合，就禁掉重选。
  // 这补上了周期检测的盲区——"措辞略变但整段复述"的复读没有固定周期，周期检测抓不到。
  const bool ngram_on = (e->m_cfg.ngram_len > 1 && e->m_cfg.ngram_ban_at > 0);
  for (int attempt = 0; ngram_on && attempt < 3; ++attempt) {
    if (tok < 0) {
      break;
    }
    uint64_t hh = e->gram_hash(tok);
    std::unordered_map<uint64_t, int>::iterator it = e->m_ng.find(hh);
    if (it != e->m_ng.end() && it->second >= e->m_cfg.ngram_ban_at) {
      e->m_banned.push_back(tok);
      ++e->m_veto_hits;
      tok = e->sample_token(logits, e->m_vocab_size);
      continue;
    }
    e->m_ng[hh] += 1;
    break;
  }
  if (tok < 0) {
    e->m_force_stop = true;
    return -1;
  }
  e->m_gen.push_back(tok);
  e->m_hist.push_back(tok);
  if (tok == e->m_eos_id) {
    e->m_eos_emitted = true;
  } else {
    for (int i = 0; i < e->m_vocab.n_special_eos_id; ++i) {
      if (tok == e->m_vocab.special_eos_id[i]) {
        e->m_eos_emitted = true;
        break;
      }
    }
  }
  if (e->m_first_token_t == 0) {
    e->m_first_token_t = now_us();
  }
  return tok;
}

void Engine::note_text(const std::string& piece) {
  if (piece.empty()) {
    return;
  }
  if (piece == m_last_piece) {
    ++m_repeat_line_run;
    if (m_repeat_line_run >= 6) {  // 同一片段连续 6 次：判定复读，收尾
      m_force_stop = true;
      ++m_loop_hits;
      if (m_res) {
        m_res->degeneration_guard = true;
      }
    }
  } else {
    m_last_piece = piece;
    m_repeat_line_run = 1;
  }
  // 长块复读检测：输出每增长若干字节就取一段尾部文本，看它在全文里出现了几次。
  // 典型失效模式是"整块自我复述"（周期上百 token），只靠 token 级周期检测容易漏。
  const size_t W = (size_t)m_cfg.loop_window_bytes;
  if (m_cfg.loop_max_occurrences > 0 && m_out.size() >= (size_t)m_cfg.loop_check_min_bytes &&
      m_out.size() >= m_next_loop_check) {
    std::string tail = m_out.substr(m_out.size() - W);
    int occ = 0;
    for (size_t pos = 0; (pos = m_out.find(tail, pos)) != std::string::npos; pos += 1) {
      ++occ;
    }
    if (occ >= m_cfg.loop_max_occurrences) {
      m_force_stop = true;
      ++m_loop_hits;
      if (m_res) {
        m_res->degeneration_guard = true;
      }
    }
    m_next_loop_check = m_out.size() + (size_t)m_cfg.loop_check_stride_bytes;
  }
}

static size_t utf8_partial_tail(const std::string& s) {
  // 末尾是否是一个"还没写完"的多字节序列起点（BPE 常把一个汉字拆成两片）
  for (size_t back = 1; back <= 3 && back <= s.size(); ++back) {
    unsigned char c = (unsigned char)s[s.size() - back];
    if (c < 0x80) {
      return 0;
    }
    if (c >= 0xC0) {
      int need = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : 2;
      return (back < (size_t)need) ? back : 0;
    }
  }
  return 0;
}

static std::string fix_utf8(const std::string& in) {
  static const char* REPL = "\xEF\xBF\xBD";  // U+FFFD
  std::string out;
  out.reserve(in.size());
  size_t i = 0, n = in.size();
  while (i < n) {
    unsigned char c = (unsigned char)in[i];
    int need;
    if (c < 0x80) {
      out += (char)c;
      ++i;
      continue;
    } else if (c >= 0xC2 && c <= 0xDF) {
      need = 2;
    } else if (c >= 0xE0 && c <= 0xEF) {
      need = 3;
    } else if (c >= 0xF0 && c <= 0xF4) {
      need = 4;
    } else {
      out += REPL;
      ++i;
      continue;
    }
    if (i + (size_t)need > n) {
      out += REPL;
      ++i;
      continue;
    }
    unsigned char b1 = (unsigned char)in[i + 1];
    bool ok = true;
    if (c == 0xE0 && b1 < 0xA0) ok = false;             // 过长编码
    if (c == 0xED && b1 >= 0xA0) ok = false;            // 代理区
    if (c == 0xF0 && b1 < 0x90) ok = false;
    if (c == 0xF4 && b1 > 0x8F) ok = false;
    for (int k = 1; ok && k < need; ++k) {
      unsigned char d = (unsigned char)in[i + k];
      if (d < 0x80 || d > 0xBF) ok = false;
    }
    if (!ok) {
      out += REPL;
      ++i;
      continue;
    }
    out.append(in, i, (size_t)need);
    i += (size_t)need;
  }
  return out;
}

int Engine::cb_result(void* ud, RKLLMResult* result, LLMCallState state) {
  Engine* e = (Engine*)ud;
  if (state == RKLLM_RUN_ERROR) {
    e->m_force_stop = true;
    if (e->m_res) {
      e->m_res->ret = -1;
    }
    return 0;
  }
  if (state != RKLLM_RUN_NORMAL || !result || result->num_tokens <= 0) {
    return 0;
  }
  std::string piece;
  if (result->num_tokens == 1) {
    piece = e->m_tokenizer->TokenToPiece(result->token_ids[0]);
  } else {
    piece = e->m_tokenizer->Decode(result->token_ids, result->num_tokens);
  }
  // 半截 UTF-8 续接 + 非法字节替换：否则后面的 json dump 会抛异常，整条请求变 500
  if (!e->m_pending_utf8.empty()) {
    piece = e->m_pending_utf8 + piece;
    e->m_pending_utf8.clear();
  }
  size_t tail = utf8_partial_tail(piece);
  if (tail > 0) {
    e->m_pending_utf8 = piece.substr(piece.size() - tail);
    piece.erase(piece.size() - tail);
  }
  piece = fix_utf8(piece);
  if (piece.empty()) {
    return 0;
  }
  e->m_out += piece;
  e->note_text(piece);
  // 命中 stop 串：截断并停止生成
  for (size_t i = 0; i < e->m_req.stop_strings.size(); ++i) {
    const std::string& s = e->m_req.stop_strings[i];
    if (!s.empty() && e->m_out.size() >= s.size() &&
        e->m_out.compare(e->m_out.size() - s.size(), s.size(), s) == 0) {
      e->m_out.erase(e->m_out.size() - s.size());
      e->m_stop_hit = true;
    }
  }
  if (e->m_sink) {
    if (!e->m_sink->on_text(piece)) {
      e->m_client_gone = true;
      return 0;
    }
  }
  return 0;
}

// ------------------------------------------------------------------ 生成
int Engine::generate(const GenRequest& req, GenResult& out, TextSink* sink) {
  if (!m_ready) {
    return -1;
  }
  m_req = req;
  m_res = &out;
  m_sink = sink;
  m_hist.clear();
  m_gen.clear();
  m_banned.clear();
  m_ng.clear();
  m_out.clear();
  if ((int)m_lb.size() != m_vocab_size) {
    m_lb.assign((size_t)m_vocab_size, 0.0f);
  }
  for (size_t i = 0; i < m_lb_touched.size(); ++i) {
    m_lb[m_lb_touched[i]] = 0.0f;
  }
  m_lb_touched.clear();
  for (size_t i = 0; i < req.logit_bias_ids.size(); ++i) {
    int id = req.logit_bias_ids[i];
    if (id >= 0 && id < m_vocab_size) {
      m_lb[id] = req.logit_bias_vals[i];
      m_lb_touched.push_back(id);
    }
  }
  out.logprobs.clear();
  m_last_piece.clear();
  m_repeat_line_run = 0;
  m_client_gone = false;
  m_force_stop = false;
  m_stop_hit = false;
  m_eos_emitted = false;
  m_deadline_hit = false;
  m_veto_hits = 0;
  m_loop_hits = 0;
  m_next_loop_check = 0;
  m_pending_utf8.clear();
  m_first_token_t = 0;
  m_deadline = req.timeout_ms > 0 ? now_us() + (long)req.timeout_ms * 1000 : 0;
  out = GenResult();
  out.ret = 0;

  int ret = rknn3_session_clear_kvcache(m_sess, RKNN3_KVCACHE_CLEAR_ALL, nullptr);
  if (ret != RKNN3_SUCCESS) {
    LOGN("clear kvcache ret=%d (ignored)", ret);
  }
  ret = rknn3_session_set_chat_template(m_sess, req.system_prompt.c_str(), req.prompt_prefix.c_str(),
                                        req.prompt_postfix.c_str());
  if (ret != RKNN3_SUCCESS) {
    LOGE("set chat template fail ret=%d", ret);
    out.ret = ret;
    return ret;
  }
  if (m_grammar_ready) {
    if (req.use_grammar) {
      ret = rknn3_session_set_grammar(m_sess, req.grammar_str.empty() ? nullptr : req.grammar_str.c_str(),
                                      (rknn3_grammar_type)req.grammar_type, NULL);
      if (ret != 0) {
        LOGN("set grammar fail ret=%d (忽略约束)", ret);
      }
    } else {
      rknn3_session_clear_grammar(m_sess);
    }
  }
  // runtime 没有"清除工具"的接口：每请求显式覆盖，避免上一个请求的工具定义残留
  if (req.use_tools && !req.tools_json.empty()) {
    ret = rknn3_session_set_function_tools(m_sess, req.tools_json.c_str());
    if (ret != 0) {
      LOGN("set function tools fail ret=%d", ret);
    }
  } else {
    ret = rknn3_session_set_function_tools(m_sess, "[]");
    if (ret != 0) {
      LOGN("reset function tools ret=%d (ignored)", ret);
    }
  }

  if (req.use_runtime_sampler) {
    rknn3_llm_param p;
    memset(&p, 0, sizeof(p));
    p.logits_name = (char*)m_cfg.logits_name.c_str();
    p.max_context_len = m_cfg.max_context_len > 0 ? m_cfg.max_context_len : m_llm_config.max_ctx_len;
    p.sampling_param.temperature = req.temperature;
    p.sampling_param.top_k = req.top_k;
    p.sampling_param.top_p = req.top_p;
    p.sampling_param.repeat_penalty = req.repeat_penalty;
    p.sampling_param.frequency_penalty = req.frequency_penalty;
    p.sampling_param.presence_penalty = req.presence_penalty;
    p.vocab_info.vocab_size = m_vocab_size;
    p.vocab_info.n_special_bos_id = m_vocab.n_special_bos_id;
    p.vocab_info.n_special_eos_id = m_vocab.n_special_eos_id;
    p.vocab_info.linefeed_id = m_vocab.linefeed_id;
    memcpy(p.vocab_info.special_bos_id, m_vocab.special_bos_id, sizeof(m_vocab.special_bos_id));
    memcpy(p.vocab_info.special_eos_id, m_vocab.special_eos_id, sizeof(m_vocab.special_eos_id));
    ret = rknn3_session_set_llm_param(m_sess, &p, 1);
    if (ret != RKNN3_SUCCESS) {
      LOGN("set llm param ret=%d (ignored)", ret);
    }
  }

  if (req.seed >= 0) {
    srand((unsigned)req.seed);
  }

  rknn3_llm_multimodal_tensor tensor;
  memset(&tensor, 0, sizeof(tensor));
  tensor.name = "input_embeds";
  tensor.prompt = req.text.c_str();
  tensor.enable_thinking = req.thinking;

  rknn3_llm_input input;
  memset(&input, 0, sizeof(input));
  input.role = req.role.empty() ? "user" : req.role.c_str();
  input.input_type = RKNN3_LLM_INPUT_MULTIMODAL;
  input.multimodal_input = tensor;

  rknn3_llm_infer_param ip;
  memset(&ip, 0, sizeof(ip));
  ip.keep_history = 0;
  if (req.max_new_tokens > 0) {
    ip.max_new_tokens = req.max_new_tokens;
  } else {
    // "不限制"：给到会话的上下文窗口长度，实际由 EOS 或剩余窗口收尾（官方 -n -1 同义）
    int room = m_max_ctx > 0 ? m_max_ctx : m_llm_config.max_ctx_len;
    if (room < 128) {
      room = 128;
    }
    ip.max_new_tokens = (uint32_t)room;
  }

  m_t0 = now_us();
  ret = rknn3_session_run(m_sess, &input, 1, &ip);
  long t1 = now_us();
  out.ret = ret;
  if (ret != RKNN3_SUCCESS) {
    LOGE("rknn3_session_run fail ret=%d", ret);
  }

  RKLLMRunState st;
  memset(&st, 0, sizeof(st));
  if (rknn3_session_query_state(m_sess, &st) == RKNN3_SUCCESS) {
    out.prompt_tokens = (long)st.n_prefill_tokens;
    out.completion_tokens = (long)st.n_decode_tokens;
  }

  long ft = m_first_token_t ? m_first_token_t : t1;
  out.prefill_ms = (double)(ft - m_t0) / 1000.0;
  out.decode_ms = (double)(t1 - ft) / 1000.0;

  out.text = m_out;
  out.gen_tokens = m_gen;
  out.veto_hits = m_veto_hits;
  out.loop_hits = m_loop_hits;
  out.client_gone = m_client_gone;

  // finish_reason 判定：自定义采样器在线时，用"采样器是否自己选中了 EOS"来区分自然收尾与被
  // max_new_tokens 截断 —— runtime 的 n_decode_tokens 与请求上限之间存在 off-by-one
  // （实测 max_new_tokens=64 时 n_decode_tokens=63），只按 token 数比较会把截断误报成 stop。
  // 退回 runtime 内置采样时没有采样回调，只能按 token 数判断。
  bool natural = m_eos_emitted || m_stop_hit || m_force_stop;
  bool capped = false;
  if (req.max_new_tokens > 0) {
    if (req.use_runtime_sampler) {
      capped = out.completion_tokens >= (long)req.max_new_tokens;
    } else {
      capped = !natural;
    }
  }
  if (!m_deadline_hit && capped) {
    out.finish_reason = "length";
  } else {
    out.finish_reason = "stop";
  }
  if (ret != RKNN3_SUCCESS) {
    out.finish_reason = "error";
  }

  rknn3_session_clear_kvcache(m_sess, RKNN3_KVCACHE_CLEAR_ALL, nullptr);
  if (m_grammar_ready) {
    rknn3_session_clear_grammar(m_sess);
  }
  m_res = NULL;
  m_sink = NULL;
  return ret;
}

int Engine::clear_kvcache() { return m_sess ? rknn3_session_clear_kvcache(m_sess, RKNN3_KVCACHE_CLEAR_ALL, nullptr) : -1; }
int Engine::save_kvcache(const std::string& path) { return m_sess ? rknn3_session_save_kvcache(m_sess, path.c_str()) : -1; }
int Engine::load_kvcache(const std::string& path) {
  return m_sess ? rknn3_session_load_kvcache_from_path(m_sess, path.c_str()) : -1;
}
int Engine::stop_generation() { return m_sess ? rknn3_session_stop(m_sess) : -1; }
int Engine::pause_session() { return m_sess ? rknn3_session_pause(m_sess) : -1; }
int Engine::resume_session() { return m_sess ? rknn3_session_resume(m_sess) : -1; }
