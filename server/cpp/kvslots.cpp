// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// kvslots.cpp — KV 槽位分页实现
#include "kvslots.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "engine.h"
#include "time_utils.h"

#define LOGK(fmt, ...)                                                                                               \
  do {                                                                                                               \
    fprintf(stderr, "[kvslot] " fmt "\n", ##__VA_ARGS__);                                                            \
    fflush(stderr);                                                                                                  \
  } while (0)

static long ms_since(long us) { return (long)((getCurrentTimeUs() - us) / 1000); }

// 丢掉一个槽的会话状态（不动镜像文件与计时统计）
static void clear_state(SlotState& s) {
  s.prompt_text.clear();
  s.gen_text.clear();
  s.kv_text.clear();
  s.prompt_tokens = 0;
  s.gen_tokens = 0;
  s.kv_tokens = 0;
  s.ids.clear();
  s.resident = false;
}

void KvSlots::configure(int max_slots, const std::string& dir, int min_prefix, long max_image_bytes,
                        long max_total_bytes) {
  std::lock_guard<std::mutex> lk(m_mu);
  if (max_slots < 0) {
    max_slots = 0;
  }
  if (max_slots > 8) {
    max_slots = 8;  // 卡上只有一个会话槽，主机侧镜像最多 8 个
  }
  if (min_prefix < 8) {
    min_prefix = 8;  // 只防住荒谬值；生产建议 1024（前缀太短时重算比拷贝镜像更便宜）
  }
  bool changed = (max_slots != m_max) || (dir != m_dir);
  m_max = max_slots;
  m_min_prefix = min_prefix;
  m_dir = dir;
  m_max_image_bytes = max_image_bytes;
  m_max_total_bytes = max_total_bytes;
  if (!changed) {
    return;
  }
  m_slots.clear();
  if (m_max <= 0) {
    return;
  }
  if (!m_dir.empty() && mkdir(m_dir.c_str(), 0755) != 0 && errno != EEXIST) {
    LOGK("mkdir %s failed", m_dir.c_str());
  }
  m_slots.resize((size_t)m_max);
  for (int i = 0; i < m_max; ++i) {
    m_slots[(size_t)i].id = i;
    m_slots[(size_t)i].path = m_dir + "/slot" + std::to_string(i) + ".kv";
  }
  LOGK("enabled: %d slots, dir=%s, min_prefix=%d tokens, image<=%ldMB total<=%ldMB", m_max, m_dir.c_str(),
       m_min_prefix, m_max_image_bytes / (1024 * 1024), m_max_total_bytes / (1024 * 1024));
}

void KvSlots::drop_image(SlotState& s) {
  if (s.saved && !s.path.empty()) {
    unlink(s.path.c_str());
  }
  s.saved = false;
  s.saved_bytes = 0;
}

long KvSlots::total_bytes_unlocked() const {
  long n = 0;
  for (size_t i = 0; i < m_slots.size(); ++i) {
    if (m_slots[i].saved) {
      n += m_slots[i].saved_bytes;
    }
  }
  return n;
}

void KvSlots::enforce_budget(SlotState* keep) {
  if (m_max_total_bytes <= 0) {
    return;
  }
  while (total_bytes_unlocked() > m_max_total_bytes) {
    SlotState* victim = NULL;
    for (size_t i = 0; i < m_slots.size(); ++i) {
      SlotState& s = m_slots[i];
      if (!s.saved || &s == keep) {
        continue;
      }
      if (!victim || s.last_used < victim->last_used) {
        victim = &s;
      }
    }
    if (!victim) {
      break;
    }
    LOGK("budget: dropping slot %d image (%ld MB, last_used=%ld)", victim->id, victim->saved_bytes / (1024 * 1024),
         victim->last_used);
    drop_image(*victim);
  }
}

bool KvSlots::save_image(Engine& eng, SlotState& s) {
  if (s.prompt_text.empty() || s.prompt_tokens < m_min_prefix) {
    drop_image(s);
    return false;  // 这段会话太短，不值得占主机内存
  }
  long t0 = getCurrentTimeUs();
  int ret = eng.save_kvcache(s.path);
  if (ret != 0) {
    LOGK("save slot %d failed ret=%d", s.id, ret);
    drop_image(s);
    return false;
  }
  struct stat st;
  s.saved = true;
  s.saved_bytes = (stat(s.path.c_str(), &st) == 0) ? (long)st.st_size : 0;
  s.save_ms = ms_since(t0);
  if (m_max_image_bytes > 0 && s.saved_bytes > m_max_image_bytes) {
    LOGK("slot %d image too large (%ld MB > %ld MB), discarded", s.id, s.saved_bytes / (1024 * 1024),
         m_max_image_bytes / (1024 * 1024));
    drop_image(s);
    return false;
  }
  enforce_budget(&s);
  LOGK("slot %d saved: %ld tokens, %ld bytes, %ld ms", s.id, s.prompt_tokens + s.gen_tokens, s.saved_bytes, s.save_ms);
  return true;
}

bool KvSlots::load_image(Engine& eng, SlotState& s) {
  if (!s.saved) {
    return false;
  }
  struct stat st;
  if (stat(s.path.c_str(), &st) != 0) {
    LOGK("load slot %d: image missing (%s)", s.id, s.path.c_str());
    s.saved = false;
    s.saved_bytes = 0;
    return false;
  }
  // 卡上可能还留着上一个会话的 KV：先清干净，避免 load 被旧状态干扰
  eng.clear_kvcache();
  long t0 = getCurrentTimeUs();
  int ret = eng.load_kvcache(s.path);
  if (ret != 0) {
    LOGK("load slot %d failed ret=%d", s.id, ret);
    s.saved = false;
    return false;
  }
  s.load_ms = ms_since(t0);
  LOGK("slot %d loaded: %ld tokens, %ld ms", s.id, s.prompt_tokens + s.gen_tokens, s.load_ms);
  return true;
}

SlotState* KvSlots::pick_victim(const SlotState* avoid) {
  SlotState* victim = NULL;
  for (size_t i = 0; i < m_slots.size(); ++i) {
    SlotState& s = m_slots[i];
    if (&s == avoid) {
      continue;
    }
    if (s.prompt_text.empty()) {
      return &s;  // 空闲槽优先
    }
    if (!victim || s.last_used < victim->last_used) {
      victim = &s;  // 否则取最久未用的（最不活跃）
    }
  }
  if (!victim && avoid) {
    victim = const_cast<SlotState*>(avoid);  // 只配了一个槽：直接换掉卡上这个
  }
  return victim;
}

SlotPlan KvSlots::acquire(Engine& eng, const std::string& prompt_text) {
  std::lock_guard<std::mutex> lk(m_mu);
  SlotPlan plan;
  if (m_max <= 0 || prompt_text.empty()) {
    return plan;
  }

  SlotState* resident = NULL;
  for (size_t i = 0; i < m_slots.size(); ++i) {
    if (m_slots[i].resident) {
      resident = &m_slots[i];
    }
  }

  // 1) 找提示词前缀命中的槽：同一会话的下一轮必然以"上一轮送进去的提示词"开头
  SlotState* best = NULL;
  for (size_t i = 0; i < m_slots.size(); ++i) {
    SlotState& s = m_slots[i];
    if (s.prompt_text.empty() || s.prompt_tokens < m_min_prefix) {
      continue;
    }
    if (prompt_text.size() <= s.prompt_text.size()) {
      continue;  // 严格续写才算命中（等长重放没有新增内容可算）
    }
    if (prompt_text.compare(0, s.prompt_text.size(), s.prompt_text) != 0) {
      continue;
    }
    if (!best || s.prompt_tokens > best->prompt_tokens) {
      best = &s;
    }
  }

  if (best) {
    bool ok = true;
    if (!best->resident) {
      if (resident) {
        save_image(eng, *resident);  // 失败就丢弃这段会话
        resident->resident = false;
      }
      if (!load_image(eng, *best)) {
        ok = false;
      } else {
        plan.loaded = true;
        plan.load_ms = best->load_ms;
        plan.save_ms = resident ? resident->save_ms : 0;
      }
    }
    if (ok) {
      // 精确末端续接：只有新提示词以"KV 对应的完整文本"（上次提示词+生成文本）开头时才复用。
      // runtime 的滑窗 KV 只在续接点正好是 KV 末尾时可用（否则要找 checkpoint）。
      if (best->kv_text.empty() || prompt_text.size() <= best->kv_text.size() ||
          prompt_text.compare(0, best->kv_text.size(), best->kv_text) != 0) {
        LOGK("slot %d: not an exact continuation (kv_text=%zu prompt=%zu) -> cold start", best->id,
             best->kv_text.size(), prompt_text.size());
        ok = false;
      } else {
        plan.prefix_tokens = (int)(best->kv_tokens > 0 ? best->kv_tokens : best->prompt_tokens);
        plan.send_text = prompt_text.substr(best->kv_text.size());
      }
    }
    if (ok) {
      for (size_t i = 0; i < m_slots.size(); ++i) {
        m_slots[i].resident = (&m_slots[i] == best);
      }
      best->hits += 1;
      best->last_used = time(NULL);
      plan.slot = best->id;
      plan.reused = true;
      LOGK("slot %d hit: prefix %d tokens, send %zu bytes (new %zu)%s", best->id, plan.prefix_tokens,
           plan.send_text.size(), prompt_text.size() - best->kv_text.size(),
           plan.loaded ? " (image loaded)" : "");
      return plan;
    }
    // 命中但镜像/续接失败：退化成冷启动，别把坏状态留在槽里
    drop_image(*best);
    clear_state(*best);
    if (resident == best) {
      resident = NULL;
    }
  }

  // 2) 没命中：先把卡上这段会话存成主机镜像（否则清 KV 后就没了），再新开或驱逐一个槽
  if (resident) {
    save_image(eng, *resident);
    resident->resident = false;
  }
  SlotState* target = pick_victim(resident);
  if (!target) {
    return plan;
  }
  drop_image(*target);
  clear_state(*target);
  target->hits = 0;
  target->last_used = time(NULL);
  for (size_t i = 0; i < m_slots.size(); ++i) {
    m_slots[i].resident = (&m_slots[i] == target);
  }
  plan.slot = target->id;
  plan.prefix_tokens = 0;  // 冷启动：引擎会清 KV
  LOGK("slot %d cold start (%d slots configured)", target->id, m_max);
  return plan;
}

void KvSlots::commit(int slot_id, const std::string& prompt_text, long prompt_tokens, const std::string& gen_text,
                     long gen_tokens, long kv_tokens, const std::vector<int32_t>& ids, bool ids_suffix,
                     int prefix_keep) {
  std::lock_guard<std::mutex> lk(m_mu);
  if (m_max <= 0 || slot_id < 0 || slot_id >= m_max) {
    return;
  }
  SlotState& s = m_slots[(size_t)slot_id];
  if (!ids.empty()) {
    if (ids_suffix && prefix_keep > 0 && (int)s.ids.size() >= prefix_keep) {
      std::vector<int32_t> merged(s.ids.begin(), s.ids.begin() + prefix_keep);
      merged.insert(merged.end(), ids.begin(), ids.end());
      s.ids.swap(merged);
    } else {
      s.ids = ids;
    }
  }
  s.prompt_text = prompt_text;
  s.prompt_tokens = prompt_tokens > 0 ? prompt_tokens : 0;
  s.kv_text = prompt_text + gen_text;
  s.gen_text = gen_text;
  s.gen_tokens = gen_tokens;
  s.kv_tokens = kv_tokens;
  s.resident = true;
  s.last_used = time(NULL);
  // 卡上现在就是最新状态，旧镜像作废（下次换出时会重新存）
  drop_image(s);
}

void KvSlots::invalidate(Engine& eng, int slot_id) {
  std::lock_guard<std::mutex> lk(m_mu);
  if (m_max <= 0 || slot_id < 0 || slot_id >= m_max) {
    return;
  }
  SlotState& s = m_slots[(size_t)slot_id];
  drop_image(s);
  s.prompt_text.clear();
  s.gen_text.clear();
  s.kv_text.clear();
  s.prompt_tokens = 0;
  s.gen_tokens = 0;
  s.kv_tokens = 0;
  s.ids.clear();
  s.resident = false;
  eng.clear_kvcache();
  LOGK("slot %d invalidated after failed run", slot_id);
}

void KvSlots::reset(Engine& eng) {
  std::lock_guard<std::mutex> lk(m_mu);
  bool any = false;
  for (size_t i = 0; i < m_slots.size(); ++i) {
    any = any || m_slots[i].resident;
    clear_state(m_slots[i]);
    m_slots[i].hits = 0;
    drop_image(m_slots[i]);
  }
  if (any) {
    eng.clear_kvcache();
  }
  if (m_max > 0) {
    LOGK("all slots cleared");
  }
}

std::vector<SlotState> KvSlots::snapshot() const {
  std::lock_guard<std::mutex> lk(m_mu);
  return m_slots;
}

long KvSlots::image_bytes() const {
  std::lock_guard<std::mutex> lk(m_mu);
  long n = 0;
  for (size_t i = 0; i < m_slots.size(); ++i) {
    if (m_slots[i].saved) {
      n += m_slots[i].saved_bytes;
    }
  }
  return n;
}

long KvSlots::resident_id() const {
  std::lock_guard<std::mutex> lk(m_mu);
  for (size_t i = 0; i < m_slots.size(); ++i) {
    if (m_slots[i].resident) {
      return m_slots[i].id;
    }
  }
  return -1;
}
