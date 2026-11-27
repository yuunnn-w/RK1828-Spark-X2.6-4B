// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// kvslots.h — KV 槽位分页：主机内存里保存多个会话的 KV Cache 镜像，卡上只放当前活跃的那一个
//
// 卡上只有一个会话槽（4B/64K 的 KV 已占满卡内存），但多个并发会话可以各自把 KV 存成主机侧镜像：
//   请求到达 → 看哪个槽的提示词文本是本请求的前缀（同一会话的历史必然逐字相同）；
//   命中且卡上就是它 → 把卡上的 KV 截断到"客户端已见的前缀"（镜像里多了上次生成的 token），
//                      只把新增的这一轮文本送给 runtime，前缀不重算；
//   命中但卡上是别的槽 → 先把卡上这个槽存成镜像，再把目标槽的镜像装回卡上；
//   没命中 → 槽数未满就新开一个，满了就按 LRU 驱逐最久未用的那个（它的镜像被丢弃）。
// 生成结束后卡上的 KV 保持原样（keep_kv_after），所以同一会话连续对话不需要任何拷贝。
//
// 为什么用文本前缀而不是 token 前缀：compose_spark 是确定性的，同一会话下一轮的
// 完整提示词必然以"上一轮送进去的提示词"开头，逐字比较既便宜又不会误判。
#ifndef _RKNN_SPARK_KVSLOTS_H_
#define _RKNN_SPARK_KVSLOTS_H_

#include <stdint.h>

#include <mutex>
#include <string>
#include <vector>

class Engine;

struct SlotState {
  int id = -1;
  bool resident = false;       // 卡上的 KV 现在就是这个槽
  std::string prompt_text;     // 上次送进 runtime 的完整提示词（KV 里已经包含这段）
  long prompt_tokens = 0;      // 上面这段提示词的 token 数（runtime 口径）
  std::string gen_text;        // 上次生成的原始文本（诊断/记录用）
  long gen_tokens = 0;
  long kv_tokens = 0;      // 镜像保存时 runtime 报告的 KV 总 token 数（真实长度）
  std::vector<int32_t> ids; // runtime 自己的 prompt token id
  std::string kv_text;     // 卡上 KV 对应的完整文本 = 提示词 + 生成文本（精确末端续接用）
  bool saved = false;          // 主机侧是否存在与上面状态对应的镜像
  long saved_bytes = 0;
  long last_used = 0;          // 秒级时间戳（LRU）
  long hits = 0;               // 命中次数
  std::string path;            // 主机侧镜像文件
  long save_ms = 0;            // 最近一次 save 耗时
  long load_ms = 0;            // 最近一次 load 耗时
};

struct SlotPlan {
  int slot = -1;               // 选用的槽（<0 = 未启用槽位）
  int prefix_tokens = 0;       // 卡上已缓存、本次可以直接复用的前缀 token 数
  size_t text_offset = 0;      // （保留）文本偏移；实际发送串见 send_text
  std::string send_text;       // 命中时本次要送给 runtime 的文本（截断后的提示词尾巴 + 新的一轮）
  bool reused = false;         // 是否命中并沿用了已有 KV
  bool loaded = false;         // 是否做了 存旧镜像 + 装新镜像
  long save_ms = 0;
  long load_ms = 0;
};

class KvSlots {
 public:
  void configure(int max_slots, const std::string& dir, int min_prefix_tokens, long max_image_bytes = 0,
                 long max_total_bytes = 0);
  bool enabled() const { return m_max > 0; }
  int max_slots() const { return m_max; }
  int min_prefix() const { return m_min_prefix; }
  const std::string& dir() const { return m_dir; }

  // 选出本次请求要用的槽，必要时把它的 KV 装回卡上并对齐前缀。调用方保证串行（同一时刻只有一个生成）。
  SlotPlan acquire(Engine& eng, const std::string& prompt_text);

  // 生成结束：记录这个槽当前的提示词与生成结果，让它成为卡上的活跃槽
  void commit(int slot_id, const std::string& prompt_text, long prompt_tokens, const std::string& gen_text,
              long gen_tokens, long kv_tokens, const std::vector<int32_t>& ids, bool ids_suffix, int prefix_keep);

  // 生成失败：卡上状态不可信，清掉 KV 并丢弃该槽
  void invalidate(Engine& eng, int slot_id);

  // 清空所有槽记录与镜像（模型重载、槽配置变化、服务退出时用）
  void reset(Engine& eng);

  std::vector<SlotState> snapshot() const;
  long image_bytes() const;
  long resident_id() const;

 private:
  bool save_image(Engine& eng, SlotState& s);
  bool load_image(Engine& eng, SlotState& s);
  void drop_image(SlotState& s);
  SlotState* pick_victim(const SlotState* avoid);
  long total_bytes_unlocked() const;
  void enforce_budget(SlotState* keep);

  int m_max = 0;
  int m_min_prefix = 1024;
  long m_max_image_bytes = 0;   // 单个镜像上限（超出就丢弃该槽镜像）
  long m_max_total_bytes = 0;   // 所有镜像总预算（超出按 LRU 丢弃）
  std::string m_dir;
  std::vector<SlotState> m_slots;
  mutable std::mutex m_mu;
};

#endif
