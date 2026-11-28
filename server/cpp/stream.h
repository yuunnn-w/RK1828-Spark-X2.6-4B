// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// stream.h — 模型原始文本流的通道分离（reasoning / content / tool_calls）
#ifndef _RKNN_SPARK_STREAM_H_
#define _RKNN_SPARK_STREAM_H_

#include "app.h"

// 把切分点退回到 UTF-8 字符边界，避免把多字节字符拆开
static inline size_t utf8_cut_back(const std::string& s, size_t pos) {
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

#endif
