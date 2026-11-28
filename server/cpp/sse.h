// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// sse.h — OpenAI 流式出口（chat.completion.chunk 与 Responses API 事件）
#ifndef _RKNN_SPARK_SSE_H_
#define _RKNN_SPARK_SSE_H_

#include "stream.h"

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

#endif
