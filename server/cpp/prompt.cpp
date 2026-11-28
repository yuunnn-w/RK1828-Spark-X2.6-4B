// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// prompt.cpp — Spark 对话模板组装
#include "prompt.h"

#include <cstring>

#include "app.h"

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
  // 官方格式：工具调用标签 + 函数名 + arg_key/arg_value 序列
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

PromptParts compose_spark(const json& messages, bool thinking, const std::string& default_system,
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

