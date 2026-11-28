// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// prompt.h — Spark-X2.6 的对话模板组装（system / 各轮标记 / 工具说明 / 生成引导）
#ifndef _RKNN_SPARK_PROMPT_H_
#define _RKNN_SPARK_PROMPT_H_

#include <string>

#include <nlohmann/json.hpp>

using nlohmann::json;
using nlohmann::ordered_json;

struct PromptParts {
  std::string system;      // 完整 system 块（system_in_prompt=off 时交给 runtime）
  std::string prefix;      // 用户轮起始标记（system_in_prompt=off 时）
  std::string postfix;     // 生成引导（system_in_prompt=off 时）
  std::string text;        // 本轮文本（多轮历史已串在里面）
  std::string last_role;   // 最后一轮的角色（user/assistant/tool）
  bool ok = true;
  std::string err;
};

// 按官方 chat_template 组装：默认 system + thinking_effort 说明 + 工具清单 + 首条 system，
// 之后是 Spark 的多轮标记。system_in_prompt=true 时整段（含 system 块）写进 text。
PromptParts compose_spark(const json& messages, bool thinking, const std::string& default_system, const json& tools,
                          const std::string& reasoning_effort, bool inject_effort, bool system_in_prompt,
                          const std::string& forced_tool, bool require_tool, bool single_tool);

#endif
