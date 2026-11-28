// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 本文件属 RK1828-Spark-X2.6-4B 项目；许可证见 LICENSE（GPL-3.0）与 NOTICE（第三方部件）。
// openai.h — OpenAI 协议层：请求参数、grammar 构造、流式解析与三种响应出口
#ifndef _RKNN_SPARK_OPENAI_H_
#define _RKNN_SPARK_OPENAI_H_

#include <string>
#include <vector>

#include "app.h"

struct ChatParams {
  float temperature, top_p;
  int top_k;
  float repeat_penalty, frequency_penalty, presence_penalty;
  int max_new;
  long seed;
  bool thinking;
  bool stream;
  bool include_usage;
  bool want_logprobs;
  std::string effort;
  int inject_effort = -1;  // -1 = 用服务端配置；0/1 = 请求级覆盖官方 thinking_effort 注入
  int thinking_grammar = -1;  // -1 = 用服务端配置；0/1 = 请求级覆盖"思考段必须闭合"的 grammar
  int top_logprobs = 0;
  std::vector<int32_t> lb_ids;
  std::vector<float> lb_vals;
  std::vector<std::string> stops;
  std::string grammar_str;
  int grammar_type = 0;
  bool use_grammar = false;
  bool use_tools = false;
  std::string tools_json;
  std::string tool_choice = "auto";
  bool require_tool = false;   // tool_choice=required 或指定了函数
  std::string forced_tool;     // 指定函数名（tool_choice={"type":"function",...}）
  bool single_tool = false;    // parallel_tool_calls=false
  bool err = false;
  std::string err_msg;
  std::string err_code;
};

ChatParams parse_params(const json& body);
std::string tools_runtime_json(const json& tools);
ordered_json tool_call_json(const std::string& id, const std::string& name, const std::string& args);
bool build_runtime_grammar(const json& rf, bool has_rf, bool thinking, const std::string& forced_tool,
                           bool require_tool, bool tool_grammar, bool thinking_grammar, std::string& out,
                           int& out_type);


#endif
