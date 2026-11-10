# Spark-X2.6 的 chat template：服务端怎么组装 prompt

这一节是给"想自己写客户端 / 自己接 runtime"的人看的。**Spark 的对话格式不是 ChatML**，
照 ChatML 拼出来的 prompt 会让模型行为退化（官方 `rkllm3-server` 的 Spark 链路坏掉后，
这是第二重要的坑）。

## 1. 词表里的控制标记（逐字）

| 用途 | 字面 | 说明 |
|---|---|---|
| 句开始 | `<｜start▁of▁sentence｜>` | 注意是**全角竖线 U+FF5C** 与 **下八分之一方块 U+2581**，不是 ASCII `|`；词表 id 0 |
| 句结束 | `<｜end▁of▁sentence｜>` | 同上；词表 id 1（也是 EOS） |
| 角色标记 | `<\|System\|>`、`<\|User\|>`、`<\|Bot\|>`、`<\|Tool\|>` | 这些用的是 **ASCII 竖线**，且必须在行首 |
| 思考通道 | `<think>` / `</think>` | 成对；没有思考内容时官方模板只写闭合标记 |
| 工具 | `<tool_call>`、`<arg_key>`/`</arg_key>`、`<arg_value>`/`</arg_value>`、`<tools>`/`</tools>`、`<tool_response>`/`<tool_response>` | 见 §3 |

（表里角色标记写成 `<\|System\|>` 只是为了避免 Markdown 把它当表格分隔符 —— 实际是 `<` + `|` + `System` + `|` + `>`。）

## 2. 一轮对话的组装顺序

服务端把 chat_template 拆成三段交给 `rknn3_session_set_chat_template(system, prefix, postfix)`，
最终 token 序列等价于 `system + prefix + text + postfix`：

```
system 块 = <句开始><|System|>\n
            you are a helpful assistant.
            [可选：官方 thinking_effort 注入文本]
            [可选：\n\n## Tools\nYou have access to the following functions:\n<tools>\n{每个 tool.function 的紧凑 JSON}\n</tools>]
            [可选：\n\n + 用户第一条 system 文本]
            <句结束>

prefix    = <句开始> + 第一个非-system 轮的角色标记
text      = 各轮正文，轮与轮之间用 <句结束><句开始> + 角色标记 分隔
postfix   = <句结束><句开始><|Bot|> + (<think> 或 </think>)      # 末轮是 assistant 时不追加（续写语义）
```

## 3. 工具相关的三种形态

1. **工具清单必须写进 system 块**（模型侧唯一能看到工具的途径）。只调
   `rknn3_session_set_function_tools()` 而不把工具渲染进 system，模型根本不知道有工具存在。
   JSON 用紧凑分隔符（`,` 与 `:` 后不加空格），非 ASCII 保留原字符 —— 这与 transformers 4.57.1
   执行官方模板 `tojson(separators=(',',':'))` 的输出逐字节一致。
2. **历史里的 assistant 工具调用**回填成：
   `<tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>…</tool_call>`
   —— 不是整块 JSON。多个参数就多对 `arg_key/arg_value`。
3. **工具返回**属于 `tool` 角色轮：`<句开始><|Tool|>` 后面跟若干成对的
   `<tool_response>{内容}</tool_response>`，连续多条工具结果合并进同一轮后再 `<句结束>` 收尾。

## 4. `thinking` 与 `reasoning_effort`

- 官方模板 `enable_thinking` 默认 **true**；为 true 时往 system 追加
  `` `thinking_effort` guides on how much to think … Now the system is invoked with `thinking_effort=high|low`. ``
  —— 这是**文本注入**，不是布尔开关。`reasoning_effort` 只接受 `high|low`，其它值模板直接
  `raise_exception`。官方 README 给的权衡：4B 从 high→low 平均 **−1.7 分 / 输出长度 −22 %**。
- 生成后缀：thinking 开 → `<|Bot|><think>`；thinking 关 → `<|Bot|></think>`。
- 本服务的映射：`reasoning_effort:"none"|"minimal"` 或 `chat_template_kwargs.enable_thinking:false` → 关闭思考；
  `chat_template_kwargs.reasoning_effort:"low"` → 注入 low 那句；`chat_template_kwargs.thinking_effort_inject:false`
  或 `--thinking-effort-inject off` → 完全不注入这段（实测有些 prompt 会把它复述出来，见 `docs/benchmarks.md` §6）。

## 5. 与官方实现的一致性是怎么验证的

不是靠眼看。方法：在转换机上用**与模型 `transformers_version` 相同的 4.57.1** 和官方仓库的
`tokenizer.apply_chat_template(...)` 生成 11 组参考 prompt（脚本 `convert/render_chat_template.py`），
再用本仓库 `convert/check_chat_template.py` 把服务端 `compose_spark()` 的等价 Python 镜像与之**逐字节 diff**。

结果：**8/11 完全一致**（含中文工具描述、工具调用历史、连续多工具结果、多 system、content 为文本数组、
thinking off、effort low）。剩下 3 处只差同一个有意偏差：末轮是 assistant 时服务端不追加生成后缀
（续写语义，等价 llama.cpp 的 `continue_final_message`）。

两处**官方更严格**、本服务端**有意更宽松**的地方：

| 情况 | 官方模板 | 本服务 |
|---|---|---|
| `tool_call.function.arguments` 传 JSON **字符串** | `TemplateError: … must be a dictionary` | 自己 `json.loads` 后正常渲染（OpenAI 客户端回传历史通常就是字符串） |
| `reasoning_effort` 传 `medium` 之类 | `TemplateError: Unexpected reasoning effort` | 非 `low` 一律按 `high` 处理 |

一处已知微小差异：历史 `tool_calls` 里**非字符串**参数值，官方渲染成 `["a", "b"]`（逗号后带空格），
本服务端渲染成 `["a","b"]`（紧凑）。只影响 tokenization 的微小差异，未修。

## 6. 自己写客户端时的最小可用拼法

如果你不想用 `/v1/chat/completions` 而要走 `/v1/completions`（本服务端支持，`prompt` 原样送入、不做 chat 包装），
可以用这段自己拼（Python，UTF-8）：

```python
SOS = "<｜start▁of▁sentence｜>"      # 全角竖线 + U+2581
EOS = "<｜end▁of▁sentence｜>"
def turn(role, text):
    return f"{SOS}<|{role}|>{text}{EOS}"
prompt = turn("System", "you are a helpful assistant.") + \
         turn("User", "17*23 等于几？只回答数字。") + \
         f"{SOS}<|Bot|><think>"       # 开思考；不开就写 </think>
```

用 `examples/client.py` 里的 `--only chat` 先确认服务正常，再对比你自己拼的 prompt 走 `/v1/completions` 的结果。
