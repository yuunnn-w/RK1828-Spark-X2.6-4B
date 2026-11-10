#!/usr/bin/env python3
# 生成 GRQ 校准集（格式与 rknn3-model-zoo/datasets/llm_quant.json 一致：[{"input","target"}]）
# 依据 examples/Qwen3_5/README.md §5：校准样本应贴近真实应用场景（含 system prompt / function call），数量 20~128。
import json, random

random.seed(20261105)
S = []

def add(text, target=""):
    S.append({"input": text, "target": target})

# 1) 中文通用问答 / 解释
zh = [
    "请用三句话解释 PCIe 5.0 x1 链路的理论带宽和实际有效带宽为什么不一样。",
    "M.2 接口的 AI 协处理器卡相比 USB 直连协处理器，在推理延迟上有什么优势？",
    "把下面这段话改写成更正式的书面语：这个板子老是莫名其妙死掉，后来发现是它自己去休眠了又醒不过来。",
    "W4A16 量化和 W8A8 量化有什么区别？分别适合什么场景？",
    "KV Cache 量化到 int4 会影响长文本的检索能力吗？为什么？",
    "请给出一个 8 核 NPU 上做流水线并行时负载均衡的常用做法。",
    "用通俗语言解释什么是滑动窗口注意力（sliding window attention）。",
    "以下说法对不对：模型上下文越长，首 token 延迟一定越高？请说明理由。",
    "帮我把这段中文技术说明压缩成 60 字以内：RK1828 是 M.2 形态的协处理器，通过 PCIe 挂载到主控 SoC，模型权重和 KV 缓存都存放在卡上的 5GB 显存里。",
    "请列出排查 PCIe 设备无法枚举的三个常见原因。",
    "一个 4B 模型在 5GB 显存的推理卡上跑 32K 上下文，显存大概怎么分配？",
    "这句话有歧义吗？如果有，请指出并改写：'模型转换工具不支持这台机器因为权限不够'。",
]
for t in zh: add(t)

# 2) 数学 / 推理（带分步要求）
math_ = [
    "笼子里有鸡和兔共 12 个头、32 只脚，问鸡和兔各几只？请分步推理并给出答案。",
    "一件商品先涨价 20% 再降价 20%，最终价格相对原价变化了多少？请给出推导。",
    "一个圆柱体积为 300 立方厘米，底面半径 5 厘米，求它的高（π 取 3.14），保留两位小数。",
    "甲乙两人相向而行，速度分别是 4 km/h 和 6 km/h，相距 30 km，多久相遇？请写出方程。",
    "计算 sum_{k=1}^{100} k^2，并说明你用的公式。",
    "某模型 TTFT 为 355 ms，TPOT 为 21.8 ms，生成 512 个 token 的总耗时是多少？",
    "把 3/7、5/11、8/17 从小到大排序，说明比较方法。",
    "一个序列每次翻倍，10 步后是 1024，第 3 步是多少？请说明推理。",
]
for t in math_: add(t, "\n答案:\n")

# 3) 代码生成 / 解释（Python / C / shell）
code = [
    "写一个 Python 函数，读取一个目录下的所有 .rknn 文件并返回它们的字节数，按从大到小排序。",
    "解释这段 C 代码的问题：`free(p); printf(\"%s\", p);`",
    "用 bash 写一个脚本，轮询某个 TCP 端口直到可连接，最多等 120 秒，超时返回非零。",
    "下面这段 Python 报 KeyError，请说明原因并给修复：\nd = {'a': 1}\nprint(d['b'] if 'b' in d else d['c'])",
    "写一个最小可运行的 FastAPI 服务，暴露 /v1/chat/completions，把请求转发到另一个 OpenAI 兼容端点。",
    "把这段 shell 改成不依赖外部命令的版本：`ls | grep '.log$' | xargs tail -n 5`",
    "用 Python 实现一个 LRU 缓存，容量 128，要求 O(1) 读写。",
    "为 `aria2c -x8 -s8 --continue=true` 写一个封装函数，下载失败时自动重试并降并发。",
]
for t in code: add(t)

# 4) function call / JSON 结构化输出
tools = [
    "你是一个设备管理助手，可用工具：{\"name\":\"reboot_device\",\"parameters\":{\"device_id\":\"string\"}} 和 {\"name\":\"get_card_status\",\"parameters\":{}}。用户说：帮我看下 0 号卡的健康状态。请只输出工具调用 JSON。",
    "系统：你可以调用工具 search_docs(query)。用户：RKNN3 里 RKNN3_ERR_INCOMPATIBLE_VERSION 是什么原因？请输出工具调用 JSON。",
    "把下面信息抽成 JSON：设备 0，型号 RM1828MC0-F，序列号 R2DMA262202273，温度 37 度，显存 36/5120 MB。字段名用 device_id/product_name/serial/temperature/memory_used/memory_total。",
    "请把这段日志结构化为一行 JSON：\"18:22:28 UP; LnkSta 5.0GT/s x1; DLActive-; firmware 1.1.0\"。",
    "系统提示：你只能回答与推理卡相关的问题，其他问题回答“超出范围”。用户：今天天气怎么样？",
    "生成一个表格（Markdown）对比 w4a16/w6a16/w8a16/w16a16 的权重体积与精度。",
]
for t in tools: add(t)

# 5) 长文本理解 / 摘要 / 检索式问答
long_doc = (
    "背景资料：某嵌入式平台使用协处理器加速推理。协处理器通过 PCIe 根端口挂载，根端口训练结果为 "
    "Width x1 / 2.5GT/s 时链路半速；驱动为 pcie-rkep 3.3.1，固件版本 1.1.0，运行时库 librknn3_api.so 与 "
    "librknn3_api_rkcp.so 版本必须与固件一致，否则返回 RKNN3_ERR_INCOMPATIBLE_VERSION(-18)。传输代理 "
    "rknn3_transfer_proxy 由 systemd 服务常驻，卡已启动时重复执行启动脚本会返回 status=234 并使服务不可用。"
)
add(long_doc + "\n请列出上文提到的三个版本必须一致的组件。")
add(long_doc + "\n如果重复执行启动脚本会发生什么？给出原文依据。")
add(long_doc + "\n请把上文压缩成 40 字以内的一句话。")
add(long_doc + "\n请判断：'链路半速一定导致推理失败' 这句话能否从文中推出？说明理由。")

# 6) 英文通用
en = [
    "Summarize in two sentences why an M.2 AI coprocessor needs a separate 12V supply.",
    "Translate to Chinese: The root port trained to x1 at 2.5 GT/s, and the endpoint did not respond to configuration space reads.",
    "Explain the difference between time-to-first-token and time-per-output-token in LLM inference.",
    "Given TTFT=192.6 ms, TPOT=21.7 ms, and 128 generated tokens, estimate the total latency and show the arithmetic.",
    "Rewrite this sentence to be more precise: 'the card is fast when the model is small'.",
    "List three likely causes of a PCIe device not enumerating, and the first command you would run for each.",
    "What does group-wise quantization (group32) do that per-channel quantization does not?",
    "Answer briefly: is weight-only quantization (w4a16) likely to hurt long-context retrieval more than short-context chat? Why?",
]
for t in en: add(t)

# 7) 指令跟随 / 约束遵守
add("只输出一个数字，不要任何解释：把 27 除以 4 取整数部分。", "\n答案:\n")
add("按顺序输出 1 到 20 中所有不能被 3 整除的数，用逗号分隔，不要多余文字。")
add("请把下面列表按字母顺序排序并输出：delta, alpha, charlie, bravo, echo")
add("用一句话回答，且必须以“结论：”开头：推理卡的显存大小主要限制什么？")
add("将 'RK1828 / RK1820 / RK3572' 这三个芯片按它们作为协处理器还是片内 NPU 分类，用两行输出。")

# 8) 多轮/带系统提示的对话样本
add("系统：你是一个严谨的技术助手，回答必须给出依据。\n用户：ModelMemory 4910 MB、卡显存 5120 MB，还剩多少余量？请给出计算。")
add("系统：你是设备运维助手。\n历史：用户此前询问过看门狗配置。\n用户：systemd 的 RuntimeWatchdogSec=40 表示什么？")
add("系统：你只能使用提供的上下文回答。\n上下文：{'device':0,'online':true,'temp':38}\n用户：这张卡现在温度多少？在线吗？")

# 9) 少量带 target 的样例（官方 llm_quant.json 里部分条目带 target）
add("半胱氨酸和酪氨酸在人体内可分别由什么转变而来。\nA. 胱氨酸和蛋氨酸\nB. 蛋氨酸和苯丙氨酸\nC. 赖氨酸和蛋氨酸\nD. 胱氨酸和苯丙氨酸",
    "\n答案:\n A\n")
add("下列关于蛋白质的叙述，错误的是____\nA. 有些蛋白质具有运输功能\nB. 有些蛋白质具有催化功能\nC. 有些蛋白质具有调节功能\nD. 有些蛋白质具有免疫功能\nE. 所有蛋白质都含有S元素",
    "\n答案:\n E\n")

# 10) 补齐到 ~96 条：变体改写（保持多样性，避免重复度过高）
prefix = ["请直接回答：", "简要说明：", "分两步回答：", "用中文回答：", "要求：先给结论再给理由。"]
base = zh + code + en
for i in range(len(S), 96):
    b = base[i % len(base)]
    add(prefix[i % len(prefix)] + " " + b)

random.shuffle(S)
assert len(S) <= 128, len(S)
out = "rk1828/datasets/spark_calib.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(S, f, ensure_ascii=False, indent=2)
print(f"样本数 {len(S)} -> {out}")
print("首条示例:", json.dumps(S[0], ensure_ascii=False)[:200])
