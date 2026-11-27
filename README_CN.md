# RK1828 + Spark-X2.6-4B：单卡 64K+ 上下文的板端推理服务

在 Rockchip **RK1828 M.2 算力卡**（5 GB 卡载内存、8 核 NPU）上跑讯飞 **Spark-X2.6-4B**，
外面套一个自研的 **OpenAI 兼容 C++ 服务**，让本地 Agent、脚本、客户端像调 OpenAI 一样调用它。

整条路都在这台真机上跑通了：模型转换 → 卡内存核算 → 编译 → 常驻服务（开机自启 + 进程守护）→
精度 / 复读 / 长上下文 / 压力测试。文中所有数字都来自 OrangePi5Max + RK1828（固件 1.2.0），
测法和脚本在 `tools/` 里。

先解释几个后面会反复出现的词：

- **token**：模型处理文本的最小单位，中文大约 1 字 1 token，英文大约 4 字符 1 token。
- **prefill**：把整段提示词一次性读进模型，算出 KV；**decode**：一个 token 一个 token 往下写。
  两者速度差很多，长对话里 prefill 往往是大头。
- **KV cache**：模型记住前文的中间结果，有了它，多轮对话不用把历史重新算一遍。
- **槽位（slot）**：把某个会话的 KV 存成一份快照。服务用 4 个槽位在多个会话之间切换。

## 为什么不用官方 `rkllm3-server`

Rockchip RKNN3 1.2.0 的 `rkllm3-server` 对 `model_type=spark2_5` 的输出链路是坏的：
换 prompt、换 chat template、开关 `--reasoning`、改采样，输出都从第一个 token 开始崩
（`We need answer answer answer…`、整片换行）。同一份模型文件走 model-zoo 的 Spark C++ demo 完全正常。
本服务就是把这条已经验证可用的路径做成常驻服务 + HTTP 接口。复现步骤见
[`docs/troubleshooting.md` 第 4 条](docs/troubleshooting.md)。

## 实测结果

| 项目 | 实测值 |
|---|---|
| 上下文 | **73,728 token 档位可用**。单次 prefill 53,166 token 用时 115.6 s，needle 能召回；38,006 token 用时 72.0 s。int4 KV 的卡内存硬上限约 79,900 token |
| 卡内存 | 4903 / 5120 MB（权重 2.37 GiB + KV + 内部缓冲） |
| 解码速度 | 短上下文 **50–52 tok/s**（19–20 ms/token）；14.8k 上下文 42.6 tok/s；38k / 53k 降到 35 / 32 tok/s |
| Prefill 速度 | 647 tok/s（88 token）→ 788（4.3k）→ 685（14.8k）→ 约 460（38k） |
| 精度（392 题留出集，贪心 + thinking） | 可判分 332 题，正确 **99 题 = 29.8 %**（这轮测于修复 system 送达之前，见 `docs/benchmarks.md` §2 / §6.2）；GRQ w4a16 与 bf16 原模型的 top-1 一致率 **96.16 %**，留出集 PPL 1.2451（bf16 是 1.2536） |
| 复读 | 392 题里真实复读 **0 条**；48 组对抗性 prompt 里 2 组失控（都是 prompt 明确要求"重复这段话"）。三层熔断默认开启，误伤时可调 |
| 工具调用 | 正常：`finish_reason=tool_calls`，`arguments` 能被 JSON 解析，不给 `tools` 时不会凭空造调用（官方 function-call 示例 8 题命中 4）。流式按 OpenAI 规范分片下发，`tool_choice=required` 生效 |
| 多轮 KV 复用 | 4 个槽位放在内存盘里。同一会话第二轮（1214 token）只 prefill **18 token**；槽位开、关两种配置下，贪心输出逐字节一致 |
| 压力 / 兼容性 | 31 项套件失败 1 项（是模型行为，不是服务缺陷）；24 并发全部成功；120 条浸泡测试零错误 |
| 温度 | 连续评测 + 压测期间卡 29–56 °C（被动散热），SoC 热区不超过 33 °C，没有热节流 |

完整数字和测法：[`docs/benchmarks.md`](docs/benchmarks.md)。

## 你需要什么

- **硬件**：带 M.2 Key-M 插槽的 aarch64 Linux 开发板（本项目用 OrangePi5Max / RK3588）+ RK1828 M.2 卡
  （**必须接 12 V 供电**）；一台做模型转换的 PC（**x86_64 Linux 或 macOS arm64**，GRQ 量化需要 NVIDIA GPU）。
- **软件**：Rockchip RKNN3 SDK **1.2.0**（固件 + 运行时 + toolkit，需要自己下载并接受 Rockchip 的许可）、
  CMake ≥ 3.16、GCC ≥ 9、Python 3.12（转换侧）。
- 固件、运行时、toolkit 必须同代，否则模型加载会失败（见 [`docs/build.md`](docs/build.md) 第 4 节）。

## 快速开始

```bash
git clone https://github.com/yuunnn-w/RK1828-Spark-X2.6-4B.git
cd RK1828-Spark-X2.6-4B

# ① 在 PC 上转换（不要在板子上跑）
model/download_model.sh                                   # 从 ModelScope 拉 XHToken/Spark-X2.6-4B
convert/run_conversion.sh --model-dir model/Spark-X2.6-4B --ctx 73728 --out model/out

# ② 在板子上编译服务
scripts/get_sdk.sh && scripts/build_server.sh

# ③ 装成开机自启的常驻服务
#   先把 model/out 里的 .rknn/.weight/.safetensors 和 .tokenizer.gguf/.embed.bin 放进 /opt/rk1828-spark/model/
sudo PREFIX=/opt/rk1828-spark deploy/install_service.sh
curl -s localhost:8080/health          # {"status":"ok"}
```

装好以后：改参数编辑 `/etc/rk1828-spark.env`，然后 `systemctl restart rk1828-spark.service`；
看日志用 `journalctl -u rk1828-spark -f`。调用示例在 `examples/`：

```bash
BASE=http://<板子IP>:8080 examples/curl.sh
python3 examples/client.py --base http://<板子IP>:8080      # 含工具调用与 4 万 token needle 测试
```

KV 槽位需要一块内存盘。`deploy/start.sh` 会检查 `KV_SLOT_DIR`：如果它还不是 tmpfs，就按
`KV_SLOT_TMPFS_SIZE`（默认 11G）挂一块；挂载失败会退回磁盘（能用，但保存/加载慢很多）。

## 目录结构

```
server/          自研推理服务（C++11）：engine 推理引擎、httpd HTTP/SSE、main OpenAI 协议 + 运维端点
convert/         转换流水线：export_rknn.py、run_conversion.sh、校准集构建、GRQ 精度自检
deploy/          systemd 单元、启动脚本、健康守护、安装脚本、env 样例
tools/           评测、压测、复读专项、长上下文、协议验收脚本
datasets/        GRQ 校准集（884 条）与留出评测集（392 条）+ 来源清单
docs/            build / conversion / server / chat_template / benchmarks / testing / troubleshooting
scripts/         get_sdk.sh、build_server.sh
examples/        curl 冒烟脚本与 Python 客户端示例
model/           download_model.sh（权重不入 git）
```

## 常用配置（长上下文 Agent / 编码）

默认采样值来自**模型厂商**的 `generation_config.json`：`temperature 1.0`、`top_p 0.95`、
`top_k` 不截断、各种惩罚全 0。输出长度默认不限制（不传 `max_tokens` 就生成到 EOS 或上下文用完）。
**刻意不加重复惩罚**，因为代码里合法的重复结构会被误伤；防复读交给三层熔断（周期 / n-gram / 长块）。

```bash
# /etc/rk1828-spark.env
TEMP=1.0 TOP_K=0 TOP_P=0.95 REPEAT_PENALTY=1.0 MAX_NEW=0
EXTRA_ARGS="--reasoning-format deepseek"      # 思考和正式回答分开返回
# 4 个 KV 槽位放在 11G 内存盘里（start.sh 负责挂载）
KV_SLOTS=4 KV_SLOT_DIR=/run/rk1828-kv KV_SLOT_TMPFS_SIZE=11G
# 熔断误伤时：EXTRA_ARGS="... --ngram-ban-at 6 --loop-max-occurrences 4"，或 --fuse-off
```

三条实测经验（详见 `docs/benchmarks.md`）：

- **要精度就显式传 `temperature: 0`**：同一批题上贪心比厂商采样高约 8 个百分点。
- **"只输出代码"这类短任务关掉思考**：用 `reasoning_effort: "none"` 或 `thinking: false`。
  thinking 开着时，模型有时会把任务复述一遍就收尾，代码根本不出现。
- **不要关 `--system-in-prompt`**：RKNN3 的 `set_chat_template` 不吃 `system_prompt`，走那条路
  客户端的 system 会被静默丢掉（`tools/sysprobe.py` 可复现：修复前 1/6，修复后 6/6）。
  Agent 的 system 里装的是指令和工具说明，丢了就是隐性失效。

## 已知限制

- **单 session，请求串行**：模型常驻 4.9 GB，卡上只有一个推理会话；并发只体现在 HTTP 层和排队
  （队列超过 `--max-queue` 返回 429 + `Retry-After`）。4 个 KV 槽位解决的是会话切换时的重复 prefill，
  不是并行推理。
- **KV 槽位要求客户端原样回放**上一轮 assistant 的 `content` 和 `reasoning_content`。回放不一致
  （比如丢了 reasoning、或工具参数重新序列化后格式变了）会退回冷启动：结果仍然正确，只是慢。
  细节见 `docs/server.md` §3.2。
- **`n>1` 直接拒绝**（返回 400，不会只回一条还不告诉你）。**已实现**：`/v1/responses`（流式+非流式）、
  `/v1/embeddings`（词嵌入均值池化，近似嵌入）。**未实现**：`beam_search`、图片/音频输入、
  `audio`/`modalities`、`rknn3_session_run_async` 重叠取样。
- **上下文长度是转换时写死的**：运行时只会把过长的请求夹回模型编译值，不会变长。
- 两个模型侧行为，不是服务缺陷但会影响体验：`thinking on` + 短预算时可能"只复述不作答"；
  选择题答错时可能输出重复字母（如 `AAAAA`）。
- 29.8 % 的正确率是**这颗 4B 模型本身**在数学/推理难题上的水平，不是部署损失：权重 int4 已证明
  近乎无损，剩下的差距属于模型能力。

## 许可

- 本仓库原创代码：**GPL-3.0**（见 `LICENSE`）。
- 随带的第三方部件和许可见 **`NOTICE`**（nlohmann/json MIT；model-zoo 的 utils/tokenizer/float16 Apache-2.0）。
- **Rockchip RKNN3 SDK、固件、驱动、运行时库不在本仓库内**，需要自行下载并接受 Rockchip 的 RKNN SDK License。
- 模型权重 `XHToken/Spark-X2.6-4B` 是 **Apache-2.0**（讯飞 SparkLLM Team），自行下载。

## 致谢 / 参考

- [airockchip/rknn3-toolkit](https://github.com/airockchip/rknn3-toolkit)（V1.2.0）、
  [airockchip/rknn3-model-zoo](https://github.com/airockchip/rknn3-model-zoo)（`examples/Spark_X2_5`）
- [XHToken/Spark-X2.6-4B @ ModelScope](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B)（Apache-2.0）
- OrangePi5Max 板与 RK1828 M.2 卡（易百纳交付包）

模型引用：

```bibtex
@misc{sparkx2.6,
  title  = {Spark-X2.6-4B&1.7B: Advancing the Frontier of On-Device Agents for Real-World Tasks},
  author = {SparkLLM Team},
  year   = {2026}
}
```
