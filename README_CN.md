# RK1828 + Spark-X2.6-4B：单卡 64K+ 上下文的板端推理服务

在 **Rockchip RK1828 M.2 算力卡**（5 GB 卡载内存，8 核 NPU）上部署讯飞 **Spark-X2.6-4B**
（Apache-2.0，混合注意力，标称 1M 上下文），并用自研的 **OpenAI 兼容 C++ 推理服务**常驻对外提供 API。

这条路线全部在本仓库实测跑通：**模型转换 → 卡内存判定 → 编译 → 板端常驻服务（开机自启 + 进程守护）→
精度/复读/长上下文/压力测试**。全部数字都在真机（OrangePi5Max + RK1828 卡，固件 1.2.0）上测得，
测法与脚本都在 `tools/` 里，可复现。

> **为什么是自研服务而不是官方 `rkllm3-server`**：Rockchip RKNN3 1.2.0 的 `rkllm3-server`
> 对 `model_type=spark2_5` 的输出链路是坏的 —— 穷举 6 个配置变体（含自定义 chat template，
> 且已确认模板确实生效、输入 token 序列完全正确）后，输出仍从第一个 token 开始崩坏，
> 48 题评测 0 正确 / 39 条复读崩坏；同一份模型文件走 model-zoo 的 C++ demo 路径完全正常。
> 证据与复现步骤见 [`docs/troubleshooting.md` 第 4 条](docs/troubleshooting.md)。

---

## 实测成绩（速览）

| 项目 | 实测值 |
|---|---|
| 上下文 | **73,728 token 档位可用**（单次 prefill 53,166 token / 115.6 s，needle 命中；38,006 token / 72.0 s 也命中）；int4 KV 的卡内存硬上限 ≈ 79,900 token |
| 卡内存 | 4903 / 5120 MB（权重 2.37 GiB + KV + 内部缓冲），`model_info` Per-Core 合计 4853.74 MB |
| 解码吞吐 | **50–52 tok/s**（短上下文，19–20 ms/token）；14.8k 上下文 42.6 tok/s；38k/53k 上下文 35/32 tok/s |
| Prefill 吞吐 | 647 tok/s（88 token，137 ms）→ 788（4.3k）→ 685（14.8k）→ 约 460（38k） |
| 精度（392 题留出集，贪心 + thinking on） | 可判分 332 题，正确 **99 = 29.8 %**；GRQ w4a16 与 bf16 原模型逐 token top-1 一致率 **96.16 %**，留出集 PPL 1.2451（bf16 为 1.2536） |
| 复读 | 392 题里**真实复读 0 条**；48 组对抗性 prompt 里 runaway 2 组（都是 prompt 明确要求"重复这段话"）；熔断三层默认开启，误伤时可调 |
| 工具调用 | 打通：`finish_reason=tool_calls`、`arguments` 可 JSON 解析、不给 `tools` 时不会伪造（官方 function-call 示例 8 题命中 4） |
| 压力/兼容性 | 31 项套件 **失败 1**（唯一失败是模型行为，不是服务缺陷，见"已知限制"）；并发 24 全成功；120 条浸泡零错误零复读 |
| 温度 | 连续评测 + 压测期间卡 29–56 °C（被动散热），SoC 热区 ≤ 33 °C，无热节流事件 |

完整数字与测法：[`docs/benchmarks.md`](docs/benchmarks.md)。

---

## 你需要什么

- **硬件**：一块带 M.2 Key-M 插槽的 aarch64 Linux 开发板（本项目用 OrangePi5Max / RK3588）+ RK1828 M.2 算力卡
  （**必须外接 12 V 供电**），以及一台做模型转换的 PC（**x86_64 Linux 或 macOS arm64**，GRQ 量化需要 NVIDIA GPU）。
- **软件**：Rockchip RKNN3 SDK **1.2.0**（固件 + 运行时 + toolkit，需自行下载并接受 Rockchip 的许可）、
  CMake ≥ 3.16、GCC ≥ 9、Python 3.12（转换侧）。
- 版本必须同代，否则模型加载失败（见 [`docs/build.md`](docs/build.md) 第 4 节）。

## 5 分钟上手

```bash
git clone https://github.com/yuunnn-w/RK1828-Spark-X2.6-4B.git
cd RK1828-Spark-X2.6-4B

# ① 转换（在 PC 上，不是在板子上）
model/download_model.sh                                   # ModelScope 拉 XHToken/Spark-X2.6-4B
convert/run_conversion.sh --model-dir model/Spark-X2.6-4B --ctx 73728 --out model/out

# ② 编译推理服务（在板子上）
scripts/get_sdk.sh && scripts/build_server.sh

# ③ 部署成开机自启的常驻服务
#   把 model/out 里的 .rknn/.weight/.safetensors 与 .tokenizer.gguf/.embed.bin 放进 /opt/rk1828-spark/model/
sudo PREFIX=/opt/rk1828-spark deploy/install_service.sh
curl -s localhost:8080/health          # {"status":"ok"}
curl -s localhost:8080/v1/models | head -c 400
```

装好之后：`/etc/rk1828-spark.env` 改参数（模型档位、端口、采样、熔断、鉴权），
`systemctl restart rk1828-spark.service` 生效；日志 `journalctl -u rk1828-spark -f`。

调用示例（`examples/`）：

```bash
BASE=http://<板子IP>:8080 examples/curl.sh
python3 examples/client.py --base http://<板子IP>:8080      # 含工具调用与 4 万 token needle 测试
```

## 目录结构

```
server/          自研推理服务源码（C++11，engine / httpd / main + CMake）
  cpp/           核心源码
  thirdparty/    随带的第三方（nlohmann/json MIT、model-zoo utils 与 tokenizer Apache-2.0）
convert/         转换流水线：export_rknn.py（带可调档位）、run_conversion.sh、校准集构建、GRQ 精度自检
deploy/          systemd 单元模板、启动包装、健康守护、安装脚本、env 样例
tools/           评测 / 压测 / 复读专项 / 长上下文 / 温度采样脚本
datasets/        GRQ 校准集（884 条）与留出评测集（392 条）+ 来源清单
docs/            build / conversion / server / benchmarks / testing / troubleshooting
scripts/         get_sdk.sh、build_server.sh
examples/        curl 冒烟脚本与 Python 客户端示例
model/           download_model.sh（权重不入 git）
```

## 关键配置（长上下文 Agent / 编码）

默认采样取自**模型厂商**的 `generation_config.json`（`temperature 1.0 / top_p 0.95 / top_k 不截断 /
惩罚全 0`），**输出长度默认不限制**（`max_tokens` 不传即生成到 EOS 或上下文用完），
并且**刻意不加任何重复惩罚** —— 代码里合法的重复结构会被重复惩罚误伤。
防复读改由三层熔断负责（周期 / n-gram / 长块），阈值可调、可整体关掉：

```bash
# /etc/rk1828-spark.env
TEMP=1.0 TOP_K=0 TOP_P=0.95 REPEAT_PENALTY=1.0 MAX_NEW=0
EXTRA_ARGS="--reasoning-format deepseek"      # 思考与答案分离
# 熔断误伤时：EXTRA_ARGS="... --ngram-ban-at 6 --loop-max-occurrences 4"，或 --fuse-off
```

两条实测经验（详见 `docs/benchmarks.md`）：

- **精度关键的调用建议显式传 `temperature 0`**：同一批题上贪心比厂商采样高约 8 个百分点。
- **"只输出代码"这类短任务用 `reasoning_effort:"none"` 或 `thinking:false`**：thinking on 时模型可能把
  复述写在回答通道里就收尾，代码根本不出现（`fib_probe` 六组对照）。

## 已知限制（诚实清单）

- **单 session 串行**：卡上模型常驻 4.9 GB，只有一个推理会话；并发体现在 HTTP 层与排队上
  （队列超过 `--max-queue` 返回 429 + `Retry-After`）。
- **明确拒绝 `n>1`**（返回 400，不会静默只回一条）。**未实现**：`beam_search`、多模态输入（图片/音频）、
  `/v1/embeddings`、`audio`/`modalities`、`rknn3_session_run_async` 重叠取样。
- **上下文长度是转换时写死的**：运行时只会把过长的请求夹回模型编译值，不会更长。
- 一个尚未解决的模型侧行为：`thinking on` + 短预算时可能"只复述不作答"；选择题答错时可能输出重复字母
  （如 `AAAAA`）。这不是服务缺陷，但会影响端到端体验。
- 评测正确率（29.8 %）是**这颗 4B 模型本身**在数学/推理难题上的水平，不是部署损失；
  权重 int4 已证明近乎无损，剩余差距属于模型能力。

## 许可

- 本仓库原创代码：**GPL-3.0**（见 `LICENSE`）。
- 随带的第三方部件与其许可见 **`NOTICE`**（nlohmann/json MIT；model-zoo 的 utils/tokenizer/float16 Apache-2.0）。
- **Rockchip RKNN3 SDK、固件、驱动、运行时库不在本仓库内**，需自行下载并接受 Rockchip 的 RKNN SDK License。
- 模型权重 `XHToken/Spark-X2.6-4B` 为 **Apache-2.0**（讯飞 SparkLLM Team），自行下载。

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
