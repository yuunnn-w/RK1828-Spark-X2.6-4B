# 转换：从 HuggingFace 模型到能在 RK1828 上跑的文件

## 产物清单（RKNN3 的权重分离格式）

| 文件 | 内容 | 放哪 |
|---|---|---|
| `Spark-X2.6-4B-<ctx>-kvi4.rknn` | 计算图 + 量化元数据（约 34 MB，随 ctx 变） | 卡 |
| `Spark-X2.6-4B-<ctx>-kvi4.weight` | 量化后的权重（2,491,176,960 B ≈ 2.37 GiB，**与 ctx 无关，多个档位共用同一份**） | 卡 |
| `Spark-X2.6-4B-<ctx>-kvi4.safetensors` | 主机侧 RoPE 缓存表（`positions × 640 × 2` 字节） | 主机内存 |
| `Spark-X2.6-4B.tokenizer.gguf` | 词表 + 特殊 token + chat template | 主机 |
| `Spark-X2.6-4B.embed.bin` | 输入词嵌入表 `131072 × 2560 × 2 B` = 671 MB（fp16，不参与量化） | **主机内存** |
| `Spark-X2.6-4B.onnx` + `.config.pkl` | 中间产物（GRQ 已烘进 ONNX） | 转换机 |

最后一条容易误解：1.2.0 转换出的 Spark 模型外部输入是 `input_embeds`（不是 `input_ids`），
词嵌入在主机侧查表算好再喂给卡，所以 `.embed.bin` **不占卡内存**。判断方法：
`rknn3_model_info <model>.rknn 0xff` 看外部输入名。

## 一键跑

```bash
convert/run_conversion.sh --model-dir model/Spark-X2.6-4B --ctx 73728 --out model/out
```

分步（等价）：

```bash
export PYTHONPATH=$PWD/.model-zoo
cd .model-zoo/examples/Spark_X2_5/python
python3 export_llm.py --model_path ../../model/Spark-X2.6-4B --model_size 4B \
       --export_llm_path ../../model/out/Spark-X2.6-4B.onnx \
       --quant --cali_dataset ../../datasets/calib_real.json
# ↑ 这一步同时产出 .config.pkl / .tokenizer.gguf / .embed.bin
cd ../../model/out/build_73728 && python3 ../../convert/export_rknn.py   # 参数用 SPARK_* 环境变量给
```

## 量化档位

- **权重**：`rknn.config(quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32')`；
  真正的精度来自 **GRQ**（`export_llm.py --quant`，走 `RKQuantizer.quantize(..., quantized_algorithm='grq')`，需要 CUDA）。
  本项目实测 GRQ w4a16/group32 + 884 条真实校准，在留出集上 PPL **1.2451**，bf16 原模型 **1.2536**（甚至略低），
  逐 token top-1 与 bf16 一致率 **96.16 %** → 权重侧 int4 不是精度瓶颈（见 `docs/benchmarks.md`）。
- ⚠️ `--quant` 在无 CUDA 时会被官方脚本**静默跳过**（条件写成 `if args.quant and torch.cuda.is_available()`），
  不报错。判断有没有生效：看日志里有没有 GRQ 相关输出，或比对 ONNX 大小/权重字节。
- ⚠️ 1.2.0 对 Spark 的 `llm_head` 配置 **匹配失败**（打印 `Match llm_head with 'auto' failed`），
  于是输出头按全局 w4a16/group32 编译，拿不到默认配置里声明的 w6a16。
  模型 ONNX 的输出头是 `MatMul node_linear_216`（权重 `model.lm_head.weight` 先过 `Transpose`，
  后面还挂着 `keep_one_logit` 的 `Gather`）。想显式指定 head 位宽需要正确的张量名，本仓库尚未验证哪个名字被接受。

## KV Cache 档位（决定卡内存与上下文上限）

`export_rknn.py` 里 `attention_config` 是**两条**（Spark 是混合注意力：36 层里 9 层 full、27 层 sliding，窗口 512）：
先 sliding 档（`kvcache_buffer_len=512`、`sliding_window_size=512`），再 full 档（`kvcache_buffer_len=CTX`）。

1.2.0 只接受这 6 个固定组合，其它报 `Expect combination as follow`：

```
(Normal,     Float16,     32, 32)
(GroupQuant, Int4_to_F16, 16, 64)
(GroupQuant, Int8_to_F16, 16, 64)
(GroupQuant, Int4_to_F16, 16, 128)
(GroupQuant, Int4_to_F16, 16, 256)
(GroupQuant, Int4_to_F16, 16, 512)
```

→ `group_size` 只能 16；`residual_depth` 只能 64/128/256/512（越大越准也越占内存）。
**改 KV 配置必须重新转换**（`.rknn` 会变，`.weight` 不变）。

## 卡内存判定：必须用 `rknn3_model_info`，别信公式

板端执行：

```bash
rknn3_model_info /opt/rk1828-spark/model/Spark-X2.6-4B-73728-kvi4.rknn 0xff
```

判据：**Per-Core 合计 ≤ 5051.80 MB**（8 核容量 639.50 / 619.90 / 614.50 MB，最小核是瓶颈），
否则 `rknn3_session_init` 直接失败。

Spark-X2.6-4B 实测（toolkit 1.2.0，8 核 `0xff`，单位 MB）：

| 档位 | ctx | KV | Command | Weight | Internal | KVCache | Per-Core 合计 | 结果 |
|---|---|---|---|---|---|---|---|---|
| `-8192-kvfp16` | 8192 | Float16 | 21.33 | 2375.77 | 58.87 | 693.00 | **3148.97** | ✅ |
| `-65536-kvi4` | 65536 | INT4/GQ/g16/r64 | 32.40 | 2375.77 | 395.06 | 1786.50 | **4589.74** | ✅ 余 462 |
| `-73728-kvi4` | 73728 | 同上 | 32.41 | 2375.77 | 443.06 | 2002.50 | **4853.74** | ✅ 余 198（**推荐**） |
| `-77824-kvi4` | 77824 | 同上 | 32.41 | 2375.77 | 467.06 | 2110.50 | **4985.74** | ❌ `model_info` 判 FIT 但 `session_init` 失败 |
| `-98304-kvi4` | 98304 | 同上 | — | 2375.77 | — | — | **5645.75** | ❌ |
| `-131072-kvi4` | 131072 | 同上 | 32.42 | 2375.77 | 779.06 | 3514.50 | **6701.75** | ❌ |

反推的真实斜率（用来选 ctx）：int4/r64 的 `KVCache ≈ 26.8 KiB/token`、`Internal ≈ 5.9 KiB/token`
→ 合计 **≈ 32.7 KiB/token**；fp16 档实测 **≈ 84 KiB/token**。
线性内插出 int4 档的 ctx 硬上限 ≈ **79,900 token**。

⚠️ 教科书公式 `全注意力层数 × kv_heads × head_dim × 2 × 字节`（Spark 是 9×4×256×2）只给 9.2 KB/token，
**比实测低 3.6 倍**，因为它忽略了滑窗层缓冲、分组量化残差与元数据 —— 只能用于横向比较，不能用来判断装不装得下。

## 已经证伪的"省内存错觉"

- `kvcache_buffer_len='real_time'`（按需分配）：1.2.0 的参数检查器直接拒绝（
  `All 'kvcache_buffer_len' in 'attention_config' must be positive integers when not None!`），改成 `None` 也报错。
- 把 `kvcache_buffer_len` 调小、`max_position_embeddings` 调大：内存确实骤降（ctx 131072 + buffer 8192 → 2741 MB），
  但 `model_info` 读回 `max_ctx_len: 8192`，请求更长直接失败 —— **`kvcache_buffer_len` 就是模型可用的上下文长度本身**，
  分页 KV 不会把 buffer 撑大。
- 换 `--tie-word-embedding` / 删 `.embed.bin`：对卡内存**无效**（Spark 的词嵌入本来就在主机侧）。

## 传文件到板子

别用 WiFi 慢推。在转换机上起 HTTP 服务，板子多线程拉：

```bash
# 转换机
cd model/out && python3 -m http.server 8090
# 板子
aria2c -x8 http://<转换机IP>:8090/Spark-X2.6-4B-73728-kvi4.rknn
aria2c -x8 http://<转换机IP>:8090/Spark-X2.6-4B-73728-kvi4.weight     # 2.5 GB，实测 10 MB/s 级
```

`.weight` 与 ctx 无关：只换上下文档位时，只需重传 34 MB 的 `.rknn` 与新的 `.safetensors`。
