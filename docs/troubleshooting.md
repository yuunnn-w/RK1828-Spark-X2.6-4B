# 踩过的坑与排查顺序

按"现象 → 根因 → 处置"排列。前 4 条是会让人浪费一整天的那种。

---

## 1. 模型加载就失败：版本不同代

| 现象 | 根因 |
|---|---|
| `RKNN3_ERR_INCOMPATIBLE_VERSION (-18)` | `librknn3_api.so` / `rknn3_transfer_proxy` / 卡固件 三者版本不一致 |
| `MODEL_SETUP fail, ack = ACK_FAIL`，或 CNN demo 直接段错误（rc=139） | 用 **1.2.0** toolkit 转的模型跑在 **1.1.0** 固件的卡上 |
| `model_init` 能过，`rknn3_session_init` 报 `model has no kvcache input name Th, Tc, Ts, Tsr!` + `input name check failed` | 用 **1.1.0** toolkit 转的 LLM 跑在 **1.2.0** 卡上（与 ctx、KV 档位、`save_ctx` 全都无关） |

处置：让四样东西同代（本项目：**固件 1.2.0 + API 1.2.0 + 驱动 3.3.1 + toolkit 1.2.0**）。
升级固件的正确姿势只有一条：**安装与版型匹配的 installer → 重启主机 → 开机时 `rknn3_startup` 自动执行
`pcie_upgrade_tool uf /lib/firmware/rknn3_rk1820.img -reset`**。

## 2. `rknn3_session_init` 失败但 `model_info` 说放得下

卡内存判定要用 `rknn3_model_info <model>.rknn 0xff` 的 **Per-Core 合计 ≤ 5051.80 MB**。
实测 `77824` 档合计 4985.74 MB（看似还剩 66 MB）却 `session_init` 失败 —— 静态估算不含运行时额外分配。
留足余量：本项目定在 `73728`（余 198 MB）。详见 `docs/conversion.md` 的实测表。

## 3. ⚠️ 千万不要用 `systemctl restart rknn3` 去"释放卡"

卡上还有残留 session 时，这个动作会把卡打进 **loader 态**：
`Device boot state: 0x00004c42, success: 1` → `check ddr load addr request failed` → `Device request state: 0xffa70000`。
此时 `pcie_upgrade_tool uf <img> -reset`、`db bootloader`、`rknn-smi reset -t hw`、`rmmod pcie_rkep` **全部无效**。

**唯一有效手段：整机断电 20–30 秒再上电**（`shutdown -r` 这类热重启不会给卡断电，因为卡的 12 V 取自板端供电端子）。
正确做法：结束所有持卡进程后等它自然释放，或直接断电。同理不要远程重启板子来"抢救"服务。

## 4. 官方 `rkllm3-server` 服务 Spark 模型会输出乱码

`rkllm3-server` 1.2.0 对 `model_type=spark2_5` 的链路是坏的：
48 题评测里 **0 正确、39/40 复读崩坏**，输出从第一个 token 就是 `<think>我们, `、`, `、`**，请，请，请…`。
穷举验证过 6 个变体：默认 / `--chat-template-file`（自定义成与 demo 逐字一致）/ template + `--reasoning auto|on|off`
/ `--rope-tensor` / 采样参数变化。其中一项决定性证据：自定义 chat template **确实生效**
（日志打印出我们的模板，`prompt_tokens` 从 26 降到 16，输入 token 序列完全正确），**输出照样崩坏**
→ chat template 假设被证伪，问题在 server 的运行时链路（server 侧走 `rknn3_session_run_async` +
`set_kvcache_policy(n_keep=0, group_size=1)`，且不调 `rknn3_session_set_chat_template`，解码只有 16 tok/s；
demo 路径 50–66 tok/s）。`/v1/completions` 在 1.2.0 里不存在（404），无法直接喂相同 prompt 对照。

结论：**Spark 系列只能走 C++ demo 路径**，也就是本仓库的 `rknn_spark_server`。

## 5. 卡枚举不到 / 链路半速

现象：根端口 `LnkSta: Speed 2.5GT/s Width x1`、`DLActive-`、`CorrErr+`，设备不应答配置空间（期望 `1d87:182a`）；
同一槽位插 NVMe SSD 满速正常。

排查顺序（本项目实际走过的路）：
1. `lspci -d 1d87:` / `lsmod | grep pcie_rkep` / `sudo modprobe pcie_rkep`；
2. 驱动是否被 Secure Boot 拒了：`dkms status | grep pcie-rkep`、`modprobe` 报 `Key was rejected by service`
   → `mokutil --import` 后在开机时确认 MOK，或 BIOS 关 Secure Boot；
3. 物理层：换槽位、重新插拔、确认卡的 **12 V 外接供电**接好（M.2 槽本身供电不够）；
4. 结构干涉（散热片/ standoff 高度导致接触不良）——本项目最后是这个原因，重新压紧后 `5GT/s x1` 满链路；
5. 以上都正常再看固件与 `rknn3.service`（`rknn-smi -v`、`pcie_upgrade_tool ld`）。

## 6. systemd 起不来：`status=203/EXEC` / `Permission denied`

`ExecStart` 指向的脚本没有可执行位（通过 SFTP/SCP 上传的文件常落 644）。
`StartLimitIntervalSec=0` + `Restart=always` 会变成疯狂重启循环。
处置：`chmod +x <start.sh>` 再 `systemctl restart`。**每次上传脚本后都要 chmod。**

## 7. env 文件里带空格的值必须加引号

`/etc/rk1828-spark.env` 里写 `EXTRA_ARGS=--reasoning-format deepseek` 会被 shell 解析成
"带临时变量 EXTRA_ARGS 执行命令 `deepseek`" → 变量根本没设上（报错 `deepseek: not found`）。
必须写 `EXTRA_ARGS="--reasoning-format deepseek"`。自检：
`sh -c '. /etc/rk1828-spark.env; echo [$EXTRA_ARGS]'`。

## 8. 重新编译报 `Text file busy`

先停服务再编译：`systemctl stop rk1828-spark.service`（用 SIGTERM，别 `kill -9`，否则可能把脏 session 留在卡上）。

## 9. HTTP 500 / 400 与非法 UTF-8

- 已修的服务端问题：模型输出里跨 token 边界截断的多字节字符会产生非法 UTF-8，
  JSON 库默认 `strict` 序列化会抛异常 → 被 catch-all 变成 500。现在所有出站 JSON 走 `replace` 模式，
  引擎里做 `fix_utf8()` 与跨片段续接。
- 客户端问题：Windows Git Bash 在某些代码页下发送中文 JSON 会产出非法 UTF-8，服务端返回 400 是**正确行为**。
  要用 Python/`curl` 在 UTF-8 环境下发。

## 10. 温控与降频

`rknn-smi info -t thermal_freq` / `thermal_critical`：默认 83 °C→850 MHz、90 °C→500 MHz（可用 400/500/850），
`thermal_critical` 100 °C 且默认**不**自动复位。SMI 下发的温控只存 DRAM，重启即复原。
实测（被动散热、无风扇）：连续评测 + 压测期间卡温 29–56 °C，SoC 热区 28–33 °C，无节流事件。

## 11. 其它小坑

- 每个 `rknn.build()` 用**独立工作目录**：缓存目录是 CWD 相对的 `./tmp`，失败构建会污染缓存，
  之后可能报 `Max_position_embeddings updated failed. rope_cos_cache_0 values check failed`。
- 板端跑任何 rknn3 程序前 `export TMPDIR=/run/rknn3`（本项目的 systemd 单元已设）。
- `rknn-smi info` 可能打印正常却返回码 251；判断可用性应以 `rknn-smi -v` / `rknn-smi info -l` 的返回码为准。
- `n_predict` / `-n` 在官方 server 里默认是 `-1 = context size`；本服务 `--max-new-tokens 0` 同义（不限制）。
