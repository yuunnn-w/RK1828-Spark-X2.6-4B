# 环境与编译

整个项目分两台机器：

| 机器 | 干什么 | 硬性要求 |
|---|---|---|
| **转换机（PC）** | 拉模型、GRQ 量化、导出 ONNX、转成 `.rknn` | **x86_64 Linux 或 macOS arm64**（RKNN3 Toolkit 只发这两种 wheel，Windows 原生不行）；Python 3.10/3.12；GRQ 需要 **NVIDIA GPU + CUDA** |
| **板子（RK1828 开发板）** | 编译并常驻运行推理服务、跑评测与压测 | aarch64 Linux + RK1828 M.2 算力卡已正确枚举；GCC ≥ 9 或交叉工具链；CMake ≥ 3.16 |

---

## 1. 取 Rockchip RKNN3 SDK（不在本仓库里）

Rockchip 的运行时与头文件受 **RKNN SDK License** 约束，不能随本仓库分发。两种取法：

```bash
scripts/get_sdk.sh                      # 克隆 airockchip/rknn3-toolkit（tag V1.2.0）到 .sdk/
# 或者已有 SDK：
export RKNN3_SDK_ROOT=/path/to/rknn3-toolkit
```

需要用到的是：

- 头文件 `rknn3-runtime/rknn3-api/include/rknn3_api.h`
- 库 `rknn3-runtime/rknn3-api/lib/Linux/aarch64/librknn3_api.so`、`librknn3_api_rkcp.so`（协处理器后端）

转换机上还要装 toolkit 的 wheel（PC 侧转换工具，同样在这个仓库里）：

```bash
cd $RKNN3_SDK_ROOT/rknn3-toolkit/packages
md5sum -c md5sum.txt
pip install -r requirements_cp312-1.2.0.txt
pip install rknn3_toolkit-1.2.0-cp312-cp312-manylinux2014_x86_64.whl
```

## 2. 转换机 Python 环境

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r convert/requirements.txt
```

版本必须对齐 model-zoo 的 Spark 示例（钉 `torch 2.8.0` / `transformers 4.57.1` / `tokenizers 0.22.2`）。
`numpy` 在 model-zoo 侧被钉 `<=1.26.4`，而 toolkit 的 requirements 要求 `>=2.0` —— 若冲突，以能装下
model-zoo Spark 示例的那套为准，转换流程实测可用。

## 3. 板端编译推理服务

```bash
# 板上（aarch64，本机编译最省事）
git clone https://github.com/<你的账号>/RK1828-Spark-X2.6-4B.git && cd RK1828-Spark-X2.6-4B
scripts/get_sdk.sh          # 或 export RKNN3_SDK_ROOT=...
scripts/build_server.sh     # 产物 server/build/rknn_spark_server + server/build/lib/*.so
```

交叉编译（在 PC 上编板端二进制）：

```bash
export CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++
scripts/build_server.sh
```

CMake 会把运行时 `.so` 收进 `server/build/lib/`，可执行文件的 RPATH 是 `$ORIGIN/lib`，
所以 `bin/` 与 `lib/` 一起拷走即可运行；若系统里已装好 RKNN3 运行时（官方 DEB 会装到 `/usr/lib`），
`lib/` 目录可以不要。

## 4. 版本一致性（最容易踩的坑）

`librknn3_api.so`、`rknn3_transfer_proxy`、协处理器固件、以及 `.rknn` 里内嵌的编译器版本必须**同代**，
否则：

- `RKNN3_ERR_INCOMPATIBLE_VERSION (-18)`，或
- `MODEL_SETUP fail, ack = ACK_FAIL`（用 1.2.0 toolkit 转的模型跑在 1.1.0 固件的卡上），或
- `rknn3_session_init` 报 `model has no kvcache input name Th, Tc, Ts, Tsr!`（1.1.0 toolkit 转的模型跑在 1.2.0 卡上）

本项目实测可用的组合：**固件 1.2.0 + rknn3 API 1.2.0 + PCIe 驱动 3.3.1 + toolkit 1.2.0**。
板端查版本：

```bash
rknn-smi -v                      # 固件 / API / 驱动版本
rknn-smi info                    # 卡状态
```

## 5. 卡不在位时的表现

```bash
lspci -d 1d87:            # 应看到 Rockchip 协处理器（RK1828 是 1d87:182a）
lsmod | grep pcie_rkep
sudo modprobe pcie_rkep   # 未加载时
sudo env TMPDIR=/run/rknn3 pcie_upgrade_tool ld
```

若根端口只训练出半条链路（`LnkSta: Width x1 / 2.5GT/s`、`DLActive-`）而设备不应答配置空间，
那是 PCIe 接触/供电/插槽问题，与本项目软件无关（`docs/troubleshooting.md` 有排查顺序）。
