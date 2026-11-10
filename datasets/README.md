# 校准集与留出评测集

| 文件 | 条数 | 用途 |
|---|---|---|
| `calib_real.json` | 884 | GRQ 量化校准（格式同 `rknn3-model-zoo/datasets/llm_quant.json`：`[{"input","target"}]`） |
| `calib_real.provenance.json` | 884 | 每条的来源标记（`_src` = 上游数据集:配置:切分，`_domain` = 领域） |
| `eval_real.json` | 392 | **留出评测集**，不参与校准，用于测转换后的精度与复读 |

领域分布（`_domain`）：`math_en` / `math_hard`（AIME 级）/ `math_live` / `reason` / `reason_live` /
`instr_live` / `code` / `zh_exam`（高考题型）/ `agent_tool` 等，中英混合，都是人工挑过的**小体量高含金量**
题集（逻辑、数学、代码、指令跟随、工具调用），不是通用大语料。

为什么这么挑：`rknn3-model-zoo/examples/Qwen3_5/README.md` §5 明确写了 GRQ 校准样本**要贴近真实业务输入**
（含 system prompt、function call 形态），数量 20~128 条即可；实测用这种"小而有代表性"的集合，
GRQ w4a16/group32 在留出集上的困惑度甚至略低于 bf16 原模型（详见 `docs/benchmarks.md`）。

## 重新生成

```bash
# 从上游公开数据集重新构建（走 hf-mirror，不需要代理；需要 pip install datasets）
WORKDIR=/tmp/rk1828-calib bash convert/build_datasets.sh
# 不依赖任何上游数据集的合成版（较小，用于打通流程）
python3 convert/make_calib_synth.py
```

## 许可提醒

这两份文件是**从多个公开数据集里挑选、改写、拼装**出来的研究用小题集，来源逐条记在
`calib_real.provenance.json` 里。上游各自的许可条款不同且可能变化 —— 用于本项目之外的用途前，
请自行核对每个来源的数据集许可与模型许可。本仓库不对这些外部数据集的许可作出声明。
