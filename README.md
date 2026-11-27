# RK1828 + Spark-X2.6-4B: 64K+ context LLM serving on one Rockchip M.2 accelerator

Run iFlytek **Spark-X2.6-4B** on a **Rockchip RK1828 M.2 AI card** (5 GB on-card memory, 8 NPU cores)
and expose it through a self-built **OpenAI-compatible C++ server** that runs as a supervised service.

The whole path was measured on real hardware (OrangePi5Max + RK1828, firmware 1.2.0):
model conversion → on-card memory sizing → build → deployed service (boot start + health watchdog) →
accuracy, repetition, long-context and stress tests. The scripts in `tools/` reproduce every number.

Terms used below:

- **token**: the unit models read and write. Roughly one Chinese character or four English characters.
- **prefill / decode**: prefill reads the whole prompt and fills the KV cache; decode generates one token
  at a time. Long conversations are usually dominated by prefill.
- **KV cache**: the model's memory of the previous tokens. With it, a new turn does not recompute the history.
- **slot**: a snapshot of one conversation's KV, stored in host RAM. The card holds one at a time.

The Chinese README ([`README_CN.md`](README_CN.md)) is the primary version.

## Why not the official `rkllm3-server`

RKNN3 1.2.0's `rkllm3-server` is broken for `model_type=spark2_5`. Six configuration variants were tried
(custom chat template included, verified token by token); the output is corrupt from the first token —
0 correct out of 48 and 39 degenerate. The same model file works through the model-zoo C++ demo
(`rknn3_session_set_chat_template` + `rknn3_session_run`). This server turns that working path into a
service. Reproduction: [`docs/troubleshooting.md`](docs/troubleshooting.md), item 4.

## Measured results

| Item | Value |
|---|---|
| Context | **73,728-token build works**. One prefill of 53,166 tokens took 115.6 s and the needle was recalled; 38,006 tokens took 72.0 s. Hard ceiling for int4 KV on this card ≈ 79,900 tokens |
| On-card memory | 4903 / 5120 MB (2.37 GiB weights + KV + internal buffers) |
| Decode | **50–52 tok/s** at short context (19–20 ms/token); 42.6 tok/s at 14.8k; 35/32 tok/s at 38k/53k |
| Prefill | 647 tok/s (88 tokens) → 788 (4.3k) → 685 (14.8k) → ~460 (38k) |
| Accuracy (392 held-out items, greedy + thinking) | 99 / 332 scoreable = **29.8 %** (measured before the system-prompt fix; see `docs/benchmarks.md` §2 / §6.2). GRQ w4a16 vs bf16 top-1 agreement **96.16 %**, held-out PPL 1.2451 vs 1.2536 |
| Repetition | **0 genuine degeneration** in 392 items; 2 of 48 adversarial prompts run away (both literally ask for repetition) |
| Tool calling | `finish_reason=tool_calls`, parseable `arguments`, no fabricated calls when `tools` is absent (4/8 on the official function-calling examples). Streaming deltas follow the OpenAI spec; `tool_choice=required` is enforced |
| Multi-turn KV reuse | 4 slots kept in RAM. The second turn of a 1214-token conversation prefills **18 tokens** instead of 1214, with byte-identical greedy output versus slots off |
| Stress / compatibility | 31-item suite: 1 failure (model behaviour, not a server defect); 24 concurrent requests all succeed; 120-request soak with zero errors |
| Thermals | Card 29–56 °C under continuous eval + stress (passive cooling); SoC zones ≤ 33 °C; no throttling |

Details and methodology: [`docs/benchmarks.md`](docs/benchmarks.md).

## Prerequisites

- **Hardware**: an aarch64 Linux board with an M.2 Key-M slot (this project uses OrangePi5Max / RK3588)
  and an RK1828 M.2 card (**12 V external power is required**). A PC for conversion
  (**x86_64 Linux or macOS arm64**; GRQ quantization needs an NVIDIA GPU).
- **Software**: Rockchip RKNN3 SDK **1.2.0** (firmware + runtime + toolkit, downloaded separately under
  the Rockchip RKNN SDK License), CMake ≥ 3.16, GCC ≥ 9, Python 3.12 on the conversion machine.
- Firmware, runtime and toolkit must be the same generation, or the model will not load
  (see [`docs/build.md`](docs/build.md), section 4).

## Quick start

```bash
git clone https://github.com/yuunnn-w/RK1828-Spark-X2.6-4B.git
cd RK1828-Spark-X2.6-4B

# 1) Convert (on the PC, not on the board)
model/download_model.sh
convert/run_conversion.sh --model-dir model/Spark-X2.6-4B --ctx 73728 --out model/out

# 2) Build the server (on the board)
scripts/get_sdk.sh && scripts/build_server.sh

# 3) Install as a boot-starting service
#    put .rknn/.weight/.safetensors + .tokenizer.gguf/.embed.bin into /opt/rk1828-spark/model/
sudo PREFIX=/opt/rk1828-spark deploy/install_service.sh
curl -s localhost:8080/health
```

Clients:

```bash
BASE=http://<board-ip>:8080 examples/curl.sh
python3 examples/client.py --base http://<board-ip>:8080   # tools + 40k-token needle test
```

Service parameters live in `/etc/rk1828-spark.env`; `systemctl restart rk1828-spark.service` applies them.

KV slots need a RAM disk. `deploy/start.sh` checks `KV_SLOT_DIR`: if it is not a tmpfs, it mounts one of
`KV_SLOT_TMPFS_SIZE` (default 11G). If the mount fails, images go to disk — still correct, but much slower.

## Repository layout

```
server/          the server (C++11): engine, httpd, main, kvslots
convert/         conversion pipeline: export_rknn.py, run_conversion.sh, calibration builder, GRQ check
deploy/          systemd units, start wrapper, health watchdog, installer, env sample
tools/           eval, stress, repetition, long-context and protocol checks
datasets/        GRQ calibration set (884) + held-out eval set (392) + provenance
docs/            build / conversion / server / chat_template / benchmarks / testing / troubleshooting
scripts/         get_sdk.sh, build_server.sh
examples/        curl smoke script and a stdlib-only Python client
model/           download_model.sh (weights are not in git)
```

## Defaults for long-context agents and coding

Sampling defaults come from the **model vendor's** `generation_config.json`:
`temperature 1.0`, `top_p 0.95`, `top_k` unbounded, all penalties off. Output length is unbounded by
default. **No repetition penalty is applied on purpose** — it damages legitimate repeated structures in
code. Repetition is handled by a three-layer circuit breaker instead (periodic / n-gram / long block).

```bash
# /etc/rk1828-spark.env
TEMP=1.0 TOP_K=0 TOP_P=0.95 REPEAT_PENALTY=1.0 MAX_NEW=0
EXTRA_ARGS="--reasoning-format deepseek"      # separate reasoning from the answer
KV_SLOTS=4 KV_SLOT_DIR=/run/rk1828-kv KV_SLOT_TMPFS_SIZE=11G
```

Three measured rules of thumb:

- For accuracy-critical calls, **pass `temperature: 0`** — greedy beat the vendor sampling defaults by
  about 8 points on the same items.
- For "output code only" tasks, **turn thinking off** (`reasoning_effort: "none"` or `thinking: false`).
  With thinking on the model sometimes restates the task in the answer channel and stops without code.
- **Do not turn `--system-in-prompt` off**: RKNN3's `set_chat_template()` ignores `system_prompt`, so the
  old path silently drops client system messages (`tools/sysprobe.py`: 1/6 before the fix, 6/6 after).
  In agent workloads the system turn carries instructions and tool specs — losing it fails silently.

## Known limitations

- **Single session, requests are serialized**: the model takes 4.9 GB of the card, so concurrency is
  HTTP-level plus queueing (429 + `Retry-After` beyond `--max-queue`). The 4 KV slots avoid re-prefilling
  when switching conversations; they do not make inference parallel.
- **KV slots require the client to replay** the previous assistant `content` and `reasoning_content`
  verbatim. Any difference (dropped reasoning, re-serialized tool arguments) falls back to a cold start:
  still correct, just slower. See `docs/server.md` §3.2.
- **`n>1` is rejected** (400, never silently one result). **Implemented**: `/v1/responses`
  (streaming + non-streaming), `/v1/embeddings` (mean-pooled token embeddings, approximate).
  **Not implemented**: `beam_search`, image/audio input, `audio`/`modalities`, async overlap sampling.
- Context length is fixed at conversion time; longer requests are clamped, not extended.
- Two model-side behaviours, not server defects: with thinking on and a short budget the model may restate
  the task and stop; on some multiple-choice questions it may emit repeated letters (`AAAAA`).
- 29.8 % accuracy is what this 4B model does on hard math/reasoning items. Weight int4 was shown to be
  near-lossless, so the gap is model capability, not deployment loss.

## License

- Original code in this repository: **GPL-3.0** (see `LICENSE`).
- Bundled third-party components: see **`NOTICE`** (nlohmann/json MIT; model-zoo utils/tokenizer/float16
  Apache-2.0).
- **Not redistributed here**: Rockchip RKNN3 SDK, firmware, drivers, runtime libraries — download them
  yourself and accept the Rockchip RKNN SDK License.
- Model weights `XHToken/Spark-X2.6-4B`: **Apache-2.0** (SparkLLM Team).

## References

- [airockchip/rknn3-toolkit](https://github.com/airockchip/rknn3-toolkit) (V1.2.0),
  [airockchip/rknn3-model-zoo](https://github.com/airockchip/rknn3-model-zoo) (`examples/Spark_X2_5`)
- [XHToken/Spark-X2.6-4B @ ModelScope](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B)

```bibtex
@misc{sparkx2.6,
  title  = {Spark-X2.6-4B&1.7B: Advancing the Frontier of On-Device Agents for Real-World Tasks},
  author = {SparkLLM Team},
  year   = {2026}
}
```
