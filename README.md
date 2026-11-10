# RK1828 + Spark-X2.6-4B: 64K+ context LLM serving on a single Rockchip M.2 accelerator

Deploy iFlytek **Spark-X2.6-4B** (Apache-2.0, hybrid attention, 1M nominal context) on a
**Rockchip RK1828 M.2 AI co-processor** (5 GB on-card memory, 8 NPU cores) and serve it through a
self-developed **OpenAI-compatible C++ inference server** that runs as a supervised system service.

The whole path is measured end to end on real hardware (OrangePi5Max + RK1828 card, firmware 1.2.0):
**model conversion → on-card memory sizing → build → deployed service (auto-start + health watchdog) →
accuracy / repetition / long-context / stress testing**. Every number below is reproducible with the
scripts in `tools/`.

English README; the Chinese one ([`README_CN.md`](README_CN.md)) is the primary version.

> **Why not the official `rkllm3-server`?** Rockchip's RKNN3 1.2.0 `rkllm3-server` is broken for
> `model_type=spark2_5`: after exhausting six configuration variants (including a custom chat template whose
> effect was verified token-by-token) the output is corrupt from the very first token — 0 correct answers out
> of 48, 39 of them degenerate. The same model file works perfectly through the C++ demo path
> (`rknn3_session_set_chat_template` + `rknn3_session_run`). This server implements that verified path, made
> resident and hardened. Evidence: [`docs/troubleshooting.md`](docs/troubleshooting.md), item 4.

---

## Measured results

| Item | Value |
|---|---|
| Context | **73,728-token build works** (single prefill of 53,166 tokens in 115.6 s with needle recall; 38,006 tokens in 72.0 s); hard ceiling for int4 KV on this card ≈ 79,900 tokens |
| On-card memory | 4903 / 5120 MB (2.37 GiB weights + KV + internal buffers); `rknn3_model_info` per-core total 4853.74 MB |
| Decode | **50–52 tok/s** short context (19–20 ms/token); 42.6 tok/s at 14.8k; 35/32 tok/s at 38k/53k |
| Prefill | 647 tok/s (88 tok, 137 ms) → 788 (4.3k) → 685 (14.8k) → ~460 (38k) |
| Accuracy (392 held-out items, greedy + thinking) | 99 / 332 scoreable = **29.8 %** (measured before the system-prompt-delivery fix, see `docs/benchmarks.md` §2 / §6.2); GRQ w4a16 vs bf16 teacher-forced top-1 agreement **96.16 %**, held-out PPL 1.2451 vs 1.2536 (bf16) |
| Repetition | **0 genuine degeneration** in 392 items; 2 of 48 adversarial cases run away (both are prompts that literally demand repetition) |
| Tool calling | Working: `finish_reason=tool_calls`, parseable `arguments`, no fabricated calls when `tools` is absent (4/8 on the official function-calling examples) |
| Stress/compat | 31-item suite: **1 failure** (a model behaviour, not a server defect); 24 concurrent requests all succeed; 120-request soak with zero errors |
| Thermals | Card 29–56 °C during continuous eval + stress (passive cooling); SoC zones ≤ 33 °C; no throttling events |

Details and methodology: [`docs/benchmarks.md`](docs/benchmarks.md).

---

## Prerequisites

- **Hardware**: an aarch64 Linux board with an M.2 Key-M slot (this project: OrangePi5Max / RK3588) plus an
  RK1828 M.2 card (**12 V external power is mandatory**), and a PC for conversion
  (**x86_64 Linux or macOS arm64**; GRQ quantization needs an NVIDIA GPU).
- **Software**: Rockchip RKNN3 SDK **1.2.0** (firmware + runtime + toolkit — downloaded separately under the
  Rockchip RKNN SDK License), CMake ≥ 3.16, GCC ≥ 9, Python 3.12 on the conversion machine.
  Versions must match generation (see [`docs/build.md`](docs/build.md), section 4).

## Quick start

```bash
git clone https://github.com/yuunnn-w/RK1828-Spark-X2.6-4B.git
cd RK1828-Spark-X2.6-4B

# 1) Convert (on the PC, not on the board)
model/download_model.sh
convert/run_conversion.sh --model-dir model/Spark-X2.6-4B --ctx 73728 --out model/out

# 2) Build the server (on the board)
scripts/get_sdk.sh && scripts/build_server.sh

# 3) Deploy as a supervised, boot-starting service
#    put .rknn/.weight/.safetensors + .tokenizer.gguf/.embed.bin into /opt/rk1828-spark/model/
sudo PREFIX=/opt/rk1828-spark deploy/install_service.sh
curl -s localhost:8080/health
```

Clients:

```bash
BASE=http://<board-ip>:8080 examples/curl.sh
python3 examples/client.py --base http://<board-ip>:8080   # tools + 40k-token needle test
```

All service parameters live in `/etc/rk1828-spark.env`; `systemctl restart rk1828-spark.service` applies them.

## Repository layout

```
server/          the OpenAI-compatible server (C++11: engine / httpd / main, CMake)
convert/         conversion pipeline: export_rknn.py, run_conversion.sh, calibration builder, GRQ quality check
deploy/          systemd unit templates, start wrapper, health watchdog, installer, env sample
tools/           eval, stress, repetition probes, long-context probe, thermal sampler
datasets/        GRQ calibration set (884) + held-out eval set (392) + provenance list
docs/            build / conversion / server / benchmarks / testing / troubleshooting
scripts/         get_sdk.sh, build_server.sh
examples/        curl smoke script and a stdlib-only Python client
model/           download_model.sh (weights are not in git)
```

## Defaults tuned for long-context agents and coding

Sampling defaults come from the **model vendor's** `generation_config.json`
(`temperature 1.0 / top_p 0.95 / top_k unbounded / all penalties off`), output length is **unbounded by default**,
and **no repetition penalty is applied on purpose** — penalties damage legitimate repeated structures in code.
Repetition is instead handled by a three-layer circuit breaker (periodic / n-gram / long-block), tunable or
switchable off with `--fuse-off`.

Three measured rules of thumb:

- For accuracy-critical calls, **pass `temperature: 0`** — greedy beat the vendor sampling defaults by ~8 points.
- For "output code only" tasks, prefer **`reasoning_effort: "none"`** (or `thinking: false`) — with thinking on the
  model sometimes restates the task in the answer channel and stops without producing any code.
- **Do not turn `--system-in-prompt` off**: RKNN3's `set_chat_template()` ignores `system_prompt`, so the old path
  silently drops client system messages (`tools/sysprobe.py` reproduces it: 1/6 before the fix, 6/6 after).
  In agent workloads the system turn carries instructions and tool specs — losing it fails silently.

## Known limitations (honest list)

- **Single session, requests are serialized**: the model occupies 4.9 GB of the card, so concurrency is HTTP-level
  plus queueing (429 + `Retry-After` beyond `--max-queue`).
- **Explicitly rejected**: `n>1` (400, never silently one result). **Not implemented**: `beam_search`,
  multimodal input, `/v1/embeddings`, audio, async overlap sampling.
- Context length is fixed at conversion time; the runtime clamps longer requests instead of failing.
- Open model-side behaviours (verbose answer channel with thinking on; repeated letters on some multiple-choice
  answers) are documented rather than hidden.
- 29.8 % accuracy reflects what this 4B model can do on hard math/reasoning items — weight int4 quantization was
  shown to be near-lossless, so the gap is model capability, not deployment loss.

## License

- Original code in this repository: **GPL-3.0** (see `LICENSE`).
- Bundled third-party components and their licenses: see **`NOTICE`** (nlohmann/json MIT; model-zoo
  utils/tokenizer/float16 Apache-2.0).
- **Not redistributed here**: Rockchip RKNN3 SDK, firmware, drivers, runtime libraries — download them yourself
  and accept the Rockchip RKNN SDK License.
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
