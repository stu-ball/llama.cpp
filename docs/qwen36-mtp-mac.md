# Qwen3.6-27B MTP on Apple Silicon Mac

This document captures the exact build, tuning, and launch steps for running `unsloth/Qwen3.6-27B-MTP-GGUF` on an Apple Silicon Mac with Metal support.

## What was validated

- Source build of `llama.cpp` in Release mode.
- Metal support enabled.
- Native CPU tuning enabled.
- Local model files already present:
  - `~/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q2_K_XL.gguf`
  - `~/models/unsloth/Qwen3.6-27B-MTP-GGUF/mmproj-F16.gguf` (vision; not needed for text-only MTP inference)
- Optimization sweep with `llama-bench` identified `--flash-attn --cache-type-v q8_0` as the fastest config.
- Confirmed decode speed: **10.45 t/s** (vs 9.60 t/s baseline = **+8.9%**).
- MTP draft acceptance rate is very high (~97 %) but generation is currently slower with MTP enabled on Apple Silicon (see [MTP note](#mtp-on-apple-silicon) below).
- `llama-server` API health and chat completion were both verified successfully.

## One-time build from source

From the repository root:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_METAL=ON \
  -DGGML_NATIVE=ON \
  -DGGML_ACCELERATE=ON

cmake --build build --config Release -j $(sysctl -n hw.logicalcpu) \
  --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
```

The build should produce these binaries:

- `build/bin/llama-cli`
- `build/bin/llama-mtmd-cli`
- `build/bin/llama-server`
- `build/bin/llama-gguf-split`

## Download the model

If you don't have the model files yet, fetch them with `huggingface-cli`:

```bash
pip install huggingface_hub   # if not already installed
huggingface-cli download unsloth/Qwen3.6-27B-MTP-GGUF \
  Qwen3.6-27B-UD-Q2_K_XL.gguf \
  mmproj-F16.gguf \
  --local-dir ~/models/unsloth/Qwen3.6-27B-MTP-GGUF
```

## Recommended runtime settings

Use these settings as a default starting point:

- model: `Qwen3.6-27B-UD-Q2_K_XL.gguf`
- multimodal projector: not loaded (the `mmproj-F16.gguf` is a CLIP vision projector; loading it suppresses speculative decoding entirely)
- MTP: disabled by default — enable with `LLAMA_MTP=1` (see [MTP note](#mtp-on-apple-silicon))
- flash attention: `--flash-attn` (**enabled** — free at short context, measurably better at long context)
- KV cache: `--cache-type-k f16 --cache-type-v q8_0` (**+8.9% decode speed**, no perceptible quality loss)
- context: `--ctx-size 8192`
- GPU layers: `--n-gpu-layers 999`
- threads: `--threads <N>` — auto-detected from perf cores (irrelevant at -ngl 999; kept for CPU fallback)
- thinking disabled: `--reasoning off`

> **Temperature guidance (per Unsloth's official docs)**:
> - Non-thinking / instruct mode (`--reasoning off`): use `--temp 0.7` for general tasks.
> - Thinking mode (`--reasoning on`): use `--temp 1.0` for general tasks, `--temp 0.6` for precise coding.
> The script defaults to `--temp 0.7` (non-thinking mode).

### Optimization benchmark (M4 Pro, Q2_K_XL, llama-bench tg200, pp32)

| Config | tg200 t/s | vs baseline |
|--------|----------:|-----------:|
| Baseline (f16 KV, no flash attn) | 9.60 | — |
| `--flash-attn` only | 9.68 | +0.8% |
| `--cache-type-v q8_0` + `-fa` | **10.45** | **+8.9%** |
| `--cache-type-v q4_0` + `-fa` | 10.20 | +6.3% |
| `--cache-type-k q8_0 --cache-type-v q8_0` + `-fa` | 9.55 | −0.5% |
| MTP n=2 (draft acceptance 97.8 %) | 9.38 | −2.3% |

Key finding: **V-cache quantization helps, K-cache quantization hurts.** Symmetric KV quantization (q8_0/q8_0) is no better than baseline because K-cache dequant overhead cancels the V-cache bandwidth saving. Flash attention is largely free at short context but beneficial at 2048+ tokens.

Thread count (1–8) has no impact on decode speed when `-ngl 999` offloads all layers to Metal.

## MTP on Apple Silicon

The MTP heads are embedded in the main GGUF (`qwen35.nextn_predict_layers: 1`). When enabled, llama.cpp initialises a separate draft context and verifies speculative tokens in a batched pass.

Benchmark results on M4 Pro (24 GB unified memory, Q2_K_XL, single slot, `--reasoning off`):

| Mode | Speed |
|------|-------|
| No MTP (baseline) | ~11 t/s |
| MTP n=2 (97.8 % acceptance) | ~9.4 t/s |

Despite near-perfect draft acceptance, MTP is currently **slower** on Apple Silicon. The Metal backend evaluates the MTP head with overhead comparable to a main-model forward pass, so the memory-bandwidth bottleneck is hit twice. NVIDIA CUDA GPUs handle batched speculative verification more efficiently, which is where Unsloth's 1.4–2.2× speedup figures come from.

There is also a **memory cost**: the MTP head weights are loaded into unified memory even when MTP is disabled, consuming roughly 1 GB that could otherwise go to a larger KV cache or context window. A standard non-MTP GGUF of equivalent quality and quant would therefore be preferable on Apple Silicon — faster, and ~1 GB lighter.

If you want to stay with this GGUF (e.g. you already have it downloaded, or you plan to move to a CUDA machine later), keep `LLAMA_MTP=0` and retest with `LLAMA_MTP=1` on future llama.cpp builds as Metal MTP support improves.

## Run the server

Use the helper script from this repo:

```bash
./scripts/run-qwen36-mtp.sh
```

The script starts `llama-server` with the tuned configuration and runs it in the foreground. Stop it with `Ctrl+C`.

Any setting can be overridden at launch time via environment variables:

```bash
LLAMA_MTP=1 LLAMA_SPEC_DRAFT_N_MAX=3 LLAMA_CTX_SIZE=4096 ./scripts/run-qwen36-mtp.sh
```

Available overrides: `LLAMA_SERVER_BIN`, `LLAMA_MODEL`, `LLAMA_MMPROJ`, `LLAMA_PORT`, `LLAMA_THREADS`, `LLAMA_CTX_SIZE`, `LLAMA_N_GPU_LAYERS`, `LLAMA_SPEC_DRAFT_N_MAX`, `LLAMA_TEMP`, `LLAMA_HOST`, `LLAMA_MTP`.

To enable thinking mode, set `LLAMA_TEMP=1.0` in your `.env` and pass `--reasoning on` manually (the script hard-codes `--reasoning off` for the non-thinking default).

Alternatively, create a `.env` file in the repo root and the script will source it automatically:

```bash
# .env  (do not commit this file)
LLAMA_SPEC_DRAFT_N_MAX=3
LLAMA_CTX_SIZE=4096
```

## Check that the server is ready

In another terminal:

```bash
curl -s http://127.0.0.1:8001/health
```

You should see a healthy response.

## Test the OpenAI-compatible API

```bash
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "unsloth/Qwen3.6-27B-MTP-GGUF",
    "messages": [
      {"role": "user", "content": "Reply with exactly: SERVER_OK"}
    ]
  }'
```

If everything is wired correctly, the reply content should be `SERVER_OK`.

## Tuning for your machine

If you want to test whether future llama.cpp builds improve MTP on Apple Silicon, run this sweep to find the best `--spec-draft-n-max` value. Do **not** pass `--mmproj` — loading the vision projector suppresses speculative decoding entirely.

```bash
MODEL="$HOME/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q2_K_XL.gguf"
THREADS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 8)

# Baseline: no MTP
echo "Measuring no-MTP baseline..."
./build/bin/llama-cli \
  --model "$MODEL" \
  --n-gpu-layers 999 --threads "$THREADS" --ctx-size 8192 \
  --reasoning off --temp 0.7 --top-k 1 --top-p 1 --min-p 0 \
  --single-turn \
  --prompt "Count from 1 to 128, one number per line." \
  --n-predict 128 2>&1 | grep -E "eval time|Generation"

# MTP sweep
for n in 1 2 3 4; do
  echo "Measuring MTP n=$n..."
  OUT=$(./build/bin/llama-cli \
    --model "$MODEL" \
    --n-gpu-layers 999 --threads "$THREADS" --ctx-size 8192 \
    --reasoning off --temp 0.7 --top-k 1 --top-p 1 --min-p 0 \
    --single-turn \
    --prompt "Count from 1 to 128, one number per line." \
    --spec-type draft-mtp --spec-draft-n-max "$n" \
    --n-predict 128 2>&1)
  TPS=$(echo "$OUT" | grep -E "eval time" | tail -1)
  echo "  n=$n  $TPS"
done
```

If any MTP `n` value beats the baseline, set `LLAMA_MTP=1 LLAMA_SPEC_DRAFT_N_MAX=<n>` in your `.env` before running the launcher script.

If you run into memory pressure, the first knob to reduce is `--ctx-size`.

## Troubleshooting

- If the model files are missing, download them again into `~/models/unsloth/Qwen3.6-27B-MTP-GGUF/`.
- If `build/bin/llama-server` is missing, rerun the build commands above.
- If the server starts but is slow, make sure MTP is disabled (default) and lower `--ctx-size` to `4096` or `6144` if you're under memory pressure.
- If you see `W Setting 'enable_thinking' via --chat-template-kwargs is deprecated`, the launch script has an old flag; the current script uses `--reasoning off` instead.
- If `"speculative.types": "none"` appears in `/v1/slots` despite MTP being enabled, check that `LLAMA_MMPROJ` is not set — loading the vision projector suppresses speculative decoding entirely.
- **MTP is currently slower on Apple Silicon** (~9.4 t/s vs ~11 t/s baseline on M4 Pro). This is a known limitation of the Metal backend's speculative verification path. Disable with `LLAMA_MTP=0` (the default) or omit the MTP flags.
