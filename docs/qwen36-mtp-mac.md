# Qwen3.6-27B MTP on Apple Silicon Mac

This document captures the exact build, tuning, and launch steps for running `unsloth/Qwen3.6-27B-MTP-GGUF` on an Apple Silicon Mac with Metal support.

## What was validated

- Source build of `llama.cpp` in Release mode.
- Metal support enabled.
- Native CPU tuning enabled.
- Local model files already present:
  - `~/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q2_K_XL.gguf`
  - `~/models/unsloth/Qwen3.6-27B-MTP-GGUF/mmproj-F16.gguf`
- An MTP sweep found `--spec-draft-n-max 2` to be the fastest setting on one Apple Silicon Mac; run the sweep below to confirm the right value for your chip.
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
- multimodal projector: `mmproj-F16.gguf`
- MTP mode: `--spec-type draft-mtp`
- draft tokens: `--spec-draft-n-max 2`
- context: `--ctx-size 8192`
- GPU layers: `--n-gpu-layers 999`
- threads: `--threads <N>` — auto-detected from your chip's performance core count
- thinking disabled: `--chat-template-kwargs '{"enable_thinking":false}'`

These defaults were chosen after a measured sweep on Apple Silicon. The optimal `--spec-draft-n-max` value is hardware-dependent — run the sweep below on your own machine to confirm the best value.

## Run the server

Use the helper script from this repo:

```bash
./scripts/run-qwen36-mtp.sh
```

The script starts `llama-server` with the tuned configuration and runs it in the foreground. Stop it with `Ctrl+C`.

Any setting can be overridden at launch time via environment variables:

```bash
LLAMA_SPEC_DRAFT_N_MAX=3 LLAMA_CTX_SIZE=4096 ./scripts/run-qwen36-mtp.sh
```

Available overrides: `LLAMA_SERVER_BIN`, `LLAMA_MODEL`, `LLAMA_MMPROJ`, `LLAMA_PORT`, `LLAMA_THREADS`, `LLAMA_CTX_SIZE`, `LLAMA_N_GPU_LAYERS`, `LLAMA_SPEC_DRAFT_N_MAX`, `LLAMA_HOST`.

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

The best `--spec-draft-n-max` value varies by chip. Run a quick sweep with your local build:

```bash
MODEL="$HOME/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q2_K_XL.gguf"
MMPROJ="$HOME/models/unsloth/Qwen3.6-27B-MTP-GGUF/mmproj-F16.gguf"

for n in 1 2 3 4 5 6; do
  OUT=$(./build/bin/llama-cli \
    --model "$MODEL" --mmproj "$MMPROJ" \
    --n-gpu-layers 999 --threads "$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 8)" --ctx-size 8192 \
    --temp 0 --top-k 1 --top-p 1 --min-p 0 \
    --single-turn \
    --prompt "Explain speculative decoding in three short sentences." \
    --spec-type draft-mtp --spec-draft-n-max "$n" \
    --n-predict 128 2>&1)
  TPS=$(echo "$OUT" | grep "Generation:" | awk '{print $2}')
  echo "n=$n  ${TPS:-NA} t/s"
done
```

Use whichever `n` gives the highest `t/s` value and set `LLAMA_SPEC_DRAFT_N_MAX=<n>` before running the launcher script.

If you run into memory pressure, the first knob to reduce is `--ctx-size`.

## Troubleshooting

- If the model files are missing, download them again into `~/models/unsloth/Qwen3.6-27B-MTP-GGUF/`.
- If `build/bin/llama-server` is missing, rerun the build commands above.
- If the server starts but is slow, keep `--spec-draft-n-max 2` and lower `--ctx-size` to `4096` or `6144`.
