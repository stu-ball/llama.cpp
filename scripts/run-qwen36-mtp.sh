#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source a local .env file if present — useful for persisting tuned values
# Example: echo 'LLAMA_SPEC_DRAFT_N_MAX=3' >> .env
if [[ -f "$ROOT_DIR/.env" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.env"
fi

BIN="${LLAMA_SERVER_BIN:-$ROOT_DIR/build/bin/llama-server}"
MODEL="${LLAMA_MODEL:-$HOME/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q2_K_XL.gguf}"
MMPROJ="${LLAMA_MMPROJ:-}"  # empty by default; set to enable vision/multimodal (disables MTP)
PORT="${LLAMA_PORT:-8001}"
_PERF_CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || echo 8)
THREADS="${LLAMA_THREADS:-$_PERF_CORES}"
CTX_SIZE="${LLAMA_CTX_SIZE:-8192}"
N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-999}"
SPEC_DRAFT_N_MAX="${LLAMA_SPEC_DRAFT_N_MAX:-2}"
# MTP is disabled by default on Apple Silicon — see docs/qwen36-mtp-mac.md for context
# Set LLAMA_MTP=1 to enable it if you want to test on a future llama.cpp build
MTP="${LLAMA_MTP:-0}"
TEMP="${LLAMA_TEMP:-0.7}"
HOST="${LLAMA_HOST:-127.0.0.1}"
ALIAS="${LLAMA_ALIAS:-unsloth/Qwen3.6-27B-MTP-GGUF}"

if [[ ! -x "$BIN" ]]; then
  echo "llama-server not found or not executable: $BIN" >&2
  echo "Build it first with: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_NATIVE=ON -DGGML_ACCELERATE=ON" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model file not found: $MODEL" >&2
  exit 1
fi

if [[ -n "$MMPROJ" && ! -f "$MMPROJ" ]]; then
  echo "mmproj file not found: $MMPROJ" >&2
  exit 1
fi

echo "Starting tuned Qwen3.6-27B MTP server"
echo "  binary: $BIN"
echo "  model:  $MODEL"
echo "  mmproj: ${MMPROJ:-disabled}"
echo "  MTP:    ${MTP:-0} (set LLAMA_MTP=1 to enable; currently slower on Apple Silicon)"
echo "  host:   $HOST"
echo "  port:   $PORT"
echo "  ctx:    $CTX_SIZE"
echo "  threads:$THREADS"
echo "  draft:  $SPEC_DRAFT_N_MAX (MTP enabled: $MTP)"
echo "  temp:   $TEMP"
echo

exec "$BIN" \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  ${MMPROJ:+--mmproj "$MMPROJ"} \
  --alias "$ALIAS" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  --threads "$THREADS" \
  --ctx-size "$CTX_SIZE" \
  --flash-attn \
  --cache-type-k f16 \
  --cache-type-v q8_0 \
  ${MTP:+--spec-type draft-mtp} \
  ${MTP:+--spec-draft-n-max "$SPEC_DRAFT_N_MAX"} \
  --reasoning off \
  --temp "$TEMP" \
  --top-p 0.8 \
  --top-k 20 \
  --presence-penalty 1.5 \
  --min-p 0.0