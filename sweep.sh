#!/bin/zsh
MODEL="$HOME/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q2_K_XL.gguf"
MMPROJ="$HOME/models/unsloth/Qwen3.6-27B-MTP-GGUF/mmproj-F16.gguf"
PROMPT="Explain speculative decoding in exactly three short sentences."

echo "| n | tok/s |"
echo "|---|-------|"

for n in 1 2 3 4 5 6; do
    # Warmup
    ./build/bin/llama-cli --model "$MODEL" --mmproj "$MMPROJ" --n-gpu-layers 999 --threads 8 --ctx-size 8192 --temp 0 --top-k 1 --top-p 1 --min-p 0 --single-turn --prompt "$PROMPT" --spec-type draft-mtp --spec-draft-n-max $n --n-predict 48 > /dev/null 2>&1
    
    # Measured run
    OUTPUT=$(./build/bin/llama-cli --model "$MODEL" --mmproj "$MMPROJ" --n-gpu-layers 999 --threads 8 --ctx-size 8192 --temp 0 --top-k 1 --top-p 1 --min-p 0 --single-turn --prompt "$PROMPT" --spec-type draft-mtp --spec-draft-n-max $n --n-predict 128 2>&1)
    
    TPS=$(echo "$OUTPUT" | grep "Generation:" | awk '{print $2}')
    if [[ -z "$TPS" ]]; then
        echo "| $n | FAILED |"
    else
        echo "| $n | $TPS |"
    fi
done
