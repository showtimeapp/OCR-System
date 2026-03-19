#!/bin/bash
# ═══════════════════════════════════════════════════
#  Watchdog — Auto-restart dead servers
#  Usage: conda activate ocr && cd ~/pipeline && bash watchdog.sh
#  Checks every 30s, restarts anything that died
# ═══════════════════════════════════════════════════

export LD_LIBRARY_PATH=$HOME/miniconda3/envs/ocr/lib:$LD_LIBRARY_PATH
cd ~/pipeline

echo "═══ Watchdog started ═══"
echo "Monitoring: GLM-OCR (8090) | Qwen (8091) | API (80)"

while true; do
    # Check GLM-OCR
    if ! curl -s http://localhost:8090/health > /dev/null 2>&1; then
        echo "$(date '+%H:%M:%S') | GLM-OCR down — restarting..."
        pkill -f "vllm.*8090" 2>/dev/null; sleep 2
        FLASHINFER_DISABLE_VERSION_CHECK=1 vllm serve zai-org/GLM-OCR \
          --port 8090 --dtype float16 --gpu-memory-utilization 0.25 --max-model-len 8192 \
          --served-model-name glm-ocr --allowed-local-media-path / > logs/glm_vllm.log 2>&1 &
        echo "  Waiting for GLM-OCR to load..."
        sleep 120
    fi

    # Check Qwen
    if ! curl -s http://localhost:8091/health > /dev/null 2>&1; then
        echo "$(date '+%H:%M:%S') | Qwen down — restarting..."
        pkill -f "vllm.*8091" 2>/dev/null; sleep 2
        FLASHINFER_DISABLE_VERSION_CHECK=1 vllm serve cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit \
          --port 8091 --dtype float16 --gpu-memory-utilization 0.45 --max-model-len 4096 \
          --served-model-name qwen-vl > logs/qwen_vllm.log 2>&1 &
        echo "  Waiting for Qwen to load..."
        sleep 120
    fi

    # Check API
    if ! curl -s http://localhost:80/health > /dev/null 2>&1; then
        echo "$(date '+%H:%M:%S') | API down — restarting..."
        sudo pkill -f api.py 2>/dev/null; sleep 2
        sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH $(which python3) ~/pipeline/api.py > logs/api.log 2>&1 &
        sleep 15
    fi

    sleep 30
done