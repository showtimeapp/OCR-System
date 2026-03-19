#!/bin/bash
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/ocr/lib:$LD_LIBRARY_PATH
mkdir -p logs uploads results charts

echo "═══ Starting Pipeline ═══"

pkill -f "vllm serve" 2>/dev/null
sudo pkill -f api.py 2>/dev/null
sleep 3

echo "Starting GLM-OCR vLLM (port 8090)..."
FLASHINFER_DISABLE_VERSION_CHECK=1 vllm serve zai-org/GLM-OCR \
  --port 8090 --dtype float16 --gpu-memory-utilization 0.25 --max-model-len 8192 \
  --served-model-name glm-ocr --allowed-local-media-path / > logs/glm_vllm.log 2>&1 &

echo "Starting Qwen vLLM (port 8091)..."
FLASHINFER_DISABLE_VERSION_CHECK=1 vllm serve cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit \
  --port 8091 --dtype float16 --gpu-memory-utilization 0.45 --max-model-len 4096 \
  --served-model-name qwen-vl > logs/qwen_vllm.log 2>&1 &

echo "Waiting for vLLM servers (~2 min)..."
for i in $(seq 1 90); do
  G=$(curl -s http://localhost:8090/health 2>/dev/null)
  Q=$(curl -s http://localhost:8091/health 2>/dev/null)
  if [ -n "$G" ] && [ -n "$Q" ]; then
    echo "  ✓ Both vLLM servers ready!"
    break
  fi
  sleep 2
done

echo "Starting API on port 80..."
sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH $(which python3) api.py &
sleep 15

if curl -s http://localhost:80/health > /dev/null 2>&1; then
  EXT_IP=$(curl -s ifconfig.me 2>/dev/null || echo "UNKNOWN")
  echo ""
  echo "═══════════════════════════════════════"
  echo "  ✓ Pipeline running!"
  echo "  API:  http://$EXT_IP/docs"
  echo "  GLM:  http://localhost:8090"
  echo "  Qwen: http://localhost:8091"
  echo "═══════════════════════════════════════"
else
  echo "  ✗ API failed. Check logs/api.log"
  tail -10 logs/api.log
fi