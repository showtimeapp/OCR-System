#!/bin/bash
source /opt/pipeline/venv/bin/activate
mkdir -p /opt/pipeline/logs

echo "Starting GLM-OCR (port 8001)..."
vllm serve zai-org/GLM-OCR \
  --port 8001 \
  --dtype float16 \
  --gpu-memory-utilization 0.15 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --served-model-name glm-ocr \
  > /opt/pipeline/logs/glm_ocr.log 2>&1 &

echo "Starting Qwen3-VL (port 8002)..."
vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --port 8002 \
  --dtype float16 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --served-model-name qwen-vl \
  > /opt/pipeline/logs/qwen_vl.log 2>&1 &

echo "Waiting for servers..."
for port in 8001 8002; do
  for i in $(seq 1 60); do
    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
      echo "  Port $port: ready"; break
    fi
    sleep 2
  done
done

echo "═══ Servers running ═══"