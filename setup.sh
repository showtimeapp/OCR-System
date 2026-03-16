#!/bin/bash
# ═══════════════════════════════════════════════════
#  Setup — Run once on fresh GCP VM (L4 GPU)
# ═══════════════════════════════════════════════════

set -e
echo "═══ PDF Pipeline Setup ═══"

# System packages
sudo apt-get update -qq
sudo apt-get install -y -qq poppler-utils python3-pip python3-venv

# Create venv
python3 -m venv /opt/pipeline/venv
source /opt/pipeline/venv/bin/activate

# Core packages
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install vllm
pip install git+https://github.com/huggingface/transformers
pip install fastapi uvicorn python-multipart
pip install ultralytics pdf2image Pillow pdfplumber
pip install huggingface_hub aiofiles aiohttp

# Download models (pre-cache so first request is fast)
echo "Pre-downloading models..."
python3 -c "
from huggingface_hub import snapshot_download, hf_hub_download
print('Downloading GLM-OCR...')
snapshot_download('zai-org/GLM-OCR')
print('Downloading Qwen3-VL-4B...')
snapshot_download('Qwen/Qwen3-VL-4B-Instruct')
print('Downloading YOLO...')
hf_hub_download('DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet',
    'yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt')
print('All models downloaded!')
"

# Create directories
mkdir -p /opt/pipeline/{uploads,results,logs}

echo "═══ Setup complete! ═══"
echo "Next: bash start_servers.sh"
