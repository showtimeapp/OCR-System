# PDF Financial Document Extraction Pipeline

**GLM-OCR + YOLO + Qwen3-VL on GCP L4 GPU with vLLM**

## Architecture

```
Client (PDF upload)
    │
    ▼
FastAPI (port 8000)
    │
    ├── PDF → Images (poppler, CPU)
    ├── YOLO Detection (GPU, 0.05s/page)
    ├── GLM-OCR Filter via vLLM (GPU, chart YES/NO)
    │
    ├── PARALLEL:
    │   ├── GLM-OCR via vLLM (port 8001) → OCR all pages
    │   └── Qwen3-VL via vLLM (port 8002) → Describe charts
    │
    └── Output: JSON + Markdown
```

## L4 GPU VRAM Layout (24GB)

```
GLM-OCR (vLLM):    ~3GB  (25% utilization)
Qwen3-VL (vLLM):  ~13GB  (55% utilization)
YOLO:              ~0.5GB
Free headroom:     ~7.5GB
```

## Setup (run once)

```bash
# SSH into your GCP VM
gcloud compute ssh your-instance-name

# Clone/upload this folder to /opt/pipeline
sudo mkdir -p /opt/pipeline
sudo cp -r pdf_pipeline/* /opt/pipeline/
cd /opt/pipeline

# Run setup
chmod +x setup.sh start_servers.sh
bash setup.sh
```

## Start (every time VM boots)

```bash
cd /opt/pipeline
source venv/bin/activate

# Step 1: Start vLLM servers (takes ~60s to load models)
bash start_servers.sh

# Step 2: Start API
python3 api.py
```

## API Usage

### Sync (small PDFs, <100 pages)
```bash
curl -X POST http://YOUR_VM_IP:8000/extract \
  -F "file=@report.pdf" \
  -o result.json

# Or with Python
python3 test_client.py report.pdf
```

### Async (large PDFs, 100+ pages)
```bash
# Start job
curl -X POST http://YOUR_VM_IP:8000/extract/async \
  -F "file=@big_report.pdf"
# Returns: {"job_id": "abc123", "status": "processing"}

# Poll status
curl http://YOUR_VM_IP:8000/status/abc123

# Download when done
curl http://YOUR_VM_IP:8000/download/abc123/extraction.json -o result.json
curl http://YOUR_VM_IP:8000/download/abc123/full.md -o result.md
curl http://YOUR_VM_IP:8000/download/abc123/all -o results.zip
```

### Python client
```python
import requests

with open('report.pdf', 'rb') as f:
    resp = requests.post('http://YOUR_VM_IP:8000/extract',
        files={'file': ('report.pdf', f, 'application/pdf')})

data = resp.json()
print(f"Pages: {data['pages_processed']}")
print(f"Speed: {data['avg_sec_per_page']}s/page")

# Download markdown
md = requests.get(f"http://YOUR_VM_IP:8000{data['download_md']}")
with open('result.md', 'wb') as f:
    f.write(md.content)
```

## Expected Speed on L4

| Phase | 384 pages |
|-------|-----------|
| PDF convert | ~30s |
| YOLO | ~22s |
| GLM filter | ~5s |
| OCR (8 concurrent) | ~3-5 min |
| Qwen charts (~15) | ~45s |
| **Total** | **~5-6 min** |
| **Per page** | **~0.8-1.0s** |

## Output Format

### JSON (extraction.json)
```json
{
  "source": "report.pdf",
  "pages_processed": 384,
  "charts_found": 15,
  "total_time_sec": 320.5,
  "avg_sec_per_page": 0.83,
  "pages": [
    {
      "page": 1,
      "ocr_text": "Annual Report 2024-25...",
      "has_chart": false
    },
    {
      "page": 14,
      "ocr_text": "NOV grew 52% YoY...",
      "has_chart": true,
      "charts": [{
        "bbox": [50, 100, 500, 400],
        "description": "Bar chart showing NOV growth from 27,735 crore in FY23..."
      }]
    }
  ]
}
```

### Markdown (full.md)
```markdown
# report

---
## Page 14

NOV (B2C business) and consolidated Adjusted Revenue grew 52% YoY...

📊 [Chart: Bar chart showing NOV growth from INR 27,735 crore in FY23
to 40,562 crore in FY24 (46% YoY) to 61,852 crore in FY25 (52% YoY)...]

Food delivery NOV grew 20% YoY to INR 32,862 crore in FY25
```

## Monitoring

```bash
# Check server logs
tail -f /opt/pipeline/logs/glm_ocr.log
tail -f /opt/pipeline/logs/qwen_vl.log

# GPU usage
nvidia-smi -l 1

# API health
curl http://localhost:8000/health
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| vLLM OOM | Reduce `--gpu-memory-utilization` in start_servers.sh |
| Slow OCR | Increase `OCR_CONCURRENCY` in pipeline.py |
| Missing charts | Lower `CHART_MIN_AREA` in pipeline.py |
| Too many false charts | Raise `CHART_MIN_AREA` to 0.08 or 0.10 |
