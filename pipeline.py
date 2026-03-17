"""
Production Pipeline v2 — GLM-OCR SDK (vLLM) + YOLO + Qwen Direct
- OCR: GLM-OCR SDK via vLLM (2.1s/page)
- Charts: YOLO detect → GLM filter → Qwen describe (direct)
"""

import os, json, time, gc, logging, shutil,re,base64
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('pipeline')

# ═══════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════

DPI = int(os.getenv("DPI", "100"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "800"))
CHART_MIN_AREA = float(os.getenv("CHART_MIN_AREA", "0.05"))
MERGE_GAP = int(os.getenv("MERGE_GAP", "80"))
QWEN_UPSCALE = int(os.getenv("QWEN_UPSCALE", "800"))
GLMOCR_CONFIG = os.getenv("GLMOCR_CONFIG", os.path.expanduser("~/pipeline/config.yaml"))

CHART_PROMPT = """Read every number carefully. Commas are thousand separators (11,323 = eleven thousand 323, NOT 11.23).
1. Chart title and type
2. EVERY bar/line/slice with EXACT label and number
3. EVERY percentage, growth rate, YoY change
4. Time periods
5. Multiple series/colors with values
6. Overall trend
Write as flowing sentences. Reader should know every data point without seeing the chart."""

FILTER_PROMPT = "Is this a bar chart, line graph, pie chart, or area chart with axes and data points? Not a table, not a photo, not an icon, not an infographic. Answer only YES or NO."


# ═══════════════════════════════════════════════════
#  MODELS — Qwen + YOLO (loaded once)
# ═══════════════════════════════════════════════════

class ChartModels:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return

        # GLM-OCR for filter (lightweight, reuse vLLM)
        log.info('Loading GLM-OCR filter client...')
        import aiohttp
        self.glm_url = os.getenv("GLM_OCR_URL", "http://localhost:8090/v1/chat/completions")
        self.glm_model = os.getenv("GLM_MODEL_NAME", "glm-ocr")

        # Qwen3-VL direct
        log.info('Loading Qwen3-VL...')
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor as QP
        self.qwen_processor = QP.from_pretrained(
            'Qwen/Qwen3-VL-4B-Instruct', min_pixels=256*256, max_pixels=1280*960
        )
        self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            'Qwen/Qwen3-VL-4B-Instruct', torch_dtype=torch.float16
        ).to('cuda')
        self.qwen_model.eval()
        log.info(f'  Qwen GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB')

        # YOLO
        log.info('Loading YOLO...')
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        yolo_path = hf_hub_download(
            repo_id='DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet',
            filename='yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt'
        )
        self.yolo = YOLO(yolo_path)
        self.picture_id = [k for k, v in self.yolo.names.items() if v == 'Picture'][0]

        self._loaded = True
        log.info(f'All chart models loaded | GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB')

    def unload_qwen(self):
        """Free Qwen from GPU to make room for vLLM."""
        if hasattr(self, 'qwen_model'):
            del self.qwen_model
            del self.qwen_processor
            torch.cuda.empty_cache()
            gc.collect()
            self._loaded = False
            log.info('Qwen unloaded from GPU')

    def load_qwen(self):
        """Reload Qwen onto GPU."""
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor as QP
        self.qwen_processor = QP.from_pretrained(
            'Qwen/Qwen3-VL-4B-Instruct', min_pixels=256*256, max_pixels=1280*960
        )
        self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            'Qwen/Qwen3-VL-4B-Instruct', torch_dtype=torch.float16
        ).to('cuda')
        self.qwen_model.eval()
        log.info(f'Qwen reloaded | GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB')

    # ── GLM filter via vLLM HTTP ──
    def is_chart(self, image):
        import requests, base64
        from io import BytesIO
        buf = BytesIO()
        image.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()

        resp = requests.post(self.glm_url, json={
            "model": self.glm_model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": FILTER_PROMPT}
            ]}],
            "max_tokens": 5,
            "temperature": 0,
        }, timeout=30)
        result = resp.json()['choices'][0]['message']['content']
        return 'yes' in result.strip().lower()

    # ── Qwen chart description (direct) ──
    def describe_chart(self, image, max_tokens=600):
        if max(image.size) < QWEN_UPSCALE:
            s = QWEN_UPSCALE / max(image.size)
            image = image.resize((int(image.width * s), int(image.height * s)), Image.LANCZOS)

        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': CHART_PROMPT}
        ]}]
        inputs = self.qwen_processor.apply_chat_template(
            messages, tokenize=True, return_dict=True,
            return_tensors='pt', add_generation_prompt=True
        ).to('cuda')
        with torch.inference_mode():
            out = self.qwen_model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=False, temperature=None, top_p=None
            )
        gen = out[:, inputs['input_ids'].shape[1]:]
        result = self.qwen_processor.batch_decode(gen, skip_special_tokens=True)[0]
        del inputs, out, gen
        torch.cuda.empty_cache()
        return result.strip()


# ═══════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════

def merge_boxes(boxes, gap=MERGE_GAP):
    if len(boxes) <= 1:
        return boxes
    merged = True
    while merged:
        merged = False
        new_boxes, used = [], [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]: continue
            x1, y1, x2, y2, conf = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]: continue
                bx1, by1, bx2, by2, bconf = boxes[j]
                if not (bx1 > x2 + gap or bx2 < x1 - gap) and not (by1 > y2 + gap or by2 < y1 - gap):
                    x1, y1 = min(x1, bx1), min(y1, by1)
                    x2, y2 = max(x2, bx2), max(y2, by2)
                    conf = max(conf, bconf); used[j] = True; merged = True
            new_boxes.append([x1, y1, x2, y2, conf]); used[i] = True
        boxes = new_boxes
    return boxes


# ═══════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════

def process_pdf(pdf_path, output_dir=None, start=1, end=None):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir or f'./results/{pdf_path.stem}')
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / 'charts'
    charts_dir.mkdir(exist_ok=True)

    log.info(f'═══ Processing: {pdf_path.name}')
    t_total = time.time()

    # ── Phase 1: GLM-OCR SDK (vLLM backend) — OCR all pages ──
    log.info('Phase 1: GLM-OCR SDK...')
    t0 = time.time()
    sdk_output = output_dir / 'sdk_raw'
    sdk_output.mkdir(exist_ok=True)

    from glmocr import parse
    result = parse(str(pdf_path), config_path=GLMOCR_CONFIG, mode="selfhosted", enable_layout=False, ocr_api_host="localhost", ocr_api_port=8090, model="glm-ocr")
    result.save(output_dir=str(sdk_output))

    # Read SDK JSON output
    sdk_json = list(sdk_output.rglob('*.json'))
    if sdk_json:
        with open(sdk_json[0], 'r', encoding='utf-8') as f:
            sdk_pages = json.load(f)
    else:
        sdk_pages = []

    # Read SDK markdown
    sdk_md_files = list(sdk_output.rglob('*.md'))
    sdk_md = ""
    if sdk_md_files:
        with open(sdk_md_files[0], 'r', encoding='utf-8') as f:
            sdk_md = f.read()

    ocr_time = time.time() - t0
    num_pages = len(sdk_pages) if isinstance(sdk_pages, list) else 0
    log.info(f'Phase 1 (OCR): {num_pages} pages in {ocr_time:.1f}s ({ocr_time/max(num_pages,1):.2f}s/page)')

    # ── Phase 2: YOLO chart detection ──
    log.info('Phase 2: YOLO + chart detection...')
    t0 = time.time()

    import pdfplumber
    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
    last = min(end or total_pages, total_pages)

    raw = convert_from_path(str(pdf_path), dpi=DPI, first_page=start, last_page=last, fmt='png', thread_count=8)
    page_images = []
    for img in raw:
        r = MAX_SIDE / max(img.size)
        if r < 1:
            img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
        page_images.append(img)
    del raw

    cm = ChartModels()
    cm.load()

    raw_boxes = {}
    for idx, img in enumerate(page_images):
        pn = start + idx
        img_w, img_h = img.size
        results = cm.yolo.predict(source=np.array(img), conf=0.3, verbose=False, imgsz=640, device='cuda:0')
        if results and results[0].boxes is not None:
            page_boxes = []
            for i in range(len(results[0].boxes)):
                if int(results[0].boxes.cls[i]) != cm.picture_id: continue
                x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[i].cpu().numpy())
                if ((x2 - x1) * (y2 - y1)) / (img_w * img_h) < CHART_MIN_AREA: continue
                conf = float(results[0].boxes.conf[i])
                page_boxes.append([x1, y1, x2, y2, conf])
            if page_boxes:
                raw_boxes[pn] = page_boxes

    raw_crops = []
    for pn, boxes in raw_boxes.items():
        img = page_images[pn - start]
        img_w, img_h = img.size
        merged = merge_boxes(boxes)
        for b in merged:
            x1, y1, x2, y2, conf = b
            fx1, fy1 = max(0, x1 - 30), max(0, y1 - 30)
            fx2, fy2 = min(img_w, x2 + 30), min(img_h, y2 + 30)
            crop = img.crop((fx1, fy1, fx2, fy2))
            raw_crops.append({'page': pn, 'crop': crop, 'bbox': [x1, y1, x2, y2], 'conf': round(conf, 3)})

    log.info(f'  YOLO: {len(raw_crops)} crops in {time.time()-t0:.1f}s')

    # ── Phase 3: GLM filter via vLLM ──
    t0 = time.time()
    chart_crops = {}
    for c in raw_crops:
        if cm.is_chart(c['crop']):
            pn = c['page']
            if pn not in chart_crops: chart_crops[pn] = []
            img = page_images[pn - start]
            img_w, img_h = img.size
            x1, y1, x2, y2 = c['bbox']
            bx1, by1 = max(0, x1 - 40), max(0, y1 - 60)
            bx2, by2 = min(img_w, x2 + 40), min(img_h, y2 + 80)
            big_crop = img.crop((bx1, by1, bx2, by2))
            crop_path = charts_dir / f'page_{pn:04d}_chart_{len(chart_crops[pn])+1}.png'
            big_crop.save(crop_path)
            chart_crops[pn].append({
                'crop_path': str(crop_path), 'bbox': [bx1, by1, bx2, by2], 'conf': c['conf']
            })
            log.info(f'  Pg {pn}: CHART')
        else:
            log.info(f'  Pg {c["page"]}: skip')

    total_charts = sum(len(v) for v in chart_crops.values())
    log.info(f'  Filter: {total_charts} charts in {time.time()-t0:.1f}s')

    # ── Phase 4: Qwen chart descriptions (direct) ──
    t0 = time.time()
    for pn, charts in chart_crops.items():
        for i, ch in enumerate(charts):
            t1 = time.time()
            crop = Image.open(ch['crop_path'])
            area = crop.width * crop.height
            max_tok = 900 if area > 500000 else 600 if area > 250000 else 400
            desc = cm.describe_chart(crop, max_tok)
            ch['description'] = desc
            log.info(f'  [Qwen] Pg {pn} chart {i+1}: {len(desc)} chars [{time.time()-t1:.1f}s]')
    log.info(f'Phase 4 (Qwen): {total_charts} charts in {time.time()-t0:.1f}s')

    # ── Phase 5: Merge + save ──
    total_time = time.time() - t_total
    all_pages = []
    if isinstance(sdk_pages, list):
        for idx, page_data in enumerate(sdk_pages):
            pn = start + idx
            img = page_images[idx] if idx < len(page_images) else None
            page_w = img.size[0] if img else 0
            page_h = img.size[1] if img else 0

            # Build text blocks with coordinates
            blocks = []
            if isinstance(page_data, list):
                for item in page_data:
                    if isinstance(item, dict):
                        block = {
                            'type': 'text',
                            'index': item.get('index', 0),
                            'label': item.get('label', 'text'),
                            'content': item.get('content', ''),
                            'bbox': item.get('bbox_2d', None),
                            'page': pn,
                            'page_size': [page_w, page_h]
                        }
                        blocks.append(block)
            elif isinstance(page_data, dict):
                blocks.append({
                    'type': 'text',
                    'index': 0,
                    'label': page_data.get('label', 'text'),
                    'content': page_data.get('content', ''),
                    'bbox': page_data.get('bbox_2d', None),
                    'page': pn,
                    'page_size': [page_w, page_h]
                })

            # Add chart blocks with coordinates
            if pn in chart_crops:
                for ci, ch in enumerate(chart_crops[pn]):
                    blocks.append({
                        'type': 'chart',
                        'index': ci,
                        'label': 'chart',
                        'content': ch.get('description', ''),
                        'bbox': ch['bbox'],
                        'conf': ch['conf'],
                        'crop_path': ch.get('crop_path', ''),
                        'page': pn,
                        'page_size': [page_w, page_h]
                    })

            # Full page text for backward compatibility
            full_text = '\n\n'.join([b['content'] for b in blocks if b['type'] == 'text'])

            page = {
                'page': pn,
                'page_size': [page_w, page_h],
                'ocr_text': full_text,
                'has_chart': pn in chart_crops,
                'blocks': blocks
            }
            all_pages.append(page)

    # Save JSON
    report = {
        'source': str(pdf_path),
        'total_pages': total_pages,
        'pages_processed': len(all_pages),
        'charts_found': total_charts,
        'total_time_sec': round(total_time, 1),
        'avg_sec_per_page': round(total_time / max(len(all_pages), 1), 2),
        'extracted_at': datetime.now().isoformat(),
        'pages': all_pages,
    }
    json_path = output_dir / 'extraction.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Split SDK markdown into pages (separated by ---)
    sdk_sections = re.split(r'\n---\n', sdk_md) if sdk_md else []
    
    # Insert chart descriptions into correct page sections
    for pn, charts in chart_crops.items():
        page_idx = pn - start  # 0-indexed
        if page_idx < len(sdk_sections):
            chart_text = ''
            for ch in charts:
                desc = ch.get('description', '')
                if desc:
                    chart_text += f'\n\n📊 **[Chart Description]**: {desc}\n'
            sdk_sections[page_idx] = sdk_sections[page_idx] + chart_text
    
    final_md = '\n---\n'.join(sdk_sections)
    md_path = output_dir / 'full.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(final_md)

    log.info(f'═══ Done: {pdf_path.name} | {len(all_pages)} pages | {total_charts} charts | {total_time:.1f}s ({total_time/max(len(all_pages),1):.2f}s/page)')

    del page_images
    gc.collect()
    torch.cuda.empty_cache()

    return json_path, md_path, report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--pages', nargs=2, type=int)
    args = parser.parse_args()
    s, e = (args.pages[0], args.pages[1]) if args.pages else (1, None)
    process_pdf(args.pdf, args.output, s, e)
