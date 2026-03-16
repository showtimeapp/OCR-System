"""
Core Pipeline — YOLO + vLLM (GLM-OCR + Qwen3-VL)
Async, concurrent, production-grade
"""

import os, json, time, gc, base64, asyncio, logging
from pathlib import Path
from io import BytesIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import aiohttp
from PIL import Image
from pdf2image import convert_from_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('pipeline')

# ═══════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════

# GLM_URL = "http://localhost:8001/v1/chat/completions"
# QWEN_URL = "http://localhost:8002/v1/chat/completions"

# DPI = 100
# MAX_SIDE = 800
# OCR_CONCURRENCY = 8       # 8 concurrent OCR requests to vLLM
# YOLO_DEVICE = 'cuda:0'
# CHART_MIN_AREA = 0.05
# MERGE_GAP = 80

import os
from dotenv import load_dotenv
load_dotenv()  # loads from .env file

GLM_URL = os.getenv("GLM_OCR_URL", "http://localhost:8001/v1/chat/completions")
QWEN_URL = os.getenv("QWEN_VL_URL", "http://localhost:8002/v1/chat/completions")
GLM_MODEL = os.getenv("GLM_MODEL_NAME", "glm-ocr")
QWEN_MODEL = os.getenv("QWEN_MODEL_NAME", "qwen-vl")

DPI = int(os.getenv("DPI", "100"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "800"))
OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "8"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cuda:0")
CHART_MIN_AREA = float(os.getenv("CHART_MIN_AREA", "0.05"))
MERGE_GAP = int(os.getenv("MERGE_GAP", "80"))

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
#  YOLO (loaded once, stays in memory)
# ═══════════════════════════════════════════════════

_yolo = None
_picture_id = None

def get_yolo():
    global _yolo, _picture_id
    if _yolo is None:
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        log.info('Loading YOLO...')
        path = hf_hub_download(
            repo_id='DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet',
            filename='yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt'
        )
        _yolo = YOLO(path)
        _picture_id = [k for k, v in _yolo.names.items() if v == 'Picture'][0]
        log.info(f'YOLO loaded | Picture ID: {_picture_id}')
    return _yolo, _picture_id


# ═══════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════

def img_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def merge_boxes(boxes, gap=MERGE_GAP):
    if len(boxes) <= 1:
        return boxes
    merged = True
    while merged:
        merged = False
        new_boxes, used = [], [False]*len(boxes)
        for i in range(len(boxes)):
            if used[i]: continue
            x1,y1,x2,y2,conf = boxes[i]
            for j in range(i+1, len(boxes)):
                if used[j]: continue
                bx1,by1,bx2,by2,bconf = boxes[j]
                if not(bx1>x2+gap or bx2<x1-gap) and not(by1>y2+gap or by2<y1-gap):
                    x1,y1 = min(x1,bx1),min(y1,by1)
                    x2,y2 = max(x2,bx2),max(y2,by2)
                    conf = max(conf,bconf); used[j]=True; merged=True
            new_boxes.append([x1,y1,x2,y2,conf]); used[i]=True
        boxes = new_boxes
    return boxes


# ═══════════════════════════════════════════════════
#  PDF → IMAGES
# ═══════════════════════════════════════════════════

def pdf_to_images(pdf_path, start=1, end=None):
    import pdfplumber
    with pdfplumber.open(str(pdf_path)) as pdf:
        total = len(pdf.pages)
    end = min(end or total, total)

    t0 = time.time()
    raw = convert_from_path(str(pdf_path), dpi=DPI, first_page=start, last_page=end,
                            fmt='png', thread_count=8)
    images = []
    for img in raw:
        r = MAX_SIDE / max(img.size)
        if r < 1:
            img = img.resize((int(img.width*r), int(img.height*r)), Image.LANCZOS)
        images.append(img)
    del raw

    log.info(f'PDF: {total} pages | Converted {len(images)} in {time.time()-t0:.1f}s')
    return images, total, start, end


# ═══════════════════════════════════════════════════
#  YOLO DETECTION
# ═══════════════════════════════════════════════════

def detect_charts(page_images, start_page):
    yolo, pic_id = get_yolo()
    t0 = time.time()

    raw_boxes = {}
    for idx, img in enumerate(page_images):
        pn = start_page + idx
        img_w, img_h = img.size
        results = yolo.predict(source=np.array(img), conf=0.3, verbose=False,
                               imgsz=640, device=YOLO_DEVICE)
        if results and results[0].boxes is not None:
            page_boxes = []
            for i in range(len(results[0].boxes)):
                if int(results[0].boxes.cls[i]) != pic_id: continue
                x1,y1,x2,y2 = map(int, results[0].boxes.xyxy[i].cpu().numpy())
                if ((x2-x1)*(y2-y1))/(img_w*img_h) < CHART_MIN_AREA: continue
                conf = float(results[0].boxes.conf[i])
                page_boxes.append([x1,y1,x2,y2,conf])
            if page_boxes:
                raw_boxes[pn] = page_boxes

    # Merge + crop (small margin for filter)
    raw_crops = []
    for pn, boxes in raw_boxes.items():
        img = page_images[pn - start_page]
        img_w, img_h = img.size
        merged = merge_boxes(boxes)
        for b in merged:
            x1,y1,x2,y2,conf = b
            fx1,fy1 = max(0,x1-30), max(0,y1-30)
            fx2,fy2 = min(img_w,x2+30), min(img_h,y2+30)
            crop = img.crop((fx1,fy1,fx2,fy2))
            raw_crops.append({'page':pn, 'crop':crop, 'bbox':[x1,y1,x2,y2], 'conf':round(conf,3)})

    log.info(f'YOLO: {len(raw_crops)} crops in {time.time()-t0:.1f}s')
    return raw_crops


# ═══════════════════════════════════════════════════
#  ASYNC vLLM CALLS
# ═══════════════════════════════════════════════════

async def call_glm(session, img, prompt="Document Parsing:", max_tokens=1024):
    b64 = img_to_base64(img)
    payload = {
        "model": GLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    async with session.post(GLM_URL, json=payload) as resp:
        data = await resp.json()
        return data['choices'][0]['message']['content']


async def call_qwen(session, img, prompt=CHART_PROMPT, max_tokens=600):
    b64 = img_to_base64(img)
    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    async with session.post(QWEN_URL, json=payload) as resp:
        data = await resp.json()
        return data['choices'][0]['message']['content']

# ═══════════════════════════════════════════════════
#  ASYNC FILTER (GLM YES/NO, concurrent)
# ═══════════════════════════════════════════════════

async def filter_charts(raw_crops, page_images, start_page, output_dir):
    """Filter crops with GLM-OCR YES/NO, return confirmed chart_crops."""
    t0 = time.time()
    sem = asyncio.Semaphore(OCR_CONCURRENCY)
    chart_crops = {}

    async def check_one(session, c, idx):
        async with sem:
            try:
                result = await call_glm(session, c['crop'], prompt=FILTER_PROMPT, max_tokens=5)
                return 'yes' in result.strip().lower()
            except Exception as e:
                log.warning(f'Filter error: {e}')
                return False

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        tasks = [check_one(session, c, i) for i, c in enumerate(raw_crops)]
        results = await asyncio.gather(*tasks)

    charts_dir = output_dir / 'charts'
    charts_dir.mkdir(parents=True, exist_ok=True)

    for idx, is_chart in enumerate(results):
        c = raw_crops[idx]
        pn = c['page']
        if is_chart:
            if pn not in chart_crops:
                chart_crops[pn] = []
            img = page_images[pn - start_page]
            img_w, img_h = img.size
            x1,y1,x2,y2 = c['bbox']
            bx1=max(0,x1-40); by1=max(0,y1-60)
            bx2=min(img_w,x2+40); by2=min(img_h,y2+80)
            big_crop = img.crop((bx1,by1,bx2,by2))
            crop_path = charts_dir / f'page_{pn:04d}_chart_{len(chart_crops[pn])+1}.png'
            big_crop.save(crop_path)
            chart_crops[pn].append({
                'crop': big_crop, 'crop_path': str(crop_path),
                'bbox': [bx1,by1,bx2,by2], 'conf': c['conf']
            })
            log.info(f'  Pg {pn}: CHART')
        else:
            log.info(f'  Pg {pn}: skip')

    total = sum(len(v) for v in chart_crops.values())
    log.info(f'Filter: {total} charts confirmed in {time.time()-t0:.1f}s')
    return chart_crops


# ═══════════════════════════════════════════════════
#  ASYNC OCR (all pages, concurrent)
# ═══════════════════════════════════════════════════

async def ocr_all_pages(page_images, start_page, chart_crops):
    """OCR all pages concurrently through vLLM."""
    t0 = time.time()
    sem = asyncio.Semaphore(OCR_CONCURRENCY)
    results = [None] * len(page_images)

    async def ocr_one(session, idx):
        pn = start_page + idx
        t1 = time.time()
        async with sem:
            try:
                text = await call_glm(session, page_images[idx])
            except Exception as e:
                log.error(f'OCR pg {pn}: {e}')
                text = ''
        dt = time.time() - t1
        tag = ' +chart' if pn in chart_crops else ''
        log.info(f'  [OCR] Pg {pn}: {len(text)} chars [{dt:.1f}s]{tag}')
        results[idx] = {'page': pn, 'ocr_text': text, 'has_chart': pn in chart_crops}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        tasks = [ocr_one(session, i) for i in range(len(page_images))]
        await asyncio.gather(*tasks)

    total = time.time() - t0
    log.info(f'OCR: {len(page_images)} pages in {total:.1f}s ({total/max(len(page_images),1):.1f}s/page)')
    return results


# ═══════════════════════════════════════════════════
#  ASYNC QWEN CHART DESCRIPTIONS
# ═══════════════════════════════════════════════════

async def describe_charts(chart_crops):
    """Describe confirmed chart crops through Qwen vLLM."""
    total = sum(len(v) for v in chart_crops.values())
    if total == 0:
        log.info('No charts to describe')
        return chart_crops

    t0 = time.time()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        for pn, charts in chart_crops.items():
            for i, ch in enumerate(charts):
                try:
                    crop = ch['crop']
                    # Upscale small crops
                    if max(crop.size) < 800:
                        s = 800 / max(crop.size)
                        crop = crop.resize((int(crop.width*s), int(crop.height*s)), Image.LANCZOS)

                    area = crop.width * crop.height
                    max_tok = 600 if area > 500000 else 400 if area > 250000 else 250

                    t1 = time.time()
                    desc = await call_qwen(session, crop, max_tokens=max_tok)
                    dt = time.time() - t1

                    ch['description'] = desc
                    del ch['crop']
                    log.info(f'  [Qwen] Pg {pn} chart {i+1}: {len(desc)} chars [{dt:.1f}s]')
                except Exception as e:
                    log.error(f'  [Qwen] Pg {pn} chart {i+1}: {e}')
                    ch['description'] = ''
                    ch.pop('crop', None)

    log.info(f'Qwen: {total} charts in {time.time()-t0:.1f}s')
    return chart_crops


# ═══════════════════════════════════════════════════
#  MARKDOWN BUILDER
# ═══════════════════════════════════════════════════

def build_markdown(all_pages, chart_crops, page_images, start_page, pdf_name):
    lines = [f'# {pdf_name}\n']

    for p in all_pages:
        pn = p['page']
        lines.append(f'\n---\n## Page {pn}\n')

        text = p.get('ocr_text', '')
        charts = []
        if pn in chart_crops:
            for ch in chart_crops[pn]:
                charts.append(ch)

        if not charts:
            lines.append(text)
            continue

        # Insert chart descriptions at correct vertical position
        img = page_images[pn - start_page]
        img_h = img.size[1]
        text_lines = text.split('\n')
        total_lines = max(len(text_lines), 1)

        inserts = {}
        for ch in charts:
            bbox = ch.get('bbox', [0,0,0,0])
            mid_y = (bbox[1] + bbox[3]) / 2
            pos = min(int((mid_y / img_h) * total_lines), total_lines - 1)
            desc = ch.get('description', '')
            if desc:
                inserts[pos] = inserts.get(pos, '') + f'\n📊 [Chart: {desc}]\n'

        result = []
        for i, line in enumerate(text_lines):
            result.append(line)
            if i in inserts:
                result.append(inserts[i])
        lines.append('\n'.join(result))

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════

async def process_pdf(pdf_path, output_dir=None, start=1, end=None):
    """Full async pipeline. Returns (json_path, md_path)."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir or f'/opt/pipeline/results/{pdf_path.stem}')
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f'═══ Processing: {pdf_path.name}')
    t_total = time.time()

    # Phase 1: PDF → images (sync, CPU-bound)
    page_images, total_pages, start_page, end_page = pdf_to_images(pdf_path, start, end)

    # Phase 2: YOLO detection (sync, GPU)
    raw_crops = detect_charts(page_images, start_page)

    # Phase 3: Filter charts (async, vLLM)
    chart_crops = await filter_charts(raw_crops, page_images, start_page, output_dir)

    # Phase 4: OCR + Qwen in parallel (async, vLLM)
    log.info('Running OCR + Qwen in parallel...')
    ocr_task = ocr_all_pages(page_images, start_page, chart_crops)
    qwen_task = describe_charts(chart_crops)
    all_pages, chart_crops = await asyncio.gather(ocr_task, qwen_task)

    # Phase 5: Merge + save
    total_charts = sum(len(v) for v in chart_crops.values())

    # Add charts to pages
    for p in all_pages:
        pn = p['page']
        if pn in chart_crops:
            p['charts'] = [{'bbox': ch['bbox'], 'conf': ch['conf'],
                'crop_path': ch.get('crop_path',''),
                'description': ch.get('description','')} for ch in chart_crops[pn]]

    total_time = time.time() - t_total

    # JSON
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
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Markdown
    md = build_markdown(all_pages, chart_crops, page_images, start_page, pdf_path.stem)
    md_path = output_dir / 'full.md'
    with open(md_path, 'w') as f:
        f.write(md)

    log.info(f'═══ Done: {pdf_path.name} | {len(all_pages)} pages | {total_charts} charts | {total_time:.1f}s ({total_time/len(all_pages):.2f}s/page)')

    # Cleanup
    del page_images
    gc.collect()

    return json_path, md_path, report


# CLI
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--pages', nargs=2, type=int)
    args = parser.parse_args()
    s, e = (args.pages[0], args.pages[1]) if args.pages else (1, None)
    asyncio.run(process_pdf(args.pdf, args.output, s, e))
