"""
Production Pipeline — Direct Inference on L4 GPU
GLM-OCR (0.22s/page) + YOLO + Qwen3-VL (7.5s/chart)
"""

import os, json, time, gc, base64, logging, asyncio
from pathlib import Path
from io import BytesIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv
import threading
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
OCR_MAX_TOKENS = int(os.getenv("OCR_MAX_TOKENS", "1024"))
QWEN_UPSCALE = int(os.getenv("QWEN_UPSCALE", "800"))

CHART_PROMPT = os.getenv("CHART_PROMPT", """Read every number carefully. Commas are thousand separators (11,323 = eleven thousand 323, NOT 11.23).
if is not chart or graph do not give any response ,else Give the concise discription of chart such that if someone reading your text can undersand everything about that chart without looking into chart, also strictly mention all the data points of the chart with small explanation""")

FILTER_PROMPT = "Is this a bar chart, line graph, pie chart, or area chart with axes and data points? Not a table, not a photo, not an icon, not an infographic. Answer only YES or NO."


# ═══════════════════════════════════════════════════
#  MODEL MANAGER — load once, keep forever
# ═══════════════════════════════════════════════════

class Models:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load_all(self):
        if self._loaded:
            return

        # GLM-OCR
        log.info('Loading GLM-OCR...')
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.ocr_processor = AutoProcessor.from_pretrained('zai-org/GLM-OCR')
        self.ocr_model = AutoModelForImageTextToText.from_pretrained(
            'zai-org/GLM-OCR', dtype=torch.float16
        ).to('cuda')
        self.ocr_model.eval()
        log.info(f'  GLM-OCR: {torch.cuda.memory_allocated()/1e9:.1f}GB')

        # Qwen3-VL
        log.info('Loading Qwen3-VL...')
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor as QP
        self.qwen_processor = QP.from_pretrained(
            'Qwen/Qwen3-VL-4B-Instruct', min_pixels=256*256, max_pixels=1280*960
        )
        self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            'Qwen/Qwen3-VL-4B-Instruct', torch_dtype=torch.float16
        ).to('cuda')
        self.qwen_model.eval()
        log.info(f'  Qwen: {torch.cuda.memory_allocated()/1e9:.1f}GB')

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
        log.info(f'All models loaded | GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB')

    # ── GLM-OCR inference ──
    def ocr(self, image):
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': 'Document Parsing:'}
        ]}]
        text = self.ocr_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.ocr_processor(text=[text], images=[image], return_tensors='pt').to('cuda')
        with torch.inference_mode():
            out = self.ocr_model.generate(**inputs, max_new_tokens=OCR_MAX_TOKENS)
        result = self.ocr_processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        del inputs, out
        return result.strip()

    # ── GLM-OCR filter ──
    def is_chart(self, image):
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': FILTER_PROMPT}
        ]}]
        text = self.ocr_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.ocr_processor(text=[text], images=[image], return_tensors='pt').to('cuda')
        with torch.inference_mode():
            out = self.ocr_model.generate(**inputs, max_new_tokens=5)
        result = self.ocr_processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        del inputs, out
        return 'yes' in result.strip().lower()

    # ── Qwen chart description ──
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


def build_markdown(all_pages, page_images, start_page, pdf_name):
    lines = [f'# {pdf_name}\n']
    for p in all_pages:
        pn = p['page']
        lines.append(f'\n---\n## Page {pn}\n')
        text = p.get('ocr_text', '')
        charts = p.get('charts', [])
        if not charts:
            lines.append(text)
            continue
        img = page_images[pn - start_page]
        img_h = img.size[1]
        text_lines = text.split('\n')
        total_lines = max(len(text_lines), 1)
        inserts = {}
        for ch in charts:
            bbox = ch.get('bbox', [0, 0, 0, 0])
            mid_y = (bbox[1] + bbox[3]) / 2
            pos = min(int((mid_y / img_h) * total_lines), total_lines - 1)
            desc = ch.get('description', '')
            if desc:
                inserts[pos] = inserts.get(pos, '') + f'\n\U0001F4CA [Chart: {desc}]\n'
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

def process_pdf(pdf_path, output_dir=None, start=1, end=None):
    """Full pipeline. Returns (json_path, md_path, report)."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir or f'./results/{pdf_path.stem}')
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / 'charts'
    charts_dir.mkdir(exist_ok=True)

    m = Models()
    m.load_all()

    log.info(f'═══ Processing: {pdf_path.name}')
    t_total = time.time()

    # ── Phase 1: PDF → images ──
    import pdfplumber
    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
    end = min(end or total_pages, total_pages)
    start_page = start

    t0 = time.time()
    raw = convert_from_path(str(pdf_path), dpi=DPI, first_page=start, last_page=end,
                            fmt='png', thread_count=8)
    page_images = []
    for img in raw:
        r = MAX_SIDE / max(img.size)
        if r < 1:
            img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
        page_images.append(img)
    del raw
    log.info(f'Phase 1 (PDF→images): {len(page_images)} pages in {time.time()-t0:.1f}s')

    # ── Phase 2: YOLO detection ──
    t0 = time.time()
    raw_boxes = {}
    for idx, img in enumerate(page_images):
        pn = start_page + idx
        img_w, img_h = img.size
        results = m.yolo.predict(source=np.array(img), conf=0.3, verbose=False,
                                 imgsz=640, device='cuda:0')
        if results and results[0].boxes is not None:
            page_boxes = []
            for i in range(len(results[0].boxes)):
                if int(results[0].boxes.cls[i]) != m.picture_id: continue
                x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[i].cpu().numpy())
                if ((x2 - x1) * (y2 - y1)) / (img_w * img_h) < CHART_MIN_AREA: continue
                conf = float(results[0].boxes.conf[i])
                page_boxes.append([x1, y1, x2, y2, conf])
            if page_boxes:
                raw_boxes[pn] = page_boxes

    # Merge + crop (small margin for filter)
    raw_crops = []
    for pn, boxes in raw_boxes.items():
        img = page_images[pn - start_page]
        img_w, img_h = img.size
        merged = merge_boxes(boxes)
        for b in merged:
            x1, y1, x2, y2, conf = b
            fx1, fy1 = max(0, x1 - 30), max(0, y1 - 30)
            fx2, fy2 = min(img_w, x2 + 30), min(img_h, y2 + 30)
            crop = img.crop((fx1, fy1, fx2, fy2))
            raw_crops.append({'page': pn, 'crop': crop, 'bbox': [x1, y1, x2, y2], 'conf': round(conf, 3)})

    log.info(f'Phase 2 (YOLO): {len(raw_crops)} crops in {time.time()-t0:.1f}s')

    # ── Phase 3: GLM-OCR filter ──
    t0 = time.time()
    chart_crops = {}
    for c in raw_crops:
        if m.is_chart(c['crop']):
            pn = c['page']
            if pn not in chart_crops: chart_crops[pn] = []
            img = page_images[pn - start_page]
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
    log.info(f'Phase 3 (filter): {total_charts} charts in {time.time()-t0:.1f}s')

    # ── Phase 4+5: OCR + Qwen in PARALLEL ──
    import threading

    ocr_results = [None]
    qwen_done = [False]

    def run_ocr():
        t0 = time.time()
        pages = []
        for idx, img in enumerate(page_images):
            pn = start_page + idx
            t1 = time.time()
            text = m.ocr(img)
            dt = time.time() - t1
            tag = ' +chart' if pn in chart_crops else ''
            log.info(f'  [OCR] Pg {pn}: {len(text)} chars [{dt:.2f}s]{tag}')
            pages.append({'page': pn, 'ocr_text': text, 'has_chart': pn in chart_crops})
            torch.cuda.empty_cache()
        ocr_time = time.time() - t0
        log.info(f'  [OCR] Done: {len(pages)} pages in {ocr_time:.1f}s ({ocr_time/len(pages):.2f}s/page)')
        ocr_results[0] = pages

    def run_qwen():
        t0 = time.time()
        for pn, charts in chart_crops.items():
            for i, ch in enumerate(charts):
                t1 = time.time()
                crop = Image.open(ch['crop_path'])
                area = crop.width * crop.height
                max_tok = 600 if area > 500000 else 400 if area > 250000 else 250
                desc = m.describe_chart(crop, max_tok)
                ch['description'] = desc
                log.info(f'  [Qwen] Pg {pn} chart {i+1}: {len(desc)} chars [{time.time()-t1:.1f}s]')
                torch.cuda.empty_cache()
        log.info(f'  [Qwen] Done: {sum(len(v) for v in chart_crops.values())} charts in {time.time()-t0:.1f}s')

    log.info('Phase 4+5: OCR + Qwen PARALLEL...')
    t0 = time.time()
    t_ocr = threading.Thread(target=run_ocr)
    t_qwen = threading.Thread(target=run_qwen)
    t_ocr.start()
    t_qwen.start()
    t_ocr.join()
    t_qwen.join()
    all_pages = ocr_results[0]
    log.info(f'Phase 4+5 total: {time.time()-t0:.1f}s (parallel)')

    # ── Phase 6: Merge + save ──
    for p in all_pages:
        pn = p['page']
        if pn in chart_crops:
            p['charts'] = [{'bbox': ch['bbox'], 'conf': ch['conf'],
                            'crop_path': ch.get('crop_path', ''),
                            'description': ch.get('description', '')} for ch in chart_crops[pn]]

    total_time = time.time() - t_total

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

    md = build_markdown(all_pages, chart_crops if chart_crops else {}, page_images, start_page, pdf_path.stem)
    md_path = output_dir / 'full.md'
    with open(md_path, 'w') as f:
        f.write(md)

    log.info(f'═══ Done: {pdf_path.name} | {len(all_pages)} pages | {total_charts} charts | {total_time:.1f}s ({total_time/len(all_pages):.2f}s/page)')

    del page_images
    gc.collect()
    torch.cuda.empty_cache()

    return json_path, md_path, report


# Fix build_markdown to accept correct args
def build_markdown(all_pages, chart_crops, page_images, start_page, pdf_name):
    lines = [f'# {pdf_name}\n']
    for p in all_pages:
        pn = p['page']
        lines.append(f'\n---\n## Page {pn}\n')
        text = p.get('ocr_text', '')
        charts = p.get('charts', [])
        if not charts:
            lines.append(text)
            continue
        img_h = 800  # default
        text_lines = text.split('\n')
        total_lines = max(len(text_lines), 1)
        inserts = {}
        for ch in charts:
            bbox = ch.get('bbox', [0, 0, 0, 0])
            mid_y = (bbox[1] + bbox[3]) / 2
            pos = min(int((mid_y / img_h) * total_lines), total_lines - 1)
            desc = ch.get('description', '')
            if desc:
                inserts[pos] = inserts.get(pos, '') + f'\n\U0001F4CA [Chart: {desc}]\n'
        result = []
        for i, line in enumerate(text_lines):
            result.append(line)
            if i in inserts:
                result.append(inserts[i])
        lines.append('\n'.join(result))
    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--pages', nargs=2, type=int)
    args = parser.parse_args()
    s, e = (args.pages[0], args.pages[1]) if args.pages else (1, None)
    process_pdf(args.pdf, args.output, s, e)
