"""
FastAPI — PDF Extraction API
Upload PDF → Get JSON + Markdown
"""

import os, json, shutil, logging
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline import process_pdf, Models

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('api')

app = FastAPI(
    title="PDF Financial Document Extractor",
    description="Upload PDF → Get structured JSON + Markdown with chart descriptions",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./results"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

jobs = {}


@app.on_event("startup")
async def startup():
    log.info("Loading models on startup...")
    Models().load_all()
    log.info("API ready!")


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/extract")
async def extract_pdf(
    file: UploadFile = File(...),
    start_page: int = 1,
    end_page: int = None,
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files accepted")

    job_id = uuid4().hex[:12]
    pdf_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    output_dir = RESULTS_DIR / job_id

    try:
        json_path, md_path, report = process_pdf(
            str(pdf_path), str(output_dir), start_page, end_page
        )
        return JSONResponse({
            "job_id": job_id,
            "source": file.filename,
            "pages_processed": report["pages_processed"],
            "charts_found": report["charts_found"],
            "total_time_sec": report["total_time_sec"],
            "avg_sec_per_page": report["avg_sec_per_page"],
            "download_json": f"/download/{job_id}/extraction.json",
            "download_md": f"/download/{job_id}/full.md",
            "download_zip": f"/download/{job_id}/all",
            "pages": report["pages"],
        })
    except Exception as e:
        log.error(f"Extract failed: {e}")
        raise HTTPException(500, str(e))
    finally:
        pdf_path.unlink(missing_ok=True)


@app.post("/extract/async")
async def extract_async(
    file: UploadFile = File(...),
    start_page: int = 1,
    end_page: int = None,
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files accepted")

    job_id = uuid4().hex[:12]
    pdf_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    output_dir = RESULTS_DIR / job_id
    jobs[job_id] = {"status": "processing", "filename": file.filename}

    import asyncio
    async def run():
        try:
            _, _, report = process_pdf(str(pdf_path), str(output_dir), start_page, end_page)
            jobs[job_id] = {
                "status": "done",
                "pages_processed": report["pages_processed"],
                "charts_found": report["charts_found"],
                "total_time_sec": report["total_time_sec"],
                "download_json": f"/download/{job_id}/extraction.json",
                "download_md": f"/download/{job_id}/full.md",
                "download_zip": f"/download/{job_id}/all",
            }
        except Exception as e:
            jobs[job_id] = {"status": "error", "error": str(e)}
        finally:
            pdf_path.unlink(missing_ok=True)

    asyncio.create_task(run())
    return {"job_id": job_id, "status": "processing", "poll": f"/status/{job_id}"}


@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/download/{job_id}/extraction.json")
async def dl_json(job_id: str):
    p = RESULTS_DIR / job_id / "extraction.json"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p, filename=f"{job_id}_extraction.json")


@app.get("/download/{job_id}/full.md")
async def dl_md(job_id: str):
    p = RESULTS_DIR / job_id / "full.md"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p, filename=f"{job_id}_full.md")


@app.get("/download/{job_id}/all")
async def dl_zip(job_id: str):
    d = RESULTS_DIR / job_id
    if not d.exists(): raise HTTPException(404)
    z = RESULTS_DIR / f"{job_id}.zip"
    shutil.make_archive(str(z).replace('.zip', ''), 'zip', str(d))
    return FileResponse(z, filename=f"{job_id}_results.zip")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
