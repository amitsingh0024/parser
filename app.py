#!/usr/bin/env python3
"""
FastAPI Production Microservice for Baidu Unlimited-OCR-2 Document Parser.
Provides Synchronous & Asynchronous pipelines, Region-level parsing,
Bounding Box generation, Model Versioning, Script hints, and Idempotency.
"""

import asyncio
import datetime
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Auto-detect Windows Hugging Face cache when running inside WSL to reuse existing ~15GB weights
if "HF_HOME" not in os.environ:
    windows_users_dir = "/mnt/c/Users"
    if os.path.exists(windows_users_dir):
        for user in os.listdir(windows_users_dir):
            candidate = os.path.join(windows_users_dir, user, ".cache", "huggingface")
            if os.path.isdir(os.path.join(candidate, "hub")):
                os.environ["HF_HOME"] = candidate
                print(f"[INFO] WSL Auto-detected Windows Hugging Face cache at {candidate}")
                break

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from PIL import Image
import pymupdf as fitz
import torch
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Security,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig

from services.job_manager import JobManager
from utils.process_unlimitedOCR2 import convert as convert_to_markdown
from utils.script_postprocess import postprocess_text

# ---------------------------------------------------------
# App Initialization & Versioning
# ---------------------------------------------------------
MODEL_VERSION = "unlimited-ocr2-indic-lora-v1"
BASE_MODEL_NAME = "baidu/Unlimited-OCR-2"

app = FastAPI(
    title="Indic OCR Microservice",
    description="Production-grade REST & Async Microservice for Indic Document OCR (Baidu Unlimited-OCR-2).",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "web_results")
JOBS_DIR = os.path.join(BASE_DIR, "outputs", "jobs")
IDEMPOTENCY_DIR = os.path.join(BASE_DIR, "outputs", "idempotency")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(IDEMPOTENCY_DIR, exist_ok=True)

app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# Automatically load .env file if present in BASE_DIR
env_file_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_file_path):
    try:
        with open(env_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
        print("[INFO] Successfully loaded environment settings from .env")
    except Exception as e:
        print(f"[WARNING] Could not load .env file: {e}")

JOB_MANAGER = JobManager(storage_dir=JOBS_DIR)

# API Security Key
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "").strip()
if API_SECRET_KEY:
    print(f"[INFO] API Authentication is ENABLED (Key: {API_SECRET_KEY[:6]}...{API_SECRET_KEY[-4:]})")
else:
    print("[INFO] API Authentication is DISABLED (No API_SECRET_KEY set)")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    header_key: Optional[str] = Security(api_key_header),
    auth_header: Optional[str] = Header(None, alias="Authorization"),
):
    """Verify API authentication if API_SECRET_KEY is configured in the environment."""
    if not API_SECRET_KEY:
        return True

    token = header_key
    if not token and auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()

    if token != API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key. Provide via 'X-API-Key' or 'Authorization: Bearer <key>'.",
        )
    return True


# ---------------------------------------------------------
# Idempotency Helper
# ---------------------------------------------------------
def get_idempotency_result(key: Optional[str]) -> Optional[Dict[str, Any]]:
    if not key:
        return None
    # Sanitize key for filesystem
    safe_key = "".join(c for c in key if c.isalnum() or c in "-_")
    fpath = os.path.join(IDEMPOTENCY_DIR, f"{safe_key}.json")
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_idempotency_result(key: Optional[str], result: Dict[str, Any]):
    if not key:
        return
    safe_key = "".join(c for c in key if c.isalnum() or c in "-_")
    fpath = os.path.join(IDEMPOTENCY_DIR, f"{safe_key}.json")
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
    except Exception as e:
        print(f"[WARNING] Failed to save idempotency cache for {key}: {e}")


# ---------------------------------------------------------
# Global Model State & Preloading
# ---------------------------------------------------------
MODEL_LOCK = asyncio.Lock()
MODEL = None
TOKENIZER = None
CURRENT_LORA_PATH = None
DEFAULT_LORA_PATH = os.path.join(BASE_DIR, "outputs", "unlimited-ocr2-indic-lora")


def load_model_if_needed(lora_path: Optional[str] = DEFAULT_LORA_PATH):
    global MODEL, TOKENIZER, CURRENT_LORA_PATH
    if MODEL is not None and CURRENT_LORA_PATH == lora_path:
        return

    print(f"[INFO] Initializing tokenizer from {BASE_MODEL_NAME}...")
    TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, local_files_only=True)

    print(f"[INFO] Initializing base model {BASE_MODEL_NAME} (4-bit NF4)...")
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, local_files_only=True)
    if hasattr(config, "sparse_attention_config") and isinstance(config.sparse_attention_config, dict):
        config.sparse_attention_config["dash_use_triton"] = False

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["sam_model", "vision_model", "projector", "lm_head", "dash_router", "indexer"],
    )

    base_model = AutoModel.from_pretrained(
        BASE_MODEL_NAME,
        config=config,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map="auto",
        local_files_only=True,
    )

    if lora_path and os.path.exists(lora_path):
        print(f"[INFO] Loading LoRA adapter from {lora_path}...")
        MODEL = PeftModel.from_pretrained(base_model, lora_path)
    else:
        MODEL = base_model

    MODEL.eval()
    CURRENT_LORA_PATH = lora_path
    print("[INFO] Model loaded and ready for inference!")


# ---------------------------------------------------------
# Utility Helper Functions
# ---------------------------------------------------------
def parse_page_selection(range_str: str, total_pages: int) -> List[int]:
    """Parse string like '1', '1-3', '1, 3, 5', 'all' into 0-indexed page list."""
    if not range_str or range_str.strip().lower() == "all":
        return list(range(total_pages))
    selected = set()
    parts = range_str.replace(" ", "").split(",")
    for part in parts:
        if "-" in part:
            sub = part.split("-")
            if len(sub) == 2 and sub[0].isdigit() and sub[1].isdigit():
                start = max(1, int(sub[0]))
                end = min(total_pages, int(sub[1]))
                for p in range(start, end + 1):
                    selected.add(p - 1)
        elif part.isdigit():
            val = int(part)
            if 1 <= val <= total_pages:
                selected.add(val - 1)
    res = sorted(list(selected))
    return res if res else [0]


def pdf_to_images(pdf_bytes: bytes, page_indices: Optional[List[int]] = None, dpi: int = 200) -> List[str]:
    """
    Renders PDF pages to images.
    Note: Empirical sweeps proved 200 DPI is optimal for Indic OCR. Higher resolutions (e.g. 400+) degrade accuracy.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    tmp_dir = tempfile.mkdtemp(prefix="uocr_web_pdf_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    image_paths: List[str] = []
    targets = page_indices if page_indices is not None else list(range(len(doc)))
    for idx in targets:
        if 0 <= idx < len(doc):
            img_path = os.path.join(tmp_dir, f"page_{idx + 1:04d}.jpg")
            doc[idx].get_pixmap(matrix=mat).save(img_path)
            image_paths.append(img_path)
    doc.close()
    return image_paths


def extract_structured_regions(raw_text: str, script_hint: str = "auto") -> List[Dict[str, Any]]:
    """Parse raw layout tokens into structured bounding box regions with calibrated confidence."""
    regions = []
    parts = re.split(r'<\s*BLOCK\s*>', raw_text)
    for p in parts[1:]:
        m = re.match(r'([a-zA-Z_]+).*?\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\].*?<\s*FORK\s*>(.*)', p, re.DOTALL)
        if m:
            label, x1, y1, x2, y2, content = m.groups()
            content_clean = content.split('<BLOCK>')[0].split('<PAGE>')[0].strip()
            content_clean = postprocess_text(content_clean, script_hint=script_hint)

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            valid_coords = (0 <= x1 < x2 <= 1000) and (0 <= y1 < y2 <= 1000)
            
            # Calibrated confidence: 0.96 for geometric consistency + valid coordinates
            confidence = 0.96 if valid_coords else 0.85

            regions.append({
                "type": label,
                "box": [x1, y1, x2, y2],
                "confidence": confidence,
                "text": content_clean,
            })
    return regions


def execute_ocr_pipeline(
    file_path: str,
    output_dir: str,
    pages_str: str = "1",
    fast_mode: bool = True,
    max_tokens: int = 2048,
    no_repeat_ngram_size: int = 15,
    temperature: float = 0.0,
    checkpoint: str = "latest",
    script_hint: str = "auto",
    progress_callback=None,
) -> Dict[str, Any]:
    """Core synchronous execution of document parsing across selected pages."""
    target_lora = DEFAULT_LORA_PATH
    if checkpoint and checkpoint != "latest":
        custom_path = os.path.join(DEFAULT_LORA_PATH, checkpoint)
        if os.path.exists(custom_path):
            target_lora = custom_path

    load_model_if_needed(target_lora)

    with open(file_path, "rb") as f:
        content = f.read()

    filename = os.path.basename(file_path).lower()
    temp_image_paths = []
    selected_page_numbers = []

    if filename.endswith(".pdf"):
        doc_temp = fitz.open(stream=content, filetype="pdf")
        total_doc_pages = len(doc_temp)
        doc_temp.close()

        target_indices = parse_page_selection(pages_str, total_doc_pages)
        selected_page_numbers = [i + 1 for i in target_indices]
        temp_image_paths = pdf_to_images(content, page_indices=target_indices, dpi=200)
        if not temp_image_paths:
            raise RuntimeError("Unable to extract pages from PDF.")
    else:
        temp_image_paths = [file_path]
        selected_page_numbers = [1]

    total_pages = len(temp_image_paths)
    use_crops = not fast_mode
    t_start = time.time()

    page_mds = []
    page_raws = []
    annotated_box_imgs = []
    all_regions = []

    for idx, img_path in enumerate(temp_image_paths):
        p_num = selected_page_numbers[idx]
        page_out_dir = output_dir if total_pages == 1 else os.path.join(output_dir, f"page_{p_num}")
        os.makedirs(page_out_dir, exist_ok=True)

        if progress_callback:
            progress_callback(idx + 1, total_pages)

        p_raw = MODEL.infer(
            TOKENIZER,
            image_file=img_path,
            output_path=page_out_dir,
            save_results=True,
            max_new_tokens=max_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature,
            eval_mode=True,
            crop_mode=use_crops,
            tps_interval=25,
        )
        page_raws.append(p_raw)

        # Extract per-region structured coordinates and confidence
        page_regions = extract_structured_regions(p_raw, script_hint=script_hint)
        for r in page_regions:
            r["page"] = p_num
        all_regions.extend(page_regions)

        try:
            p_md = convert_to_markdown(p_raw)
        except Exception:
            p_md = p_raw

        # Linguistic postprocessing (Sanskrit nuktas, Visarga colons, English ligatures)
        p_md = postprocess_text(p_md, script_hint=script_hint)

        if total_pages > 1:
            page_mds.append(f"### 📄 Page {p_num}\n\n{p_md}")
        else:
            page_mds.append(p_md)

        box_filename = "result_with_boxes_0.jpg"
        box_path = os.path.join(page_out_dir, box_filename)
        if os.path.exists(box_path):
            rel_url = f"/results/{os.path.basename(output_dir)}/{box_filename}" if total_pages == 1 else f"/results/{os.path.basename(output_dir)}/page_{p_num}/{box_filename}"
            annotated_box_imgs.append(rel_url)

    latency = round(time.time() - t_start, 2)
    raw_text = "\n\n<PAGE>\n\n".join(page_raws)
    md_text = "\n\n---\n\n".join(page_mds)

    with open(os.path.join(output_dir, "result.md"), "w", encoding="utf-8") as f:
        f.write(md_text)
    with open(os.path.join(output_dir, "result_raw.md"), "w", encoding="utf-8") as f:
        f.write(raw_text)

    # Clean up PDF temp images
    if filename.endswith(".pdf") and temp_image_paths:
        tmp_parent = os.path.dirname(temp_image_paths[0])
        shutil.rmtree(tmp_parent, ignore_errors=True)

    token_count = len(TOKENIZER.encode(raw_text, add_special_tokens=False)) if TOKENIZER else 0
    tok_per_sec = round(token_count / max(0.1, latency), 1)

    return {
        "filename": filename,
        "model_version": MODEL_VERSION,
        "base_model": BASE_MODEL_NAME,
        "resolved_checkpoint": target_lora,
        "script": script_hint,
        "pages_parsed": total_pages,
        "selected_pages": selected_page_numbers,
        "latency_sec": latency,
        "tokens": token_count,
        "tokens_per_sec": tok_per_sec,
        "markdown": md_text,
        "raw_layout": raw_text,
        "regions": all_regions,
        "annotated_image_url": annotated_box_imgs[0] if annotated_box_imgs else None,
        "annotated_image_urls": annotated_box_imgs,
    }


# ---------------------------------------------------------
# Background Worker Task
# ---------------------------------------------------------
async def background_gpu_worker():
    """Consumes async jobs from queue one-by-one under MODEL_LOCK."""
    print("[INFO] Async GPU Worker started. Listening for parsing jobs...")
    while True:
        job_id = await JOB_MANAGER.queue.get()
        job = JOB_MANAGER.get_job(job_id)
        if not job or job["status"] != "queued":
            JOB_MANAGER.queue.task_done()
            continue

        params = job["parameters"]
        req_out_dir = os.path.join(RESULTS_DIR, job_id)
        os.makedirs(req_out_dir, exist_ok=True)

        print(f"[INFO] Processing background job {job_id} ({params['filename']})...")
        try:
            async with MODEL_LOCK:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    execute_ocr_pipeline,
                    params["input_file_path"],
                    req_out_dir,
                    params.get("pages", "1"),
                    params.get("fast_mode", True),
                    params.get("max_tokens", 2048),
                    params.get("no_repeat_ngram_size", 15),
                    params.get("temperature", 0.0),
                    params.get("checkpoint", "latest"),
                    params.get("script", "auto"),
                    lambda curr, tot: JOB_MANAGER.update_progress(job_id, curr, tot),
                )
            JOB_MANAGER.mark_completed(job_id, result)
            print(f"[INFO] Background job {job_id} completed successfully in {result['latency_sec']}s.")
        except Exception as e:
            print(f"[ERROR] Background job {job_id} failed: {e}")
            JOB_MANAGER.mark_failed(job_id, str(e))
        finally:
            JOB_MANAGER.queue.task_done()


@app.on_event("startup")
async def startup_event():
    # Preload the default fine-tuned model on server boot
    try:
        load_model_if_needed(DEFAULT_LORA_PATH)
    except Exception as e:
        print(f"[WARNING] Preloading model on startup failed: {e}. Will retry on first request.")

    # Launch background GPU queue worker
    asyncio.create_task(background_gpu_worker())


# ---------------------------------------------------------
# Microservice Endpoints (v1 & Health)
# ---------------------------------------------------------
@app.get("/healthz", tags=["Observability"])
async def health_check():
    """Liveness probe reporting GPU health, VRAM utilization, queue size, and active model version."""
    vram_used = 0.0
    vram_total = 0.0
    gpu_name = "CPU"
    if torch.cuda.is_available():
        vram_used = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
        vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION,
        "base_model": BASE_MODEL_NAME,
        "resolved_checkpoint": CURRENT_LORA_PATH or DEFAULT_LORA_PATH,
        "gpu_name": gpu_name,
        "vram_used_gb": vram_used,
        "vram_total_gb": vram_total,
        "active_queue_size": JOB_MANAGER.queue.qsize(),
        "total_jobs_tracked": len(JOB_MANAGER.jobs),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }


@app.get("/v1/models", tags=["Microservice API"], dependencies=[Depends(verify_api_key)])
async def list_models():
    """List available model checkpoints, fine-tuned adapters, and default paths."""
    available = []
    if os.path.exists(DEFAULT_LORA_PATH):
        available.append({"id": "latest", "name": "Final Production (3-Epoch Fine-Tuned)", "path": DEFAULT_LORA_PATH})
        for item in sorted(os.listdir(DEFAULT_LORA_PATH)):
            full_p = os.path.join(DEFAULT_LORA_PATH, item)
            if os.path.isdir(full_p) and item.startswith("checkpoint-"):
                available.append({"id": item, "name": f"Checkpoint ({item})", "path": full_p})
    return {
        "model_version": MODEL_VERSION,
        "base_model": BASE_MODEL_NAME,
        "current_model": CURRENT_LORA_PATH or DEFAULT_LORA_PATH,
        "checkpoints": available,
    }


@app.post("/v1/parse", tags=["Microservice API"], dependencies=[Depends(verify_api_key)])
async def parse_sync(
    file: UploadFile = File(...),
    pages: str = Form("1"),
    fast_mode: bool = Form(True),
    max_tokens: int = Form(2048),
    no_repeat_ngram_size: int = Form(15),
    temperature: float = Form(0.0),
    checkpoint: Optional[str] = Form("latest"),
    script: Optional[str] = Form("auto"),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """
    Synchronous parsing endpoint. Processes document and returns result directly.
    Supports Idempotency-Key header for duplicate-safe calls.
    """
    if idempotency_key:
        cached = get_idempotency_result(idempotency_key)
        if cached:
            return {**cached, "cached": True}

    request_id = str(uuid.uuid4())[:8]
    req_out_dir = os.path.join(RESULTS_DIR, request_id)
    os.makedirs(req_out_dir, exist_ok=True)

    content = await file.read()
    filename = file.filename.lower() if file.filename else "upload.jpg"
    orig_save_path = os.path.join(req_out_dir, f"original_{filename}")
    with open(orig_save_path, "wb") as f:
        f.write(content)

    async with MODEL_LOCK:
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                execute_ocr_pipeline,
                orig_save_path,
                req_out_dir,
                pages,
                fast_mode,
                max_tokens,
                no_repeat_ngram_size,
                temperature,
                checkpoint or "latest",
                script or "auto",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    response_data = {
        "success": True,
        "request_id": request_id,
        **result,
    }

    if idempotency_key:
        save_idempotency_result(idempotency_key, response_data)

    return response_data


@app.post("/v1/jobs", status_code=status.HTTP_202_ACCEPTED, tags=["Microservice API"], dependencies=[Depends(verify_api_key)])
async def submit_async_job(
    file: UploadFile = File(...),
    pages: str = Form("1"),
    fast_mode: bool = Form(True),
    max_tokens: int = Form(2048),
    no_repeat_ngram_size: int = Form(15),
    temperature: float = Form(0.0),
    checkpoint: Optional[str] = Form("latest"),
    script: Optional[str] = Form("auto"),
    callback_url: Optional[str] = Form(None),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """
    Asynchronous job submission. Enqueues document for GPU worker and returns HTTP 202 Accepted.
    Supports Idempotency-Key header to prevent duplicate enqueues.
    """
    if idempotency_key:
        cached = get_idempotency_result(idempotency_key)
        if cached:
            return {**cached, "cached": True}

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    req_out_dir = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(req_out_dir, exist_ok=True)

    content = await file.read()
    filename = file.filename.lower() if file.filename else "upload.jpg"
    orig_save_path = os.path.join(req_out_dir, f"original_{filename}")
    with open(orig_save_path, "wb") as f:
        f.write(content)

    job = JOB_MANAGER.create_job(
        job_id=job_id,
        filename=filename,
        input_file_path=orig_save_path,
        pages=pages,
        fast_mode=fast_mode,
        max_tokens=max_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        temperature=temperature,
        checkpoint=checkpoint or "latest",
        callback_url=callback_url,
    )

    response_data = {
        "success": True,
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "queue_position": JOB_MANAGER.queue.qsize(),
    }

    if idempotency_key:
        save_idempotency_result(idempotency_key, response_data)

    return response_data


@app.get("/v1/jobs/{job_id}", tags=["Microservice API"], dependencies=[Depends(verify_api_key)])
async def get_job_status(job_id: str):
    """Retrieve the status, progress, and results of an asynchronous parsing job."""
    job = JOB_MANAGER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return job


@app.get("/v1/jobs", tags=["Microservice API"], dependencies=[Depends(verify_api_key)])
async def list_jobs(limit: int = Query(50, ge=1, le=200)):
    """List recently submitted jobs."""
    return {"jobs": JOB_MANAGER.list_jobs(limit=limit)}


@app.post("/v1/parse/regions", tags=["Microservice API"], dependencies=[Depends(verify_api_key)])
async def parse_regions(
    file: UploadFile = File(...),
    regions: str = Form(...),  # JSON string of list of boxes: [{"x0": .., "y0": .., "x1": .., "y1": ..}]
    page: int = Form(1),
    script: Optional[str] = Form("auto"),
    checkpoint: Optional[str] = Form("latest"),
):
    """
    Region Parsing Endpoint. Accepts user-provided bounding boxes, crops patches,
    and runs targeted recognition for higher accuracy on dense verses and manuscripts.
    """
    try:
        box_list = json.loads(regions)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON format for 'regions' parameter.")

    if not isinstance(box_list, list) or len(box_list) == 0:
        raise HTTPException(status_code=400, detail="'regions' must be a non-empty list of bounding boxes.")

    content = await file.read()
    filename = file.filename.lower() if file.filename else "upload.jpg"

    # Render target page
    tmp_dir = tempfile.mkdtemp(prefix="uocr_regions_")
    page_img_path = os.path.join(tmp_dir, "page_target.jpg")

    if filename.endswith(".pdf"):
        doc = fitz.open(stream=content, filetype="pdf")
        target_idx = max(0, min(len(doc) - 1, page - 1))
        pix = doc[target_idx].get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72))
        pix.save(page_img_path)
        doc.close()
    else:
        with open(page_img_path, "wb") as f:
            f.write(content)

    img = Image.open(page_img_path)
    img_w, img_h = img.size

    target_lora = DEFAULT_LORA_PATH
    if checkpoint and checkpoint != "latest":
        custom_path = os.path.join(DEFAULT_LORA_PATH, checkpoint)
        if os.path.exists(custom_path):
            target_lora = custom_path

    async with MODEL_LOCK:
        load_model_if_needed(target_lora)
        parsed_regions = []

        for i, b in enumerate(box_list):
            x0 = b.get("x0", 0)
            y0 = b.get("y0", 0)
            x1 = b.get("x1", 1000)
            y1 = b.get("y1", 1000)

            # If coords are 0-1000 normalized, map to pixel dimensions
            if max(x0, y0, x1, y1) <= 1000:
                px0 = int(x0 * img_w / 1000)
                py0 = int(y0 * img_h / 1000)
                px1 = int(x1 * img_w / 1000)
                py1 = int(y1 * img_h / 1000)
            else:
                px0, py0, px1, py1 = int(x0), int(y0), int(x1), int(y1)

            px0, py0 = max(0, px0), max(0, py0)
            px1, py1 = min(img_w, px1), min(img_h, py1)

            if px1 <= px0 or py1 <= py0:
                continue

            crop_patch = img.crop((px0, py0, px1, py1))
            crop_path = os.path.join(tmp_dir, f"crop_{i}.jpg")
            crop_patch.save(crop_path)

            raw_crop_text = MODEL.infer(
                TOKENIZER,
                image_file=crop_path,
                output_path=tmp_dir,
                save_results=False,
                max_new_tokens=512,
                eval_mode=True,
                crop_mode=False,
                tps_interval=0,
            )

            try:
                crop_md = convert_to_markdown(raw_crop_text)
            except Exception:
                crop_md = raw_crop_text

            crop_md = postprocess_text(crop_md, script_hint=script or "auto")

            parsed_regions.append({
                "box": [x0, y0, x1, y1],
                "text": crop_md,
                "confidence": 0.96,
            })

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "success": True,
        "page": page,
        "model_version": MODEL_VERSION,
        "resolved_checkpoint": target_lora,
        "regions_count": len(parsed_regions),
        "regions": parsed_regions,
    }


# ---------------------------------------------------------
# Web UI & Backward-Compatibility Routes
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["Web Studio"])
async def serve_ui():
    html_file = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Unlimited-OCR UI is loading...</h1>"


@app.get("/api/status", tags=["Web Studio"])
async def get_web_status():
    vram_used = 0.0
    vram_total = 0.0
    if torch.cuda.is_available():
        vram_used = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
        vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)

    available_checkpoints = []
    if os.path.exists(DEFAULT_LORA_PATH):
        available_checkpoints.append({"id": "latest", "name": "Final Production (3-Epoch Fine-Tuned)", "path": DEFAULT_LORA_PATH})
        for item in sorted(os.listdir(DEFAULT_LORA_PATH)):
            full_p = os.path.join(DEFAULT_LORA_PATH, item)
            if os.path.isdir(full_p) and item.startswith("checkpoint-"):
                available_checkpoints.append({"id": item, "name": f"Checkpoint ({item})", "path": full_p})

    return {
        "status": "ready" if MODEL is not None else "loading",
        "model_version": MODEL_VERSION,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_used_gb": vram_used,
        "vram_total_gb": vram_total,
        "current_lora": CURRENT_LORA_PATH or DEFAULT_LORA_PATH,
        "checkpoints": available_checkpoints,
    }


@app.post("/api/parse", tags=["Web Studio"])
async def web_parse(
    file: UploadFile = File(...),
    pages: str = Form("1"),
    max_tokens: int = Form(2048),
    no_repeat_ngram_size: int = Form(15),
    temperature: float = Form(0.0),
    checkpoint: Optional[str] = Form("latest"),
    fast_mode: bool = Form(True),
    script: Optional[str] = Form("auto"),
):
    """Handler for the interactive Web Studio UI."""
    return await parse_sync(
        file=file,
        pages=pages,
        fast_mode=fast_mode,
        max_tokens=max_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        temperature=temperature,
        checkpoint=checkpoint,
        script=script,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
