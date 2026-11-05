#!/usr/bin/env python3
"""
FastAPI Server for Baidu Unlimited-OCR-2 Document Parser.
Supports PDF and Image parsing, Markdown extraction, and Visual Bounding Box overlays.
"""

import asyncio
import os
import shutil
import sys
import tempfile
import time
import uuid
from typing import List, Optional

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

import pymupdf as fitz
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig

from utils.process_unlimitedOCR2 import convert as convert_to_markdown

app = FastAPI(title="Unlimited-OCR-2 Parser", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory for processed web results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "web_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# Global model state
MODEL_LOCK = asyncio.Lock()
MODEL = None
TOKENIZER = None
CURRENT_LORA_PATH = None
BASE_MODEL_NAME = "baidu/Unlimited-OCR-2"
DEFAULT_LORA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "unlimited-ocr2-indic-lora")


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


def pdf_to_images(pdf_bytes: bytes, dpi: int = 180) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    tmp_dir = tempfile.mkdtemp(prefix="uocr_web_pdf_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    image_paths: List[str] = []
    for i, page in enumerate(doc):
        img_path = os.path.join(tmp_dir, f"page_{i + 1:04d}.jpg")
        page.get_pixmap(matrix=mat).save(img_path)
        image_paths.append(img_path)
    doc.close()
    return image_paths


@app.on_event("startup")
async def startup_event():
    # Preload the default final fine-tuned model
    try:
        load_model_if_needed(DEFAULT_LORA_PATH)
    except Exception as e:
        print(f"[WARNING] Preloading model on startup failed: {e}. Will retry on first request.")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Unlimited-OCR UI is loading...</h1>"


@app.get("/api/status")
async def get_status():
    vram_used = 0
    vram_total = 0
    if torch.cuda.is_available():
        vram_used = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
        vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)

    available_checkpoints = []
    base_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "unlimited-ocr2-indic-lora")
    if os.path.exists(base_out):
        available_checkpoints.append({"id": "latest", "name": "Final Production (3-Epoch Fine-Tuned)", "path": base_out})
        for item in sorted(os.listdir(base_out)):
            full_p = os.path.join(base_out, item)
            if os.path.isdir(full_p) and item.startswith("checkpoint-"):
                available_checkpoints.append({"id": item, "name": f"Checkpoint ({item})", "path": full_p})

    return {
        "status": "ready" if MODEL is not None else "loading",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_used_gb": vram_used,
        "vram_total_gb": vram_total,
        "current_lora": CURRENT_LORA_PATH,
        "checkpoints": available_checkpoints,
    }


@app.post("/api/parse")
async def parse_document(
    file: UploadFile = File(...),
    max_tokens: int = Form(2048),
    no_repeat_ngram_size: int = Form(15),
    temperature: float = Form(0.0),
    checkpoint: Optional[str] = Form("latest"),
    fast_mode: bool = Form(True),
):
    request_id = str(uuid.uuid4())[:8]
    req_out_dir = os.path.join(RESULTS_DIR, request_id)
    os.makedirs(req_out_dir, exist_ok=True)

    content = await file.read()
    filename = file.filename.lower() if file.filename else "upload.jpg"

    # Save original uploaded file
    orig_save_path = os.path.join(req_out_dir, f"original_{filename}")
    with open(orig_save_path, "wb") as f:
        f.write(content)

    temp_image_paths = []
    if filename.endswith(".pdf"):
        temp_image_paths = pdf_to_images(content)
        if not temp_image_paths:
            raise HTTPException(status_code=400, detail="Unable to extract pages from PDF.")
    else:
        temp_image_paths = [orig_save_path]

    async with MODEL_LOCK:
        target_lora = DEFAULT_LORA_PATH
        if checkpoint and checkpoint != "latest":
            custom_path = os.path.join(DEFAULT_LORA_PATH, checkpoint)
            if os.path.exists(custom_path):
                target_lora = custom_path

        load_model_if_needed(target_lora)

        t_start = time.time()
        use_crops = not fast_mode
        print(f"[INFO] Processing request (fast_mode={fast_mode}, crop_mode={use_crops})...")
        try:
            if len(temp_image_paths) == 1:
                raw_text = MODEL.infer(
                    TOKENIZER,
                    image_file=temp_image_paths[0],
                    output_path=req_out_dir,
                    save_results=True,
                    max_new_tokens=max_tokens,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    temperature=temperature,
                    eval_mode=True,
                    crop_mode=use_crops,
                    tps_interval=25,
                )
            else:
                raw_text, _ = MODEL.infer_multi(
                    TOKENIZER,
                    image_files=temp_image_paths,
                    output_path=req_out_dir,
                    save_results=True,
                    max_new_tokens=max_tokens,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    temperature=temperature,
                    eval_mode=True,
                    crop_mode=use_crops,
                    tps_interval=25,
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

        latency = round(time.time() - t_start, 2)

    # Convert to Markdown
    try:
        md_text = convert_to_markdown(raw_text)
    except Exception:
        md_text = raw_text

    # Locate generated bounding box image if created by Baidu's _uocr_save_results
    annotated_box_img = None
    box_filename = "result_with_boxes_0.jpg"
    box_path = os.path.join(req_out_dir, box_filename)
    if os.path.exists(box_path):
        annotated_box_img = f"/results/{request_id}/{box_filename}"

    # Clean up PDF temp images if created
    if filename.endswith(".pdf") and temp_image_paths:
        tmp_parent = os.path.dirname(temp_image_paths[0])
        shutil.rmtree(tmp_parent, ignore_errors=True)

    token_count = len(TOKENIZER.encode(raw_text, add_special_tokens=False)) if TOKENIZER else 0
    tok_per_sec = round(token_count / max(0.1, latency), 1)

    return {
        "success": True,
        "request_id": request_id,
        "filename": file.filename,
        "pages_parsed": len(temp_image_paths),
        "latency_sec": latency,
        "tokens": token_count,
        "tokens_per_sec": tok_per_sec,
        "markdown": md_text,
        "raw_layout": raw_text,
        "annotated_image_url": annotated_box_img,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
