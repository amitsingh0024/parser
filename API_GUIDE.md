# Indic OCR Microservice — API Integration Guide

This guide explains how to run, secure, and integrate the **Baidu Unlimited-OCR-2 Indic Parser** as a standalone microservice in your other applications and pipelines.

---

## 1. Generating an API Key

To generate a cryptographically secure, random 256-bit API key, run any of the following commands:

### Using Python (Recommended — works anywhere):
```bash
python3 -c "import secrets; print('ocr_' + secrets.token_urlsafe(32))"
```
*(Example output: `ocr_8xK2Lm9Qp_4z7vY1aB5cE3wR6tU0sD9fG2hJ4kL6n`)*

### Using OpenSSL (Linux / WSL / macOS):
```bash
openssl rand -hex 32
```

### Using PowerShell (Windows native):
```powershell
-join ((65..90) + (97..122) + (48..57) | Get-Random -Count 40 | ForEach-Object {[char]$_})
```

---

## 2. Setting Up the Microservice

### A. Set the Environment Variable
Set `API_SECRET_KEY` so the server enforces authentication on all `/v1/*` endpoints:

**In Linux / WSL / macOS:**
```bash
export API_SECRET_KEY="your-generated-api-key"
```

**In Windows (PowerShell):**
```powershell
$env:API_SECRET_KEY="your-generated-api-key"
```

*(Note: If `API_SECRET_KEY` is not set, authentication is disabled by default for zero-friction local development).*

### B. Launch the Microservice

**In Linux / WSL (Native Triton enabled):**
```bash
source .venv-wsl/bin/activate
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**In Windows (PowerShell):**
```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**Via Docker:**
```bash
docker compose up --build
```

---

## 3. Authentication Header

When `API_SECRET_KEY` is configured, pass the key using either of these HTTP headers:

* **Header 1 (Recommended)**: `X-API-Key: <your-key>`
* **Header 2 (Standard Bearer)**: `Authorization: Bearer <your-key>`

---

## 4. API Endpoints Reference

### 1. Health & GPU Monitoring
* **`GET /healthz`**
* **Auth**: Public (No auth required — suitable for Kubernetes / Docker liveness probes).
* **Response**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
    "vram_used_gb": 4.82,
    "vram_total_gb": 6.0,
    "active_queue_size": 0,
    "total_jobs_tracked": 12,
    "timestamp": "2026-11-05T13:10:00Z"
  }
  ```

---

### 2. Available Models & Adapters
* **`GET /v1/models`**
* **Auth**: Required
* **Response**:
  ```json
  {
    "current_model": ".../outputs/unlimited-ocr2-indic-lora",
    "checkpoints": [
      { "id": "latest", "name": "Final Production (3-Epoch Fine-Tuned)" },
      { "id": "checkpoint-300", "name": "Checkpoint (checkpoint-300)" },
      { "id": "checkpoint-250", "name": "Checkpoint (checkpoint-250)" }
    ]
  }
  ```

---

### 3. Synchronous Document OCR
Best for single-page documents, forms, receipts, or images where you need the result immediately.

* **`POST /v1/parse`**
* **Content-Type**: `multipart/form-data`
* **Form Parameters**:
  * `file` (File, required): The PDF or image file (`.pdf`, `.png`, `.jpg`, `.jpeg`).
  * `pages` (String, default: `"1"`): Page selection for PDFs (e.g. `"1"`, `"1-3"`, `"all"`).
  * `fast_mode` (Boolean, default: `true`): `true` uses 273 visual tokens (~15–20s); `false` uses 1,513 tokens for dense fine print.
  * `max_tokens` (Integer, default: `2048`): Maximum generated token length.
  * `checkpoint` (String, default: `"latest"`): Model checkpoint to use.
* **Example Response**:
  ```json
  {
    "success": true,
    "request_id": "a1b2c3d4",
    "filename": "question_paper.pdf",
    "pages_parsed": 1,
    "selected_pages": [1],
    "latency_sec": 16.4,
    "tokens": 420,
    "tokens_per_sec": 25.6,
    "markdown": "# 📄 Page 1\n\n1. किस शब्द में 'वि' उपसर्ग नहीं है ?\n(1) विशाखा...",
    "raw_layout": "<PAGE>\n<BLOCK>text[95,57,380,81]<FORK>1. किस शब्द में...",
    "annotated_image_url": "/results/a1b2c3d4/result_with_boxes_0.jpg",
    "annotated_image_urls": ["/results/a1b2c3d4/result_with_boxes_0.jpg"]
  }
  ```

---

### 4. Asynchronous Task Pipeline (Multi-Page Documents)
Best for large multi-page PDFs or background workers. Returns a `job_id` in `<100ms`.

* **`POST /v1/jobs`**
* **Form Parameters**: Same as `/v1/parse`, plus optional:
  * `callback_url` (String, optional): Webhook URL the server will POST to upon job completion.
* **Response**:
  ```json
  {
    "success": true,
    "job_id": "job_e7a192bf80",
    "status": "queued",
    "created_at": "2026-11-05T13:12:00Z",
    "queue_position": 1
  }
  ```

---

### 5. Query Job Status & Progress
* **`GET /v1/jobs/{job_id}`**
* **Response (While Processing)**:
  ```json
  {
    "job_id": "job_e7a192bf80",
    "status": "processing",
    "progress": {
      "current_page": 2,
      "total_pages": 5,
      "percentage": 40.0
    }
  }
  ```
* **Response (When Completed)**:
  ```json
  {
    "job_id": "job_e7a192bf80",
    "status": "completed",
    "created_at": "2026-11-05T13:12:00Z",
    "completed_at": "2026-11-05T13:13:15Z",
    "progress": { "current_page": 5, "total_pages": 5, "percentage": 100.0 },
    "result": {
      "markdown": "### 📄 Page 1\n...\n---\n### 📄 Page 2\n...",
      "latency_sec": 75.3,
      "tokens": 2150,
      "annotated_image_urls": [...]
    }
  }
  ```

---

## 5. Client Code Examples

### A. Python Client (Zero Dependencies)
Copy `client/indic_ocr.py` into your project:

```python
from indic_ocr import IndicOCRClient

client = IndicOCRClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# 1. Synchronous OCR (Page 1)
result = client.parse("sample.pdf", pages="1", fast_mode=True)
print(result.markdown)
result.save_markdown("output.md")

# 2. Asynchronous Job (Pages 1-5)
job = client.submit_job("booklet.pdf", pages="1-5", fast_mode=True)
print(f"Submitted Job ID: {job.job_id}")

# Wait for completion (automatically polls)
completed = job.wait(poll_interval=2.0)
print(f"Finished in {completed.latency_sec}s!")
print(completed.markdown)
```

---

### B. Python via standard `requests`
```python
import time
import requests

headers = {"X-API-Key": "your-api-key"}

# Submit async job
with open("paper.pdf", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/jobs",
        headers=headers,
        files={"file": f},
        data={"pages": "1-3", "fast_mode": "true"}
    )
job_id = resp.json()["job_id"]

# Poll for result
while True:
    status_resp = requests.get(f"http://localhost:8000/v1/jobs/{job_id}", headers=headers).json()
    if status_resp["status"] == "completed":
        print(status_resp["result"]["markdown"])
        break
    elif status_resp["status"] == "failed":
        print("Job failed:", status_resp.get("error"))
        break
    time.sleep(2)
```

---

### C. Node.js / TypeScript (`axios` & `form-data`)
```typescript
import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';

const client = axios.create({
  baseURL: 'http://localhost:8000',
  headers: { 'X-API-Key': 'your-api-key' }
});

async function parseDocument() {
  const form = new FormData();
  form.append('file', fs.createReadStream('sample.pdf'));
  form.append('pages', '1');
  form.append('fast_mode', 'true');

  const { data } = await client.post('/v1/parse', form, {
    headers: form.getHeaders()
  });

  console.log('Markdown output:\n', data.markdown);
}

parseDocument();
```

---

### D. cURL (Command Line)
```bash
# Synchronous Parse
curl -X POST http://localhost:8000/v1/parse \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf" \
  -F "pages=1" \
  -F "fast_mode=true"

# Asynchronous Job Submission with Webhook
curl -X POST http://localhost:8000/v1/jobs \
  -H "X-API-Key: your-api-key" \
  -F "file=@booklet.pdf" \
  -F "pages=1-5" \
  -F "fast_mode=true" \
  -F "callback_url=https://my-app.com/api/ocr-callback"
```

---

## 6. Webhook Format

If you provide a `callback_url` when submitting a job, the microservice will send an HTTP `POST` to that URL once parsing finishes:

```json
{
  "job_id": "job_e7a192bf80",
  "status": "completed",
  "created_at": "2026-11-05T13:12:00Z",
  "completed_at": "2026-11-05T13:13:15Z",
  "parameters": {
    "filename": "booklet.pdf",
    "pages": "1-5",
    "fast_mode": true
  },
  "result": {
    "filename": "booklet.pdf",
    "pages_parsed": 5,
    "markdown": "...",
    "annotated_image_urls": [...]
  }
}
```
