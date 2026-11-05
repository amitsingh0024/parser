# Indic OCR Microservice — API Integration Guide

This guide explains how to run, secure, and integrate the **Baidu Unlimited-OCR-2 Indic Parser** as a standalone microservice in production systems and pipelines.

---

## 1. Generating an API Key

To generate a cryptographically secure 256-bit API key, run any of the following:

### Using Python:
```bash
python -c "import secrets; print('ocr_' + secrets.token_urlsafe(32))"
```
*(Example output: `ocr_8xK2Lm9Qp_4z7vY1aB5cE3wR6tU0sD9fG2hJ4kL6n`)*

### Using OpenSSL:
```bash
openssl rand -hex 32
```

### Using PowerShell:
```powershell
-join ((65..90) + (97..122) + (48..57) | Get-Random -Count 40 | ForEach-Object {[char]$_})
```

---

## 2. Setting Up the Microservice

### A. Environment Configuration (`.env`)
Save your key in a `.env` file in the project root:
```ini
API_SECRET_KEY=ocr_8xK2Lm9Qp_4z7vY1aB5cE3wR6tU0sD9fG2hJ4kL6n
```
The server automatically loads `.env` on boot. *(If `API_SECRET_KEY` is empty or unset, authentication is disabled for zero-friction local development).*

### B. Launching the Microservice

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

## 3. Permanent Stable Hostname (Production Cloudflare Named Tunnel)

If you are using Cloudflare Tunnels for remote hosting, ephemeral quick tunnels (`*.trycloudflare.com`) rotate their hostname on every restart. 

To give your service a **permanent, static domain** (e.g. `ocr.yourdomain.com` or a persistent Cloudflare URL):

1. **Login to Cloudflare in WSL/Linux:**
   ```bash
   cloudflared tunnel login
   ```
2. **Create a persistent named tunnel:**
   ```bash
   cloudflared tunnel create indic-ocr-tunnel
   ```
3. **Route DNS to your domain:**
   ```bash
   cloudflared tunnel route dns indic-ocr-tunnel ocr.yourdomain.com
   ```
4. **Run the persistent tunnel:**
   ```bash
   cloudflared tunnel run --url http://127.0.0.1:8000 indic-ocr-tunnel
   ```
   *Your URL `https://ocr.yourdomain.com` will now remain static indefinitely across all server and machine restarts.*

---

## 4. Authentication & Idempotency Headers

* **Authentication (pass either)**:
  * `X-API-Key: <your-key>`
  * `Authorization: Bearer <your-key>`
* **Idempotency**:
  * `Idempotency-Key: <uuid-or-unique-string>`
  * If a request with an existing idempotency key is repeated within 24 hours, the service immediately returns the cached result in `<5ms` without re-running GPU inference.

---

## 5. Input Resolution & Script Guidance

* **Optimal Resolution**: Empirical benchmarking on classical Indic manuscripts shows that **200 DPI is optimal (94.6% accuracy)**. Rendering higher (e.g., 400–600 DPI) degrades recognition accuracy down to 86.2%. The service defaults to 200 DPI internally.
* **Script Hints (`script`)**:
  * `auto` (Default): Multilingual detection.
  * `sanskrit`: Strips Devanagari nuktas (which never occur in Sanskrit orthography), repairs ASCII Visarga colons (`:` $\to$ `ः`), and disambiguates domain terms (e.g. `व्रण`).
  * `hindi`: Preserves Hindi/Urdu nuktas (`ड़`, `ढ़`, `ज़`, `फ़`).
  * `english`: Restores dropped ligature characters (`ffi`, `ff`, `fl`).

---

## 6. API Endpoints Reference

### 1. Health & Observability Probe
* **`GET /healthz`**
* **Auth**: Public (No auth required — ideal for Docker/K8s liveness checks).
* **Response**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "model_version": "unlimited-ocr2-indic-lora-v1",
    "base_model": "baidu/Unlimited-OCR-2",
    "resolved_checkpoint": ".../outputs/unlimited-ocr2-indic-lora",
    "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
    "vram_used_gb": 2.76,
    "vram_total_gb": 6.0,
    "active_queue_size": 0,
    "total_jobs_tracked": 12,
    "timestamp": "2026-11-05T13:10:00Z"
  }
  ```

---

### 2. Model Registry
* **`GET /v1/models`**
* **Auth**: Required
* **Response**:
  ```json
  {
    "model_version": "unlimited-ocr2-indic-lora-v1",
    "base_model": "baidu/Unlimited-OCR-2",
    "current_model": ".../outputs/unlimited-ocr2-indic-lora",
    "checkpoints": [
      { "id": "latest", "name": "Final Production (3-Epoch Fine-Tuned)" },
      { "id": "checkpoint-300", "name": "Checkpoint (checkpoint-300)" },
      { "id": "checkpoint-250", "name": "Checkpoint (checkpoint-250)" }
    ]
  }
  ```

---

### 3. Synchronous Document Parsing
Processes single-page documents, images, or forms and returns results immediately.

* **`POST /v1/parse`**
* **Headers**: `X-API-Key`, `Idempotency-Key` (optional)
* **Form Parameters**:
  * `file` (File, required): PDF or image (`.pdf`, `.png`, `.jpg`, `.webp`).
  * `pages` (String, default: `"1"`): Page selection (e.g. `"1"`, `"1-3"`, `"all"`).
  * `script` (String, default: `"auto"`): Language hint: `"sanskrit"`, `"hindi"`, `"auto"`.
  * `fast_mode` (Boolean, default: `true`): `true` uses 273 visual tokens (~15–20s); `false` uses 1,513 tokens for dense fine print.
  * `max_tokens` (Integer, default: `2048`): Maximum generated token length.
  * `checkpoint` (String, default: `"latest"`): Model checkpoint to pin.
* **Response**:
  ```json
  {
    "success": true,
    "request_id": "a1b2c3d4",
    "model_version": "unlimited-ocr2-indic-lora-v1",
    "base_model": "baidu/Unlimited-OCR-2",
    "resolved_checkpoint": ".../outputs/unlimited-ocr2-indic-lora",
    "script": "sanskrit",
    "filename": "manuscript.pdf",
    "pages_parsed": 1,
    "selected_pages": [1],
    "latency_sec": 16.4,
    "tokens": 420,
    "tokens_per_sec": 25.6,
    "markdown": "### 📄 Page 1\n\n1. किस शब्द में 'वि' उपसर्ग नहीं है ?\n(1) विशाखा...",
    "raw_layout": "<PAGE><BLOCK>text[95,57,380,81]<FORK>1. किस शब्द में...",
    "regions": [
      {
        "type": "text",
        "box": [95, 57, 380, 81],
        "confidence": 0.96,
        "text": "1. किस शब्द में 'वि' उपसर्ग नहीं है ?"
      }
    ],
    "annotated_image_urls": ["/results/a1b2c3d4/result_with_boxes_0.jpg"]
  }
  ```

---

### 4. Asynchronous Task Pipeline (Multi-Page Documents)
Enqueues document for GPU worker and returns `202 Accepted` in `<100ms`.

* **`POST /v1/jobs`**
* **Status Code**: `202 Accepted`
* **Form Parameters**: Same as `/v1/parse`, plus:
  * `callback_url` (String, optional): Webhook URL called upon job completion.
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
* **Response (In Progress)**:
  ```json
  {
    "job_id": "job_e7a192bf80",
    "status": "processing",
    "progress": {
      "pages_done": 2,
      "pages_total": 5,
      "current_page": 2,
      "total_pages": 5,
      "percentage": 40.0
    }
  }
  ```
* **Response (Completed)**:
  ```json
  {
    "job_id": "job_e7a192bf80",
    "status": "completed",
    "created_at": "2026-11-05T13:12:00Z",
    "completed_at": "2026-11-05T13:13:15Z",
    "progress": { "pages_done": 5, "pages_total": 5, "percentage": 100.0 },
    "result": {
      "markdown": "### 📄 Page 1\n...\n---\n### 📄 Page 2\n...",
      "latency_sec": 75.3,
      "tokens": 2150,
      "regions": [...],
      "annotated_image_urls": [...]
    }
  }
  ```

---

### 6. Targeted Region Recognition
Accepts pre-computed bounding boxes and crops individual patches for **+3% higher accuracy** on dense pages:

* **`POST /v1/parse/regions`**
* **Form Parameters**:
  * `file` (File, required): PDF or image.
  * `regions` (String, required): JSON string of bounding boxes, e.g. `[{"x0": 95, "y0": 57, "x1": 380, "y1": 81}]`. Coordinates can be normalized (0..1000) or pixel dimensions.
  * `page` (Integer, default: `1`): Target page.
  * `script` (String, default: `"auto"`): `"sanskrit"`, `"hindi"`, `"auto"`.
* **Response**:
  ```json
  {
    "success": true,
    "page": 1,
    "model_version": "unlimited-ocr2-indic-lora-v1",
    "regions_count": 1,
    "regions": [
      {
        "box": [95, 57, 380, 81],
        "text": "1. किस शब्द में 'वि' उपसर्ग नहीं है ?",
        "confidence": 0.96
      }
    ]
  }
  ```

---

## 7. Error Semantics & Operating Limits

| HTTP Status | Semantics | Client Action |
| :--- | :--- | :--- |
| **`200 OK` / `202 Accepted`** | Success / Job Enqueued | Process result or begin polling. |
| **`400 Bad Request`** | Invalid file format, corrupted PDF, or bad page range. | **Fatal**. Do not retry without modifying parameters. |
| **`401 Unauthorized`** | Missing or incorrect API key. | **Fatal**. Verify `X-API-Key` header. |
| **`404 Not Found`** | Job ID does not exist. | **Fatal**. Check `job_id`. |
| **`422 Unprocessable`** | Form schema validation failure. | **Fatal**. Check parameter names. |
| **`429 Too Many Requests`** | GPU queue capacity saturated. | **Retryable**. Back off using `Retry-After` header. |
| **`500 Internal Error`** | Transient inference or worker error. | **Retryable**. Exponential back-off. |

### Operational Limits:
- **Maximum File Size**: 100 MB.
- **Maximum Synchronous Pages**: 5 pages per `/v1/parse` request (for $>5$ pages, always use `/v1/jobs` to prevent HTTP gateway timeouts).
- **GPU Queue Concurrency**: Automatically serialized via async worker locks to guarantee zero Out-of-Memory (OOM) crashes on 6 GB VRAM GPUs.

---

## 8. Client Code Examples

### A. Python Client SDK (`client/indic_ocr.py`)
```python
from indic_ocr import IndicOCRClient

client = IndicOCRClient("http://localhost:8000", api_key="your-api-key")

# 1. Full Page OCR with Sanskrit script hint
result = client.parse("manuscript.pdf", pages="1", script="sanskrit")
print("Markdown:\n", result.markdown)
for region in result.regions:
    print(f"Box {region['box']} (Conf: {region['confidence']}): {region['text']}")

# 2. Targeted Region OCR (Higher Accuracy on Dense Verses)
boxes = [{"x0": 95, "y0": 57, "x1": 380, "y1": 81}]
region_res = client.parse_regions("manuscript.pdf", regions=boxes, script="sanskrit")
print("Cropped text:", region_res["regions"][0]["text"])

# 3. Async Job Pipeline (Multi-Page Books)
job = client.submit_job("book.pdf", pages="1-10", script="sanskrit")
completed = job.wait(poll_interval=2.0)
print(f"Finished 10 pages in {completed.latency_sec}s!")
```

### B. Node.js / TypeScript (`axios`)
```typescript
import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';

const client = axios.create({
  baseURL: 'http://localhost:8000',
  headers: { 'X-API-Key': 'your-api-key' }
});

async function parseBooklet() {
  const form = new FormData();
  form.append('file', fs.createReadStream('booklet.pdf'));
  form.append('pages', '1-5');
  form.append('script', 'sanskrit');

  const { data } = await client.post('/v1/jobs', form, {
    headers: form.getHeaders()
  });

  console.log(`Job enqueued: ${data.job_id}`);
}
```

### C. cURL (Command Line)
```bash
# Synchronous with Sanskrit Hint & Idempotency
curl -X POST http://localhost:8000/v1/parse \
  -H "X-API-Key: your-api-key" \
  -H "Idempotency-Key: req_001_p1" \
  -F "file=@manuscript.pdf" \
  -F "pages=1" \
  -F "script=sanskrit"

# Targeted Region Parsing
curl -X POST http://localhost:8000/v1/parse/regions \
  -H "X-API-Key: your-api-key" \
  -F "file=@manuscript.pdf" \
  -F 'regions=[{"x0": 95, "y0": 57, "x1": 380, "y1": 81}]' \
  -F "script=sanskrit"
```
