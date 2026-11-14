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

## 4. Guardrails: Hallucination, Degeneration, & Timeout Prevention

### A. Fail Loudly — Never Emit Fluent Inventions
Autoregressive vision-language decoders can occasionally lose visual grounding on dense, unpunctuated lists (such as botanical formulations or herb enumerations) and drop into high-frequency conversational Hindi attractor loops (`प्रत्याशन के लिए... प्रतिनिधि के लिए...`).

The server enforces an **active double-layer guardrail**:
1. **Real-time Decoder Repetition Interceptor**: 
   Monitors token generation in real-time. If a token cycle of length 3 to 25 repeats 3 times consecutively, decoding is **immediately halted in ~2 seconds** by forcing an `EOS` token. This completely prevents the model from generating 2048 tokens and guarantees that Cloudflare's 120s timeout (**HTTP 524**) will **never** be triggered.
2. **Server-Side Hallucination & Degeneration Detector**:
   Analyzes the resulting text for n-gram repetition loops, repetition ratios, and script consistency.
   When a loop or intrusion is detected, the server **fails loudly**:
   - `"success": false`
   - `"degenerated": true`
   - `"hallucination_flag": true`
   - `"confidence": 0.20` *(or 0.35)*
   - `"degeneration_reason": "Repetition loop: 3-gram '...' repeated 3 times | Hindi language intrusion under Sanskrit constraint"`
   
   Downstream clients (like OpenPDFSpecs) can safely inspect `success === false` or `confidence < 0.50` and immediately fall back to the existing document text layer, preventing corrupt inventions from entering clinical or scholarly knowledgebases.

### B. Script & Language Constraints (`script=devanagari-sanskrit`)
Pass `script="devanagari-sanskrit"` (or `script="sanskrit"`) to activate strict classical Sanskrit rules:
- **Hindi Intrusion Rejection**: Detects modern conversational Hindi markers (`के लिए`, `प्रतिनिधि`, `करता है`, `होता है`) and flags them as hallucinations.
- **Nukta Stripping**: Classical Sanskrit has no nuktas. Automatically normalizes all Hindi nuktas (`ड़` $\to$ `ड`, `ढ़` $\to$ `ढ`, `क़` $\to$ `क`, `ख़` $\to$ `ख`, `ग़` $\to$ `ग`, `ज़` $\to$ `ज`, `फ़` $\to$ `फ`).
- **Visarga Restoration**: Converts ASCII colons (`:`) following Devanagari text into genuine Visargas (`ः`).
- **Domain Lexicon Disambiguation**: Corrects pervasive minimal-pair OCR confusions in Ayurvedic texts:
  - `ब्रण` / `ब्रणी` $\to$ `व्रण` / `व्रणी` (wound/ulcer)
  - `बातव्याधि` / `बातपित्त` $\to$ `वातव्याधि` / `वातपित्त` (Vata dosha)
  - `ब्याधि` $\to$ `व्याधि` (disease)
  - `बृद्धि` $\to$ `वृद्धि` (enlargement)
  - `बिपाक` $\to$ `विपाक` (post-digestive effect)
  - `बिरेचन` $\to$ `विरेचन` (purgation)
  - `बीर्य` $\to$ `वीर्य` (potency)

---

## 5. Input Resolution & Script Guidance

* **Optimal Resolution**: Empirical benchmarking on classical Indic manuscripts shows that **200 DPI is optimal (94.6% accuracy)**. Rendering higher (e.g., 400–600 DPI) degrades recognition accuracy down to 86.2%. The service defaults to 200 DPI internally.
* **Script Hints (`script`)**:
  * `auto` (Default): Multilingual detection.
  * `devanagari-sanskrit` or `sanskrit`: Strict Sanskrit script constraints, nukta stripping, visarga restoration, domain minimal-pair disambiguation, and conversational Hindi intrusion rejection.
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
    "total_jobs_tracked": 14,
    "timestamp": "2026-11-14T11:00:00Z"
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
  * `script` (String, default: `"auto"`): Language hint: `"devanagari-sanskrit"`, `"sanskrit"`, `"hindi"`, `"auto"`.
  * `fast_mode` (Boolean, default: `true`): `true` uses 273 visual tokens (~15–20s); `false` uses 1,513 tokens for dense fine print.
  * `max_tokens` (Integer, default: `1024`): Maximum generated token length.
  * `no_repeat_ngram_size` (Integer, default: `7`): Repetition suppression n-gram window.
  * `checkpoint` (String, default: `"latest"`): Model checkpoint to pin.
* **Response (Normal Case)**:
  ```json
  {
    "success": true,
    "request_id": "a1b2c3d4",
    "confidence": 0.95,
    "degenerated": false,
    "hallucination_flag": false,
    "degeneration_reason": null,
    "model_version": "unlimited-ocr2-indic-lora-v1",
    "base_model": "baidu/Unlimited-OCR-2",
    "resolved_checkpoint": ".../outputs/unlimited-ocr2-indic-lora",
    "script": "devanagari-sanskrit",
    "filename": "manuscript.pdf",
    "pages_parsed": 1,
    "selected_pages": [1],
    "latency_sec": 16.4,
    "tokens": 420,
    "tokens_per_sec": 25.6,
    "markdown": "त्र्यूषणं धान्यकं चव्यं चित्रकं गजपिप्पली...",
    "regions": [
      {
        "type": "text",
        "box": [95, 57, 380, 81],
        "confidence": 0.96,
        "text": "त्र्यूषणं धान्यकं चव्यं..."
      }
    ]
  }
  ```
* **Response (Degeneration / Hallucination Detected — Fails Loudly)**:
  ```json
  {
    "success": false,
    "request_id": "f9e8d7c6",
    "confidence": 0.20,
    "degenerated": true,
    "hallucination_flag": true,
    "degeneration_reason": "Repetition loop: 3-gram 'प्रत्याशन के लिए' repeated 3 times | Hindi language intrusion under Sanskrit constraint: के लिए, प्रतिनिधि",
    "model_version": "unlimited-ocr2-indic-lora-v1",
    "latency_sec": 2.4,
    "tokens": 35,
    "markdown": "प्रत्याशन के लिए प्रत्याशन के लिए प्रत्याशन के लिए..."
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
    "created_at": "2026-11-14T11:02:00Z",
    "queue_position": 1
  }
  ```

---

### 5. Batch Job Submission (Multiple Documents / Failure Isolation)
Submits multiple documents under a single batch, processing each independently so one difficult page or failure does not fail the entire batch.

* **`POST /v1/jobs/batch`**
* **Status Code**: `202 Accepted`
* **Form Parameters**:
  * `files` (List of Files, required): Multiple PDF or image uploads.
  * `pages` (String, default: `"1"`): Page selection per document.
  * `script` (String, default: `"auto"`): e.g. `"devanagari-sanskrit"`.
* **Response**:
  ```json
  {
    "success": true,
    "batch_id": "batch_98b7c6d5",
    "total_submitted": 3,
    "jobs": [
      { "job_id": "job_111", "filename": "page1.pdf", "status": "queued" },
      { "job_id": "job_222", "filename": "page2.pdf", "status": "queued" },
      { "job_id": "job_333", "filename": "page3.pdf", "status": "queued" }
    ],
    "status_url": "/v1/jobs/batch/batch_98b7c6d5"
  }
  ```

* **`GET /v1/jobs/batch/{batch_id}`**
* **Response**:
  ```json
  {
    "batch_id": "batch_98b7c6d5",
    "total_jobs": 3,
    "completed": 2,
    "failed": 0,
    "processing": 1,
    "queued": 0,
    "is_completed": false,
    "jobs": [ ... ]
  }
  ```

---

### 6. Targeted Region Recognition
Accepts pre-computed bounding boxes and crops individual patches for **+3% higher accuracy** on dense pages:

* **`POST /v1/parse/regions`**
* **Form Parameters**:
  * `file` (File, required): PDF or image.
  * `regions` (String, required): JSON string of bounding boxes, e.g. `[{"x0": 95, "y0": 57, "x1": 380, "y1": 81}]`.
  * `page` (Integer, default: `1`): Target page.
  * `script` (String, default: `"auto"`): `"devanagari-sanskrit"`, `"hindi"`, `"auto"`.
* **Response**:
  ```json
  {
    "success": true,
    "page": 1,
    "model_version": "unlimited-ocr2-indic-lora-v1",
    "script": "devanagari-sanskrit",
    "degenerated": false,
    "hallucination_flag": false,
    "regions_count": 1,
    "regions": [
      {
        "box": [95, 57, 380, 81],
        "text": "त्र्यूषणं धान्यकं चव्यं चित्रकं गजपिप्पली।",
        "confidence": 0.96,
        "degenerated": false,
        "hallucination_flag": false
      }
    ]
  }
  ```

---

## 7. Retraining vs. Runtime Defenses: The Full Strategy

When evaluating whether retraining is necessary, the problem divides into two distinct layers:

### 1. What Runtime Defenses Solve (Available Right Now, Zero Downtime)
- **Preventing HTTP 524 timeouts**: The real-time loop detector cuts off repeating loops in $\approx 2.5\text{s}$ instead of letting decoding run for $126\text{s}$.
- **Failing loudly on hallucinations**: The detector identifies repetitive phrases and Hindi functional intrusions under Sanskrit constraints, returning `"success": false`, `"degenerated": true`, and `"confidence": 0.20`. Clients can safely preserve the existing PDF text layer rather than ingesting hallucinated text.
- **Nukta and Visarga repairs**: Stripping Hindi nuktas and restoring ASCII colons to Visargas is cleanly handled by post-processing.
- **Ayurvedic domain minimal pairs**: Disambiguating `ब्रण` $\to$ `व्रण` and `बात` $\to$ `वात` in medical compounds is resolved by linguistic rules.

### 2. When Retraining IS Required (LoRA V2 Fine-Tuning)
Runtime defenses stop the bleeding and fail safely, but they cannot convert an ungrounded hallucination into ground-truth text. To achieve $\ge 95\%$ accuracy on dense Sanskrit verse lists, a **V2 Retraining** is required:
1. **Synthetic Verse Lists**: 500–1,000 synthesized images of dense, unpunctuated Sanskrit enumerations (herbs, formulations, metrics) without conversational prose structure.
2. **Hard-Negative Minimal Pairs**: Oversampling visually confusable Devanagari character pairs:
   - `ब` vs `व`
   - `त` vs `ल`
   - `र` vs `न`
   - Short `ि` vs Long `ी`
   - `ु` vs `ृ`
3. **Negative Conditioning**: Penalty on high-frequency conversational Hindi tokens in classical Sanskrit contexts.

---

## 8. Client Code Examples

### A. Python Client SDK (`client/indic_ocr.py`)
```python
from indic_ocr import IndicOCRClient

client = IndicOCRClient("http://localhost:8000", api_key="your-api-key")

# 1. Full Page OCR with Sanskrit script hint and degeneration checking
result = client.parse_file("manuscript.pdf", pages="21", script="devanagari-sanskrit")
if not result.success or result.degenerated:
    print(f"Warning: Model degeneration detected ({result.degeneration_reason}). Keeping fallback text layer.")
else:
    print("Markdown:\n", result.markdown)

# 2. Targeted Region OCR (Higher Accuracy on Dense Verses)
boxes = [{"x0": 95, "y0": 57, "x1": 380, "y1": 81}]
region_res = client.parse_regions("manuscript.pdf", regions=boxes, script="devanagari-sanskrit")
print("Cropped text:", region_res["regions"][0]["text"])

# 3. Async Job Pipeline (Multi-Page Books)
job = client.submit_job("book.pdf", pages="1-10", script="devanagari-sanskrit")
completed = job.wait(poll_interval=2.0)
print(f"Finished 10 pages in {completed.latency_sec}s (Conf: {completed.confidence})")

# 4. Batch Submission (Isolated Sequential Processing)
batch = client.submit_batch(["p1.pdf", "p2.pdf", "p3.pdf"], script="devanagari-sanskrit")
print(f"Batch submitted: {batch['batch_id']}, total jobs: {batch['total_submitted']}")
```
