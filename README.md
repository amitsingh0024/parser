<div align="center">

<img src="assets/logo.png" width="40%" alt="Baidu Inc." />

</div>

<hr>

<h1 align="center">Unlimited OCR Works</h1>

<div align="center">
<a href="https://trendshift.io/repositories/62053?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-62053" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/62053/daily" alt="baidu%2FUnlimited-OCR | Trendshift" width="250" height="55"/></a>
</div>

<div align="center">
  <a href="https://github.com/baidu/Unlimited-OCR">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white" />
  </a>
  <a href="https://huggingface.co/baidu/Unlimited-OCR">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Unlimited--OCR-ffc107?color=ffc107&logoColor=white" />
  </a>
  <a href="https://huggingface.co/baidu/Unlimited-OCR-2">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Unlimited--OCR--2-ffc107?color=ffc107&logoColor=white" />
  </a>
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2606.23050">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Unlimited OCR Works-b31b1b?logo=arxiv&logoColor=white" />
  </a>
    <a href="https://arxiv.org/abs/2610.39308">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Unlimited OCR Works 2-b31b1b?logo=arxiv&logoColor=white" />
  </a>
  <a href="https://x.com/Baidu_Inc" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-Baidu Inc.-white?logo=x&logoColor=white" />
  </a>
</div>

<h3 align="center">Pioneering the OCR-3.0 Era: From Long- to Unlimited-Horizon Parsing.</h3>

<div align="center">

<img src="assets/Unlimited-OCR2.png" width="600" alt="Unlimited OCR 2 overview" />

</div>

---

## Local Training & Inference Guide (Hindi, Sanskrit & English — Windows / 6 GB VRAM)

This repository is configured to fine-tune and run **Baidu Unlimited-OCR-2** (`baidu/Unlimited-OCR-2`, 3B parameters) locally on a consumer Windows GPU (e.g., **NVIDIA GeForce RTX 4050 6 GB VRAM**) using **4-bit NF4 QLoRA**, with datasets and pipelines built for **Hindi, Sanskrit (Devanagari), and English** documents.

### 1. Project Structure

```text
baidu-parser/
├── .venv/                                    # Local Python 3.11 + CUDA 12.8 virtual environment
├── requirements.txt                          # Pinned dependencies for Unlimited-OCR-2 + QLoRA
├── train.py                                  # Single-GPU 4-bit QLoRA training script with live tqdm progress bar
├── local_infer.py                            # Local Windows Transformers inference (images, PDFs, LoRA adapters)
├── infer.py                                  # Upstream batch SGLang client for MPDocBench / MarathonDocBench
├── scripts/
│   ├── download_and_prepare_datasets.py      # Downloads HF Sanskrit/Devanagari datasets & builds trilingual pages
│   └── pdf_to_dataset.py                     # Converts custom digital PDFs into Unlimited-OCR-2 training JSONL
├── utils/
│   ├── process_unlimitedOCR2.py              # Converts raw <PAGE><BLOCK>...<FORK> output to clean Markdown
│   └── README.md                             # Post-processing & layout block specification
└── data/
    ├── sample_train.jsonl                    # Minimal 2-sample reference JSONL
    ├── raw_hf/                               # Downloaded Hugging Face Sanskrit & Devanagari OCR datasets
    │   ├── Process-Venue__Sanskrit-OCR-Typed-Dataset/
    │   └── Malathip72__devanagari-ocr-dataset/
    └── unlimited_ocr/                        # Ready-to-train Unlimited-OCR-2 dataset (250 documents)
        ├── images/                           # Full-page composite & structured trilingual page renders
        ├── train.jsonl                       # 225 training documents (single-page & multi-page)
        ├── val.jsonl                         # 25 validation documents
        └── dataset_stats.json                # Dataset summary counts
```

---

### 2. Environment Setup

If setting up from scratch on Windows (PowerShell):

```powershell
# 1. Create and activate the virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install CUDA-enabled PyTorch (CUDA 12.8)
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install project & QLoRA dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. Verify GPU detection
.\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

### 3. Dataset Format & Preparation (Hindi, Sanskrit & English)

#### A. Unlimited-OCR-2 Ground-Truth Schema
Each line in `train.jsonl` / `val.jsonl` is a JSON object:
```json
{
  "images": ["images/trilingual_docs/structured_multi_0001_p1.jpg", "images/trilingual_docs/structured_multi_0001_p2.jpg"],
  "prompt": "<PAGE><image><PAGE><image>Multi page parsing.",
  "response": "<PAGE><BLOCK>title<LEVEL1>[76,57,924,105]<FORK>अध्यायः १ — श्रीमद्भगवद्गीता उपदेशः\n<BLOCK>text[76,125,924,260]<FORK>धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः...\n<BLOCK>cross_text[76,720,924,910]<FORK>Paragraph starting on page 1...<PAGE><BLOCK>append_text[76,57,924,210]<FORK>...and continuing on page 2."
}
```
* **Coordinates**: `[x1, y1, x2, y2]` are normalized to the `0–1000` range relative to page width and height.
* **Block Labels**: `title<LEVEL1..n>`, `text`, `table` (HTML `<table>...</table>`), `equation`, `image`, `figure`, `chart`, `header`, `footer`, `page_number`.
* **Cross-Page Continuations**: A block split across page $p$ and $p+1$ uses `cross_text` / `cross_table` at the end of page $p$ and `append_text` / `append_table` at the start of page $p+1$.

#### B. Rebuild or Expand the Included Datasets
To re-run or increase the sample count from the Hugging Face Sanskrit & Devanagari datasets and trilingual document generator:
```powershell
.\.venv\Scripts\python.exe scripts/download_and_prepare_datasets.py --hf-max-samples 300
```

#### C. Convert Your Own Digital PDFs (Hindi / Sanskrit / English) into Training Data
Drop any digital PDFs (with embedded text) into a folder (e.g., `my_pdfs/`) and run:
```powershell
.\.venv\Scripts\python.exe scripts/pdf_to_dataset.py --pdf-dir my_pdfs --out-dir data/custom_pdfs --dpi 180
```
This extracts page images (`data/custom_pdfs/images/*.jpg`), normalizes bounding boxes to `0–1000`, builds both single-page and multi-page sliding-window samples, and writes `data/custom_pdfs/train.jsonl`.

---

### 4. Fine-Tuning (`train.py`)

#### A. Run 4-Bit QLoRA Training on 6 GB VRAM
```powershell
.\.venv\Scripts\python.exe train.py `
  --train-data data/unlimited_ocr/train.jsonl `
  --eval-data data/unlimited_ocr/val.jsonl `
  --output-dir outputs/unlimited-ocr2-indic-lora `
  --batch-size 1 `
  --grad-accum 8 `
  --max-length 2048 `
  --lora-r 16 `
  --lora-alpha 32 `
  --lr 2e-4 `
  --epochs 3 `
  --save-steps 50 `
  --logging-steps 1
```

#### B. Live Progress Bar & Monitoring
During training, `train.py` displays:
1. **Dataset loading progress bar** (`doc/s`) when reading `train.jsonl` and `val.jsonl`.
2. **Live `tqdm` training bar** updated every step (`--logging-steps 1`) showing:
   * Current `Step / Total Steps`, percentage bar, elapsed time, and **ETA**
   * `epoch` (e.g., `1.20/3`)
   * `loss` (training loss) & `val_loss` (validation loss)
   * `lr` (current learning rate)
   * `vram` (live GPU memory allocated/reserved vs total 6.0 GB)
3. **Persistent JSONL log**: Every step is appended to `<output-dir>/training_progress.jsonl`. You can inspect it anytime from another terminal:
   ```powershell
   Get-Content outputs\unlimited-ocr2-indic-lora\training_progress.jsonl -Wait -Tail 10
   ```

#### C. `train.py` CLI Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model-name-or-path` | `baidu/Unlimited-OCR-2` | Hugging Face model ID or local checkpoint directory |
| `--train-data` | *(Required)* | Path to training `.jsonl` file |
| `--eval-data` | `None` | Optional path to validation `.jsonl` file |
| `--output-dir` | `outputs/unlimited-ocr2-lora` | Directory to save LoRA adapters, checkpoints, and `training_progress.jsonl` |
| `--max-length` | `4096` | Max sequence token length (use `2048` on 6 GB VRAM for Devanagari pages) |
| `--use-4bit` / `--no-4bit` | `--use-4bit` | Enable/disable 4-bit NF4 QLoRA quantization via `bitsandbytes` |
| `--gradient-checkpointing` | `True` | Recompute activations during backward pass to save VRAM |
| `--lora-r` | `16` | LoRA rank (`8`, `16`, or `32`) |
| `--lora-alpha` | `32` | LoRA scaling alpha |
| `--lora-target-modules` | `q_proj,k_proj,v_proj,o_proj` | Comma-separated modules to adapt (can add `gate_proj,up_proj,down_proj`) |
| `--batch-size` | `1` | Per-device batch size (keep at `1` for 6 GB VRAM) |
| `--grad-accum` | `8` | Gradient accumulation steps (effective batch size = `batch_size * grad_accum`) |
| `--lr` | `2e-4` | Peak learning rate |
| `--epochs` | `3` | Total training epochs |
| `--save-steps` | `50` | Save LoRA checkpoint every $N$ steps |
| `--logging-steps` | `1` | Update live `tqdm` progress bar and log file every $N$ steps |

---

### 5. Running Local Inference (`local_infer.py`)

`local_infer.py` runs `baidu/Unlimited-OCR-2` directly on Windows via Hugging Face `transformers` (with 4-bit quantization enabled by default for 6 GB VRAM) and automatically converts the raw `<PAGE><BLOCK>...<FORK>` output into clean Markdown using `utils/process_unlimitedOCR2.py`.

```powershell
# 1. Single-page image inference (base model)
.\.venv\Scripts\python.exe local_infer.py `
  --images data/unlimited_ocr/images/trilingual_docs/structured_single_0000.jpg `
  --out-dir outputs/infer_single

# 2. Multi-page image inference (cross-page context preserved)
.\.venv\Scripts\python.exe local_infer.py `
  --images data/unlimited_ocr/images/trilingual_docs/structured_multi_0000_p1.jpg data/unlimited_ocr/images/trilingual_docs/structured_multi_0000_p2.jpg `
  --out-dir outputs/infer_multi

# 3. Direct PDF inference with your fine-tuned LoRA adapter
.\.venv\Scripts\python.exe local_infer.py `
  --pdf your_document.pdf `
  --lora-path outputs/unlimited-ocr2-indic-lora `
  --out-dir outputs/infer_pdf
```
Outputs written to `--out-dir`:
* `raw_output.txt` — Raw layout blocks (`<PAGE><BLOCK>...[x1,y1,x2,y2]<FORK>...`)
* `output.md` — Final merged Markdown with headings, HTML tables, and cross-page text stitched together.

---

## Release
- [2026/11/02] 📄 The paper of **Unlimited OCR 2** is now available on [arXiv](https://arxiv.org/abs/2610.39308).
- [2026/10/28] Thanks to the [ms-swift community](https://github.com/modelscope/ms-swift) for supporting training of **Unlimited OCR 2**.
- [2026/10/27] 📊 We release [**MarathonDocBench**](https://huggingface.co/datasets/baidu/MarathonDocBench), a new benchmark for ultra-long document parsing.
- [2026/10/27] 🤝 Thanks to the [ModelScope community](https://github.com/modelscope) for their support. Our model is now available at [ModelScope](https://modelscope.cn/models/PaddlePaddle/Unlimited-OCR-2).
- [2026/10/27] 🤝 Thanks to the Baidu Cloud team for their support. Our model is now available on [Baidu Cloud](https://cloud.baidu.com/doc/OCR/s/fmr1p39gb).
- [2026/10/27] 🤝 Thanks to [AK](https://x.com/_akhaliq) for creating a new demo for us. Try it at [Unlimited-OCR-2](https://huggingface.co/spaces/baidu/Unlimited-OCR-2).
- [2026/10/27] 🚀 We release [**Unlimited OCR 2**](https://github.com/baidu/Unlimited-OCR), achieving unlimited-horizon parsing with SR-SWA.
- [2026/07/21] 🤝 Thanks to the [ms-swift community](https://github.com/modelscope/ms-swift) for their support, our model now supports training with [ms-swift](https://github.com/modelscope/ms-swift).
- [2026/07/03] 🤝 Thanks to the Baidu Cloud team for their support. Our model is now available on [Baidu Cloud](https://cloud.baidu.com/doc/OCR/s/fmr1p39gb).
- [2026/06/28] 🤝 Thanks to the [vLLM community](https://github.com/vllm-project/vllm) and [Tianyu Guo](https://github.com/gty111) for their support, our model now supports vLLM inference.
- [2026/06/24] 🤝 Thanks to [AK](https://x.com/_akhaliq) for creating a demo for us. It is now available at [Hugging Face Spaces](https://huggingface.co/spaces/baidu/Unlimited-OCR).
- [2026/06/23] 📄 Our paper is now available on [arXiv](https://arxiv.org/abs/2606.23050).
- [2026/06/23] 🤝 Thanks to the [ModelScope community](https://github.com/modelscope) for their support. Our model is now available at [ModelScope](https://modelscope.cn/models/PaddlePaddle/Unlimited-OCR).
- [2026/06/22] 🚀 We present [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR/tree/UnlimitedOCR1.0), aiming to push [Deepseek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) one step further.

## What's New in 2
* Sparse Reference Sliding Window Attention (SR-SWA): Compresses text-visual attention interactions from ultra-long sequences down to a fixed-size set of working references (e.g., 512 tokens). This bounds computational complexity, keeping per-step decoding latency and memory footprint nearly constant regardless of document length.
* From Long- to Unlimited-Horizon Parsing: Powered by SR-SWA, Unlimited-OCR 2 achieves true, unbounded document parsing within a finite context window—effectively eliminating quadratic attention scaling and memory bottlenecks as documents grow.
* Ultra-Long Document Parsing SOTA: Achieves state-of-the-art performance across both dozens-of-pages benchmarks (e.g., MPDocBench) and extreme hundred-page benchmarks (e.g., MarathonDocBench).

<div align="center">
  <img src="assets/leaderboard.png" width="1000" alt="Performance" />
</div>

## Inference

> Unlimited OCR 2 supports **Transformers** (native, for quick tests) and **SGLang** (recommended for serving).

### Transformers

Tested with Python 3.10, `torch==2.13.0`, `transformers==4.52.1`, `huggingface-hub==0.36.0`, `tokenizers==0.21.4`, on an NVIDIA A800-80GB:
```shell
pip install torch==2.13.0 torchvision pillow transformers==4.52.1 huggingface-hub==0.36.0 \
  tokenizers==0.21.4 addict==2.4.0 einops==0.8.2 easydict==1.13
```

Load the model once:
```python
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "baidu/Unlimited-OCR-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True,
                                  torch_dtype=torch.bfloat16).eval().cuda()
```

**Multi-page inference** (`infer_multi`). All pages of a document are parsed in one request, so the model can use cross-page context. Pass the page images in page order:
```python
pages = [f"your_doc/page_{i}.jpg" for i in range(1, 8)]

text, num_tokens = model.infer_multi(tokenizer, image_files=pages,
                                     output_path="output/multi", save_results=True)
```
The prompt is built automatically as `"<PAGE><image>" * num_pages + "Multi page parsing."`, and each page is encoded as one global view (base mode). The output contains one `<PAGE>` per input page, in the same order as `image_files`.

**Single-page inference** (`infer`). A single page is encoded with dynamic tiling (gundam mode) by default:
```python
text = model.infer(tokenizer, image_file="your_doc/page_1.jpg",
                   output_path="output/single", save_results=True)
```


### SGLang

Set up the environment (uv-managed virtualenv, Python 3.10) and install the customized SGLang wheel
`sglang-0.5.20.dev13+g75b9b06b02-py3-none-any.whl`, plus PyMuPDF for PDF-to-image conversion:
```shell
uv venv --python 3.10
source .venv/bin/activate

uv pip install wheel/sglang-0.5.20.dev13+g75b9b06b02-py3-none-any.whl
uv pip install pymupdf==1.27.2.2
uv pip install fastokens==0.3.2
```

Start the SGLang server:
```shell
# Page-window scheduling for very long documents (the KV tail of the previous window is reused).
export SGLANG_UOCR_EANABLE_DOC_LIMIT=1          # 1 = enable, 0 = whole document in one window
export SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW=10    # pages per window (gundam mode)
export SGLANG_UOCR_DOC_BASE_PAGE_WINDOW=10      # pages per window (base mode)
export SGLANG_UOCR_MAX_PAGE_TOKENS=20000        # per-page output cap; must be > 0
# Chunk cap for the SAM attention bias. The built-in default (512) allows a single
# ~462 MiB allocation and can OOM; 64 is equivalent with a lower peak.
export SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB=64
# Pass large crop tensors through /dev/shm instead of pickling them into the
# tokenizer -> scheduler frame.
export SGLANG_UOCR_SHM_WRAP_MODEL_DATA=1
export SGLANG_UOCR_SHM_WRAP_MIN_MB=1
export SGLANG_VLM_CACHE_SIZE_MB=0
# Reduces fragmentation failures on the large contiguous allocations.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m sglang.launch_server \
    --model-path baidu/Unlimited-OCR-2 \
    --served-model-name Unlimited-OCR-2 \
    --chat-template unlimited-ocr \
    --tokenizer-backend fastokens \
    --tokenizer-worker-num 8 \
    --mem-fraction-static 0.7 \
    --max-running-requests 64 \
    --max-prefill-tokens 2048 \
    --enable-custom-logit-processor \
    --enable-streaming-session \
    --allow-auto-truncate \
    --disable-radix-cache \
    --mm-preprocess-cache-size-mb 0 \
    --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 64 \
    --cuda-graph-backend-prefill full \
    --cuda-graph-max-bs-prefill 16384 \
    --cuda-graph-bs-prefill 1024 2048 \
    --cuda-graph-config '{"prefill": {"backend": "full", "full_prefill_max_req": 64}}' \
    --host 0.0.0.0 \
    --port 10000
```

Notes on the knobs that matter for memory:
* `--mem-fraction-static` is the share reserved for weights + KV. `0.75` leaves too little headroom at 32+ concurrent requests; `0.7` is stable.
* `--max-prefill-tokens` drives a temporary tensor of roughly `[tokens, context/32]`, so memory grows linearly with it: `4096 -> 2048` halves that peak.
* The largest `--cuda-graph-bs-prefill` bucket must equal `--max-prefill-tokens`. A smaller one makes oversized batches fall back to eager; a larger one is never used and only wastes capture memory.
* `--cuda-graph-config`'s `full_prefill_max_req` is the request-slot count for the full-prefill CUDA graph. The default (8) makes 16/32 concurrency fall back to eager.
* A smaller page window lowers peak memory but hurts cross-page merging, since blocks that continue across a window boundary lose their context.

To parse a whole document in one window instead (`SGLANG_UOCR_EANABLE_DOC_LIMIT=0`).

Send requests to the OpenAI-compatible API:


```python
import base64
import json
import os
import tempfile

import fitz
import requests
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor

server_url = "http://127.0.0.1:10000"
NGRAM_PROCESSOR = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()

session = requests.Session()
session.trust_env = False


def pdf_to_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    image_paths = []
    for i, page in enumerate(doc):
        image_path = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(image_path)
        image_paths.append(image_path)
    doc.close()
    return image_paths


def encode_image(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def build_content(image_paths):
    prompt = "<PAGE><image>" * len(image_paths) + "Multi page parsing."
    return [encode_image(path) for path in image_paths] + [{"type": "text", "text": prompt}]


def generate(image_paths, crop_mode=True, tokens_per_page=8000):
    # A wider n-gram window for multi-page documents, as in our evaluation setup.
    window_size = 1024 if len(image_paths) > 1 else 128
    payload = {
        "model": "Unlimited-OCR-2",
        "messages": [{"role": "user", "content": build_content(image_paths)}],
        # Must be positive: it is the global output budget for the whole document.
        "max_tokens": tokens_per_page * len(image_paths),
        "temperature": 0,
        "skip_special_tokens": False,
        "images_config": {"crop_mode": crop_mode},  # True = gundam (tiles), False = base
        "custom_logit_processor": NGRAM_PROCESSOR,
        "custom_params": {
            "ngram_size": 35,
            "window_size": window_size,
            "page_token_id": 128827,   # <PAGE> marker id (= sparse_attention_config.page_token_id)
            "eos_token_id": 1,         # = eos_token_id in config.json
            "max_page_tokens": 20000,  # per-page output cap, <= 0 disables the page guard
        },
        "stream": True,
    }
    response = session.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=6000,
        stream=True,
    )
    response.raise_for_status()

    chunks = []
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        choices = event.get("choices") or []  # the final chunk carries usage only
        if not choices:
            continue
        delta = (choices[0].get("delta") or {}).get("content") or ""
        if delta:
            print(delta, end="", flush=True)
            chunks.append(delta)
    return "".join(chunks)


# Multi page: all pages of a document, parsed in one request
pages = [f"your_doc/page_{i}.jpg" for i in range(1, 8)]
generate(pages)

# Single page
generate(["your_doc/page_1.jpg"])

# PDF
generate(pdf_to_images("your_doc.pdf", dpi=300))
```

## Evaluation

`infer.py` is the batch client we use for MPDocBench / MarathonDocBench: it sends each document to the SGLang server above as a single request and saves the raw output, which `utils/process_unlimitedOCR2.py` then converts to the evaluation Markdown.

```shell
python infer.py --url 127.0.0.1:10000 --benchmark MPDocBench \
    --dataset-root /path/to/MPDocBench --all --raw-dir raw --concurrency 4
python utils/process_unlimitedOCR2.py --infer-dir raw --out-dir pred
```

See [utils/README.md](utils/README.md) for the output format, the merging rules and the post-processing details.

## Visualization

Unlimited-horizon parsing in action: a full-length book parsed in a single request (replay).

<div align="center">

![Harry Potter replay](assets/HarryPotter.gif)

</div>

## Acknowledgement

We would like to thank [Deepseek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR), [Deepseek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for their valuable models and ideas, and [SGLang](https://github.com/sgl-project/sglang) for the serving framework.

## Citation
```bibtex
@misc{wei2026unlimitedocrworks2,
      title={Unlimited OCR Works 2}, 
      author={Baidu ERNIE Team},
      year={2026},
      eprint={2610.39308},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.39308}, 
}

@misc{yin2026unlimitedocrworks,
      title={Unlimited OCR Works}, 
      author={Youyang Yin and Huanhuan Liu and YY and Qunyi Xie and Chaorun Liu and Shiqi Yang and Shaohua Wang and Zhanlong Liu and Hao Zou and Jinyue Chen and Shu Wei and Jingjing Wu and Mingxin Huang and Zhen Wu and Guibin Wang and Tengyu Du and Lei Jia},
      year={2026},
      eprint={2606.23050},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.23050}, 
}
```
