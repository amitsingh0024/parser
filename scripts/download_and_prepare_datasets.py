#!/usr/bin/env python3
"""
Download & Prepare Hindi, Sanskrit, and English OCR Datasets for Baidu Unlimited-OCR-2.

This script:
1. Downloads public Sanskrit, Hindi (Devanagari), and English OCR datasets from Hugging Face
   and saves them into `data/raw_hf/`.
2. Converts real cropped OCR images into:
   - Single-image OCR training samples
   - Full-page composite document layouts with exact [x1, y1, x2, y2] (0-1000 normalized)
     bounding boxes in Unlimited-OCR-2 `<PAGE><BLOCK>...<FORK>` format.
3. Synthesizes rich single-page and multi-page structured documents (titles, paragraphs,
   bilingual/trilingual tables, cross-page text continuations) in Hindi, Sanskrit, and English
   using PyMuPDF's HarfBuzz shaper (`Nirmala.ttc`) with exact normalized bounding boxes.
4. Verifies all generated annotations with `utils/process_unlimitedOCR2.py` and writes:
   - `data/unlimited_ocr/train.jsonl`
   - `data/unlimited_ocr/val.jsonl`
   - `data/unlimited_ocr/dataset_stats.json`
"""

import argparse
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageOps
from tqdm.auto import tqdm

# Ensure project root is importable
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from utils.process_unlimitedOCR2 import convert as verify_unlimited_ocr2_convert

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Curated Hindi, Sanskrit & English Document Corpora for Multi-Block/Multi-Page
# Layout & Table Ground-Truth Generation
# ---------------------------------------------------------------------------

SANSKRIT_TITLES = [
    "अध्यायः १ — श्रीमद्भगवद्गीता उपदेशः",
    "प्रकरणम् २ — पञ्चतन्त्रस्य नीतिवाक्यानि",
    "अध्यायः ३ — पाणिनीयव्याकरणसूत्रम्",
    "चतुर्थः सर्गः — रघुवंशमहाकाव्यम्",
    "पञ्चमः भागः — आयुर्वेदसंहिता एवं चरकसूत्राणि",
    "षष्ठः अध्यायः — अर्थशास्त्रस्य राजधर्मः",
    "सप्तमः खण्डः — उपनिषदां ब्रह्मविद्या",
]

SANSKRIT_PARAGRAPHS = [
    "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥ दृष्ट्वा तु पाण्डवानीकं व्यूढं दुर्योधनस्तदा। आचार्यमुपसङ्गम्य राजा वचनमब्रवीत्॥",
    "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥ योगस्थः कुरु कर्माणि सङ्गं त्यक्त्वा धनञ्जय। सिद्ध्यसिद्ध्योः समो भूत्वा समत्वं योग उच्यते॥",
    "विद्या ददाति विनयं विनयाद्याति पात्रताम्। पात्रत्वाद्धनमाप्नोति धनाद्धर्मं ततः सुखम्॥ न चोरहार्यं न च राजहार्यं न भ्रातृभाज्यं न च भारकारि। व्यये कृते वर्धते एव नित्यं विद्याधनं सर्वधनप्रधानम्॥",
    "असतो मा सद्गमय। तमसो मा ज्योतिर्गमय। मृत्योर्मा अमृतं गमय॥ ॐ सह नाववतु। सह नौ भुनक्तु। सह वीर्यं करवावहै। तेजस्वि नावधीतमस्तु मा विद्विषावहै॥",
    "उद्यमेन हि सिध्यन्ति कार्याणि न मनोरथैः। न हि सुप्तस्य सिंहस्य प्रविशन्ति मुखे मृगाः॥ काव्यशास्त्रविनोदेन कालो गच्छति धीमताम्। व्यसनेन च मूर्खाणां निद्रया कलहेन वा॥",
    "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत। अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम्॥ परित्राणाय साधूनां विनाशाय च दुष्कृताम्। धर्मसंस्थापनार्थाय सम्भवामि युगे युगे॥",
]

HINDI_TITLES = [
    "अध्याय १: भारतीय भाषाओं का डिजिटल दस्तावेज़ीकरण",
    "खंड २: देवनागरी लिपि और संगणकीय भाषाविज्ञान",
    "प्रकरण ३: बहु-पृष्ठीय प्रकाशकीय वर्ण पहचान (OCR) प्रणाली",
    "भाग ४: प्राचीन पांडुलिपियों का संरक्षण एवं विश्लेषण",
    "अध्याय ५: वैज्ञानिक तथा तकनीकी शब्दावली आयोग",
    "खंड ६: भारतीय संविधान एवं राजभाषा प्रावधान",
]

HINDI_PARAGRAPHS = [
    "भारत एक बहुभाषी देश है जहाँ अनेक समृद्ध भाषाएँ और लिपियाँ प्रचलित हैं। देवनागरी लिपि में संस्कृत, हिन्दी, मराठी, नेपाली तथा कोंकणी जैसी प्रमुख भाषाएँ लिखी जाती हैं। आधुनिक कृत्रिम बुद्धिमत्ता के युग में इन दस्तावेज़ों का सटीक डिजिटलीकरण अत्यंत महत्वपूर्ण है।",
    "लंबे दस्तावेज़ों और पुस्तकों के विश्लेषण में अक्सर तालिकाएँ, शीर्षक और अनुच्छेद एक पृष्ठ से दूसरे पृष्ठ तक फैले होते हैं। पारंपरिक ओसीआर प्रणालियाँ प्रत्येक पृष्ठ को अलग-अलग पढ़ती हैं जिससे पृष्ठों के बीच का संदर्भ समाप्त हो जाता है।",
    "संयुक्त अक्षरों, मात्राओं और शिरोरेखा के कारण देवनागरी लिपि की पहचान में विशेष सावधानी की आवश्यकता होती है। उच्च-स्तरीय दृश्य-भाषा मॉडल (VLM) इस चुनौती को हल करने में सक्षम हैं और जटिल पृष्ठ संरचनाओं को सुरक्षित रखते हैं।",
    "डिजिटल पुस्तकालयों में लाखों ऐतिहासिक पुस्तकें, शासकीय राजपत्र और शोध-पत्र पीडीएफ प्रारूप में उपलब्ध हैं। इन्हें संरचित मार्कडाउन और एचटीएमएल तालिकाओं में परिवर्तित करने से सूचना खोज और ज्ञान निष्कर्षण सरल हो जाता है।",
    "शिक्षा, न्यायपालिका और प्रशासन के क्षेत्र में द्विभाषी एवं त्रिभाषी दस्तावेज़ों का व्यापक उपयोग होता है। एक ही पृष्ठ पर हिन्दी, संस्कृत उद्धरण और अंग्रेज़ी तकनीकी शब्दों का समन्वय सामान्य रूप से देखा जाता है।",
]

ENGLISH_TITLES = [
    "Section 1: Unlimited-Horizon Multilingual Document Parsing",
    "Chapter 2: Devanagari & Latin Layout Structure Extraction",
    "Section 3: Cross-Page Table and Paragraph Continuation Analysis",
    "Chapter 4: Sparse Reference Sliding Window Attention (SR-SWA)",
    "Section 5: Evaluation on Indic & Classical Sanskrit Corpora",
]

ENGLISH_PARAGRAPHS = [
    "Real-world documents in South Asian archives frequently interleave English exposition with Hindi commentary and classical Sanskrit shlokas in Devanagari script. Parsing such documents requires preserving reading order, hierarchical section headings, and cross-page continuations.",
    "Unlimited-OCR-2 encodes document pages into layout blocks tagged with semantic labels, normalized bounding boxes [x1, y1, x2, y2] in the 0-1000 coordinate space, and structural markers such as <PAGE>, <BLOCK>, <LEVELn>, and <FORK>.",
    "When a paragraph or table spans across a page boundary, the leading segment is annotated as cross_text or cross_table on page N, and the continuing segment on page N+1 is annotated as append_text or append_table.",
    "Fine-tuning with 4-bit QLoRA allows adaptation to complex Devanagari conjuncts (samyuktakshara) and multilingual layouts on consumer GPUs while keeping the vision encoder frozen.",
]

TABLE_SAMPLES = [
    (
        "<table><tr><th>क्रमांक (No.)</th><th>संस्कृतम् (Sanskrit)</th><th>हिन्दी (Hindi)</th><th>English Meaning</th></tr>"
        "<tr><td>१</td><td>ज्ञानम्</td><td>ज्ञान / बोध</td><td>Knowledge</td></tr>"
        "<tr><td>२</td><td>धर्मः</td><td>कर्तव्य / धर्म</td><td>Righteousness / Duty</td></tr>"
        "<tr><td>३</td><td>सत्यम्</td><td>सत्य / यथार्थ</td><td>Truth</td></tr>"
        "<tr><td>४</td><td>अहिंसा</td><td>अहिंसा / करुणा</td><td>Non-violence</td></tr></table>"
    ),
    (
        "<table><tr><th>भाषा (Language)</th><th>लिपि (Script)</th><th>परिवार (Family)</th><th>उदाहरण (Sample)</th></tr>"
        "<tr><td>संस्कृतम्</td><td>देवनागरी</td><td>Indo-Aryan</td><td>सत्यमेव जयते</td></tr>"
        "<tr><td>हिन्दी</td><td>देवनागरी</td><td>Indo-Aryan</td><td>वसुधैव कुटुम्बकम्</td></tr>"
        "<tr><td>English</td><td>Latin</td><td>Indo-European</td><td>Truth Alone Triumphs</td></tr></table>"
    ),
    (
        "<table><tr><th>अध्याय (Chapter)</th><th>विषय (Topic)</th><th>पृष्ठ संख्या (Pages)</th><th>स्थिति (Status)</th></tr>"
        "<tr><td>अध्याय १</td><td>प्रस्तावना एवं मूल सिद्धान्त</td><td>1 - 12</td><td>पूर्ण (Completed)</td></tr>"
        "<tr><td>अध्याय २</td><td>व्याकरण तथा संधि-विच्छेद</td><td>13 - 28</td><td>पूर्ण (Completed)</td></tr>"
        "<tr><td>अध्याय ३</td><td>साहित्यिक अनुप्रयोग</td><td>29 - 45</td><td>समीक्षाधीन (In Review)</td></tr></table>"
    ),
]


def norm_bbox(x0: float, y0: float, x1: float, y1: float, w: float, h: float) -> Tuple[int, int, int, int]:
    """Normalize bounding box coordinates to [0, 1000] integer range."""
    nx0 = max(0, min(1000, int(round(x0 * 1000.0 / w))))
    ny0 = max(0, min(1000, int(round(y0 * 1000.0 / h))))
    nx1 = max(0, min(1000, int(round(x1 * 1000.0 / w))))
    ny1 = max(0, min(1000, int(round(y1 * 1000.0 / h))))
    return nx0, ny0, max(nx0 + 1, nx1), max(ny0 + 1, ny1)


def extract_pil_and_text(row: Dict[str, Any]) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Extract PIL image and text label from a HuggingFace dataset row."""
    img = None
    for k in ("image", "img", "file", "page"):
        if k in row and row[k] is not None:
            val = row[k]
            if isinstance(val, Image.Image):
                img = val.convert("RGB")
                break
            elif isinstance(val, dict) and "bytes" in val and val["bytes"]:
                img = Image.open(io.BytesIO(val["bytes"])).convert("RGB")
                break
            elif isinstance(val, dict) and "path" in val and val["path"] and os.path.exists(val["path"]):
                img = Image.open(val["path"]).convert("RGB")
                break

    txt = None
    for k in ("text", "label", "transcription", "ground_truth", "sentence", "ocr"):
        if k in row and isinstance(row[k], str) and row[k].strip():
            txt = row[k].strip()
            break
    return img, txt


def download_hf_ocr_datasets(raw_hf_dir: Path, max_per_dataset: int = 400) -> Dict[str, List[Tuple[Image.Image, str]]]:
    """
    Download public Sanskrit & Devanagari/Hindi OCR datasets from Hugging Face
    and save raw copies under `data/raw_hf/`.
    """
    raw_hf_dir.mkdir(parents=True, exist_ok=True)
    collected: Dict[str, List[Tuple[Image.Image, str]]] = {
        "sanskrit": [],
        "devanagari": [],
    }

    candidates = [
        ("sanskrit", "Process-Venue/Sanskrit-OCR-Typed-Dataset", "train"),
        ("devanagari", "Malathip72/devanagari-ocr-dataset", "train"),
    ]

    for category, repo_id, split in candidates:
        safe_name = repo_id.replace("/", "__")
        save_dir = raw_hf_dir / safe_name
        save_dir.mkdir(parents=True, exist_ok=True)
        jsonl_meta = save_dir / "samples.jsonl"
        img_dir = save_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Reuse cached samples on disk if already downloaded
        if jsonl_meta.exists() and jsonl_meta.stat().st_size > 0:
            loaded = 0
            with open(jsonl_meta, "r", encoding="utf-8") as mf:
                for line in mf:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    img_p = raw_hf_dir / item["image"]
                    if img_p.exists():
                        collected[category].append((Image.open(img_p).convert("RGB"), item["text"]))
                        loaded += 1
            if loaded > 0:
                print(f"[HF] Loaded {loaded} cached samples from {save_dir}")
                continue

        print(f"[HF] Fetching {repo_id} ({split})...")
        try:
            ds = load_dataset(repo_id, split=split, streaming=True)
            count = 0
            with open(jsonl_meta, "w", encoding="utf-8") as mf:
                for idx, row in enumerate(ds):
                    if count >= max_per_dataset:
                        break
                    img, txt = extract_pil_and_text(row)
                    if img is None or not txt:
                        continue
                    img_path = img_dir / f"sample_{count:05d}.jpg"
                    img.save(img_path, quality=92)
                    mf.write(
                        json.dumps(
                            {"id": count, "image": str(img_path.relative_to(raw_hf_dir)), "text": txt},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    collected[category].append((img, txt))
                    count += 1
            print(f"[HF] Saved {count} real OCR samples from {repo_id} -> {save_dir}")
        except Exception as e:
            print(f"[WARN] Could not stream {repo_id}: {e}")

    return collected


def build_composite_pages_from_hf(
    hf_samples: List[Tuple[Image.Image, str]],
    out_img_dir: Path,
    prefix: str,
    num_pages: int = 80,
) -> List[Dict[str, Any]]:
    """
    Compose multiple real cropped OCR line/block images from Hugging Face datasets
    onto full document pages (1024x1448) with exact [x1, y1, x2, y2] bounding boxes
    in Unlimited-OCR-2 `<PAGE><BLOCK>...<FORK>` format.
    """
    if not hf_samples:
        return []

    out_img_dir.mkdir(parents=True, exist_ok=True)
    page_w, page_h = 1024, 1448
    records: List[Dict[str, Any]] = []

    for p_idx in tqdm(range(num_pages), desc=f"Building {prefix} composite pages", unit="page"):
        canvas = Image.new("RGB", (page_w, page_h), color=(252, 251, 248))
        draw = ImageDraw.Draw(canvas)
        y_cursor = 70
        margin_x = 70
        max_block_w = page_w - 2 * margin_x

        blocks_per_page = random.randint(4, 8)
        chosen = random.choices(hf_samples, k=blocks_per_page)
        block_strs: List[str] = []

        for b_idx, (crop_img, txt) in enumerate(chosen):
            cw, ch = crop_img.size
            if cw <= 0 or ch <= 0:
                continue
            scale = min(max_block_w / cw, 110.0 / max(ch, 1))
            scale = max(0.5, min(scale, 2.2))
            new_w = max(40, min(max_block_w, int(cw * scale)))
            new_h = max(24, int(ch * scale))

            if y_cursor + new_h + 40 > page_h - 60:
                break

            resized = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x0 = margin_x
            y0 = y_cursor
            x1 = x0 + new_w
            y1 = y0 + new_h

            canvas.paste(resized, (x0, y0))
            bx0, by0, bx1, by1 = norm_bbox(x0, y0, x1, y1, page_w, page_h)

            if b_idx == 0:
                block_strs.append(
                    f"<BLOCK>title<LEVEL1>[{bx0},{by0},{bx1},{by1}]<FORK>{txt}"
                )
            else:
                block_strs.append(
                    f"<BLOCK>text[{bx0},{by0},{bx1},{by1}]<FORK>{txt}"
                )

            y_cursor = y1 + random.randint(28, 55)

        if not block_strs:
            continue

        img_path = out_img_dir / f"{prefix}_page_{p_idx:04d}.jpg"
        canvas.save(img_path, quality=92)
        response = "<PAGE>" + "\n".join(block_strs)

        # Verify compatibility with Unlimited-OCR-2 post-processor
        _ = verify_unlimited_ocr2_convert(response)

        records.append(
            {
                "images": [str(img_path)],
                "prompt": "<PAGE><image>Multi page parsing.",
                "response": response,
                "source": f"hf_composite_{prefix}",
                "languages": ["sa", "hi"],
            }
        )

    return records


def render_structured_page_pymupdf(
    blocks_spec: List[Dict[str, Any]],
    out_path: Path,
    dpi: int = 150,
) -> Tuple[str, str]:
    """
    Render a structured page (titles, Hindi/Sanskrit/English paragraphs, HTML tables)
    using PyMuPDF's HarfBuzz HTML box renderer so Devanagari ligatures are shaped accurately
    and exact normalized [x1, y1, x2, y2] coordinates are recorded.
    """
    doc = pymupdf.open()
    pw, ph = 595.0, 842.0  # A4 points
    page = doc.new_page(width=pw, height=ph)

    css = """
    * { font-family: 'Nirmala UI', sans-serif; color: #111111; }
    h1 { font-size: 16pt; font-weight: bold; margin: 0; padding: 0; }
    h2 { font-size: 13pt; font-weight: bold; margin: 0; padding: 0; }
    p { font-size: 10.5pt; line-height: 1.45; margin: 0; padding: 0; }
    table { border-collapse: collapse; width: 100%; font-size: 9.5pt; }
    th, td { border: 1px solid #333333; padding: 4px 6px; text-align: left; }
    th { background-color: #f0f0f0; font-weight: bold; }
    """

    y_cursor = 48.0
    margin_x = 45.0
    box_w = pw - 2 * margin_x
    out_blocks: List[str] = []

    for spec in blocks_spec:
        btype = spec["type"]
        content = spec["content"]
        level = spec.get("level", 1)

        if btype == "title":
            tag = "h1" if level == 1 else "h2"
            html = f"<{tag}>{content}</{tag}>"
            max_h = 65.0
        elif btype == "table":
            html = content
            max_h = 180.0
        else:
            html = f"<p>{content}</p>"
            max_h = 140.0

        if y_cursor + 40.0 > ph - 45.0:
            break

        rect = pymupdf.Rect(margin_x, y_cursor, margin_x + box_w, min(ph - 36.0, y_cursor + max_h))
        spare_h, _ = page.insert_htmlbox(rect, html, css=css)
        used_h = max(16.0, (rect.height - spare_h) if spare_h >= 0 else rect.height)
        actual_y1 = min(ph - 36.0, y_cursor + used_h + 4.0)

        bx0, by0, bx1, by1 = norm_bbox(margin_x, y_cursor, margin_x + box_w, actual_y1, pw, ph)

        if btype == "title":
            out_blocks.append(f"<BLOCK>title<LEVEL{level}>[{bx0},{by0},{bx1},{by1}]<FORK>{content}")
        else:
            out_blocks.append(f"<BLOCK>{btype}[{bx0},{by0},{bx1},{by1}]<FORK>{content}")

        y_cursor = actual_y1 + random.uniform(14.0, 24.0)

    mat = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))
    doc.close()

    page_str = "<PAGE>" + "\n".join(out_blocks)
    return str(out_path), page_str


def generate_trilingual_documents(
    out_img_dir: Path,
    num_single_page: int = 90,
    num_multi_page: int = 25,
) -> List[Dict[str, Any]]:
    """
    Generate single-page and multi-page Hindi + Sanskrit + English structured documents
    (following Baidu Unlimited-OCR-2's single-page + multi-page training mix, including
    cross_text -> append_text continuations across pages).
    """
    records: List[Dict[str, Any]] = []

    # 1. Single-page structured documents (Hindi, Sanskrit, English, and Mixed Trilingual)
    for idx in tqdm(range(num_single_page), desc="Rendering single-page trilingual docs", unit="doc"):
        mode = idx % 4  # 0: Sanskrit, 1: Hindi, 2: English, 3: Trilingual mixed
        blocks_spec: List[Dict[str, Any]] = []

        if mode == 0:
            blocks_spec.append({"type": "title", "level": 1, "content": random.choice(SANSKRIT_TITLES)})
            for _ in range(random.randint(2, 3)):
                blocks_spec.append({"type": "text", "content": random.choice(SANSKRIT_PARAGRAPHS)})
            if random.random() < 0.45:
                blocks_spec.append({"type": "table", "content": random.choice(TABLE_SAMPLES)})
            langs = ["sa"]
        elif mode == 1:
            blocks_spec.append({"type": "title", "level": 1, "content": random.choice(HINDI_TITLES)})
            for _ in range(random.randint(2, 3)):
                blocks_spec.append({"type": "text", "content": random.choice(HINDI_PARAGRAPHS)})
            if random.random() < 0.45:
                blocks_spec.append({"type": "table", "content": random.choice(TABLE_SAMPLES)})
            langs = ["hi"]
        elif mode == 2:
            blocks_spec.append({"type": "title", "level": 1, "content": random.choice(ENGLISH_TITLES)})
            for _ in range(random.randint(2, 3)):
                blocks_spec.append({"type": "text", "content": random.choice(ENGLISH_PARAGRAPHS)})
            langs = ["en"]
        else:
            blocks_spec.append({"type": "title", "level": 1, "content": random.choice(HINDI_TITLES)})
            blocks_spec.append({"type": "text", "content": random.choice(SANSKRIT_PARAGRAPHS)})
            blocks_spec.append({"type": "title", "level": 2, "content": random.choice(ENGLISH_TITLES)})
            blocks_spec.append({"type": "text", "content": random.choice(HINDI_PARAGRAPHS)})
            blocks_spec.append({"type": "table", "content": random.choice(TABLE_SAMPLES)})
            blocks_spec.append({"type": "text", "content": random.choice(ENGLISH_PARAGRAPHS)})
            langs = ["hi", "sa", "en"]

        img_path = out_img_dir / f"structured_single_{idx:04d}.jpg"
        saved_path, page_str = render_structured_page_pymupdf(blocks_spec, img_path)
        _ = verify_unlimited_ocr2_convert(page_str)

        records.append(
            {
                "images": [saved_path],
                "prompt": "<PAGE><image>Multi page parsing.",
                "response": page_str,
                "source": "trilingual_structured_single",
                "languages": langs,
            }
        )

    # 2. Multi-page documents with cross_text -> append_text across page boundaries
    for m_idx in tqdm(range(num_multi_page), desc="Rendering multi-page trilingual docs", unit="doc"):
        n_pages = random.randint(2, 3)
        doc_images: List[str] = []
        doc_page_strs: List[str] = []

        for p in range(n_pages):
            blocks_spec = []
            if p == 0:
                blocks_spec.append({"type": "title", "level": 1, "content": random.choice(SANSKRIT_TITLES)})
            else:
                # Continuation from previous page's cross_text
                cont_pool = HINDI_PARAGRAPHS if m_idx % 2 == 0 else SANSKRIT_PARAGRAPHS
                blocks_spec.append({"type": "append_text", "content": random.choice(cont_pool)})
                blocks_spec.append({"type": "title", "level": 2, "content": random.choice(HINDI_TITLES)})

            blocks_spec.append({"type": "text", "content": random.choice(HINDI_PARAGRAPHS)})
            if p == 0 and random.random() < 0.6:
                blocks_spec.append({"type": "table", "content": random.choice(TABLE_SAMPLES)})
            blocks_spec.append({"type": "text", "content": random.choice(ENGLISH_PARAGRAPHS)})

            if p < n_pages - 1:
                # Cross-page paragraph at the bottom of page p
                lead_pool = HINDI_PARAGRAPHS if m_idx % 2 == 0 else SANSKRIT_PARAGRAPHS
                blocks_spec.append({"type": "cross_text", "content": random.choice(lead_pool)})

            img_path = out_img_dir / f"structured_multi_{m_idx:04d}_p{p + 1}.jpg"
            saved_path, page_str = render_structured_page_pymupdf(blocks_spec, img_path)
            doc_images.append(saved_path)
            doc_page_strs.append(page_str)

        full_response = "".join(doc_page_strs)
        _ = verify_unlimited_ocr2_convert(full_response)

        records.append(
            {
                "images": doc_images,
                "prompt": "<PAGE><image>" * len(doc_images) + "Multi page parsing.",
                "response": full_response,
                "source": "trilingual_structured_multi",
                "languages": ["hi", "sa", "en"],
            }
        )

    return records


def main():
    parser = argparse.ArgumentParser(description="Download and build Hindi, Sanskrit, and English Unlimited-OCR-2 datasets")
    parser.add_argument("--data-dir", default=str(ROOT_DIR / "data"), help="Base data directory")
    parser.add_argument("--hf-max-samples", type=int, default=300, help="Max samples to pull per Hugging Face dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)
    raw_hf_dir = data_dir / "raw_hf"
    uocr_dir = data_dir / "unlimited_ocr"
    uocr_img_dir = uocr_dir / "images"
    uocr_dir.mkdir(parents=True, exist_ok=True)
    uocr_img_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download real Sanskrit & Devanagari/Hindi OCR datasets from Hugging Face
    hf_collected = download_hf_ocr_datasets(raw_hf_dir, max_per_dataset=args.hf_max_samples)

    all_records: List[Dict[str, Any]] = []

    # Step 2: Build composite full-page Unlimited-OCR-2 samples from real HF datasets
    if hf_collected["sanskrit"]:
        sa_records = build_composite_pages_from_hf(
            hf_collected["sanskrit"], uocr_img_dir / "hf_sanskrit", "sanskrit", num_pages=60
        )
        all_records.extend(sa_records)
        print(f"[BUILD] Created {len(sa_records)} full-page Sanskrit OCR composite pages from HF data.")

    if hf_collected["devanagari"]:
        dev_records = build_composite_pages_from_hf(
            hf_collected["devanagari"], uocr_img_dir / "hf_devanagari", "devanagari", num_pages=60
        )
        all_records.extend(dev_records)
        print(f"[BUILD] Created {len(dev_records)} full-page Hindi/Devanagari OCR composite pages from HF data.")

    # Step 3: Build structured single-page and multi-page Hindi + Sanskrit + English documents
    structured_records = generate_trilingual_documents(
        uocr_img_dir / "trilingual_docs",
        num_single_page=100,
        num_multi_page=30,
    )
    all_records.extend(structured_records)
    print(f"[BUILD] Created {len(structured_records)} structured Hindi/Sanskrit/English single & multi-page documents.")

    # Convert image paths to paths relative to `uocr_dir` so JSONL is portable across machines
    for rec in all_records:
        rec["images"] = [
            os.path.relpath(p, str(uocr_dir)).replace("\\", "/") for p in rec["images"]
        ]

    random.shuffle(all_records)
    val_size = max(10, int(len(all_records) * 0.1))
    val_records = all_records[:val_size]
    train_records = all_records[val_size:]

    train_jsonl = uocr_dir / "train.jsonl"
    val_jsonl = uocr_dir / "val.jsonl"

    with open(train_jsonl, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_jsonl, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "total_documents": len(all_records),
        "train_documents": len(train_records),
        "val_documents": len(val_records),
        "hf_raw_sanskrit_samples": len(hf_collected["sanskrit"]),
        "hf_raw_devanagari_samples": len(hf_collected["devanagari"]),
        "multi_page_documents": sum(1 for r in all_records if len(r["images"]) > 1),
        "single_page_documents": sum(1 for r in all_records if len(r["images"]) == 1),
        "train_jsonl": str(train_jsonl.relative_to(ROOT_DIR)).replace("\\", "/"),
        "val_jsonl": str(val_jsonl.relative_to(ROOT_DIR)).replace("\\", "/"),
    }
    stats_path = uocr_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n[DONE] Dataset preparation complete!")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
