#!/usr/bin/env python3
"""
Convert digital Hindi, Sanskrit, and English PDFs into Baidu Unlimited-OCR-2
training JSONL format (<PAGE><BLOCK>label[x1,y1,x2,y2]<FORK>text).

Usage:
  python scripts/pdf_to_dataset.py --pdf-dir path/to/pdfs --out-dir data/custom_pdfs
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pymupdf


def norm_coord(val: float, total: float) -> int:
    return max(0, min(1000, int(round(val * 1000.0 / max(total, 1.0)))))


def extract_page_blocks(page: pymupdf.Page) -> str:
    pw, ph = page.rect.width, page.rect.height
    d = page.get_text("dict")
    blocks_out: List[str] = []

    for b in d.get("blocks", []):
        if b.get("type") != 0:  # 0 = text block
            continue
        x0, y0, x1, y1 = b["bbox"]
        bx0 = norm_coord(x0, pw)
        by0 = norm_coord(y0, ph)
        bx1 = norm_coord(x1, pw)
        by1 = norm_coord(y1, ph)

        lines_txt: List[str] = []
        max_size = 0.0
        for line in b.get("lines", []):
            spans_txt = []
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    spans_txt.append(t)
                    max_size = max(max_size, float(span.get("size", 10.0)))
            if spans_txt:
                lines_txt.append(" ".join(spans_txt))

        content = " ".join(lines_txt).strip()
        if not content:
            continue

        if max_size >= 15.0:
            blocks_out.append(f"<BLOCK>title<LEVEL1>[{bx0},{by0},{bx1},{by1}]<FORK>{content}")
        elif max_size >= 12.8:
            blocks_out.append(f"<BLOCK>title<LEVEL2>[{bx0},{by0},{bx1},{by1}]<FORK>{content}")
        else:
            blocks_out.append(f"<BLOCK>text[{bx0},{by0},{bx1},{by1}]<FORK>{content}")

    return "<PAGE>" + "\n".join(blocks_out)


def process_pdf(pdf_path: Path, images_dir: Path, base_dir: Path, dpi: int = 180, window_pages: int = 2) -> List[Dict[str, Any]]:
    doc = pymupdf.open(str(pdf_path))
    stem = pdf_path.stem
    mat = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)

    page_imgs: List[str] = []
    page_strs: List[str] = []

    for i, page in enumerate(doc):
        p_str = extract_page_blocks(page)
        if p_str == "<PAGE>":
            continue
        img_file = images_dir / f"{stem}_page_{i + 1:04d}.jpg"
        page.get_pixmap(matrix=mat).save(str(img_file))
        rel_img = os.path.relpath(str(img_file), str(base_dir)).replace("\\", "/")
        page_imgs.append(rel_img)
        page_strs.append(p_str)

    doc.close()
    records: List[Dict[str, Any]] = []

    # Single-page samples
    for img, p_str in zip(page_imgs, page_strs):
        records.append(
            {
                "images": [img],
                "prompt": "<PAGE><image>Multi page parsing.",
                "response": p_str,
                "source": f"pdf_single:{stem}",
            }
        )

    # Multi-page sliding window samples
    if len(page_imgs) >= window_pages:
        for start in range(0, len(page_imgs) - window_pages + 1, window_pages):
            chunk_imgs = page_imgs[start : start + window_pages]
            chunk_strs = page_strs[start : start + window_pages]
            records.append(
                {
                    "images": chunk_imgs,
                    "prompt": "<PAGE><image>" * len(chunk_imgs) + "Multi page parsing.",
                    "response": "".join(chunk_strs),
                    "source": f"pdf_multi:{stem}",
                }
            )

    return records


def main():
    parser = argparse.ArgumentParser(description="Convert Hindi/Sanskrit/English PDFs to Unlimited-OCR-2 JSONL")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--out-dir", default="data/custom_pdfs", help="Output dataset directory")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        print(f"[PDF] Processing {pdf_file.name}...")
        recs = process_pdf(pdf_file, img_dir, out_dir, dpi=args.dpi)
        all_records.extend(recs)

    out_jsonl = out_dir / "train.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved {len(all_records)} training samples to {out_jsonl}")


if __name__ == "__main__":
    main()
