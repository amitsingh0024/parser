#!/usr/bin/env python3
"""
Local Transformers Inference Script for Baidu Unlimited-OCR-2 (Windows / 6GB VRAM compatible).
Supports 4-bit quantization via bitsandbytes so the 3B model runs comfortably on a 6GB GPU,
as well as loading a fine-tuned PEFT LoRA adapter and converting raw layout output to Markdown.

Examples:
  # Single image inference (4-bit mode)
  python local_infer.py --images page_1.jpg --out-dir outputs/demo

  # Multi-page PDF inference
  python local_infer.py --pdf document.pdf --out-dir outputs/pdf_demo

  # Inference with a fine-tuned LoRA adapter
  python local_infer.py --images page_1.jpg --lora-path outputs/unlimited-ocr2-lora
"""

import argparse
import os
import tempfile
from typing import List

import pymupdf as fitz  # PyMuPDF
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from utils.process_unlimitedOCR2 import convert as convert_to_markdown


def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[str]:
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="uocr_pdf_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    image_paths: List[str] = []
    for i, page in enumerate(doc):
        img_path = os.path.join(tmp_dir, f"page_{i + 1:04d}.jpg")
        page.get_pixmap(matrix=mat).save(img_path)
        image_paths.append(img_path)
    doc.close()
    return image_paths


def main():
    parser = argparse.ArgumentParser(description="Local Unlimited-OCR-2 Inference")
    parser.add_argument("--model", default="baidu/Unlimited-OCR-2", help="Model ID or local path")
    parser.add_argument("--lora-path", default=None, help="Optional path to fine-tuned LoRA adapter")
    parser.add_argument("--images", nargs="+", default=None, help="One or more page image paths in order")
    parser.add_argument("--pdf", default=None, help="Path to a PDF file to convert and parse")
    parser.add_argument("--dpi", type=int, default=200, help="DPI when rendering PDF pages")
    parser.add_argument("--out-dir", default="outputs/infer", help="Output directory for raw and markdown results")
    parser.add_argument("--use-4bit", action="store_true", default=True, help="Load model in 4-bit NF4 (recommended for 6GB VRAM)")
    parser.add_argument("--no-4bit", action="store_false", dest="use_4bit", help="Load model in native bfloat16")
    args = parser.parse_args()

    if not args.images and not args.pdf:
        raise SystemExit("Please provide either --images <img1> [img2 ...] or --pdf <file.pdf>")

    image_files = list(args.images) if args.images else pdf_to_images(args.pdf, dpi=args.dpi)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    quant_config = None
    if args.use_4bit:
        print("[INFO] Using 4-bit quantization for low-VRAM GPU...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"[INFO] Loading model from {args.model}...")
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map="auto" if args.use_4bit else None,
    )
    if not args.use_4bit and torch.cuda.is_available():
        model = model.cuda()

    if args.lora_path:
        from peft import PeftModel
        print(f"[INFO] Loading LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()

    if len(image_files) == 1:
        print(f"[INFO] Running single-page inference on {image_files[0]}...")
        raw_text = model.infer(
            tokenizer,
            image_file=image_files[0],
            output_path=args.out_dir,
            save_results=True,
        )
    else:
        print(f"[INFO] Running multi-page inference on {len(image_files)} pages...")
        raw_text, num_tokens = model.infer_multi(
            tokenizer,
            image_files=image_files,
            output_path=args.out_dir,
            save_results=True,
        )
        print(f"[INFO] Generated {num_tokens} tokens across {len(image_files)} pages.")

    if raw_text:
        raw_file = os.path.join(args.out_dir, "raw_output.txt")
        md_file = os.path.join(args.out_dir, "output.md")
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(raw_text)
        md_text = convert_to_markdown(raw_text)
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"[INFO] Saved raw layout output -> {raw_file}")
        print(f"[INFO] Saved post-processed Markdown -> {md_file}")


if __name__ == "__main__":
    main()
