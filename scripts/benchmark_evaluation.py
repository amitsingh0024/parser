#!/usr/bin/env python3
"""
Benchmark & Metrics Evaluation Script for Baidu Unlimited-OCR-2 Indic Model.
Evaluates Character Error Rate (CER), Word Error Rate (WER), Normalized Edit Distance (NED),
and throughput on the held-out validation set across Hindi, Sanskrit, and English.
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig


def levenshtein(seq1, seq2):
    """Compute standard Levenshtein distance."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def clean_text_for_eval(text: str) -> str:
    """Normalize text by stripping layout tags for pure text CER/WER evaluation."""
    # Remove block markers and coordinates like <BLOCK>text[10, 20, 30, 40]<FORK>
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def compute_metrics(gt: str, pred: str) -> Tuple[float, float, float]:
    """Compute CER, WER, and NED between ground truth and predicted text."""
    gt_clean = clean_text_for_eval(gt)
    pred_clean = clean_text_for_eval(pred)

    if not gt_clean:
        return 0.0, 0.0, 1.0

    # Character Error Rate
    char_dist = levenshtein(gt_clean, pred_clean)
    cer = char_dist / max(1, len(gt_clean))

    # Normalized Edit Distance (Accuracy = 1 - norm_dist)
    max_len = max(len(gt_clean), len(pred_clean), 1)
    ned_score = max(0.0, 1.0 - (char_dist / max_len))

    # Word Error Rate
    gt_words = gt_clean.split()
    pred_words = pred_clean.split()
    word_dist = levenshtein(gt_words, pred_words)
    wer = word_dist / max(1, len(gt_words))

    return cer, wer, ned_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate Unlimited-OCR-2 Indic Benchmarks")
    parser.add_argument("--val-data", default="data/unlimited_ocr/val.jsonl", help="Validation JSONL path")
    parser.add_argument("--lora-path", default="outputs/unlimited-ocr2-indic-lora", help="Trained LoRA adapter path")
    parser.add_argument("--model", default="baidu/Unlimited-OCR-2", help="Base model identifier")
    parser.add_argument("--num-samples", type=int, default=15, help="Number of balanced validation samples to evaluate")
    parser.add_argument("--out-json", default="outputs/evaluation_results.json", help="Path to save evaluation JSON")
    args = parser.parse_args()

    print("[INFO] Loading validation samples from", args.val_data)
    with open(args.val_data, "r", encoding="utf-8") as f:
        val_samples = [json.loads(line) for line in f if line.strip()]

    # Stratify samples across English, Hindi, Sanskrit
    en_samples = [s for s in val_samples if "en" in s.get("languages", [])]
    hi_samples = [s for s in val_samples if "hi" in s.get("languages", [])]
    sa_samples = [s for s in val_samples if "sa" in s.get("languages", [])]

    if args.num_samples > 0:
        per_lang = max(1, args.num_samples // 3)
        selected = []
        selected.extend(en_samples[:per_lang])
        selected.extend(hi_samples[:per_lang])
        selected.extend(sa_samples[:per_lang])
        # Deduplicate
        seen_img = set()
        test_samples = []
        for s in selected:
            img = s["images"][0]
            if img not in seen_img:
                seen_img.add(img)
                test_samples.append(s)
    else:
        test_samples = val_samples

    print(f"[INFO] Selected {len(test_samples)} evaluation samples.")

    # Load Model
    print(f"[INFO] Loading base model {args.model} with 4-bit quantization...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if hasattr(config, "sparse_attention_config") and isinstance(config.sparse_attention_config, dict):
        config.sparse_attention_config["dash_use_triton"] = False

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["sam_model", "vision_model", "projector", "lm_head", "dash_router", "indexer"],
    )

    model = AutoModel.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map="auto",
        local_files_only=True,
    )

    if args.lora_path:
        print(f"[INFO] Merging LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()

    results = []
    lang_stats: Dict[str, Dict[str, list]] = {
        "en": {"cer": [], "wer": [], "ned": []},
        "hi": {"cer": [], "wer": [], "ned": []},
        "sa": {"cer": [], "wer": [], "ned": []},
        "overall": {"cer": [], "wer": [], "ned": []},
    }

    t0_total = time.time()
    total_tokens_gen = 0

    print("[INFO] Running evaluation...")
    for idx, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
        img_rel = sample["images"][0]
        img_path = os.path.join("data", "unlimited_ocr", img_rel)
        gt_text = sample["response"]
        langs = sample.get("languages", ["overall"])

        t_start = time.time()
        pred_text = model.infer(
            tokenizer,
            image_file=img_path,
            output_path="",
            save_results=False,
            max_new_tokens=2048,
            no_repeat_ngram_size=15,
            temperature=0.0,
            crop_mode=True,
            eval_mode=True,
        )
        gen_time = time.time() - t_start
        pred_tokens = len(tokenizer.encode(pred_text, add_special_tokens=False))
        total_tokens_gen += pred_tokens

        cer, wer, ned = compute_metrics(gt_text, pred_text)

        for l in langs:
            if l in lang_stats:
                lang_stats[l]["cer"].append(cer)
                lang_stats[l]["wer"].append(wer)
                lang_stats[l]["ned"].append(ned)
        lang_stats["overall"]["cer"].append(cer)
        lang_stats["overall"]["wer"].append(wer)
        lang_stats["overall"]["ned"].append(ned)

        results.append({
            "image": img_rel,
            "languages": langs,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "ned": round(ned, 4),
            "gen_time_sec": round(gen_time, 2),
            "tokens_gen": pred_tokens,
            "pred_preview": pred_text[:120].replace("\n", " "),
        })

    total_time = time.time() - t0_total

    summary = {
        "total_evaluated": len(test_samples),
        "total_time_sec": round(total_time, 2),
        "avg_tokens_per_sec": round(total_tokens_gen / max(1.0, total_time), 2),
        "overall": {
            "cer": round(sum(lang_stats["overall"]["cer"]) / len(lang_stats["overall"]["cer"]), 4),
            "wer": round(sum(lang_stats["overall"]["wer"]) / len(lang_stats["overall"]["wer"]), 4),
            "ned_accuracy": round(sum(lang_stats["overall"]["ned"]) / len(lang_stats["overall"]["ned"]) * 100, 2),
        },
    }

    for l in ["hi", "sa", "en"]:
        if lang_stats[l]["cer"]:
            summary[l] = {
                "cer": round(sum(lang_stats[l]["cer"]) / len(lang_stats[l]["cer"]), 4),
                "wer": round(sum(lang_stats[l]["wer"]) / len(lang_stats[l]["wer"]), 4),
                "ned_accuracy": round(sum(lang_stats[l]["ned"]) / len(lang_stats[l]["ned"]) * 100, 2),
                "sample_count": len(lang_stats[l]["cer"]),
            }

    print("\n" + "=" * 60)
    print("🎯 BENCHMARK EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Evaluated: {summary['total_evaluated']} documents")
    print(f"Overall CER (Character Error Rate): {summary['overall']['cer']:.4f} ({(1.0 - summary['overall']['cer'])*100:.1f}% accuracy)")
    print(f"Overall WER (Word Error Rate):      {summary['overall']['wer']:.4f}")
    print(f"Overall NED Normalized Accuracy:   {summary['overall']['ned_accuracy']:.2f}%")
    for l, name in [("hi", "Hindi"), ("sa", "Sanskrit"), ("en", "English")]:
        if l in summary:
            print(f"  - {name:10s} CER: {summary[l]['cer']:.4f} | NED Accuracy: {summary[l]['ned_accuracy']:.2f}% ({summary[l]['sample_count']} samples)")
    print(f"Throughput: {summary['avg_tokens_per_sec']} tokens/sec")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": results}, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved full evaluation details to {args.out_json}")


if __name__ == "__main__":
    main()
