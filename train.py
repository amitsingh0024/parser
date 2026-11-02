#!/usr/bin/env python3
"""
Local QLoRA / LoRA Fine-Tuning Script for Baidu Unlimited-OCR-2
Optimized for single consumer GPU (e.g., NVIDIA RTX 4050 6GB VRAM on Windows).

Dataset format (JSONL):
Each line is a JSON object with:
  {
    "images": ["path/to/page_1.jpg", "path/to/page_2.jpg"],
    "prompt": "<PAGE><image><PAGE><image>Multi page parsing.",  # optional, auto-built if omitted
    "response": "<PAGE><BLOCK>title<LEVEL1>[x1,y1,x2,y2]<FORK>..."
  }
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class UnlimitedOCRDataset(Dataset):
    """Dataset for single-page and multi-page Unlimited-OCR-2 fine-tuning."""

    def __init__(self, jsonl_path: str, tokenizer: Any, max_length: int = 4096):
        self.samples: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.base_dir = os.path.dirname(os.path.abspath(jsonl_path))

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image_paths = item.get("images") or [item["image"]]
        resolved_paths = [
            p if os.path.isabs(p) else os.path.join(self.base_dir, p)
            for p in image_paths
        ]
        images = [Image.open(p).convert("RGB") for p in resolved_paths]

        prompt = item.get("prompt")
        if not prompt:
            if len(images) > 1:
                prompt = "<PAGE><image>" * len(images) + "Multi page parsing."
            else:
                prompt = "<image>\nFree OCR."

        response = item["response"]
        full_text = f"{prompt}{response}"

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        full_ids = self.tokenizer.encode(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        if full_ids[-1] != self.tokenizer.eos_token_id and len(full_ids) < self.max_length:
            full_ids.append(self.tokenizer.eos_token_id)

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "images": images,
            "image_paths": resolved_paths,
        }


@dataclass
class DataCollatorForUnlimitedOCR:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = input_ids_padded.ne(self.pad_token_id).long()

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }


def build_model_and_tokenizer(args: argparse.Namespace):
    print(f"[INFO] Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.use_4bit:
        print("[INFO] Enabling 4-bit QLoRA quantization (NF4, bfloat16 compute)...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"[INFO] Loading model {args.model_name_or_path}...")
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map="auto",
    )

    # Freeze vision towers to save VRAM on 6GB GPU
    for name, param in model.named_parameters():
        if any(vt in name.lower() for vt in ("sam", "vision", "clip", "image_newline", "view_seperator")):
            param.requires_grad = False

    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    elif args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Baidu Unlimited-OCR-2 on a single GPU")
    parser.add_argument("--model-name-or-path", default="baidu/Unlimited-OCR-2", help="HuggingFace model ID or local path")
    parser.add_argument("--train-data", required=True, help="Path to training JSONL file")
    parser.add_argument("--eval-data", default=None, help="Optional path to evaluation JSONL file")
    parser.add_argument("--output-dir", default="outputs/unlimited-ocr2-lora", help="Directory to save checkpoints and adapter")
    parser.add_argument("--max-length", type=int, default=4096, help="Max token sequence length (lower = less VRAM)")
    parser.add_argument("--use-4bit", action="store_true", default=True, help="Use 4-bit QLoRA quantization (recommended for 6GB VRAM)")
    parser.add_argument("--no-4bit", action="store_false", dest="use_4bit", help="Disable 4-bit quantization")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated modules to apply LoRA to",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log metrics every N steps")
    args = parser.parse_args()

    model, tokenizer = build_model_and_tokenizer(args)
    train_dataset = UnlimitedOCRDataset(args.train_data, tokenizer, max_length=args.max_length)
    eval_dataset = (
        UnlimitedOCRDataset(args.eval_data, tokenizer, max_length=args.max_length)
        if args.eval_data
        else None
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForUnlimitedOCR(pad_token_id=tokenizer.pad_token_id),
    )

    print("[INFO] Starting training...")
    trainer.train()

    print(f"[INFO] Saving LoRA adapter & tokenizer to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
