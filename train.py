#!/usr/bin/env python3
"""
Local QLoRA / LoRA Fine-Tuning Script for Baidu Unlimited-OCR-2
Optimized for single consumer GPU (e.g., NVIDIA RTX 4050 6GB VRAM on Windows).
Includes live tqdm progress bars with Loss, Learning Rate, ETA, and GPU VRAM tracking.

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
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
BOS_TOKEN_ID = 0
BASE_IMAGE_SIZE = 1024
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4

_IMG_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)
_PAD_COLOR = (127, 127, 127)


def get_gpu_vram_str() -> str:
    """Return formatted current/total GPU VRAM usage string."""
    if not torch.cuda.is_available():
        return "CPU"
    alloc_gb = torch.cuda.memory_allocated(0) / (1024**3)
    res_gb = torch.cuda.memory_reserved(0) / (1024**3)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return f"{alloc_gb:.2f}G(res {res_gb:.2f}G)/{total_gb:.1f}G"


class LiveTrainingProgressCallback(TrainerCallback):
    """
    Custom tqdm progress bar + step logger that shows:
    - Step / Total Steps (%)
    - Current Epoch
    - Training Loss & Eval Loss
    - Learning Rate
    - Live GPU VRAM consumption on the RTX 4050
    - Saves step-by-step progress to `<output_dir>/training_progress.jsonl`
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.pbar: Optional[tqdm] = None
        self.start_time: float = 0.0
        self.last_loss: Optional[float] = None
        self.last_lr: Optional[float] = None
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, "training_progress.jsonl")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.start_time = time.time()
        self.pbar = tqdm(
            total=state.max_steps,
            desc="Training Unlimited-OCR-2",
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )
        tqdm.write(
            f"[PROGRESS] Training started | Total steps: {state.max_steps} | "
            f"Epochs: {args.num_train_epochs} | GPU VRAM: {get_gpu_vram_str()}"
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.pbar is not None:
            delta = state.global_step - self.pbar.n
            if delta > 0:
                self.pbar.update(delta)
            postfix: Dict[str, Any] = {
                "epoch": f"{state.epoch or 0.0:.2f}/{args.num_train_epochs:.0f}",
                "vram": get_gpu_vram_str(),
            }
            if self.last_loss is not None:
                postfix["loss"] = f"{self.last_loss:.4f}"
            if self.last_lr is not None:
                postfix["lr"] = f"{self.last_lr:.2e}"
            self.pbar.set_postfix(postfix, refresh=True)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if not logs:
            return
        if "loss" in logs:
            self.last_loss = float(logs["loss"])
        if "learning_rate" in logs:
            self.last_lr = float(logs["learning_rate"])

        if self.pbar is not None:
            postfix: Dict[str, Any] = {
                "epoch": f"{state.epoch or 0.0:.2f}/{args.num_train_epochs:.0f}",
                "vram": get_gpu_vram_str(),
            }
            if self.last_loss is not None:
                postfix["loss"] = f"{self.last_loss:.4f}"
            if self.last_lr is not None:
                postfix["lr"] = f"{self.last_lr:.2e}"
            if "eval_loss" in logs:
                postfix["val_loss"] = f"{float(logs['eval_loss']):.4f}"
            self.pbar.set_postfix(postfix, refresh=True)

        record = {
            "step": state.global_step,
            "max_steps": state.max_steps,
            "progress_pct": round(100.0 * state.global_step / max(state.max_steps, 1), 2),
            "epoch": round(float(state.epoch or 0.0), 3),
            "vram": get_gpu_vram_str(),
            "elapsed_sec": round(time.time() - self.start_time, 1),
            **logs,
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        elapsed = time.time() - self.start_time
        tqdm.write(
            f"[PROGRESS] Training finished in {elapsed / 60.0:.2f} min | "
            f"Final step: {state.global_step}/{state.max_steps} | "
            f"Progress log: {self.log_file}"
        )


def build_uocr_multimodal_sample(
    tokenizer: Any,
    prompt: str,
    response: str,
    pil_images: List[Image.Image],
    max_length: int = 2048,
) -> Dict[str, Any]:
    """
    Build Unlimited-OCR-2 multimodal inputs matching `_uocr_build_inputs` in `modeling_unlimitedocr.py`:
    - Each `<image>` in `prompt` is expanded into 273 `IMAGE_TOKEN_ID` (128815) tokens (base mode).
    - `images_seq_mask` is True at those 273 * N_pages positions so `UnlimitedOCRModel.forward`
      scatters the vision encoder features into `inputs_embeds`.
    - `labels` masks prompt + image tokens with -100 and supervises the response tokens.
    """
    n_img = len(pil_images)
    splits = prompt.split(IMAGE_TOKEN)
    if len(splits) != n_img + 1:
        prompt = "<PAGE><image>" * n_img + "Multi page parsing."
        splits = prompt.split(IMAGE_TOKEN)

    nqb = math.ceil((BASE_IMAGE_SIZE // PATCH_SIZE) / DOWNSAMPLE_RATIO)  # 16
    per_page_img_tokens = nqb * (nqb + 1) + 1  # 273 tokens per page image

    ids: List[int] = [BOS_TOKEN_ID]
    seq_mask: List[bool] = [False]
    globals_list: List[torch.Tensor] = []
    spatial_list: List[List[int]] = []

    for text_sep, img in zip(splits, pil_images):
        sep_ids = tokenizer.encode(text_sep, add_special_tokens=False)
        ids.extend(sep_ids)
        seq_mask.extend([False] * len(sep_ids))

        padded_img = ImageOps.pad(img, (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE), color=_PAD_COLOR)
        globals_list.append(_IMG_TRANSFORM(padded_img).to(torch.bfloat16))
        spatial_list.append([1, 1])

        img_toks = [IMAGE_TOKEN_ID] * per_page_img_tokens
        ids.extend(img_toks)
        seq_mask.extend([True] * per_page_img_tokens)

    tail_ids = tokenizer.encode(splits[-1], add_special_tokens=False)
    ids.extend(tail_ids)
    seq_mask.extend([False] * len(tail_ids))

    prompt_len = len(ids)

    resp_ids = tokenizer.encode(response, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    if not resp_ids or resp_ids[-1] != eos_id:
        resp_ids.append(eos_id)

    # Ensure we never truncate inside the image token region
    max_resp_len = max(64, max_length - prompt_len)
    resp_ids = resp_ids[:max_resp_len]

    full_ids = ids + resp_ids
    full_seq_mask = seq_mask + [False] * len(resp_ids)
    labels = [-100] * prompt_len + resp_ids

    images_ori = torch.stack(globals_list, dim=0)  # [num_pages, 3, 1024, 1024]
    images_crop = torch.zeros((1, 3, BASE_IMAGE_SIZE, BASE_IMAGE_SIZE), dtype=torch.bfloat16)
    spatial = torch.tensor(spatial_list, dtype=torch.long)

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "images_seq_mask": torch.tensor(full_seq_mask, dtype=torch.bool),
        "image_tuple": (images_crop, images_ori, spatial),
    }


class UnlimitedOCRDataset(Dataset):
    """Dataset for single-page and multi-page Unlimited-OCR-2 fine-tuning."""

    def __init__(self, jsonl_path: str, tokenizer: Any, max_length: int = 2048, split_name: str = "data"):
        self.samples: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.base_dir = os.path.dirname(os.path.abspath(jsonl_path))

        with open(jsonl_path, "r", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        for line in tqdm(raw_lines, desc=f"Loading {split_name} ({os.path.basename(jsonl_path)})", unit="doc"):
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
        pil_images = [Image.open(p).convert("RGB") for p in resolved_paths]

        prompt = item.get("prompt")
        if not prompt:
            prompt = "<PAGE><image>" * len(pil_images) + "Multi page parsing."

        return build_uocr_multimodal_sample(
            tokenizer=self.tokenizer,
            prompt=prompt,
            response=item["response"],
            pil_images=pil_images,
            max_length=self.max_length,
        )


@dataclass
class DataCollatorForUnlimitedOCR:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        seq_masks = [f["images_seq_mask"] for f in features]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        seq_mask_padded = torch.nn.utils.rnn.pad_sequence(
            seq_masks, batch_first=True, padding_value=False
        )
        attention_mask = input_ids_padded.ne(self.pad_token_id).long()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        images_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for f in features:
            crop_t, ori_t, spatial_t = f["image_tuple"]
            images_batch.append(
                (
                    crop_t.to(device, non_blocking=True),
                    ori_t.to(device, non_blocking=True),
                    spatial_t.to(device, non_blocking=True),
                )
            )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "images": images_batch,
            "images_seq_mask": seq_mask_padded,
        }


def build_model_and_tokenizer(args: argparse.Namespace):
    print(f"[INFO] Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading config from {args.model_name_or_path}...")
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    # Use native PyTorch SDPA path on Windows (avoids runtime MSVC cl.exe Triton JIT requirement)
    if hasattr(config, "sparse_attention_config") and isinstance(config.sparse_attention_config, dict):
        config.sparse_attention_config["dash_use_triton"] = False

    quant_config = None
    if args.use_4bit:
        print("[INFO] Enabling 4-bit QLoRA quantization (NF4, bfloat16 compute)...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=[
                "sam_model",
                "vision_model",
                "projector",
                "lm_head",
                "dash_router",
                "indexer",
            ],
        )

    print(f"[INFO] Loading model {args.model_name_or_path}...")
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        config=config,
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
    parser.add_argument("--max-length", type=int, default=2048, help="Max token sequence length (lower = less VRAM)")
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
    parser.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=1, help="Update progress bar metrics every N steps")
    args = parser.parse_args()

    model, tokenizer = build_model_and_tokenizer(args)
    train_dataset = UnlimitedOCRDataset(args.train_data, tokenizer, max_length=args.max_length, split_name="train")
    eval_dataset = (
        UnlimitedOCRDataset(args.eval_data, tokenizer, max_length=args.max_length, split_name="val")
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
        logging_first_step=True,
        save_steps=args.save_steps,
        save_total_limit=2,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        disable_tqdm=True,  # Handled by LiveTrainingProgressCallback with VRAM + loss + ETA
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForUnlimitedOCR(pad_token_id=tokenizer.pad_token_id),
        callbacks=[LiveTrainingProgressCallback(output_dir=args.output_dir)],
    )

    print("[INFO] Starting training with live tqdm progress bar...")
    trainer.train()

    print(f"[INFO] Saving LoRA adapter & tokenizer to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
