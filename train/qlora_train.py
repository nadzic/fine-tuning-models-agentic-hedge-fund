#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILE = ROOT / "data" / "processed" / "train.jsonl"
DEFAULT_EVAL_FILE = ROOT / "data" / "processed" / "eval.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "qlora-memo"
DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_SEED = 3407

SYSTEM_PROMPT = (
    "You are a buy-side equity research writer. Write concise hedge-fund style stock memos. "
    "Use only the facts provided in the prompt. Do not invent numbers, catalysts, or claims. "
    "Always preserve the required section order and headings exactly."
)


@dataclass
class Row:
    instruction: str
    input: str
    output: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_prompt(row: Dict[str, str], eos_token: str) -> Dict[str, str]:
    instruction = str(row["instruction"]).strip()
    input_text = str(row["input"]).strip()
    output_text = str(row["output"]).strip()

    text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Instruction: {instruction}\n\n"
        f"Input:\n{input_text}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"{output_text}{eos_token}"
    )
    return {"text": text}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for finance memo generation with Unsloth.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Base instruct model to fine-tune.")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE, help="Path to train.jsonl")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE, help="Path to eval.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to save adapters and tokenizer")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH, help="Maximum sequence length")
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="Load the base model in 4-bit")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit", help="Disable 4-bit loading")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=2, help="Per-device eval batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--logging-steps", type=int, default=1, help="Logging interval")
    parser.add_argument("--save-steps", type=int, default=10, help="Checkpoint save interval")
    parser.add_argument("--eval-steps", type=int, default=10, help="Eval interval")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--save-merged-16bit", action="store_true", help="Also export a merged 16-bit model")
    parser.add_argument("--save-gguf", action="store_true", help="Also export a GGUF model if supported")
    return parser.parse_args()


def ensure_files(train_file: Path, eval_file: Path) -> None:
    missing = [str(p) for p in [train_file, eval_file] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {', '.join(missing)}")


def load_jsonl_dataset(train_file: Path, eval_file: Path) -> Dict[str, Dataset]:
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "eval": str(eval_file),
        },
    )
    return dataset


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_files(args.train_file, args.eval_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    raw_dataset = load_jsonl_dataset(args.train_file, args.eval_file)
    eos_token = tokenizer.eos_token or ""
    train_dataset = raw_dataset["train"].map(lambda row: format_prompt(row, eos_token))
    eval_dataset = raw_dataset["eval"].map(lambda row: format_prompt(row, eos_token))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=SFTConfig(
            output_dir=str(args.output_dir),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            save_strategy="steps",
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            report_to="none",
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        ),
    )

    train_result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics = train_result.metrics
    if eval_dataset.num_rows > 0:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if args.save_merged_16bit:
        merged_dir = args.output_dir / "merged-16bit"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    if args.save_gguf:
        gguf_dir = args.output_dir / "gguf"
        model.save_pretrained_gguf(str(gguf_dir), tokenizer)

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "train_rows": train_dataset.num_rows,
        "eval_rows": eval_dataset.num_rows,
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
    }, indent=2))


if __name__ == "__main__":
    main()
