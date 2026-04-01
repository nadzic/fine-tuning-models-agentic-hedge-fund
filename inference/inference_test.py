#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from unsloth import FastLanguageModel

SYSTEM_PROMPT = (
    "You are a buy-side equity research writer. Write concise hedge-fund style stock memos. "
    "Use only the facts provided in the prompt. Do not invent numbers, catalysts, or claims. "
    "Always preserve the required section order and headings exactly."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned finance memo adapter.")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("artifacts/smoke-run"),
        help="Path to the saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Model max sequence length.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=400,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling.",
    )
    parser.add_argument(
        "--input-text",
        type=str,
        default=None,
        help="Inline input facts for the memo.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional text file containing the input facts.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Write a hedge-fund style stock memo.",
        help="Instruction prefix used during training.",
    )
    return parser.parse_args()


def load_input_text(args: argparse.Namespace) -> str:
    if args.input_text:
        return args.input_text.strip()
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    raise ValueError("Provide either --input-text or --prompt-file.")


def build_prompt(instruction: str, input_text: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Instruction: {instruction}\n\nInput:\n{input_text}",
        },
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Instruction: {instruction}\n\n"
        f"Input:\n{input_text}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


def main() -> None:
    args = parse_args()
    input_text = load_input_text(args)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter_path),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    prompt = build_prompt(args.instruction, input_text, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            use_cache=True,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n=== Generated memo ===\n")
    print(response.strip())


if __name__ == "__main__":
    main()