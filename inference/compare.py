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
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned finance memo generation outputs.")
    parser.add_argument(
        "--base-model",
        default="unsloth/Qwen3.5-4B",
        help="Base model name used before fine-tuning.",
    )
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
        help="Sampling temperature when --do-sample is enabled.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling when --do-sample is enabled.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling. By default, generation is deterministic for easier comparison.",
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


def generate_response(
    model_name: str,
    instruction: str,
    input_text: str,
    max_seq_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    prompt = build_prompt(instruction, input_text, tokenizer)
    inputs = tokenizer(
        text=prompt,
        return_tensors="pt",
    ).to(model.device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return response


def main() -> None:
    args = parse_args()
    input_text = load_input_text(args)

    base_output = generate_response(
        model_name=args.base_model,
        instruction=args.instruction,
        input_text=input_text,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    fine_tuned_output = generate_response(
        model_name=str(args.adapter_path),
        instruction=args.instruction,
        input_text=input_text,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n=== Prompt ===\n")
    print(input_text)
    print("\n=== Base model output ===\n")
    print(base_output)
    print("\n=== Fine-tuned output ===\n")
    print(fine_tuned_output)


if __name__ == "__main__":
    main()
