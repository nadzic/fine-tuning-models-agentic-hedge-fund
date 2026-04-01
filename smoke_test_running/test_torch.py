#!/usr/bin/env python3
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-4B"

SYSTEM_PROMPT = (
    "You are a buy-side equity research writer. Write concise hedge-fund style stock memos. "
    "Use only the facts provided in the prompt. Do not invent numbers, catalysts, or claims. "
    "Always preserve the required section order and headings exactly."
)

USER_PROMPT = """Instruction: Write a short stock memo.

Input:
Ticker: NVDA
Facts:
- Revenue grew year over year
- AI demand remains strong
- Valuation is elevated
"""

def main() -> None:
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)

if __name__ == "__main__":
    main()