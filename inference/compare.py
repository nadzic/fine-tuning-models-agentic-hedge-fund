#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned finance memo generation outputs.")
    parser.add_argument("--prompt-file", type=Path, required=False, help="Optional file containing a full prompt to test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("compare.py scaffold created.")
    if args.prompt_file:
        print(f"Prompt file: {args.prompt_file}")
        print("Implement model loading / side-by-side generation here once the fine-tuned checkpoint is available.")


if __name__ == "__main__":
    main()
