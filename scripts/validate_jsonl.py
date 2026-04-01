#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [
    ROOT / "data" / "processed" / "all_examples.jsonl",
    ROOT / "data" / "processed" / "train.jsonl",
    ROOT / "data" / "processed" / "eval.jsonl",
]
REQUIRED_KEYS = {"instruction", "input", "output"}
REQUIRED_SECTIONS = [
    "Business Overview:",
    "Bullish Thesis:",
    "Bearish Thesis:",
    "Key Risks:",
    "Conclusion:",
]


def validate_file(path: Path) -> List[str]:
    errors: List[str] = []
    if not path.exists():
        errors.append(f"{path}: file does not exist")
        return errors

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: invalid JSON - {exc}")
                continue

            missing_keys = REQUIRED_KEYS - set(row)
            if missing_keys:
                errors.append(f"{path}:{line_no}: missing keys {sorted(missing_keys)}")
                continue

            output = row.get("output", "")
            for section in REQUIRED_SECTIONS:
                if section not in output:
                    errors.append(f"{path}:{line_no}: missing required section '{section}'")

    return errors


def main() -> int:
    paths = [Path(arg).resolve() for arg in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_FILES
    all_errors: List[str] = []
    for path in paths:
        all_errors.extend(validate_file(path))

    if all_errors:
        print("Validation failed:")
        for error in all_errors:
            print(f"- {error}")
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
