from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_jsonl import REQUIRED_SECTIONS, validate_file


def _valid_row() -> dict[str, str]:
    output = "\n\n".join(f"{section}\nExample content." for section in REQUIRED_SECTIONS)
    return {
        "instruction": "Write a hedge-fund style stock memo.",
        "input": "Ticker: NVDA",
        "output": output,
    }


def test_validate_file_passes_with_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "valid.jsonl"
    path.write_text(json.dumps(_valid_row()) + "\n", encoding="utf-8")
    assert validate_file(path) == []


def test_validate_file_reports_missing_required_section(tmp_path: Path) -> None:
    path = tmp_path / "invalid_section.jsonl"
    row = _valid_row()
    row["output"] = row["output"].replace("Conclusion:", "Final Thoughts:")
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    errors = validate_file(path)

    assert errors
    assert "missing required section 'Conclusion:'" in errors[0]


def test_validate_file_reports_invalid_json_line(tmp_path: Path) -> None:
    path = tmp_path / "invalid_json.jsonl"
    path.write_text('{"instruction":"x","input":"y","output":"z"\n', encoding="utf-8")

    errors = validate_file(path)

    assert errors
    assert "invalid JSON" in errors[0]
