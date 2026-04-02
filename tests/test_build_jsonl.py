from __future__ import annotations

from scripts.build_jsonl import (
    REQUIRED_SECTIONS,
    build_input,
    deterministic_eval,
    generate_examples,
    safe_text,
)


def _sample_record() -> dict[str, object]:
    return {
        "ticker": "NVDA",
        "company_name": "NVIDIA Corp",
        "metrics": {
            "revenue_growth": "increasing",
            "operating_margin_trend": "stable",
            "free_cash_flow_trend": "increasing",
            "segment_growth": {"AI": "strong"},
            "valuation_note": "elevated",
        },
        "news": ["Demand remains strong."],
        "context": ["Sector: TECHNOLOGY", "Industry: SEMICONDUCTORS"],
    }


def test_generate_examples_creates_three_angle_variants() -> None:
    examples, missing_notes = generate_examples([_sample_record()])

    assert len(examples) == 3
    assert missing_notes == []
    for row in examples:
        assert row["instruction"]
        assert row["input"]
        for section in REQUIRED_SECTIONS:
            assert section in row["output"]


def test_build_input_contains_metrics_and_angle() -> None:
    result = build_input(_sample_record(), "growth / upside angle")

    assert "Ticker: NVDA" in result
    assert "Revenue growth: increasing" in result
    assert "Angle emphasis: growth / upside angle" in result


def test_deterministic_eval_is_stable() -> None:
    example = {"input": "A", "output": "B"}

    first = deterministic_eval(example)
    second = deterministic_eval(example)

    assert first == second


def test_safe_text_normalizes_empty_values() -> None:
    assert safe_text(None) == "unknown"
    assert safe_text("   ") == "unknown"
    assert safe_text({"a": 1}).startswith("{")
