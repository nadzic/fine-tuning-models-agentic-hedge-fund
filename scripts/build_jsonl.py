#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

INSTRUCTION = "Write a hedge-fund style stock memo."
REQUIRED_SECTIONS = [
    "Business Overview:",
    "Bullish Thesis:",
    "Bearish Thesis:",
    "Key Risks:",
    "Conclusion:",
]


def safe_text(value: Any, fallback: str = "unknown") -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        text = value.strip()
        return text if text else fallback
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def bullet_list(items: List[str], fallback: str) -> List[str]:
    cleaned = []
    for item in items:
        text = safe_text(item, fallback="").strip()
        if text:
            cleaned.append(text)
    return cleaned if cleaned else [fallback]


def format_segment_growth(segment_growth: Any) -> List[str]:
    if not isinstance(segment_growth, dict) or not segment_growth:
        return ["Segment growth details: unknown"]
    lines = []
    for key in sorted(segment_growth):
        val = safe_text(segment_growth.get(key))
        lines.append(f"Segment growth - {key}: {val}")
    return lines


def build_input(record: Dict[str, Any], angle: str) -> str:
    ticker = safe_text(record.get("ticker"))
    company = safe_text(record.get("company_name"))
    metrics = record.get("metrics") or {}

    metric_lines = [
        f"Revenue growth: {safe_text(metrics.get('revenue_growth'))}",
        f"Operating margin trend: {safe_text(metrics.get('operating_margin_trend'))}",
        f"Free cash flow trend: {safe_text(metrics.get('free_cash_flow_trend'))}",
        f"Valuation note: {safe_text(metrics.get('valuation_note'))}",
        f"Angle emphasis: {angle}",
    ]
    metric_lines.extend(format_segment_growth(metrics.get("segment_growth")))

    news_items = bullet_list(record.get("news") or [], "No recent news items available; rely on company and financial context.")
    context_items = bullet_list(record.get("context") or [], "Limited context available from structured company and financial disclosures.")

    return "\n".join(
        [
            f"Ticker: {ticker}",
            f"Company: {company}",
            "Metrics:",
            *[f"- {line}" for line in metric_lines],
            "News:",
            *[f"- {line}" for line in news_items],
            "Context:",
            *[f"- {line}" for line in context_items],
        ]
    )


def memo_section(section: str, body: str) -> str:
    return f"{section}\n{body.strip()}"


def build_output(record: Dict[str, Any], angle_key: str) -> str:
    ticker = safe_text(record.get("ticker"))
    company = safe_text(record.get("company_name"))
    metrics = record.get("metrics") or {}
    revenue_growth = safe_text(metrics.get("revenue_growth"))
    operating_margin_trend = safe_text(metrics.get("operating_margin_trend"))
    free_cash_flow_trend = safe_text(metrics.get("free_cash_flow_trend"))
    valuation_note = safe_text(metrics.get("valuation_note"))
    segment_lines = format_segment_growth(metrics.get("segment_growth"))
    segment_summary = "; ".join(segment_lines)
    context_items = bullet_list(record.get("context") or [], "Limited context available from structured company and financial disclosures.")
    context_summary = "; ".join(context_items[:2])

    angle_templates = {
        "growth": {
            "bull": f"The main upside case rests on revenue growth of {revenue_growth} and segment commentary of {segment_summary}. If these trends prove durable, {company} could compound from a stronger operating base.",
            "bear": f"The growth case is only as strong as the disclosed trajectory. Revenue growth is described as {revenue_growth}, and limited incremental detail beyond {segment_summary} may constrain conviction.",
            "risks": f"Primary risks include growth slowing from current levels, segment performance diverging from the disclosed pattern, and context factors such as {context_summary} undermining the upside narrative.",
            "conclusion": f"On balance, {ticker} screens as a growth-oriented idea only to the extent the disclosed revenue and segment trends continue to support the business trajectory. Positioning should remain evidence-driven.",
        },
        "margin": {
            "bull": f"The constructive view centers on operating margin trend of {operating_margin_trend} and free cash flow trend of {free_cash_flow_trend}. If efficiency is improving sustainably, earnings quality may strengthen even without outsized top-line acceleration.",
            "bear": f"The efficiency case is incomplete if margin gains are temporary or if free cash flow trend of {free_cash_flow_trend} does not confirm underlying improvement. Investors should avoid overstating operating leverage from limited facts.",
            "risks": f"Key risks include cost inflation, reinvestment needs, and execution issues that could reverse the reported operating margin trend of {operating_margin_trend} or weaken cash conversion.",
            "conclusion": f"{ticker} appears most interesting as a margin story if the disclosed operating discipline and cash flow profile remain intact. The memo should stay anchored to evidence rather than extrapolated profitability.",
        },
        "risk": {
            "bull": f"The stock can still merit attention if existing business quality and context support resilience, particularly where valuation note of {valuation_note} suggests expectations may already reflect some caution.",
            "bear": f"The principal downside argument is that valuation and risk may not leave much room for error. The disclosed valuation note is {valuation_note}, while revenue growth is {revenue_growth} and margin trend is {operating_margin_trend}, which may not justify a more aggressive stance.",
            "risks": f"Primary risks include multiple compression, deterioration in revenue growth or profitability, and any company-specific issues hinted at by {context_summary}.", 
            "conclusion": f"Overall, {ticker} is best framed as a risk-aware idea: potentially investable, but only with discipline around valuation, business quality, and the limits of the available facts.",
        },
    }

    template = angle_templates[angle_key]
    business_overview = f"{company} ({ticker}) is evaluated using structured company and financial disclosure data. Available facts indicate revenue growth of {revenue_growth}, operating margin trend of {operating_margin_trend}, free cash flow trend of {free_cash_flow_trend}, and segment detail summarized as {segment_summary}."

    return "\n\n".join(
        [
            memo_section("Business Overview:", business_overview),
            memo_section("Bullish Thesis:", template["bull"]),
            memo_section("Bearish Thesis:", template["bear"]),
            memo_section("Key Risks:", template["risks"]),
            memo_section("Conclusion:", template["conclusion"]),
        ]
    )


def example_specs() -> List[Tuple[str, str]]:
    return [
        ("growth", "growth / upside angle"),
        ("margin", "margin / efficiency angle"),
        ("risk", "risk / valuation angle"),
    ]


def deterministic_eval(example: Dict[str, Any], ratio: float = 0.2) -> bool:
    key = example["input"] + "\n" + example["output"]
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < ratio


def load_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not RAW_DIR.exists():
        return records
    for path in sorted(RAW_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            record = json.load(f)
        records.append(record)
    return records


def generate_examples(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], List[str]]:
    examples: List[Dict[str, str]] = []
    missing_notes: List[str] = []

    for record in records:
        ticker = safe_text(record.get("ticker"))
        company = safe_text(record.get("company_name"))
        if company == "unknown":
            missing_notes.append(f"{ticker}: company_name missing")

        metrics = record.get("metrics") or {}
        for field in ["revenue_growth", "operating_margin_trend", "free_cash_flow_trend", "valuation_note"]:
            if safe_text(metrics.get(field)) == "unknown":
                missing_notes.append(f"{ticker}: {field} unknown")

        if not isinstance(metrics.get("segment_growth"), dict) or not metrics.get("segment_growth"):
            missing_notes.append(f"{ticker}: segment_growth unknown")
        if not (record.get("news") or []):
            missing_notes.append(f"{ticker}: news unavailable; using context facts")
        if not (record.get("context") or []):
            missing_notes.append(f"{ticker}: context missing")

        for angle_key, angle_label in example_specs():
            examples.append(
                {
                    "instruction": INSTRUCTION,
                    "input": build_input(record, angle_label),
                    "output": build_output(record, angle_key),
                }
            )
    return examples, missing_notes


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    examples, missing_notes = generate_examples(records)

    train_rows: List[Dict[str, str]] = []
    eval_rows: List[Dict[str, str]] = []
    for row in examples:
        if deterministic_eval(row):
            eval_rows.append(row)
        else:
            train_rows.append(row)

    write_jsonl(PROCESSED_DIR / "all_examples.jsonl", examples)
    write_jsonl(PROCESSED_DIR / "train.jsonl", train_rows)
    write_jsonl(PROCESSED_DIR / "eval.jsonl", eval_rows)

    summary = {
        "tickers_processed": len(records),
        "raw_files_detected": len(records),
        "total_examples": len(examples),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "missing_fields_or_skipped_data": sorted(set(missing_notes)),
        "required_sections": REQUIRED_SECTIONS,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
