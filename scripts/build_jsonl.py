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


def first_meaningful(items: List[str], fallback: str) -> str:
    for item in items:
        text = safe_text(item, fallback="").strip()
        if text and not text.lower().startswith("no recent news items available"):
            return text
    return fallback


def compress_context(record: Dict[str, Any]) -> Dict[str, str]:
    context_items = bullet_list(record.get("context") or [], "Limited context available from structured company and financial disclosures.")
    summary = first_meaningful(context_items, "Limited context available from structured company and financial disclosures.")
    sector = "unknown"
    industry = "unknown"
    for item in context_items:
        if item.startswith("Sector:"):
            sector = item.split(":", 1)[1].strip() or "unknown"
        elif item.startswith("Industry:"):
            industry = item.split(":", 1)[1].strip() or "unknown"
    return {
        "summary": summary,
        "sector": sector,
        "industry": industry,
    }


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
    context = compress_context(record)
    context_summary = context["summary"]
    sector = context["sector"]
    industry = context["industry"]

    business_overview = (
        f"{company} ({ticker}) operates in {sector}"
        f"{'' if industry == 'unknown' else f', specifically {industry}'}"
        f". The factual setup is mixed: revenue growth is {revenue_growth}, operating margin trend is {operating_margin_trend}, "
        f"and free cash flow trend is {free_cash_flow_trend}. Valuation context is {valuation_note}."
    )

    angle_templates = {
        "growth": {
            "bull": (
                f"The long case rests primarily on the reported revenue trajectory. Revenue growth is {revenue_growth}, "
                f"while free cash flow is {free_cash_flow_trend}. If top-line momentum is durable and cash generation continues to confirm the trend, "
                f"the setup can support further upside even without detailed segment disclosure."
            ),
            "bear": (
                f"The pushback is that the growth evidence is incomplete. Segment detail remains {segment_summary}, and operating margin trend is {operating_margin_trend}. "
                f"That leaves open the risk that revenue direction does not translate into better incremental economics."
            ),
            "risks": (
                f"Key risks are a break in the reported revenue trend, weaker conversion of growth into cash flow, and business-specific execution issues tied to the disclosed context: {context_summary}."
            ),
            "conclusion": (
                f"Net, {ticker} fits a growth-oriented memo only if the reported revenue trajectory remains intact and cash generation continues to back it up. The evidence supports interest, not aggressive extrapolation."
            ),
        },
        "margin": {
            "bull": (
                f"The more credible bull case is around operating discipline. Operating margin trend is {operating_margin_trend}, and free cash flow trend is {free_cash_flow_trend}. "
                f"If cost control and cash conversion are holding, earnings quality may be improving even if revenue growth is only {revenue_growth}."
            ),
            "bear": (
                f"The counterpoint is that margin narratives are fragile when top-line support is limited. Revenue growth is {revenue_growth}, and the dataset does not provide segment evidence beyond {segment_summary}. "
                f"That makes it hard to underwrite operating leverage with high confidence."
            ),
            "risks": (
                "The main risks are reversal in margin trend, weaker cash conversion, and a need to reinvest more heavily than the current operating profile implies."
            ),
            "conclusion": (
                f"Viewed through a buy-side lens, {ticker} is most compelling as an efficiency story if the reported margin and free cash flow trends prove durable. Without that confirmation, the thesis weakens quickly."
            ),
        },
        "risk": {
            "bull": (
                f"The constructive angle is that the business still shows support from core operating facts. Revenue growth is {revenue_growth}, free cash flow trend is {free_cash_flow_trend}, and valuation context is {valuation_note}. "
                f"If expectations are already calibrated to these conditions, downside may be more balanced than headline multiples suggest."
            ),
            "bear": (
                f"The more cautious view is straightforward: valuation already matters here. The disclosed valuation context is {valuation_note}, while operating margin trend is {operating_margin_trend} and segment detail is {segment_summary}. "
                f"If execution slips, the setup leaves limited room for error."
            ),
            "risks": (
                f"Primary risks are multiple compression, deterioration in the reported operating profile, and company-specific issues implied by the business context: {context_summary}."
            ),
            "conclusion": (
                f"Bottom line: {ticker} belongs in a risk-aware memo. The business may be investable, but the case should be framed around valuation discipline and the limits of the disclosed evidence."
            ),
        },
    }

    template = angle_templates[angle_key]

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
