#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TICKERS_FILE = ROOT / "tickers.txt"
SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
USER_AGENT = os.getenv("SEC_USER_AGENT", "openclaw-dataset-pipeline/1.0 nik@example.com")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("AV_API_KEY")


def http_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_tickers() -> List[str]:
    if not TICKERS_FILE.exists():
        raise FileNotFoundError(f"Missing {TICKERS_FILE}")
    tickers = []
    for line in TICKERS_FILE.read_text(encoding="utf-8").splitlines():
        symbol = line.strip().upper()
        if symbol:
            tickers.append(symbol)
    return tickers


def load_sec_mapping() -> Dict[str, Dict[str, Any]]:
    payload = http_json(SEC_TICKER_URL)
    mapping = {}
    for _, item in payload.items():
        ticker = str(item.get("ticker", "")).upper()
        if ticker:
            mapping[ticker] = item
    return mapping


def flatten_units(units: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, entries in (units or {}).items():
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    rows.append(entry)
    return rows


def pick_recent(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    usable = [e for e in entries if isinstance(e.get("val"), (int, float))]

    def sort_key(entry: Dict[str, Any]) -> tuple:
        fy = entry.get("fy")
        fy_num = fy if isinstance(fy, int) else -1
        fp = str(entry.get("fp") or "")
        end = str(entry.get("end") or "")
        filed = str(entry.get("filed") or "")
        return (fy_num, fp, end, filed)

    usable.sort(key=sort_key)
    return usable[-8:]


def describe_series(entries: List[Dict[str, Any]], label: str) -> str:
    recent = pick_recent(entries)
    if len(recent) < 2:
        return "unknown"
    values = [e["val"] for e in recent]
    first = values[0]
    last = values[-1]
    if first == 0:
        direction = "changed"
    elif last > first:
        direction = "increasing"
    elif last < first:
        direction = "decreasing"
    else:
        direction = "stable"
    periods = []
    for e in recent[-2:]:
        fy = e.get("fy", "unknown")
        fp = e.get("fp", "")
        periods.append(f"FY{fy}{fp}")
    return f"{direction} based on SEC companyfacts across recent reported periods ({', '.join(periods)})"


def derive_segment_growth(facts: Dict[str, Any]) -> Dict[str, str]:
    segment_growth: Dict[str, str] = {}
    us_gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    candidates = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    ]
    for concept in candidates:
        concept_data = us_gaap.get(concept) or {}
        for unit_name, entries in (concept_data.get("units") or {}).items():
            for entry in entries if isinstance(entries, list) else []:
                frame = str(entry.get("frame", ""))
                if "" in frame:
                    pass
                if re.search(r"[A-Z]{2,}I", frame):
                    segment_growth[frame] = f"reported value available in SEC companyfacts unit {unit_name}"
        if segment_growth:
            break
    return segment_growth


def alpha_vantage_overview(ticker: str) -> Optional[Dict[str, Any]]:
    if not ALPHA_VANTAGE_KEY:
        return None
    params = urllib.parse.urlencode({"function": "OVERVIEW", "symbol": ticker, "apikey": ALPHA_VANTAGE_KEY})
    url = f"https://www.alphavantage.co/query?{params}"
    try:
        data = http_json(url)
        if isinstance(data, dict) and data and "Note" not in data and "Information" not in data:
            time.sleep(12)
            return data
    except Exception:
        return None
    return None


def build_record(ticker: str, mapping: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    entry = mapping.get(ticker)
    if not entry:
        return {
            "ticker": ticker,
            "company_name": "unknown",
            "metrics": {
                "revenue_growth": "unknown",
                "operating_margin_trend": "unknown",
                "free_cash_flow_trend": "unknown",
                "segment_growth": {},
                "valuation_note": "unknown",
            },
            "news": [],
            "context": ["Ticker not found in SEC ticker mapping."],
            "source_notes": {
                "financials_source": "SEC mapping unavailable for this ticker",
                "news_source": "unavailable",
            },
        }

    cik = str(entry["cik_str"]).zfill(10)
    facts = http_json(SEC_FACTS_URL.format(cik=cik))
    us_gaap = (facts.get("facts") or {}).get("us-gaap") or {}

    revenue_entries = flatten_units((us_gaap.get("Revenues") or {}).get("units") or {})
    if not revenue_entries:
        revenue_entries = flatten_units((us_gaap.get("RevenueFromContractWithCustomerExcludingAssessedTax") or {}).get("units") or {})
    if not revenue_entries:
        revenue_entries = flatten_units((us_gaap.get("SalesRevenueNet") or {}).get("units") or {})

    op_income_entries = flatten_units((us_gaap.get("OperatingIncomeLoss") or {}).get("units") or {})
    fcf_entries = flatten_units((us_gaap.get("NetCashProvidedByUsedInOperatingActivities") or {}).get("units") or {})

    av = alpha_vantage_overview(ticker)
    valuation_note = "unknown"
    context: List[str] = []
    if av:
        pe = av.get("PERatio")
        ev_rev = av.get("EVToRevenue")
        market_cap = av.get("MarketCapitalization")
        parts = []
        if pe and pe != "None":
            parts.append(f"PERatio reported by Alpha Vantage: {pe}")
        if ev_rev and ev_rev != "None":
            parts.append(f"EVToRevenue reported by Alpha Vantage: {ev_rev}")
        if market_cap and market_cap != "None":
            parts.append(f"MarketCapitalization reported by Alpha Vantage: {market_cap}")
        if parts:
            valuation_note = "; ".join(parts)
        if av.get("Description"):
            context.append(str(av["Description"]).strip())
        if av.get("Sector"):
            context.append(f"Sector: {av['Sector']}")
        if av.get("Industry"):
            context.append(f"Industry: {av['Industry']}")

    if not context:
        sic = facts.get("sicDescription")
        if sic:
            context.append(f"SIC description: {sic}")
        entity = facts.get("entityName")
        if entity:
            context.append(f"Entity name from SEC: {entity}")

    record = {
        "ticker": ticker,
        "company_name": facts.get("entityName") or entry.get("title") or "unknown",
        "metrics": {
            "revenue_growth": describe_series(revenue_entries, "revenue"),
            "operating_margin_trend": describe_series(op_income_entries, "operating income"),
            "free_cash_flow_trend": describe_series(fcf_entries, "operating cash flow"),
            "segment_growth": derive_segment_growth(facts),
            "valuation_note": valuation_note,
        },
        "news": [],
        "context": context,
        "source_notes": {
            "financials_source": f"SEC companyfacts CIK {cik}",
            "news_source": "unavailable; context derived from SEC/Alpha Vantage business facts",
        },
    }
    return record


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_tickers()
    mapping = load_sec_mapping()

    created = 0
    missing: List[str] = []
    for ticker in tickers:
        try:
            record = build_record(ticker, mapping)
        except Exception as exc:
            missing.append(f"{ticker}: fetch failed ({exc})")
            continue
        path = RAW_DIR / f"{ticker}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        created += 1

    summary = {
        "tickers_requested": len(tickers),
        "raw_files_created": created,
        "missing_fields_or_skipped_data": missing,
        "alpha_vantage_enabled": bool(ALPHA_VANTAGE_KEY),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
