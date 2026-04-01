# Hedge-Fund Memo Fine-Tuning Dataset Pipeline

This project lives in the `fine-tuning-reranker-finance/` folder and should be treated as the repo root for dataset work.

It contains a reproducible pipeline for building a small instruction-tuning dataset that teaches a model to write concise hedge-fund style stock memos with the following fixed structure:

- Business Overview
- Bullish Thesis
- Bearish Thesis
- Key Risks
- Conclusion

## Project layout

```text
fine-tuning-reranker-finance/
├── README.md
├── tickers.txt
├── data/
│   ├── raw/
│   └── processed/
└── scripts/
    ├── fetch_ticker_data.py
    ├── build_jsonl.py
    └── validate_jsonl.py
```

## Data sources

The pipeline is designed to use structured, traceable sources only:

- **SEC EDGAR / data.sec.gov** for company identity and companyfacts-based financial disclosures
- **Alpha Vantage** for supplemental company overview / valuation fields when `ALPHA_VANTAGE_API_KEY` is configured

If Alpha Vantage is not configured, the pipeline runs in SEC-only mode. News is intentionally left empty unless a structured approved source is added; when news is unavailable, the dataset uses context bullets derived from company and financial facts.

## Raw normalized schema

Each ticker is stored as one JSON file under `data/raw/{ticker}.json` using this schema:

```json
{
  "ticker": "...",
  "company_name": "...",
  "metrics": {
    "revenue_growth": "...",
    "operating_margin_trend": "...",
    "free_cash_flow_trend": "...",
    "segment_growth": {},
    "valuation_note": "..."
  },
  "news": [],
  "context": [],
  "source_notes": {
    "financials_source": "...",
    "news_source": "..."
  }
}
```

Rules:

- no invented hard numbers
- unavailable metrics must be set to `"unknown"`
- facts should remain traceable to the source notes and raw disclosures

## Instruction-tuning dataset schema

Each JSONL row contains:

- `instruction`
- `input`
- `output`

The instruction is always:

```text
Write a hedge-fund style stock memo.
```

The generated `output` always contains these exact sections in this order:

- `Business Overview:`
- `Bullish Thesis:`
- `Bearish Thesis:`
- `Key Risks:`
- `Conclusion:`

The build script creates exactly **3 examples per ticker**:

- Example A: growth / upside angle
- Example B: margin / efficiency angle
- Example C: risk / valuation angle

## Files

- `scripts/fetch_ticker_data.py` — reads `tickers.txt` and writes normalized raw JSON files into `data/raw/`
- `scripts/build_jsonl.py` — reads raw JSON files and generates:
  - `data/processed/all_examples.jsonl`
  - `data/processed/train.jsonl`
  - `data/processed/eval.jsonl`
- `scripts/validate_jsonl.py` — validates JSONL structure and required memo sections

## How to rebuild the dataset

Run all commands from inside `fine-tuning-reranker-finance/`.

1. Put one ticker per line in `tickers.txt`
2. Optionally export an Alpha Vantage key:

```bash
export ALPHA_VANTAGE_API_KEY=your_key_here
```

3. Fetch raw data:

```bash
python3 scripts/fetch_ticker_data.py
```

4. Build JSONL files:

```bash
python3 scripts/build_jsonl.py
```

5. Validate outputs:

```bash
python3 scripts/validate_jsonl.py
```

## Current dataset status

The current `tickers.txt` contains 10 symbols:

- AMZN
- MSFT
- GOOGL
- META
- NVDA
- AMD
- AVGO
- AAPL
- NFLX
- COIN

Current output counts:

- Tickers processed: 10
- Raw files created: 10
- Total examples: 30
- Train examples: 25
- Eval examples: 5

Current limitations:

- Alpha Vantage is now configured and `valuation_note` is populated from Alpha Vantage overview fields for all current tickers
- no approved structured news source is configured, so `news` is empty and `context` is derived from SEC and Alpha Vantage business facts
- `segment_growth` is currently `unknown` for these tickers based on the available extraction logic and companyfacts fields

These counts rebuild deterministically from the raw files currently present. If the raw inputs change, the deterministic train/eval split may also change because it is hash-based on example content.
