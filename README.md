# Fine-Tuning Reranker Finance

![Unsloth](https://img.shields.io/badge/Unsloth-Encoder%2FReranker%20FT-111827?logo=lightning&logoColor=yellow)
![QLoRA](https://img.shields.io/badge/QLoRA-4bit%20PEFT-16a34a)
![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Transformers](https://img.shields.io/badge/Transformers-4.0%2B-2563eb)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)

Fine-tuning a cross-encoder reranker for financial document retrieval using SEC filings and earnings call transcripts.

## Overview
This repository contains the data pipeline, training scripts, evaluation utilities, and export workflow for a finance-domain reranker.

## Use Case
The goal is to improve retrieval quality for financial RAG systems by reranking candidate chunks from filings and earnings transcripts.

## Features
- SEC filings and earnings transcript ingestion
- Chunking and metadata enrichment
- Query/positive/negative training pair generation
- Cross-encoder reranker fine-tuning
- Evaluation with ranking metrics
- Adapter + merged model export

## Project Structure
.
├── configs/
│   └── train.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       └── train.jsonl
└── scripts/
    ├── train_reranker.py
    └── evaluate_reranker.py

## Installation
```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

Run training on a supported GPU environment (e.g. RunPod) for Unsloth acceleration.

## Training Pipeline
1. Prepare documents
2. Chunk documents
3. Build training pairs
4. Train reranker
5. Evaluate model
6. Export merged model

## Example Commands
```bash
uv run python scripts/train_reranker.py --config configs/train.yaml
uv run python scripts/evaluate_reranker.py --config configs/train.yaml --k 10
```

## Results
- Baseline vs pretrained reranker vs fine-tuned reranker
- MRR / NDCG / Recall@K

## Future Improvements
- More companies
- Harder negatives
- Better section-aware query generation
- Integration into agentic-hedge-fund