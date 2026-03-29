# Fine-Tuning Reranker Finance

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
...

## Installation
...

## Training Pipeline
1. Prepare documents
2. Chunk documents
3. Build training pairs
4. Train reranker
5. Evaluate model
6. Export merged model

## Example Commands
...

## Results
- Baseline vs pretrained reranker vs fine-tuned reranker
- MRR / NDCG / Recall@K

## Future Improvements
- More companies
- Harder negatives
- Better section-aware query generation
- Integration into agentic-hedge-fund