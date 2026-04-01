# Fine-Tuning Reranker Finance

This repo now contains two related pieces:

1. a reproducible dataset pipeline for hedge-fund style stock memo instruction tuning
2. a QLoRA training script for fine-tuning an instruction model on that dataset

## Folder structure

```text
fine-tuning-reranker-finance/
├── data/
│   ├── raw/
│   └── processed/
├── inference/
│   └── compare.py
├── train/
│   └── qlora_train.py
├── scripts/
│   ├── build_jsonl.py
│   ├── fetch_ticker_data.py
│   └── validate_jsonl.py
├── requirements.txt
└── README.md
```

## Dataset

Processed instruction-tuning files live in:

- `data/processed/all_examples.jsonl`
- `data/processed/train.jsonl`
- `data/processed/eval.jsonl`

Each row contains:

- `instruction`
- `input`
- `output`

The target output always uses this section order:

- `Business Overview:`
- `Bullish Thesis:`
- `Bearish Thesis:`
- `Key Risks:`
- `Conclusion:`

## Training

The main training entrypoint is:

- `train/qlora_train.py`

It is designed around the Unsloth QLoRA workflow and uses the local JSONL files in `data/processed/`.

### Example

```bash
python train/qlora_train.py \
  --model-name unsloth/Llama-3.2-3B-Instruct-bnb-4bit \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/eval.jsonl \
  --output-dir artifacts/qlora-memo \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --epochs 3
```

### Optional exports

You can also request additional saves:

- `--save-merged-16bit`
- `--save-gguf`

## Install

```bash
pip install -r requirements.txt
```

If you are on Apple Silicon, some GPU-oriented packages may need adjustment depending on your local stack. The script is primarily aimed at CUDA environments typically used for QLoRA fine-tuning.

## Notes

- the current dataset is small, so overfitting is a real risk
- keep prompts grounded in the provided facts
- do not use the fine-tuned model for unsupported financial claims
