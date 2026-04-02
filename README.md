# Fine-Tuning Financial Memo Models

This repo is an end-to-end workflow for building a small financial memo dataset, fine-tuning a QLoRA adapter, and comparing the resulting model against the base model.

At a high level, the flow is:

1. fetch raw company facts into `data/raw/`
2. turn those facts into instruction-tuning JSONL files in `data/processed/`
3. validate the generated dataset
4. run a smoke-test or full QLoRA training job
5. load the saved adapter for inference
6. compare base-model vs fine-tuned outputs on the same prompt

## Repo layout

```text
fine-tuning-models-agentic-hedge-fund/
├── data/
│   ├── raw/
│   └── processed/
├── eval_prompts/
├── inference/
│   ├── compare.py
│   └── inference_test.py
├── runs/
├── scripts/
│   ├── build_jsonl.py
│   ├── fetch_ticker_data.py
│   └── validate_jsonl.py
├── smoke_test_running/
├── train/
│   └── qlora_train.py
├── tickers.txt
├── pyproject.toml
├── run_ci_checks.sh
└── README.md
```

## Install

The easiest setup uses `uv`:

```bash
uv sync --group dev
```

If you prefer `pip`, you can still do:

```bash
pip install -r requirements.txt
```

If you are on Apple Silicon, some GPU-oriented packages may need adjustment depending on your local stack. The training script is primarily aimed at CUDA environments, so local Mac runs should be treated as smoke tests unless you have a compatible setup.

## End-to-End Flow

### 1. Choose tickers

Put one ticker per line in `tickers.txt`.

Example:

```text
NVDA
MSFT
ADBE
INTC
TSLA
```

### 2. Fetch raw company data

This script reads `tickers.txt`, pulls company facts, and writes one JSON file per ticker into `data/raw/`.

```bash
uv run python scripts/fetch_ticker_data.py
```

The raw records contain fields such as:

- `ticker`
- `company_name`
- `metrics`
- `context`
- `source_notes`

### 3. Build the instruction-tuning dataset

This script converts the raw records into JSONL training examples.

```bash
uv run python scripts/build_jsonl.py
```

It writes:

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

### 4. Validate the dataset

Before training, validate that the generated JSONL files are well-formed and contain the required sections.

```bash
uv run python scripts/validate_jsonl.py
```

### 5. Run a smoke-test fine-tune

The main training entrypoint is `train/qlora_train.py`. It fine-tunes a causal language model on the processed JSONL files.

For a quick local smoke test:

```bash
uv run python train/qlora_train.py \
  --model-name unsloth/Qwen3.5-4B \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/eval.jsonl \
  --output-dir artifacts/qwen35-test \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-seq-length 1024 \
  --epochs 1
```

If your local environment struggles with 4-bit loading, you can also try `--no-load-in-4bit`, but that will increase memory usage.

### 6. Run a smoke-test on RunPod

On RunPod or similar remote GPU environments, it helps to redirect Hugging Face and temporary caches into `/workspace` so downloads do not fill the root filesystem.

```bash
HF_HOME=/workspace/.cache/huggingface \
HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
XDG_CACHE_HOME=/workspace/.cache \
TMPDIR=/workspace/tmp \
TRITON_CACHE_DIR=/workspace/.cache/triton \
HF_HUB_DISABLE_XET=1 \
uv run python train/qlora_train.py \
  --model-name unsloth/Qwen3.5-4B \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/eval.jsonl \
  --output-dir /workspace/fine-tuning-models-agentic-hedge-fund/artifacts/smoke-run \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-seq-length 1024 \
  --epochs 1 \
  --logging-steps 1 \
  --save-steps 1000 \
  --eval-steps 1000
```

This is still a real fine-tuning run, but it is intentionally small and mainly meant to confirm that the full pipeline works end to end.

The training output is saved into the folder passed via `--output-dir`. In a smoke test this is typically:

- `artifacts/smoke-run/`

That folder contains the adapter weights, adapter config, tokenizer files, and `metrics.json`.

Example smoke-test output:

![Smoke test training output](runs/smoke-test.png)

[Open the full-size smoke-test screenshot](./runs/smoke-test.png)

## Inference and Comparison

### Single-model inference

Use `inference/inference_test.py` to load the saved adapter and generate one memo from either inline input text or a prompt file.

Example:

```bash
uv run python inference/inference_test.py \
  --adapter-path artifacts/smoke-run \
  --prompt-file eval_prompts/prompt_1_growth_nvda.txt
```

### Base vs fine-tuned comparison

Use `inference/compare.py` to run the same prompt through:

1. the base model
2. the fine-tuned adapter

Example:

```bash
uv run python inference/compare.py \
  --adapter-path artifacts/smoke-run \
  --prompt-file eval_prompts/prompt_1_growth_nvda.txt
```

This prints:

- `=== Prompt ===`
- `=== Base model output ===`
- `=== Fine-tuned output ===`

By default, comparison is deterministic to make the difference easier to judge. You can enable sampling with:

```bash
uv run python inference/compare.py \
  --adapter-path artifacts/smoke-run \
  --prompt-file eval_prompts/prompt_1_growth_nvda.txt \
  --do-sample \
  --temperature 0.7 \
  --top-p 0.9
```

### RunPod evaluation workflow

If you are testing multiple prompts on RunPod, it is convenient to set the cache-related environment variables once per shell and then use shorter commands.

Set these once:

```bash
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export XDG_CACHE_HOME=/workspace/.cache
export TMPDIR=/workspace/tmp
export TRITON_CACHE_DIR=/workspace/.cache/triton
export HF_HUB_DISABLE_XET=1
```

Then run comparisons with shorter commands, for example:

```bash
.venv/bin/python inference/compare.py --adapter-path artifacts/smoke-run --prompt-file eval_prompts/prompt_1_growth_nvda.txt
```

To save the output for later analysis:

```bash
mkdir -p eval_results
.venv/bin/python inference/compare.py --adapter-path artifacts/smoke-run --prompt-file eval_prompts/prompt_1_growth_nvda.txt > eval_results/prompt_1_growth_nvda.txt
```

To both display and save the output:

```bash
mkdir -p eval_results
.venv/bin/python inference/compare.py --adapter-path artifacts/smoke-run --prompt-file eval_prompts/prompt_1_growth_nvda.txt | tee eval_results/prompt_1_growth_nvda.txt
```

To run all prompts in `eval_prompts/` and save each result into `eval_results/`:

```bash
mkdir -p eval_results
for f in eval_prompts/*.txt; do
  name=$(basename "$f")
  .venv/bin/python inference/compare.py --adapter-path artifacts/smoke-run --prompt-file "$f" | tee "eval_results/$name"
done
```

Example inference-test output:

![Inference test output](runs/inference-run-test.png)

[Open the full-size inference-test screenshot](./runs/inference-run-test.png)

## Full Training

Once the smoke test works and the dataset looks good, you can run a larger training job.

Example:

```bash
uv run python train/qlora_train.py \
  --model-name unsloth/Qwen3.5-4B \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/eval.jsonl \
  --output-dir artifacts/qlora-memo \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --epochs 3
```

The training script formats prompts via the tokenizer chat template when available, which makes it safer to switch between chat-tuned model families such as Qwen and Llama without hardcoding model-specific special tokens.

Optional exports:

- `--save-merged-16bit`
- `--save-gguf`

## CI and Local Checks

To run the same checks locally that the Python CI job runs:

```bash
bash run_ci_checks.sh
```

This runs:

- `ruff`
- `basedpyright`
- `pytest` if a `tests/` directory exists
- `compileall` over the repo's Python directories

## Notes

- the current dataset is still small, so overfitting is a real risk
- the repo is currently strongest as a proof of concept and workflow demo
- keep prompts grounded in the provided facts
- do not use the fine-tuned model for unsupported financial claims
