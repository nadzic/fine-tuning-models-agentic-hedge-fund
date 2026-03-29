import argparse
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml
from datasets import DatasetDict, load_dataset

# Unsloth recommends importing before transformers/peft for full patching.
# On unsupported local hardware (e.g. Apple Silicon), this import can fail,
# so we ignore it here and emit a clear runtime error later when training starts.
try:
    import unsloth  # noqa: F401
except Exception:
    pass
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments


@dataclass
class TrainConfig:
    model_name: str
    train_file: str
    valid_file: str | None
    output_dir: str
    max_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    seed: int
    full_finetuning: bool
    use_4bit: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    use_gradient_checkpointing: str | bool
    optim: str
    lora_target_modules: list[str]
    valid_split_ratio: float


class RerankerTrainer(Trainer):
    def compute_loss(self, model: torch.nn.Module, inputs: dict[str, Any], return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if logits.shape[-1] == 1:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.view(-1), labels.float().view(-1))
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.long().view(-1))

        return (loss, outputs) if return_outputs else loss


def load_config(config_path: str) -> TrainConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(cfg: TrainConfig) -> DatasetDict:
    if cfg.valid_file:
        data_files = {"train": cfg.train_file, "validation": cfg.valid_file}
        return load_dataset("json", data_files=data_files)  # type: ignore[return-value]

    full_train = load_dataset("json", data_files={"train": cfg.train_file})["train"]
    split = full_train.train_test_split(test_size=cfg.valid_split_ratio, seed=cfg.seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def build_model_and_tokenizer(cfg: TrainConfig):
    try:
        from transformers import AutoModelForSequenceClassification
        from unsloth import FastModel, is_bfloat16_supported
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Unsloth stack. Install dependencies with "
            "`uv sync` and run this script on a supported GPU environment "
            "(RunPod / Linux with NVIDIA, AMD, or Intel GPU)."
        ) from exc

    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=cfg.model_name,
            auto_model=AutoModelForSequenceClassification,
            max_seq_length=cfg.max_length,
            dtype=None,
            num_labels=2,
            full_finetuning=cfg.full_finetuning,
            load_in_4bit=cfg.use_4bit,
        )
    except NotImplementedError as exc:
        raise RuntimeError(
            "Unsloth cannot initialize on this machine. "
            "Use this training script on RunPod or another supported GPU instance."
        ) from exc

    if not cfg.full_finetuning:
        model = FastModel.get_peft_model(
            model,
            r=cfg.lora_r,
            target_modules=cfg.lora_target_modules,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
            random_state=cfg.seed,
            use_rslora=False,
            loftq_config=None,
            task_type="SEQ_CLS",
        )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, is_bfloat16_supported


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    if logits.ndim == 1:
        logits = logits[:, None]

    if logits.shape[-1] == 1:
        probs = 1 / (1 + np.exp(-logits[:, 0]))
    else:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores[:, 1] / exp_scores.sum(axis=1)

    preds = (probs >= 0.5).astype(int)
    accuracy = float((preds == labels).mean())
    return {"accuracy": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    datasets = load_data(cfg)
    model, tokenizer, is_bfloat16_supported = build_model_and_tokenizer(cfg)

    def preprocess(batch):
        tokenized = tokenizer(
            batch["query"],
            batch["passage"],
            max_length=cfg.max_length,
            truncation=True,
        )
        tokenized["labels"] = batch["label"]
        return tokenized

    processed = datasets.map(preprocess, batched=True, remove_columns=datasets["train"].column_names)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim=cfg.optim,
        seed=cfg.seed,
    )

    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
