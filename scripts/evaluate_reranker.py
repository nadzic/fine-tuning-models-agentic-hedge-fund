import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def score_pairs(model, tokenizer, pairs: list[tuple[str, str]], max_length: int, batch_size: int, device: torch.device):
    all_scores = []
    model.eval()

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        queries = [x[0] for x in batch]
        passages = [x[1] for x in batch]

        inputs = tokenizer(
            queries,
            passages,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        if logits.shape[-1] == 1:
            scores = logits.squeeze(-1).detach().cpu().numpy()
        else:
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores = probs.detach().cpu().numpy()
        all_scores.extend(scores.tolist())

    return all_scores


def mrr_at_k(relevances: list[int], k: int) -> float:
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            return 1.0 / idx
    return 0.0


def dcg_at_k(relevances: list[int], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        score += rel / np.log2(i + 1)
    return score


def ndcg_at_k(relevances: list[int], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal_dcg


def recall_at_k(relevances: list[int], k: int) -> float:
    total_positives = sum(relevances)
    if total_positives == 0:
        return 0.0
    return sum(relevances[:k]) / total_positives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--eval_file", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_file = args.eval_file or cfg.get("valid_file") or cfg["train_file"]
    checkpoint = args.checkpoint or cfg["output_dir"]
    max_length = int(cfg["max_length"])

    frame = pd.read_json(eval_file, lines=True)
    if "qid" not in frame.columns:
        frame["qid"] = np.arange(len(frame)).astype(str)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pairs = list(zip(frame["query"].tolist(), frame["passage"].tolist()))
    frame["score"] = score_pairs(model, tokenizer, pairs, max_length=max_length, batch_size=args.batch_size, device=device)

    per_query_relevances: dict[str, list[int]] = defaultdict(list)
    for qid, group in frame.groupby("qid"):
        ranked = group.sort_values("score", ascending=False)
        per_query_relevances[str(qid)] = ranked["label"].astype(int).tolist()

    mrr_scores = []
    ndcg_scores = []
    recall_scores = []
    for rels in per_query_relevances.values():
        mrr_scores.append(mrr_at_k(rels, args.k))
        ndcg_scores.append(ndcg_at_k(rels, args.k))
        recall_scores.append(recall_at_k(rels, args.k))

    print(f"Queries: {len(per_query_relevances)}")
    print(f"MRR@{args.k}: {np.mean(mrr_scores):.4f}")
    print(f"NDCG@{args.k}: {np.mean(ndcg_scores):.4f}")
    print(f"Recall@{args.k}: {np.mean(recall_scores):.4f}")


if __name__ == "__main__":
    main()
