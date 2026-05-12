"""Time-capped BERT fine-tuning for FactLens (``train_bert``).

Keeps the existing TF-IDF + Logistic/Passive-Aggressive artifacts intact and
writes a separate optional BERT model under ``models/train_bert``. Defaults
cap wall-clock training at 10 minutes and use a balanced subset for a quick
CPU-friendly run.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model import BASE_DIR, load_datasets


BERT_MODEL_NAME = "prajjwal1/bert-tiny"
BERT_OUTPUT_DIR = BASE_DIR / "models" / "train_bert"


class NewsDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def balanced_sample(df, max_examples: int, random_state: int):
    per_class = max_examples // 2
    parts = []
    for label in sorted(df["label"].unique()):
        label_df = df[df["label"] == label]
        take = min(per_class, len(label_df))
        parts.append(label_df.sample(n=take, random_state=random_state))
    sampled = (
        np.random.default_rng(random_state)
        .permutation(np.concatenate([part.index.to_numpy() for part in parts]))
        .tolist()
    )
    return df.loc[sampled].reset_index(drop=True)


def evaluate(model, loader, device) -> tuple[float, str]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(batch["labels"].tolist())
    report = classification_report(labels, predictions, target_names=["Fake", "Real"], zero_division=0)
    return accuracy_score(labels, predictions), report


def fine_tune_bert(
    model_name: str = BERT_MODEL_NAME,
    output_dir: Path = BERT_OUTPUT_DIR,
    max_examples: int = 600,
    max_minutes: float = 10.0,
    batch_size: int = 8,
    max_length: int = 160,
    learning_rate: float = 2e-5,
    random_state: int = 42,
) -> None:
    set_seed(random_state)
    df = balanced_sample(load_datasets(), max_examples=max_examples, random_state=random_state)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_data = NewsDataset(train_df["content"].tolist(), train_df["label"].astype(int).tolist(), tokenizer, max_length)
    test_data = NewsDataset(test_df["content"].tolist(), test_df["label"].astype(int).tolist(), tokenizer, max_length)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    deadline = time.monotonic() + (max_minutes * 60)
    steps = 0
    model.train()
    for epoch in range(20):
        for batch in train_loader:
            if time.monotonic() >= deadline:
                print(f"Time budget reached after {steps} steps.")
                break
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % 10 == 0:
                print(f"step={steps} epoch={epoch + 1} loss={loss.item():.4f}")
        else:
            continue
        break

    accuracy, report = evaluate(model, test_loader, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "base_model": model_name,
        "labels": {"0": "Fake", "1": "Real"},
        "max_examples": int(max_examples),
        "train_examples": int(len(train_df)),
        "test_examples": int(len(test_df)),
        "steps": int(steps),
        "accuracy": float(accuracy),
        "max_length": int(max_length),
    }
    (output_dir / "train_bert_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nBERT accuracy: {accuracy:.4f}")
    print(report)
    print(f"Saved BERT model to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained BERT for FactLens (default: ≤10 min wall clock, writes models/train_bert).",
    )
    parser.add_argument("--model-name", default=BERT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=BERT_OUTPUT_DIR)
    parser.add_argument("--max-examples", type=int, default=600)
    parser.add_argument("--max-minutes", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    fine_tune_bert(**vars(args))


if __name__ == "__main__":
    main()
