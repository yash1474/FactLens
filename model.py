"""Training pipeline for the FactLens fake news detector.

Run this file whenever you need to rebuild ``model.pkl`` and ``tfidf.pkl``.

The classifier is a soft-voting ensemble of **Logistic Regression** and
**calibrated PassiveAggressive**, trained on **mean-pooled BERT**
embeddings. ``tfidf.pkl`` is still
trained for **related-article / evidence similarity** in the Flask app, not for
the main fake/real score.
"""

from __future__ import annotations

import argparse
import os
import re
import string
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    import torch


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "model.pkl"
TFIDF_PATH = BASE_DIR / "tfidf.pkl"

# BERT model id used internally by the encoder.
# Override with env ``FACTLENS_BERT_HUB_ID`` for a different BERT model.
_DEFAULT_BERT_HUB_ID = "prajjwal1/bert-tiny"
_hub_override = (os.environ.get("FACTLENS_BERT_HUB_ID") or "").strip()
BERT_MODEL_NAME = _hub_override or _DEFAULT_BERT_HUB_ID
BERT_DISPLAY_NAME = "BERT Model"
BERT_MAX_LENGTH = 256

_encoder_cache: dict[str, tuple[object, object, "torch.device"]] = {}

PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Normalize article text before vectorization."""
    text = "" if pd.isna(value) else str(value)
    text = text.lower().translate(PUNCT_TRANSLATION)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def get_bert_encoder(
    model_name: str = BERT_MODEL_NAME,
) -> tuple[object, object, "torch.device"]:
    """Load BERT once per ``model_name`` for text encoding."""
    if model_name in _encoder_cache:
        return _encoder_cache[model_name]

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    _encoder_cache[model_name] = (model, tokenizer, device)
    return _encoder_cache[model_name]


def mean_pool_embeddings(
    texts: list[str],
    *,
    batch_size: int = 16,
    max_length: int = BERT_MAX_LENGTH,
    model_name: str = BERT_MODEL_NAME,
) -> np.ndarray:
    """Return float32 matrix (n, hidden) using masked mean pooling over last hidden states."""
    import torch

    model, tokenizer, device = get_bert_encoder(model_name=model_name)
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            pooled = (summed / denom).cpu().numpy().astype(np.float32)
            rows.append(pooled)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(rows)


def _standardize_frame(df: pd.DataFrame, label: int | None = None) -> pd.DataFrame:
    df = df.rename(columns={column: column.strip().lower() for column in df.columns})

    if "title" not in df.columns:
        df["title"] = ""
    if "text" not in df.columns:
        df["text"] = ""

    standardized = df[["title", "text"]].copy()
    if label is None:
        standardized["label"] = pd.to_numeric(df["label"], errors="coerce")
    else:
        standardized["label"] = label
    return standardized.dropna(subset=["label"])


def load_datasets(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and merge True, Fake, and WELFake datasets into one clean frame."""
    true_df = _standardize_frame(pd.read_csv(data_dir / "True.csv"), label=1)
    fake_df = _standardize_frame(pd.read_csv(data_dir / "Fake.csv"), label=0)

    welfake_df = _standardize_frame(pd.read_csv(data_dir / "WELFake_Dataset.csv"), label=None)
    # WELFake labels are 1=fake and 0=real, so invert to project convention:
    # REAL=1, FAKE=0.
    welfake_df["label"] = welfake_df["label"].map({0: 1, 1: 0})

    merged = pd.concat([true_df, fake_df, welfake_df], ignore_index=True)
    merged = merged.dropna(subset=["label"]).drop_duplicates(subset=["title", "text"])
    merged["label"] = merged["label"].astype(int)
    merged["title_clean"] = merged["title"].fillna("").map(clean_text)
    merged["content"] = (merged["title"].fillna("") + " " + merged["text"].fillna("")).map(clean_text)
    merged = merged[merged["content"].str.len() > 20]

    full_articles = merged[["content", "label"]].copy()
    headline_examples = merged.loc[merged["title_clean"].str.len() > 20, ["title_clean", "label"]].rename(
        columns={"title_clean": "content"}
    )

    # The app often receives only headlines from NewsAPI or short pasted text.
    # Adding title-only rows prevents the model from treating headline brevity as a fake-news signal.
    training_frame = pd.concat([full_articles, headline_examples], ignore_index=True)
    training_frame = training_frame.drop_duplicates(subset=["content", "label"])
    return training_frame[["content", "label"]]


def _stratified_subsample(frame: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    """Return at most ``n`` rows with class balance; uses sklearn (pandas ``sample`` may lack ``stratify``)."""
    if len(frame) <= n:
        return frame.reset_index(drop=True)
    sampled, _ = train_test_split(
        frame,
        train_size=n,
        stratify=frame["label"],
        random_state=random_state,
    )
    return sampled.reset_index(drop=True)


def train_model(
    test_size: float = 0.2,
    random_state: int = 42,
    max_bert_train: int = 12000,
    max_bert_eval: int | None = 8000,
    bert_batch_size: int = 16,
    bert_model_name: str = BERT_MODEL_NAME,
) -> tuple[VotingClassifier, TfidfVectorizer]:
    df = load_datasets()
    print(f"Loaded {len(df):,} usable articles")
    print(df["label"].value_counts().rename(index={1: "REAL", 0: "FAKE"}).to_string())

    x_train, x_test, y_train, y_test = train_test_split(
        df["content"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
        dtype=np.float32,
    )
    tfidf.fit(x_train)

    train_frame = pd.DataFrame({"content": x_train.tolist(), "label": y_train.tolist()})
    if len(train_frame) > max_bert_train:
        train_frame = _stratified_subsample(train_frame, max_bert_train, random_state)

    print(
        f"\nEncoding {len(train_frame):,} training texts with '{BERT_DISPLAY_NAME}' "
        f"(batch_size={bert_batch_size})...",
        flush=True,
    )
    x_train_emb = mean_pool_embeddings(
        train_frame["content"].tolist(),
        batch_size=bert_batch_size,
        model_name=bert_model_name,
    )
    y_train_emb = train_frame["label"].astype(int).to_numpy()

    logistic_model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state)
    passive_aggressive = CalibratedClassifierCV(
        PassiveAggressiveClassifier(max_iter=1000, random_state=random_state, early_stopping=True),
        cv=3,
        method="sigmoid",
    )
    model = VotingClassifier(
        estimators=[
            ("logistic", logistic_model),
            ("passive_aggressive", passive_aggressive),
        ],
        voting="soft",
        weights=[1.0, 1.15],
    )
    model.fit(x_train_emb, y_train_emb)

    eval_frame = pd.DataFrame({"content": x_test.tolist(), "label": y_test.tolist()})
    if max_bert_eval is not None and len(eval_frame) > max_bert_eval:
        eval_frame = _stratified_subsample(eval_frame, max_bert_eval, random_state)

    print(f"Encoding {len(eval_frame):,} held-out texts for evaluation...", flush=True)
    x_eval_emb = mean_pool_embeddings(
        eval_frame["content"].tolist(),
        batch_size=bert_batch_size,
        model_name=bert_model_name,
    )
    y_eval = eval_frame["label"].astype(int).to_numpy()

    predictions = model.predict(x_eval_emb)
    print(f"\nAccuracy (BERT-embedding ensemble, eval subset): {accuracy_score(y_eval, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_eval, predictions, target_names=["Fake", "Real"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_eval, predictions))

    joblib.dump(model, MODEL_PATH, compress=3)
    joblib.dump(tfidf, TFIDF_PATH, compress=3)
    print(f"\nSaved classifier to {MODEL_PATH}")
    print(f"Saved TF-IDF (evidence similarity) to {TFIDF_PATH}")
    return model, tfidf


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FactLens fake news detector.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-bert-train",
        type=int,
        default=12000,
        help="Max training rows used for BERT embeddings (stratified sample).",
    )
    parser.add_argument(
        "--max-bert-eval",
        type=int,
        default=8000,
        help="Max held-out rows for evaluation metrics; use 0 for full test split (slow).",
    )
    parser.add_argument("--bert-batch-size", type=int, default=16)
    parser.add_argument("--bert-model-name", default=BERT_MODEL_NAME)
    args = parser.parse_args()
    max_eval = None if args.max_bert_eval == 0 else args.max_bert_eval
    train_model(
        test_size=args.test_size,
        random_state=args.random_state,
        max_bert_train=args.max_bert_train,
        max_bert_eval=max_eval,
        bert_batch_size=args.bert_batch_size,
        bert_model_name=args.bert_model_name,
    )


if __name__ == "__main__":
    main()

