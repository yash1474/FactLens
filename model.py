"""Training pipeline for the FactLens fake news detector.

Run this file whenever you need to rebuild ``model.pkl`` and ``tfidf.pkl``.
The model intentionally uses TF-IDF + Logistic Regression for fast CPU-only
training and low-latency inference.
"""

from __future__ import annotations

import argparse
import re
import string
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "model.pkl"
TFIDF_PATH = BASE_DIR / "tfidf.pkl"

PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Normalize article text before vectorization."""
    text = "" if pd.isna(value) else str(value)
    text = text.lower().translate(PUNCT_TRANSLATION)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


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


def train_model(test_size: float = 0.2, random_state: int = 42) -> tuple[VotingClassifier, TfidfVectorizer]:
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
    x_train_tfidf = tfidf.fit_transform(x_train)
    x_test_tfidf = tfidf.transform(x_test)

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
    model.fit(x_train_tfidf, y_train)

    predictions = model.predict(x_test_tfidf)
    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["Fake", "Real"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(model, MODEL_PATH, compress=3)
    joblib.dump(tfidf, TFIDF_PATH, compress=3)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {TFIDF_PATH}")
    return model, tfidf


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FactLens fake news detector.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    train_model(test_size=args.test_size, random_state=args.random_state)


if __name__ == "__main__":
    main()
