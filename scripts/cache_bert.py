"""Download the BERT encoder used by FactLens during deployment builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import BERT_MODEL_NAME


def main() -> None:
    from transformers import AutoModel, AutoTokenizer

    cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or "default cache"
    print(f"Caching {BERT_MODEL_NAME} in {cache_dir}...")
    AutoTokenizer.from_pretrained(BERT_MODEL_NAME, use_fast=False)
    AutoModel.from_pretrained(BERT_MODEL_NAME)
    print("BERT cache is ready.")


if __name__ == "__main__":
    main()
