"""
Stage 0 — Cleaning & normalisation.

Reads data/raw/posts_raw.csv (needs at least: brand, text)
Writes data/processed/posts_clean.csv (adds: text_clean, plus keeps text, brand, created_utc if present)

IMPORTANT design choice:
  We do only *light* cleaning here and KEEP the original casing and punctuation.
  VADER and the transformer both USE capitalisation ("GREAT" vs "great") and
  punctuation ("good!!!") as intensity signals, so stripping them would throw
  away sentiment information. The heavy tokenisation (lowercasing, stop-word
  removal, lemmatisation) needed for topic modelling happens later, in topics.py.
"""

import re
import sys
from pathlib import Path

import pandas as pd

# make "import config" work when run from anywhere
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

URL_RE = re.compile(r"http\S+|www\.\S+")
MARKUP_RE = re.compile(r"&[a-z]+;|\[.*?\]\(.*?\)|[*_>#`~]")  # html entities + markdown
WHITESPACE_RE = re.compile(r"\s+")


def light_clean(text: str) -> str:
    """Remove URLs, markdown/HTML markup and collapse whitespace. Keep case & punctuation."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = MARKUP_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_english(text: str) -> bool:
    """Best-effort language filter. langdetect is imported lazily so the file
    still runs if you haven't installed it yet (it just keeps everything)."""
    try:
        from langdetect import detect, LangDetectException
    except ImportError:
        return True
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def main():
    if not config.RAW_FILE.exists():
        raise SystemExit(
            f"No raw data at {config.RAW_FILE}.\n"
            "Run collect_reddit.py first, or drop a CSV with 'brand' and 'text' "
            "columns there (e.g. a Kaggle download)."
        )

    df = pd.read_csv(config.RAW_FILE)
    start_n = len(df)
    print(f"Loaded {start_n:,} rows from {config.RAW_FILE.name}")

    required = {"brand", "text"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Raw file is missing required column(s): {missing}")

    # 1. light clean
    df["text_clean"] = df["text"].apply(light_clean)

    # 2. drop empties and near-duplicates
    df = df[df["text_clean"].str.len() > 0]
    df = df.drop_duplicates(subset=["brand", "text_clean"])
    print(f"  after dedup / empty removal: {len(df):,}")

    # 3. minimum length (in whitespace tokens)
    df = df[df["text_clean"].str.split().str.len() >= config.MIN_TOKENS]
    print(f"  after min-length ({config.MIN_TOKENS} tokens): {len(df):,}")

    # 4. English-only (slow on huge files; runs on the cleaned text)
    print("  running language filter (this can take a minute)...")
    df = df[df["text_clean"].apply(is_english)]
    print(f"  after English-only filter: {len(df):,}")

    keep = [c for c in ["brand", "text", "text_clean", "created_utc", "stars", "gold_sentiment"] if c in df.columns]
    df = df[keep].reset_index(drop=True)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_FILE, index=False)
    print(f"\nSaved {len(df):,} clean rows -> {config.CLEAN_FILE}")
    print(f"Removed {start_n - len(df):,} rows overall ({start_n:,} -> {len(df):,}).")
    print("\nPer-brand counts:")
    print(df["brand"].value_counts().to_string())


if __name__ == "__main__":
    main()
