"""
Prepare data from the Kaggle 'Twitter Entity Sentiment Analysis' dataset:
https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

That download gives twitter_training.csv (and twitter_validation.csv), each with
FOUR columns and NO header row:
    tweet_id , entity , sentiment , text
e.g.  2401, Borderlands, Positive, "im getting on borderlands and i will..."

This script:
  1. loads the file, names the columns,
  2. keeps only the brands (entities) you choose,
  3. writes data/raw/posts_raw.csv with columns: brand, text, gold_sentiment

'gold_sentiment' is the dataset's own label — a bonus ground-truth you can use
to validate VADER / the transformer, the same way Yelp's stars would have.

NOTE ON YOUR PROPOSAL: these entities are consumer/tech brands, not fast food,
so if you use this you'll reframe your industry and justify it in the
Methodology as a data-availability decision. Everything downstream is identical.

Usage:
  1. Put twitter_training.csv somewhere, e.g. ~/Downloads/
  2. Set INPUT_FILE below to point at it.
  3. Edit BRANDS to the entities you want to compare.
  4. python src/prepare_kaggle.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
INPUT_FILE = Path.home() / "Downloads" / "twitter_training.csv"

# The entities to treat as your competing "brands". Pick 3-5 well-populated ones.
# Common high-volume entities in this dataset include:
#   Amazon, Google, Microsoft, Verizon, Facebook, Nvidia,
#   Xbox(Xseries), PlayStation5(PS5), CallOfDuty, Battlefield, ...
BRANDS = ["Amazon", "Google", "Microsoft", "Verizon", "Facebook"]

DROP_IRRELEVANT = True  # the dataset has an 'Irrelevant' class; usually drop it


def main():
    if not INPUT_FILE.exists():
        raise SystemExit(
            f"Couldn't find {INPUT_FILE}.\n"
            "Download the dataset from Kaggle, unzip, and set INPUT_FILE."
        )

    # The file has no header, so read with header=None and name the columns.
    df = pd.read_csv(INPUT_FILE, header=None,
                     names=["tweet_id", "entity", "sentiment", "text"])
    print(f"Loaded {len(df):,} rows; {df['entity'].nunique()} distinct entities")

    df = df[df["entity"].isin(BRANDS)].copy()
    if DROP_IRRELEVANT:
        df = df[df["sentiment"].str.lower() != "irrelevant"]

    df = df.dropna(subset=["text"])
    out = pd.DataFrame({
        "brand": df["entity"],
        "text": df["text"],
        "gold_sentiment": df["sentiment"].str.lower(),
    }).reset_index(drop=True)

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.RAW_FILE, index=False)
    print(f"\nWrote {len(out):,} rows -> {config.RAW_FILE}")
    print("\nPer-brand counts:")
    print(out["brand"].value_counts().to_string())
    if len(out) == 0:
        print("\n(No rows matched. Check the BRANDS list against the entity names "
              "actually in the file.)")


if __name__ == "__main__":
    main()
