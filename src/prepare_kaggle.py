
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  
INPUT_FILE = Path.home() / "Downloads" / "twitter_training.csv"

BRANDS = ["Amazon", "Google", "Microsoft", "Verizon", "Facebook"]

DROP_IRRELEVANT = True  # the dataset has an 'Irrelevant' class; usually drop it


def main():
    if not INPUT_FILE.exists():
        raise SystemExit(
            f"Couldn't find {INPUT_FILE}.\n"
            "Download the dataset from Kaggle, unzip, and set INPUT_FILE."
        )
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
