"""
Prepare data from the Yelp Open Dataset (https://www.yelp.com/dataset).

The download gives you several large JSON-lines files. We only need two:
  - yelp_academic_dataset_business.json   (business name + categories)
  - yelp_academic_dataset_review.json     (review text + star rating)

This script:
  1. finds businesses belonging to your target fast-food brands,
  2. pulls their reviews,
  3. writes data/raw/posts_raw.csv with columns: brand, text, stars, created_utc

The 'stars' column is a bonus: it becomes your validation ground-truth
(>=4 = positive, 3 = neutral, <=2 = negative), so you don't have to hand-label.

Usage:
  1. Unzip the Yelp download somewhere, e.g. ~/Downloads/yelp/
  2. Set YELP_DIR below to that folder.
  3. python src/prepare_yelp.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# EDIT THIS: where you unzipped the Yelp dataset
# ---------------------------------------------------------------------------
YELP_DIR = Path.home() / "Downloads" / "yelp"

BUSINESS_FILE = YELP_DIR / "yelp_academic_dataset_business.json"
REVIEW_FILE = YELP_DIR / "yelp_academic_dataset_review.json"

# Map a canonical brand label -> lowercase name fragments that identify it.
# Edit to match the brands your supervisor confirms.
BRAND_NAME_MATCHES = {
    "McDonalds":  ["mcdonald"],
    "KFC":        ["kfc", "kentucky fried chicken"],
    "BurgerKing": ["burger king"],
    "Subway":     ["subway"],
    "TacoBell":   ["taco bell"],
}

# Only treat a business as in-scope if it also looks like fast food, to avoid
# e.g. a "Subway" transit stop. Yelp stores a comma-separated 'categories' string.
FASTFOOD_HINTS = ["fast food", "burgers", "chicken", "sandwiches", "restaurants"]


def match_brand(name: str) -> str | None:
    low = (name or "").lower()
    for brand, fragments in BRAND_NAME_MATCHES.items():
        if any(f in low for f in fragments):
            return brand
    return None


def main():
    if not BUSINESS_FILE.exists() or not REVIEW_FILE.exists():
        raise SystemExit(
            f"Couldn't find the Yelp files in {YELP_DIR}.\n"
            "Download from https://www.yelp.com/dataset, unzip, and set YELP_DIR."
        )

    # 1. business_id -> brand, for in-scope fast-food businesses
    print("Scanning businesses...")
    biz_to_brand = {}
    with open(BUSINESS_FILE, encoding="utf-8") as fh:
        for line in fh:
            b = json.loads(line)
            brand = match_brand(b.get("name", ""))
            if not brand:
                continue
            cats = (b.get("categories") or "").lower()
            if any(h in cats for h in FASTFOOD_HINTS):
                biz_to_brand[b["business_id"]] = brand
    print(f"  matched {len(biz_to_brand):,} businesses across {len(set(biz_to_brand.values()))} brands")

    # 2. stream reviews, keep only those for in-scope businesses
    print("Scanning reviews (this is the big file, give it a minute)...")
    rows = []
    with open(REVIEW_FILE, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            brand = biz_to_brand.get(r.get("business_id"))
            if brand:
                rows.append({
                    "brand": brand,
                    "text": r.get("text", ""),
                    "stars": r.get("stars"),
                    "created_utc": r.get("date"),
                })

    df = pd.DataFrame(rows)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_FILE, index=False)
    print(f"\nWrote {len(df):,} reviews -> {config.RAW_FILE}")
    print("\nPer-brand counts:")
    print(df["brand"].value_counts().to_string())


if __name__ == "__main__":
    main()
