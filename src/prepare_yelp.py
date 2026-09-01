
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

YELP_DIR = Path.home() / "Downloads" / "yelp"

BUSINESS_FILE = YELP_DIR / "yelp_academic_dataset_business.json"
REVIEW_FILE = YELP_DIR / "yelp_academic_dataset_review.json"

BRAND_NAME_MATCHES = {
    "McDonalds":  ["mcdonald"],
    "KFC":        ["kfc", "kentucky fried chicken"],
    "BurgerKing": ["burger king"],
    "Subway":     ["subway"],
    "TacoBell":   ["taco bell"],
}
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
