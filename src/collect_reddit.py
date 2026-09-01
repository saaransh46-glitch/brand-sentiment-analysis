
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def get_reddit():
    import praw

    cid = os.environ.get("REDDIT_CLIENT_ID")
    secret = os.environ.get("REDDIT_CLIENT_SECRET")
    agent = os.environ.get("REDDIT_USER_AGENT", "msc-brand-study")
    if not (cid and secret):
        raise SystemExit(
            "Missing credentials. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET "
            "as environment variables (see the setup notes at the top of this file)."
        )
    return praw.Reddit(client_id=cid, client_secret=secret, user_agent=agent)


def tag_brand(text: str) -> str | None:
    """Return the brand a piece of text mentions, or None. Used for general subreddits."""
    low = text.lower()
    for brand, kws in config.BRAND_KEYWORDS.items():
        if any(kw in low for kw in kws):
            return brand
    return None


def collect():
    reddit = get_reddit()
    rows = []

    
    for brand, subs in config.BRAND_SUBREDDITS.items():
        for sub in subs:
            print(f"[{brand}] r/{sub} ...")
            try:
                for post in reddit.subreddit(sub).top(
                    time_filter=config.TIME_FILTER, limit=config.POSTS_PER_SUBREDDIT
                ):
                    body = f"{post.title}. {post.selftext}".strip()
                    rows.append({"brand": brand, "text": body,
                                 "created_utc": post.created_utc, "source": f"r/{sub}"})
                    # a handful of top comments add volume and opinion
                    post.comments.replace_more(limit=0)
                    for c in post.comments[:10]:
                        rows.append({"brand": brand, "text": c.body,
                                     "created_utc": c.created_utc, "source": f"r/{sub}"})
                time.sleep(1)  # be polite to the API
            except Exception as e:  # noqa: BLE001
                print(f"  skipped r/{sub}: {e}")

    # 2. General subreddits -> tag by keyword, keep only brand-mentioning posts
    for sub in config.GENERAL_SUBREDDITS:
        print(f"[general] r/{sub} ...")
        try:
            for post in reddit.subreddit(sub).top(
                time_filter=config.TIME_FILTER, limit=config.POSTS_PER_SUBREDDIT
            ):
                body = f"{post.title}. {post.selftext}".strip()
                brand = tag_brand(body)
                if brand:
                    rows.append({"brand": brand, "text": body,
                                 "created_utc": post.created_utc, "source": f"r/{sub}"})
            time.sleep(1)
        except Exception as e:  
            print(f"  skipped r/{sub}: {e}")

    df = pd.DataFrame(rows)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_FILE, index=False)
    print(f"\nCollected {len(df):,} rows -> {config.RAW_FILE}")
    print(df["brand"].value_counts().to_string())


if __name__ == "__main__":
    collect()
