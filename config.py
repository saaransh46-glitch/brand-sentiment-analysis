"""
Central configuration for the brand sentiment & topic analysis project.
Everything the scripts need to know lives here, so you change settings in ONE place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file, so the project works on any machine)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"

RAW_FILE = RAW_DIR / "posts_raw.csv"          # collected/downloaded data lands here
CLEAN_FILE = PROCESSED_DIR / "posts_clean.csv"  # output of clean.py
SENTIMENT_FILE = PROCESSED_DIR / "posts_sentiment.csv"  # output of sentiment.py
TOPICS_FILE = PROCESSED_DIR / "posts_topics.csv"        # output of topics.py

# ---------------------------------------------------------------------------
# Brands under study and the subreddits to collect them from.
# Edit these to match the brands/industry your supervisor confirms.
# ---------------------------------------------------------------------------
BRAND_SUBREDDITS = {
    "McDonalds":  ["mcdonalds"],
    "KFC":        ["KFC"],
    "BurgerKing": ["burgerking", "BurgerKing"],
}

# Also scan a general UK food subreddit and tag posts by keyword mention
GENERAL_SUBREDDITS = ["britishfood", "CasualUK", "unitedkingdom"]

# Keywords used to attribute a general-subreddit post to a brand
BRAND_KEYWORDS = {
    "McDonalds":  ["mcdonald", "maccies", "mcds", "mcd"],
    "KFC":        ["kfc", "kentucky fried"],
    "BurgerKing": ["burger king", "bk "],
}

# ---------------------------------------------------------------------------
# Collection parameters (keep modest to respect the API and your time budget)
# ---------------------------------------------------------------------------
POSTS_PER_SUBREDDIT = 1500     # cap per subreddit
TIME_FILTER = "year"           # 'day' | 'week' | 'month' | 'year' | 'all'
MIN_TOKENS = 3                 # drop posts shorter than this after cleaning

# ---------------------------------------------------------------------------
# Sentiment models
# ---------------------------------------------------------------------------
# VADER label thresholds on the compound score (the standard cut-offs)
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

# Hugging Face transformer tuned for social-media sentiment (3-class)
TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Which method drives the Stage-3 analysis: "tf" (transformer) or "vader".
# The transformer validated far better against Yelp stars, so it's primary.
PRIMARY_METHOD = "tf"

# How many posts to hand-label for the accuracy/precision/recall/F1 validation
VALIDATION_SAMPLE_SIZE = 100
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Topic modelling (Stage 2)
# ---------------------------------------------------------------------------
N_TOPICS = 10                 # number of LDA topics
N_TOP_WORDS = 12              # top words to show/label each topic
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # BERTopic sentence embedder

# Map each topic to a business "aspect" by matching its top words to these seeds.
# Edit freely — this is where your domain judgement comes in.
ASPECT_SEEDS = {
    "Food quality": ["food", "taste", "tasty", "fresh", "quality", "delicious",
                      "flavor", "flavour", "cold", "stale", "burger", "fries",
                      "chicken", "meat", "cheese", "hot", "greasy", "bland"],
    "Service":      ["service", "staff", "rude", "friendly", "manager", "employee",
                     "worker", "attitude", "customer", "wrong", "order", "polite"],
    "Speed / wait": ["wait", "waited", "slow", "fast", "minutes", "line", "drive",
                     "quick", "long", "quickly", "queue"],
    "Price / value": ["price", "expensive", "cheap", "value", "deal", "money",
                      "worth", "overpriced", "cost", "dollar", "pricey"],
    "Cleanliness":  ["clean", "dirty", "bathroom", "table", "mess", "filthy",
                     "gross", "sticky"],
}
