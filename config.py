

from pathlib import Path
ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"

RAW_FILE = RAW_DIR / "posts_raw.csv"          # collected/downloaded data lands here
CLEAN_FILE = PROCESSED_DIR / "posts_clean.csv"  # output of clean.py
SENTIMENT_FILE = PROCESSED_DIR / "posts_sentiment.csv"  # output of sentiment.py
TOPICS_FILE = PROCESSED_DIR / "posts_topics.csv"        # output of topics.py

BRAND_SUBREDDITS = {
    "McDonalds":  ["mcdonalds"],
    "KFC":        ["KFC"],
    "BurgerKing": ["burgerking", "BurgerKing"],
}

GENERAL_SUBREDDITS = ["britishfood", "CasualUK", "unitedkingdom"]

BRAND_KEYWORDS = {
    "McDonalds":  ["mcdonald", "maccies", "mcds", "mcd"],
    "KFC":        ["kfc", "kentucky fried"],
    "BurgerKing": ["burger king", "bk "],
}
POSTS_PER_SUBREDDIT = 1500     # cap per subreddit
TIME_FILTER = "year"           # 'day' | 'week' | 'month' | 'year' | 'all'
MIN_TOKENS = 3                 # drop posts shorter than this after cleaning

# VADER label thresholds on the compound score (the standard cut-offs)
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

PRIMARY_METHOD = "tf"

VALIDATION_SAMPLE_SIZE = 100
RANDOM_SEED = 42
N_TOPICS = 10                 
N_TOP_WORDS = 12              
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   

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
