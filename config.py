

from pathlib import Path
ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"

RAW_FILE = RAW_DIR / "posts_raw.csv"          
CLEAN_FILE = PROCESSED_DIR / "posts_clean.csv"  
SENTIMENT_FILE = PROCESSED_DIR / "posts_sentiment.csv"  
TOPICS_FILE = PROCESSED_DIR / "posts_topics.csv"        
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
POSTS_PER_SUBREDDIT = 1500     
TIME_FILTER = "year"           
MIN_TOKENS = 3                 
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

PRIMARY_METHOD = "tf"

VALIDATION_SAMPLE_SIZE = 100
RANDOM_SEED = 42
N_TOPICS = 10                 
N_TOP_WORDS = 12              
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   

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
