# Listening to the Digital Consumer

Sentiment and topic analysis of social-media discourse to inform brand-marketing
strategy. MSc project (BEMM828, University of Exeter).

This repository collects public posts about competing UK fast-food brands,
classifies their **sentiment** (two methods), extracts the **topics/aspects**
being discussed (two methods), and combines the two into a sentiment-by-topic
view that supports marketing recommendations.

## Pipeline

| Stage | Script | Input | Output |
|------|--------|-------|--------|
| 0. Collect (optional) | `src/collect_reddit.py` | Reddit API | `data/raw/posts_raw.csv` |
| 1. Clean | `src/clean.py` | `posts_raw.csv` | `data/processed/posts_clean.csv` |
| 2. Sentiment | `src/sentiment.py` | `posts_clean.csv` | `data/processed/posts_sentiment.csv` |
| 3. Topics | `src/topics.py` | `posts_sentiment.csv` | topics + `outputs/` *(coming)* |
| 4. Analysis | `src/stats_analysis.py` | above | tables + figures *(coming)* |

If you use a ready-made dataset instead of Reddit, just drop a CSV with
`brand` and `text` columns at `data/raw/posts_raw.csv` and start at Stage 1.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python src/prepare_yelp.py        # Yelp -> data/raw/posts_raw.csv (or prepare_kaggle.py)
python src/clean.py               # -> posts_clean.csv (keeps Yelp 'stars' for validation)

python src/sentiment.py vader     # instant, offline
python src/sentiment.py transformer   # uses your GPU (CUDA/MPS), model downloads once
python src/sentiment.py validate  # accuracy/precision/recall/F1 vs Yelp stars

python src/topics.py lda          # scikit-learn LDA + aspect mapping
python src/topics.py bertopic     # neural topic model (optional 2nd method)

python src/stats_analysis.py      # figures + tables for your Findings
```

Outputs land in `outputs/figures/` (Fig 1 sentiment-by-brand, Fig 2 brand x aspect
matrix) and `outputs/tables/` (sentiment %, topic words, inferential results with
effect sizes, pairwise tests).

## Notes

- **Data is never committed** (see `.gitignore`) — it stays on your machine,
  which also keeps you inside the platform's terms and UK GDPR.
- **Credentials are read from environment variables**, never hard-coded.
- Configuration (brands, subreddits, thresholds, model names) lives in
  `config.py` — change settings there, not scattered through the scripts.
