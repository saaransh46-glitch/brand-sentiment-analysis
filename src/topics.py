

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _load(cmd: str):
    # 'lda' always starts fresh from the sentiment file (so re-runs pick up
    # updated sentiment columns); 'bertopic' augments the existing lda output.
    if cmd == "bertopic" and config.TOPICS_FILE.exists():
        src = config.TOPICS_FILE
    else:
        src = config.SENTIMENT_FILE
    if not src.exists():
        raise SystemExit("Run sentiment.py first (need posts_sentiment.csv).")
    return pd.read_csv(src)


def map_aspect(top_words) -> str:
    """Assign the aspect whose seed list overlaps most with a topic's top words."""
    words = set(w.lower() for w in top_words)
    best, best_hits = "Other", 0
    for aspect, seeds in config.ASPECT_SEEDS.items():
        hits = len(words & set(seeds))
        if hits > best_hits:
            best, best_hits = aspect, hits
    return best

def run_lda(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    texts = df["text_clean"].fillna("").astype(str)

    vec = CountVectorizer(
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",  # words of 3+ letters
        max_df=0.5,     # drop terms in >50% of docs (too generic)
        min_df=10,      # drop very rare terms
    )
    dtm = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=config.N_TOPICS,
        learning_method="online",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    doc_topics = lda.fit_transform(dtm)

    # top words + aspect per topic
    rows = []
    topic_aspect = {}
    print(f"\n{config.N_TOPICS} LDA topics (top {config.N_TOP_WORDS} words):\n")
    for k, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[::-1][: config.N_TOP_WORDS]
        top_words = [vocab[i] for i in top_idx]
        aspect = map_aspect(top_words)
        topic_aspect[k] = aspect
        print(f"  Topic {k:>2} [{aspect}]: {', '.join(top_words)}")
        rows.append({"topic": k, "aspect": aspect, "top_words": " ".join(top_words)})

    df["lda_topic"] = doc_topics.argmax(axis=1)
    df["aspect"] = df["lda_topic"].map(topic_aspect)

    config.TAB_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.TAB_DIR / "lda_topic_words.csv", index=False)
    print(f"\nPerplexity: {lda.perplexity(dtm):.0f}  (lower is better; use to compare N_TOPICS)")
    print(f"Saved topic words -> {config.TAB_DIR / 'lda_topic_words.csv'}")
    print("\nAspect distribution:")
    print(df["aspect"].value_counts().to_string())
    return df


def run_bertopic(df: pd.DataFrame) -> pd.DataFrame:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    import torch

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Embedding on {device}...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=device)

    docs = df["text_clean"].fillna("").astype(str).tolist()
    model = BERTopic(embedding_model=embedder, verbose=True, min_topic_size=50)
    topics, _ = model.fit_transform(docs)
    df["bertopic_topic"] = topics

    info = model.get_topic_info()
    config.TAB_DIR.mkdir(parents=True, exist_ok=True)
    info.to_csv(config.TAB_DIR / "bertopic_topics.csv", index=False)
    print(f"\nFound {len(info) - 1} topics (excluding outliers).")
    print(f"Saved topic summary -> {config.TAB_DIR / 'bertopic_topics.csv'}")
    return df


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "lda"
    df = _load(cmd)

    if cmd == "lda":
        df = run_lda(df)
    elif cmd == "bertopic":
        df = run_bertopic(df)
    else:
        raise SystemExit(f"Unknown command '{cmd}'. Use: lda | bertopic")

    df.to_csv(config.TOPICS_FILE, index=False)
    print(f"\nSaved -> {config.TOPICS_FILE}")


if __name__ == "__main__":
    main()
