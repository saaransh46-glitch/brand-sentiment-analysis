

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def run_vader(df: pd.DataFrame) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyser = SentimentIntensityAnalyzer()

    def label(compound: float) -> str:
        if compound >= config.VADER_POS_THRESHOLD:
            return "positive"
        if compound <= config.VADER_NEG_THRESHOLD:
            return "negative"
        return "neutral"

    df["vader_compound"] = df["text_clean"].apply(
        lambda t: analyser.polarity_scores(t)["compound"]
    )
    df["vader_label"] = df["vader_compound"].apply(label)
    print("VADER label distribution:")
    print(df["vader_label"].value_counts().to_string())
    return df


def _select_device():
    """Pick the fastest available backend: CUDA (e.g. an L4) -> Apple MPS (M-series) -> CPU."""
    import torch

    if torch.cuda.is_available():
        print("Using CUDA GPU.")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS).")
        return torch.device("mps")
    print("Using CPU.")
    return torch.device("cpu")


def run_transformer(df: pd.DataFrame) -> pd.DataFrame:
    from transformers import pipeline

    clf = pipeline(
        "sentiment-analysis",
        model=config.TRANSFORMER_MODEL,
        truncation=True,
        max_length=512,
        device=_select_device(),
        top_k=None,   
    )
    texts = df["text_clean"].fillna("").astype(str).tolist()
    results = clf(texts, batch_size=32)

    negs, neus, poss, labels = [], [], [], []
    for scores in results:
        d = {s["label"].lower(): s["score"] for s in scores}
        neg, neu, pos = d.get("negative", 0.0), d.get("neutral", 0.0), d.get("positive", 0.0)
        negs.append(neg); neus.append(neu); poss.append(pos)
        labels.append(max((("negative", neg), ("neutral", neu), ("positive", pos)),
                          key=lambda x: x[1])[0])

    df["tf_neg"], df["tf_neu"], df["tf_pos"] = negs, neus, poss
    df["tf_label"] = labels
    df["tf_compound"] = [p - n for p, n in zip(poss, negs)]

    print("Transformer label distribution:")
    print(df["tf_label"].value_counts().to_string())
    return df


def export_sample():
    df = pd.read_csv(config.SENTIMENT_FILE)
    sample = df.sample(
        n=min(config.VALIDATION_SAMPLE_SIZE, len(df)),
        random_state=config.RANDOM_SEED,
    ).copy()
    sample["gold_label"] = ""  
    out = config.PROCESSED_DIR / "validation_sample.csv"
    sample[["brand", "text", "text_clean", "vader_label", "gold_label"]].to_csv(out, index=False)
    print(f"Wrote {len(sample)} rows to {out}")
    print("Open it, fill the 'gold_label' column by hand, save, then run: python src/sentiment.py score")


def score_against_gold():
    from sklearn.metrics import classification_report, accuracy_score

    labelled = pd.read_csv(config.PROCESSED_DIR / "validation_sample.csv")
    labelled = labelled[labelled["gold_label"].notna() & (labelled["gold_label"] != "")]
    if labelled.empty:
        raise SystemExit("No hand labels found — fill the 'gold_label' column first.")

    for method_col in ["vader_label", "tf_label"]:
        if method_col not in labelled.columns:
            continue
        y_true = labelled["gold_label"].str.lower()
        y_pred = labelled[method_col].str.lower()
        print(f"\n=== {method_col} vs hand labels (n={len(labelled)}) ===")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
        print(classification_report(y_true, y_pred, digits=3, zero_division=0))

def validate_auto():
    """Validate VADER / transformer against a ready-made ground truth:
    Yelp 'stars' (>=4 positive, 3 neutral, <=2 negative) or the Kaggle
    'gold_sentiment' column. No hand-labelling needed."""
    from sklearn.metrics import classification_report, accuracy_score

    df = pd.read_csv(config.SENTIMENT_FILE)

    if "stars" in df.columns and df["stars"].notna().any():
        def star_to_label(s):
            if s >= 4:
                return "positive"
            if s <= 2:
                return "negative"
            return "neutral"
        df["gold"] = df["stars"].apply(star_to_label)
        source = "Yelp star ratings"
    elif "gold_sentiment" in df.columns:
        df["gold"] = df["gold_sentiment"].str.lower()
        source = "dataset gold_sentiment"
    else:
        raise SystemExit("No 'stars' or 'gold_sentiment' column to validate against. "
                         "Use 'sample' then 'score' to hand-label instead.")

    print(f"Validating against {source} (n={len(df):,})")
    for col in ["vader_label", "tf_label"]:
        if col not in df.columns:
            continue
        y_true, y_pred = df["gold"], df[col]
        print(f"\n=== {col} vs {source} ===")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
        print(classification_report(y_true, y_pred, digits=3, zero_division=0))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "vader"

    if cmd == "sample":
        export_sample(); return
    if cmd == "score":
        score_against_gold(); return
    if cmd == "validate":
        validate_auto(); return

    src = config.SENTIMENT_FILE if config.SENTIMENT_FILE.exists() else config.CLEAN_FILE
    df = pd.read_csv(src)

    if cmd == "vader":
        df = run_vader(df)
    elif cmd == "transformer":
        df = run_transformer(df)
    else:
        raise SystemExit(f"Unknown command '{cmd}'. Use: vader | transformer | sample | score")

    df.to_csv(config.SENTIMENT_FILE, index=False)
    print(f"\nSaved -> {config.SENTIMENT_FILE}")


if __name__ == "__main__":
    main()
