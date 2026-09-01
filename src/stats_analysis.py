"""
Stage 3 — Comparative analysis + figures (your Findings section).

Reads  data/processed/posts_topics.csv (needs sentiment + topics run first)
Writes figures to outputs/figures/ and tables to outputs/tables/

Produces, following your four-stage plan:
  1. Descriptive : sentiment distribution per brand (table + stacked bar).
  2. Inferential : Kruskal-Wallis on VADER compound across brands (+ epsilon^2
                   effect size) and chi-square on sentiment label x brand
                   (+ Cramer's V). Effect sizes matter here because with
                   ~40k rows almost anything is "significant".
  3. Combined    : brand x aspect sentiment matrix (heatmap) - the core output
                   linking WHAT is discussed to HOW people feel.

Every figure is saved with a title/caption and a source note.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

SOURCE_NOTE = "Source: author's analysis of Yelp Open Dataset review text."

# Which columns drive the analysis (transformer by default; see config.PRIMARY_METHOD)
PRIMARY = getattr(config, "PRIMARY_METHOD", "tf")
LABEL_COL = f"{PRIMARY}_label"
SCORE_COL = f"{PRIMARY}_compound"
METHOD_NAME = "transformer" if PRIMARY == "tf" else "VADER"


def _load():
    if not config.TOPICS_FILE.exists():
        raise SystemExit("Run sentiment.py then topics.py first (need posts_topics.csv).")
    df = pd.read_csv(config.TOPICS_FILE)
    for col in [LABEL_COL, SCORE_COL, "brand"]:
        if col not in df.columns:
            raise SystemExit(
                f"Missing column '{col}'. Run 'sentiment.py {PRIMARY if PRIMARY!='tf' else 'transformer'}' "
                "then re-run topics.py so the column carries through.")
    return df


# ---------------------------------------------------------------------------
# 1. Descriptive
# ---------------------------------------------------------------------------
def descriptive(df):
    ct = pd.crosstab(df["brand"], df[LABEL_COL], normalize="index") * 100
    ct = ct[[c for c in ["positive", "neutral", "negative"] if c in ct.columns]]
    config.TAB_DIR.mkdir(parents=True, exist_ok=True)
    ct.round(1).to_csv(config.TAB_DIR / "sentiment_by_brand_pct.csv")
    print("Sentiment distribution by brand (% within brand):")
    print(ct.round(1).to_string())

    colors = {"positive": "#4c9f70", "neutral": "#c9c9c9", "negative": "#c0504d"}
    ax = ct.plot(kind="bar", stacked=True, figsize=(9, 5.5),
                 color=[colors.get(c, "#888") for c in ct.columns])
    ax.set_ylabel("Share of reviews (%)")
    ax.set_xlabel("")
    ax.set_title(f"Figure 1. Sentiment composition by brand ({METHOD_NAME})", loc="left")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.figtext(0.01, -0.02, SOURCE_NOTE, ha="left", fontsize=8, style="italic")
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "fig1_sentiment_by_brand.png", dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 2. Inferential  (with effect sizes)
# ---------------------------------------------------------------------------
def inferential(df):
    from scipy import stats

    groups = [g[SCORE_COL].values for _, g in df.groupby("brand")]
    H, p = stats.kruskal(*groups)
    n = len(df)
    k = len(groups)
    epsilon_sq = (H - k + 1) / (n - k)   # effect size for Kruskal-Wallis
    print(f"\n--- Kruskal-Wallis: {METHOD_NAME} compound across brands ---")
    print(f"H = {H:.1f}, p = {p:.2e}, epsilon^2 = {epsilon_sq:.4f} "
          f"({_interpret_eps(epsilon_sq)} effect)")

    ct = pd.crosstab(df["brand"], df[LABEL_COL])
    chi2, pc, dof, _ = stats.chi2_contingency(ct)
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    print("\n--- Chi-square: sentiment label x brand ---")
    print(f"chi2 = {chi2:.1f}, dof = {dof}, p = {pc:.2e}, Cramer's V = {cramers_v:.4f} "
          f"({_interpret_v(cramers_v)} association)")

    # pairwise Mann-Whitney with Bonferroni correction
    brands = sorted(df["brand"].unique())
    print("\n--- Pairwise Mann-Whitney on compound (Bonferroni-corrected p) ---")
    pairs = [(a, b) for i, a in enumerate(brands) for b in brands[i + 1:]]
    m = len(pairs)
    rows = []
    for a, b in pairs:
        u, pu = stats.mannwhitneyu(df.loc[df.brand == a, SCORE_COL],
                                   df.loc[df.brand == b, SCORE_COL])
        rows.append({"brand_a": a, "brand_b": b, "p_adj": min(pu * m, 1.0)})
    pw = pd.DataFrame(rows)
    print(pw.to_string(index=False))

    with open(config.TAB_DIR / "inferential_results.txt", "w") as fh:
        fh.write(f"Kruskal-Wallis H={H:.2f} p={p:.3e} epsilon^2={epsilon_sq:.4f}\n")
        fh.write(f"Chi-square chi2={chi2:.2f} dof={dof} p={pc:.3e} CramersV={cramers_v:.4f}\n\n")
        fh.write(pw.to_string(index=False))
    pw.to_csv(config.TAB_DIR / "pairwise_mannwhitney.csv", index=False)
    print(f"\nSaved -> {config.TAB_DIR / 'inferential_results.txt'}")


def _interpret_eps(e):
    return "negligible" if e < 0.01 else "small" if e < 0.08 else "moderate" if e < 0.26 else "large"


def _interpret_v(v):
    return "negligible" if v < 0.1 else "small" if v < 0.3 else "moderate" if v < 0.5 else "large"


# ---------------------------------------------------------------------------
# 3. Combined: brand x aspect sentiment matrix
# ---------------------------------------------------------------------------
def sentiment_by_aspect(df):
    if "aspect" not in df.columns:
        print("No 'aspect' column (run topics.py) - skipping matrix.")
        return

    mat = df.pivot_table(index="brand", columns="aspect",
                         values=SCORE_COL, aggfunc="mean")
    mat.round(3).to_csv(config.TAB_DIR / "brand_aspect_sentiment.csv")
    print(f"\nMean sentiment ({METHOD_NAME} score) by brand x aspect:")
    print(mat.round(3).to_string())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(mat.values, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"Figure 2. Mean sentiment by brand and aspect ({METHOD_NAME})", loc="left")
    fig.colorbar(im, ax=ax, label=f"Mean {METHOD_NAME} score (\u22121 to +1)")
    plt.figtext(0.01, -0.02, SOURCE_NOTE, ha="left", fontsize=8, style="italic")
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "fig2_brand_aspect_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figures -> {config.FIG_DIR}")


def main():
    df = _load()
    descriptive(df)
    inferential(df)
    sentiment_by_aspect(df)
    print("\nDone. Figures in outputs/figures/, tables in outputs/tables/.")


if __name__ == "__main__":
    main()
