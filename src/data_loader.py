"""
Loads and explores the Kindle Reviews dataset.

Sentiment mapping:
  ratings 1-2  → 0 (negative)
  ratings 4-5  → 1 (positive)
  rating  3    → dropped (neutral/ambiguous)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = os.path.join("data", "kindle_reviews.csv")
SAMPLE_SIZE = 50_000   # cap to keep training time reasonable; set None to use all


def load_dataset(path: str = DATA_PATH, sample: int | None = SAMPLE_SIZE) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Run `python download_data.py` first."
        )

    print(f"Loading dataset from {path}...")
    df = pd.read_csv(path, usecols=["reviewText", "overall"])
    print(f"  Raw rows: {len(df):,}")

    df = df.dropna(subset=["reviewText", "overall"])
    df["overall"] = df["overall"].astype(int)

    # Drop neutral reviews (rating == 3)
    df = df[df["overall"] != 3].copy()

    # Binary sentiment label
    df["sentiment"] = (df["overall"] >= 4).astype(int)

    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
        print(f"  Sampled to: {len(df):,} rows")

    print(f"  Final rows : {len(df):,}")
    print(f"  Positive   : {df['sentiment'].sum():,}  ({df['sentiment'].mean()*100:.1f}%)")
    print(f"  Negative   : {(df['sentiment'] == 0).sum():,}  ({(1-df['sentiment'].mean())*100:.1f}%)")
    return df


def explore_dataset(df: pd.DataFrame, save_dir: str = "outputs") -> None:
    os.makedirs(save_dir, exist_ok=True)

    # --- Rating distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.countplot(x="overall", hue="overall", data=df, ax=axes[0], palette="viridis", legend=False)
    axes[0].set_title("Rating Distribution (after removing 3s)")
    axes[0].set_xlabel("Star Rating")
    axes[0].set_ylabel("Count")

    sns.countplot(x="sentiment", hue="sentiment", data=df, ax=axes[1], palette="Set2", legend=False)
    axes[1].set_title("Sentiment Distribution")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Negative (0)", "Positive (1)"])
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    path = os.path.join(save_dir, "01_dataset_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")

    # --- Review length distribution ---
    df["review_len"] = df["reviewText"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 4))
    df["review_len"].clip(upper=500).hist(bins=50, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Review Length Distribution (words, capped at 500)")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    path = os.path.join(save_dir, "02_review_lengths.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")

    # --- Basic stats ---
    print("\n  Review length stats:")
    print(df["review_len"].describe().to_string())
