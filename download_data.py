"""
Download the Kindle Reviews dataset from Kaggle.

Requirements:
  1. Install kagglehub:  pip install kagglehub
  2. Authenticate once:  run `kaggle` CLI or place your kaggle.json at ~/.kaggle/kaggle.json
     Get it at: https://www.kaggle.com/settings -> API -> Create New Token

Usage:
    python download_data.py
"""

import os
import shutil

DEST = os.path.join("data", "kindle_reviews.csv")


def download():
    try:
        import kagglehub

        print("Downloading Kindle Reviews dataset via kagglehub...")
        path = kagglehub.dataset_download("bharadwaj6/kindle-reviews")
        print(f"Downloaded to cache: {path}")

        # Locate the CSV inside the downloaded folder
        csv_path = None
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".csv"):
                    csv_path = os.path.join(root, f)
                    break
            if csv_path:
                break

        if csv_path is None:
            raise FileNotFoundError("No CSV found in downloaded dataset.")

        os.makedirs("data", exist_ok=True)
        shutil.copy(csv_path, DEST)
        print(f"Dataset saved to: {DEST}")

    except Exception as e:
        print(f"Download failed: {e}")
        print("\nManual option:")
        print("  1. Go to https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews")
        print("  2. Download kindle_reviews.csv")
        print(f"  3. Place it at: {DEST}")


if __name__ == "__main__":
    download()
