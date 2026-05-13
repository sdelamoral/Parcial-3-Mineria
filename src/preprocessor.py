"""
NLP preprocessing pipeline.

Supports two modes (controlled by `method` parameter):
  - 'stem'  : PorterStemmer
  - 'lemma' : WordNetLemmatizer

Usage:
    from src.preprocessor import NLPPreprocessor

    prep = NLPPreprocessor(method="stem")
    texts_clean = prep.fit_transform(raw_texts)
"""

import re
import string
import time
from typing import Literal

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data once
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

STOPWORDS = set(stopwords.words("english"))


class NLPPreprocessor:
    def __init__(self, method: Literal["stem", "lemma"] = "stem"):
        self.method = method
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    # ------------------------------------------------------------------
    # Individual steps (public so they can be tested/compared in isolation)
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """Lowercase, remove URLs, punctuation, and non-alphabetic characters."""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)           # URLs
        text = re.sub(r"[^a-z\s]", " ", text)                 # keep only letters
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    def apply_stemming(self, tokens: list[str]) -> list[str]:
        return [self.stemmer.stem(t) for t in tokens]

    def apply_lemmatization(self, tokens: list[str]) -> list[str]:
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process_single(self, text: str) -> str:
        """Apply full pipeline to one text; return a single cleaned string."""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)

        if self.method == "stem":
            tokens = self.apply_stemming(tokens)
        else:
            tokens = self.apply_lemmatization(tokens)

        return " ".join(tokens)

    def fit_transform(self, texts: list[str]) -> list[str]:
        """Process a list of texts; prints progress."""
        total = len(texts)
        print(f"  Preprocessing {total:,} texts with method='{self.method}'...")
        t0 = time.time()
        result = [self.process_single(t) for t in texts]
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")
        return result

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def show_example(self, text: str) -> None:
        """Print step-by-step transformation of a single text."""
        print(f"  Original   : {text[:120]}")
        cleaned = self.clean_text(text)
        print(f"  Cleaned    : {cleaned[:120]}")
        tokens = self.tokenize(cleaned)
        print(f"  Tokens     : {tokens[:20]}")
        tokens_ns = self.remove_stopwords(tokens)
        print(f"  No stops   : {tokens_ns[:20]}")
        if self.method == "stem":
            final = self.apply_stemming(tokens_ns)
            print(f"  Stemmed    : {final[:20]}")
        else:
            final = self.apply_lemmatization(tokens_ns)
            print(f"  Lemmatized : {final[:20]}")
