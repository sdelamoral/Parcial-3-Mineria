"""
Text vectorization: Bag-of-Words and TF-IDF.

Returns (X_train, X_test, fitted_vectorizer) tuples so the same
vectorizer can be reused for prediction or inspection.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_bow(
    train_texts: list[str],
    test_texts: list[str],
    max_features: int = 20_000,
    ngram_range: tuple[int, int] = (1, 2),
):
    """
    Bag-of-Words with unigrams + bigrams.
    Fitted only on train_texts to avoid data leakage.
    """
    vec = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,          # ignore terms that appear in fewer than 2 docs
    )
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    print(f"  BoW  shape: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, vec


def build_tfidf(
    train_texts: list[str],
    test_texts: list[str],
    max_features: int = 20_000,
    ngram_range: tuple[int, int] = (1, 2),
):
    """
    TF-IDF with unigrams + bigrams.
    Fitted only on train_texts to avoid data leakage.
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        sublinear_tf=True,   # replace tf with 1+log(tf) to dampen high counts
    )
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    print(f"  TF-IDF shape: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, vec


def top_features(vectorizer, n: int = 20) -> list[str]:
    """Return the n most common feature names (by vocabulary index)."""
    return vectorizer.get_feature_names_out()[:n].tolist()
