"""
Classification models for sentiment analysis.

Three models are implemented:
  - Naive Bayes (MultinomialNB)
  - Logistic Regression
  - Support Vector Machine (LinearSVC)

Each `train_*` function returns the fitted estimator.
"""

import time
from scipy.sparse import spmatrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def train_naive_bayes(X_train, y_train, alpha: float = 0.5):
    print("  Training Naive Bayes...")
    t0 = time.time()
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    print(f"    Done in {time.time()-t0:.2f}s")
    return model


def train_logistic_regression(X_train, y_train, C: float = 1.0, max_iter: int = 1000):
    print("  Training Logistic Regression...")
    t0 = time.time()
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"    Done in {time.time()-t0:.2f}s")
    return model


def train_svm(X_train, y_train, C: float = 1.0):
    """
    LinearSVC wrapped in CalibratedClassifierCV so it exposes
    predict_proba (needed for ROC/AUC if added later).
    """
    print("  Training SVM (LinearSVC)...")
    t0 = time.time()
    base = LinearSVC(C=C, max_iter=2000, random_state=42)
    model = CalibratedClassifierCV(base, cv=3)
    model.fit(X_train, y_train)
    print(f"    Done in {time.time()-t0:.2f}s")
    return model
