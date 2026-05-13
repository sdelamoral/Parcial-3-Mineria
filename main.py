"""
Sentiment Analysis Pipeline — Kindle Reviews
=============================================

Pipeline steps:
  1. Load & explore dataset
  2. Preprocess with Stemming  → vectorize with BoW & TF-IDF
  3. Preprocess with Lemmatization → vectorize with BoW & TF-IDF
  4. Train 3 classifiers on each (6 configurations total)
  5. Evaluate all configs and print comparative analysis
  6. Show misclassified examples for error analysis

Run:
    pip install -r requirements.txt
    python download_data.py        # first time only
    python main.py
"""

import os
import joblib
from sklearn.model_selection import train_test_split

from src.data_loader   import load_dataset, explore_dataset
from src.preprocessor  import NLPPreprocessor
from src.vectorizer    import build_bow, build_tfidf
from src.models        import train_naive_bayes, train_logistic_regression, train_svm
from src.evaluator     import evaluate_model, compare_results, print_misclassified

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Data ────────────────────────────────────────────────────────────────
print("\n[1/5] Loading dataset")
print("-" * 50)
df = load_dataset()

print("\n[1/5] Exploring dataset")
print("-" * 50)
explore_dataset(df, save_dir=OUTPUT_DIR)


# ── 2. Train/test split on raw text ────────────────────────────────────────
print("\n[2/5] Train / test split (80/20, stratified)")
print("-" * 50)
X_raw = df["reviewText"].tolist()
y     = df["sentiment"]

X_raw_train, X_raw_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_raw_train):,}   Test: {len(X_raw_test):,}")


# ── 3. Preprocessing ────────────────────────────────────────────────────────
print("\n[3/5] Preprocessing")
print("-" * 50)

print("\n  [Stem] Sample transformation:")
stem_prep = NLPPreprocessor(method="stem")
stem_prep.show_example(X_raw_train[0])

print("\n  [Lemma] Sample transformation:")
lemma_prep = NLPPreprocessor(method="lemma")
lemma_prep.show_example(X_raw_train[0])

print("\n  Processing train/test sets...")
X_stem_train  = stem_prep.fit_transform(X_raw_train)
X_stem_test   = stem_prep.fit_transform(X_raw_test)

X_lemma_train = lemma_prep.fit_transform(X_raw_train)
X_lemma_test  = lemma_prep.fit_transform(X_raw_test)


# ── 4. Vectorization ────────────────────────────────────────────────────────
print("\n[4/5] Vectorization")
print("-" * 50)

print("\n  Stem + BoW:")
stem_bow_train, stem_bow_test, stem_bow_vec = build_bow(X_stem_train, X_stem_test)

print("\n  Stem + TF-IDF:")
stem_tfidf_train, stem_tfidf_test, stem_tfidf_vec = build_tfidf(X_stem_train, X_stem_test)

print("\n  Lemma + BoW:")
lemma_bow_train, lemma_bow_test, lemma_bow_vec = build_bow(X_lemma_train, X_lemma_test)

print("\n  Lemma + TF-IDF:")
lemma_tfidf_train, lemma_tfidf_test, lemma_tfidf_vec = build_tfidf(X_lemma_train, X_lemma_test)


# ── 5. Train & evaluate ─────────────────────────────────────────────────────
print("\n[5/5] Training models & evaluation")
print("-" * 50)

results = []

configurations = [
    ("Stem + BoW",     stem_bow_train,    stem_bow_test),
    ("Stem + TF-IDF",  stem_tfidf_train,  stem_tfidf_test),
    ("Lemma + BoW",    lemma_bow_train,   lemma_bow_test),
    ("Lemma + TF-IDF", lemma_tfidf_train, lemma_tfidf_test),
]

trainers = [
    ("Naive Bayes",           train_naive_bayes),
    ("Logistic Regression",   train_logistic_regression),
    ("SVM",                   train_svm),
]

# Store best model info for misclassified analysis
best_f1         = 0.0
best_model      = None
best_label      = ""
best_vectorizer = None
best_prep_method = ""

vec_objects = {
    "Stem + BoW":     (stem_bow_vec,   "stem"),
    "Stem + TF-IDF":  (stem_tfidf_vec, "stem"),
    "Lemma + BoW":    (lemma_bow_vec,  "lemma"),
    "Lemma + TF-IDF": (lemma_tfidf_vec,"lemma"),
}

for vec_label, X_tr, X_te in configurations:
    for model_name, trainer in trainers:
        label = f"{model_name} / {vec_label}"

        model = trainer(X_tr, y_train)
        metrics = evaluate_model(model, X_te, y_test, label=label, save_dir=OUTPUT_DIR)
        results.append(metrics)

        if metrics["f1"] > best_f1:
            best_f1          = metrics["f1"]
            best_model       = model
            best_label       = label
            best_X_te        = X_te
            best_vectorizer, best_prep_method = vec_objects[vec_label]


# ── 6. Comparative summary ──────────────────────────────────────────────────
print("\n" + "="*55)
print("  FINAL COMPARISON")
print("="*55)
summary_df = compare_results(results, save_dir=OUTPUT_DIR)

# Save summary CSV
csv_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
summary_df.to_csv(csv_path)
print(f"  Results table saved: {csv_path}")


# ── 7. Error analysis ───────────────────────────────────────────────────────
print(f"\n  Best model: {best_label}  (F1={best_f1:.4f})")
print_misclassified(
    best_model, best_X_te, y_test,
    original_texts=X_raw_test,
    n=10,
    label=best_label,
)

# ── 8. Save best model artifacts for the app ────────────────────────────────
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best_model,      os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(best_vectorizer, os.path.join(MODEL_DIR, "best_vectorizer.pkl"))
with open(os.path.join(MODEL_DIR, "best_config.txt"), "w") as f:
    f.write(f"label={best_label}\n")
    f.write(f"prep={best_prep_method}\n")
    f.write(f"f1={best_f1:.4f}\n")
print(f"\n  Model saved to '{MODEL_DIR}/'  ({best_label})")

print("\nAll outputs saved to:", os.path.abspath(OUTPUT_DIR))
print("Done.")
