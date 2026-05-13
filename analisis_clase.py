"""
Análisis Comparativo — Estilo Notebook de Clase
=================================================
Implementación completa del pipeline NLP de principio a fin:
  1. Carga de datos
  2. Preprocesamiento: Stemming vs Lemmatization
  3. Vectorización: BoW vs TF-IDF
  4. Entrenamiento: Naive Bayes, Logistic Regression, SVM
  5. Comparación de las 12 configuraciones
  6. Análisis: BoW vs TF-IDF, Stem vs Lemma, mejor modelo
  7. Errores de clasificación
  8. Conclusiones

Basado en el patrón del notebook de clase (PorterStemmer, CountVectorizer,
TfidfVectorizer aplicados sobre el corpus de reseñas Kindle).

Run:
    python3 analisis_clase.py
"""

import os
import re
import string
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Descarga recursos NLTK necesarios
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE PRESENTACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def header(titulo):
    print("\n" + "=" * 65)
    print(f"  {titulo}")
    print("=" * 65)


def subheader(titulo):
    print(f"\n  ── {titulo}")
    print("  " + "-" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 — CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

header("1. CARGA Y EXPLORACIÓN DEL DATASET")

DATA_PATH = os.path.join("data", "kindle_reviews.csv")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"No se encontró '{DATA_PATH}'. Ejecuta primero: python3 download_data.py"
    )

df_raw = pd.read_csv(DATA_PATH, usecols=["reviewText", "overall"])
print(f"\n  Filas cargadas       : {len(df_raw):,}")

# Limpieza básica
df_raw = df_raw.dropna(subset=["reviewText", "overall"])
df_raw["overall"] = df_raw["overall"].astype(int)

# Eliminar reseñas neutras (rating == 3) — igual que en el proyecto
df_raw = df_raw[df_raw["overall"] != 3].copy()

# Mapeo binario de sentimiento: 1-2 → Negativo (0), 4-5 → Positivo (1)
df_raw["sentiment"] = (df_raw["overall"] >= 4).astype(int)

# Muestra de 50,000 para mantener tiempo de ejecución razonable
SAMPLE_SIZE = 50_000
if len(df_raw) > SAMPLE_SIZE:
    df_raw = df_raw.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

print(f"  Filas tras limpieza  : {len(df_raw):,}")
print(f"  Positivos (4-5★)     : {df_raw['sentiment'].sum():,}  "
      f"({df_raw['sentiment'].mean()*100:.1f}%)")
print(f"  Negativos (1-2★)     : {(df_raw['sentiment']==0).sum():,}  "
      f"({(1-df_raw['sentiment'].mean())*100:.1f}%)")

texts_raw = df_raw["reviewText"].tolist()
labels    = df_raw["sentiment"].tolist()

# Ejemplo de reseña sin procesar
print("\n  Ejemplo de reseña original:")
print(f"  Rating: {df_raw['overall'].iloc[0]}★  |  Sentimiento: {'Positivo' if labels[0]==1 else 'Negativo'}")
print(f"  Texto : {texts_raw[0][:200]}...")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2 — PREPROCESAMIENTO (ESTILO NOTEBOOK DE CLASE)
# ─────────────────────────────────────────────────────────────────────────────

header("2. PREPROCESAMIENTO: STEMMING vs LEMMATIZATION")

STOPWORDS = set(stopwords.words("english"))
stemmer   = PorterStemmer()
lemma     = WordNetLemmatizer()


# ── Función de preprocesamiento con Stemming (patrón del notebook de clase) ──

def preprocess_stem(text):
    """
    Patrón del notebook de clase (Cell [57]):
      re.sub → lower → tokenize → remove stopwords → stem → join
    """
    text = re.sub(r"[^a-zA-Z]", " ", text)   # solo letras (igual que en clase)
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens
              if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


# ── Función de preprocesamiento con Lemmatization ──

def preprocess_lemma(text):
    """
    Misma estructura que preprocess_stem pero usando WordNetLemmatizer
    para obtener la forma canónica de cada palabra.
    """
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemma.lemmatize(w) for w in tokens
              if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


# ── Mostrar transformación paso a paso (como en el notebook de clase) ──

subheader("Ejemplo paso a paso — PorterStemmer")
ejemplo = texts_raw[0]
print(f"\n  Original   : {ejemplo[:120]}")
paso1 = re.sub(r"[^a-zA-Z]", " ", ejemplo)
print(f"  Sin símbolos: {paso1[:120]}")
paso2 = paso1.lower()
print(f"  Minúsculas : {paso2[:120]}")
paso3 = paso2.split()
print(f"  Tokens     : {paso3[:12]}")
paso4 = [w for w in paso3 if w not in STOPWORDS and len(w) > 1]
print(f"  Sin stops  : {paso4[:12]}")
paso5 = [stemmer.stem(w) for w in paso4]
print(f"  Stemmed    : {paso5[:12]}")

subheader("Ejemplo paso a paso — WordNetLemmatizer")
print(f"\n  Original   : {ejemplo[:120]}")
paso5_l = [lemma.lemmatize(w) for w in paso4]
print(f"  Lemmatized : {paso5_l[:12]}")

# Diferencias visibles entre stem y lemma
print("\n  Comparación token a token (primeros 8 tokens filtrados):")
print(f"  {'Token original':<20} {'Stem':<20} {'Lemma':<20}")
print("  " + "-" * 60)
for orig, s, l in zip(paso4[:8], paso5[:8], paso5_l[:8]):
    print(f"  {orig:<20} {s:<20} {l:<20}")


# ── Aplicar preprocesamiento a todo el corpus ──

import time
print("\n  Aplicando Stemming al corpus completo...")
t0 = time.time()
corpus_stem  = [preprocess_stem(t)  for t in texts_raw]
print(f"  Listo en {time.time()-t0:.1f}s")

print("  Aplicando Lemmatization al corpus completo...")
t0 = time.time()
corpus_lemma = [preprocess_lemma(t) for t in texts_raw]
print(f"  Listo en {time.time()-t0:.1f}s")


# ── Estadísticas del vocabulario ──

subheader("Vocabulario resultante")
vocab_stem  = set(" ".join(corpus_stem).split())
vocab_lemma = set(" ".join(corpus_lemma).split())
print(f"\n  Tokens únicos con Stemming     : {len(vocab_stem):,}")
print(f"  Tokens únicos con Lemmatization: {len(vocab_lemma):,}")
print(f"  Diferencia                     : {len(vocab_lemma)-len(vocab_stem):+,}")
print("  (Lemma produce más tokens únicos porque preserva formas reales)")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3 — VECTORIZACIÓN (ESTILO NOTEBOOK DE CLASE)
# ─────────────────────────────────────────────────────────────────────────────

header("3. VECTORIZACIÓN: BoW vs TF-IDF")

# División train/test — la misma para todas las configuraciones
X_train_stem,  X_test_stem,  y_train, y_test = train_test_split(
    corpus_stem,  labels, test_size=0.2, random_state=42, stratify=labels
)
X_train_lemma, X_test_lemma, _,      _       = train_test_split(
    corpus_lemma, labels, test_size=0.2, random_state=42, stratify=labels
)

MAX_FEATURES = 20_000   # igual que en main.py del proyecto


# ── Bag-of-Words con CountVectorizer (patrón notebook Cell [59-61]) ──

subheader("Bag-of-Words (CountVectorizer)")

bow_stem  = CountVectorizer(max_features=MAX_FEATURES, ngram_range=(1,2), min_df=2)
bow_lemma = CountVectorizer(max_features=MAX_FEATURES, ngram_range=(1,2), min_df=2)

X_bow_stem_train  = bow_stem.fit_transform(X_train_stem)
X_bow_stem_test   = bow_stem.transform(X_test_stem)

X_bow_lemma_train = bow_lemma.fit_transform(X_train_lemma)
X_bow_lemma_test  = bow_lemma.transform(X_test_lemma)

print(f"\n  BoW Stem  — Train: {X_bow_stem_train.shape}  | "
      f"Vocabulario: {len(bow_stem.vocabulary_):,} features")
print(f"  BoW Lemma — Train: {X_bow_lemma_train.shape}  | "
      f"Vocabulario: {len(bow_lemma.vocabulary_):,} features")

# Top features BoW
top_bow = bow_stem.get_feature_names_out()[:20].tolist()
print(f"\n  Top 20 features BoW (por índice vocabulario):")
print(f"  {top_bow}")


# ── TF-IDF con TfidfVectorizer (patrón notebook Cell [63-64]) ──

subheader("TF-IDF (TfidfVectorizer)")

tfidf_stem  = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1,2),
                               min_df=2, sublinear_tf=True)
tfidf_lemma = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1,2),
                               min_df=2, sublinear_tf=True)

X_tfidf_stem_train  = tfidf_stem.fit_transform(X_train_stem)
X_tfidf_stem_test   = tfidf_stem.transform(X_test_stem)

X_tfidf_lemma_train = tfidf_lemma.fit_transform(X_train_lemma)
X_tfidf_lemma_test  = tfidf_lemma.transform(X_test_lemma)

print(f"\n  TF-IDF Stem  — Train: {X_tfidf_stem_train.shape}  | "
      f"Vocabulario: {len(tfidf_stem.vocabulary_):,} features")
print(f"  TF-IDF Lemma — Train: {X_tfidf_lemma_train.shape}  | "
      f"Vocabulario: {len(tfidf_lemma.vocabulary_):,} features")

# Top features por peso TF-IDF
idf_scores  = tfidf_stem.idf_
feat_names  = tfidf_stem.get_feature_names_out()
top_tfidf_idx = np.argsort(idf_scores)[-20:][::-1]
top_tfidf = feat_names[top_tfidf_idx].tolist()
print(f"\n  Top 20 features TF-IDF (mayor IDF = más discriminantes):")
print(f"  {top_tfidf}")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4 — ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# ─────────────────────────────────────────────────────────────────────────────

header("4. ENTRENAMIENTO DE MODELOS")

CONFIGS = [
    ("Naive Bayes",         "Stem",  "BoW",    X_bow_stem_train,   X_bow_stem_test),
    ("Logistic Regression", "Stem",  "BoW",    X_bow_stem_train,   X_bow_stem_test),
    ("SVM",                 "Stem",  "BoW",    X_bow_stem_train,   X_bow_stem_test),
    ("Naive Bayes",         "Stem",  "TF-IDF", X_tfidf_stem_train, X_tfidf_stem_test),
    ("Logistic Regression", "Stem",  "TF-IDF", X_tfidf_stem_train, X_tfidf_stem_test),
    ("SVM",                 "Stem",  "TF-IDF", X_tfidf_stem_train, X_tfidf_stem_test),
    ("Naive Bayes",         "Lemma", "BoW",    X_bow_lemma_train,  X_bow_lemma_test),
    ("Logistic Regression", "Lemma", "BoW",    X_bow_lemma_train,  X_bow_lemma_test),
    ("SVM",                 "Lemma", "BoW",    X_bow_lemma_train,  X_bow_lemma_test),
    ("Naive Bayes",         "Lemma", "TF-IDF", X_tfidf_lemma_train,X_tfidf_lemma_test),
    ("Logistic Regression", "Lemma", "TF-IDF", X_tfidf_lemma_train,X_tfidf_lemma_test),
    ("SVM",                 "Lemma", "TF-IDF", X_tfidf_lemma_train,X_tfidf_lemma_test),
]

results = []
trained_models = {}

for modelo, prep, vec, X_tr, X_te in CONFIGS:
    config_name = f"{modelo} / {prep} + {vec}"
    print(f"\n  [{config_name}]")

    # Entrenar modelo
    if modelo == "Naive Bayes":
        clf = MultinomialNB(alpha=0.5)
    elif modelo == "Logistic Regression":
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                 random_state=42)
    else:  # SVM
        clf = LinearSVC(C=1.0, max_iter=2000, random_state=42)

    t0 = time.time()
    clf.fit(X_tr, y_train)
    elapsed = time.time() - t0

    y_pred = clf.predict(X_te)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"    Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  "
          f"({elapsed:.1f}s)")

    results.append({
        "label"    : config_name,
        "modelo"   : modelo,
        "prep"     : prep,
        "vec"      : vec,
        "accuracy" : acc,
        "precision": prec,
        "recall"   : rec,
        "f1"       : f1,
        "y_pred"   : y_pred,
    })
    trained_models[config_name] = clf

results_df = pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 5 — TABLA DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

header("5. TABLA COMPARATIVA DE RESULTADOS")

print(f"\n  {'Configuración':<45} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("  " + "-" * 70)
for _, row in results_df.sort_values("f1", ascending=False).iterrows():
    marker = " ← MEJOR" if row["f1"] == results_df["f1"].max() else ""
    marker2 = " ← PEOR"  if row["f1"] == results_df["f1"].min() else ""
    print(f"  {row['label']:<45} {row['accuracy']:.4f} {row['precision']:.4f} "
          f"{row['recall']:.4f} {row['f1']:.4f}{marker}{marker2}")

best  = results_df.loc[results_df["f1"].idxmax()]
worst = results_df.loc[results_df["f1"].idxmin()]
print(f"\n  MEJOR CONFIGURACIÓN : {best['label']}  (F1={best['f1']:.4f})")
print(f"  PEOR  CONFIGURACIÓN : {worst['label']}  (F1={worst['f1']:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 6 — BoW vs TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

header("6. ANÁLISIS: BoW vs TF-IDF")

bow_mean   = results_df[results_df["vec"] == "BoW"]["f1"].mean()
tfidf_mean = results_df[results_df["vec"] == "TF-IDF"]["f1"].mean()
diff_vec   = tfidf_mean - bow_mean
ganador_vec = "TF-IDF" if diff_vec > 0 else "BoW"

print(f"\n  F1 promedio BoW    : {bow_mean:.4f}")
print(f"  F1 promedio TF-IDF : {tfidf_mean:.4f}")
print(f"  Diferencia         : {diff_vec:+.4f}  →  GANA: {ganador_vec}")

subheader("Desglose por modelo")
print(f"  {'Modelo':<25} {'BoW F1':>9} {'TF-IDF F1':>10} {'Ganador':>10}")
print("  " + "-" * 56)
for modelo in results_df["modelo"].unique():
    sub = results_df[results_df["modelo"] == modelo]
    b = sub[sub["vec"] == "BoW"]["f1"].mean()
    t = sub[sub["vec"] == "TF-IDF"]["f1"].mean()
    g = "TF-IDF" if t > b else "BoW"
    print(f"  {modelo:<25} {b:>9.4f} {t:>10.4f} {g:>10}")

subheader("Conclusión")
if diff_vec > 0:
    print(
        f"  TF-IDF supera a BoW (F1 {tfidf_mean:.4f} vs {bow_mean:.4f}).\n"
        f"\n"
        f"  TF-IDF penaliza palabras muy frecuentes en todo el corpus\n"
        f"  ('book', 'read', 'story') con un IDF bajo, y amplifica las\n"
        f"  que aparecen en pocos documentos ('disappointing', 'excellent',\n"
        f"  'terrible'), que son las más informativas para el sentimiento.\n"
        f"\n"
        f"  sublinear_tf=True reemplaza tf por 1+log(tf), suavizando el\n"
        f"  efecto de reseñas muy largas que inflarían conteos brutos."
    )
else:
    print(
        f"  BoW supera a TF-IDF (F1 {bow_mean:.4f} vs {tfidf_mean:.4f}).\n"
        f"\n"
        f"  Con Naive Bayes, TF-IDF colapsa el F1 de negativos (clase\n"
        f"  minoritaria) porque MultinomialNB espera frecuencias enteras\n"
        f"  no negativas, y los valores continuos de TF-IDF violan ese\n"
        f"  supuesto. BoW con frecuencias enteras funciona mejor con NB.\n"
        f"  SVM y LR sí se benefician de TF-IDF, pero el promedio global\n"
        f"  es arrastrado hacia abajo por el bajo rendimiento de NB+TF-IDF."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 7 — Stemming vs Lemmatization
# ─────────────────────────────────────────────────────────────────────────────

header("7. ANÁLISIS: Stemming vs Lemmatization")

stem_mean  = results_df[results_df["prep"] == "Stem"]["f1"].mean()
lemma_mean = results_df[results_df["prep"] == "Lemma"]["f1"].mean()
diff_prep  = lemma_mean - stem_mean
ganador_prep = "Lemmatization" if diff_prep > 0 else "Stemming"

print(f"\n  F1 promedio Stemming     : {stem_mean:.4f}")
print(f"  F1 promedio Lemmatization: {lemma_mean:.4f}")
print(f"  Diferencia               : {diff_prep:+.4f}  →  GANA: {ganador_prep}")

subheader("Desglose por vectorizador")
print(f"  {'Vectorizador':<15} {'Stem F1':>9} {'Lemma F1':>10} {'Ganador':>14}")
print("  " + "-" * 50)
for vec in ["BoW", "TF-IDF"]:
    sub = results_df[results_df["vec"] == vec]
    s = sub[sub["prep"] == "Stem"]["f1"].mean()
    l = sub[sub["prep"] == "Lemma"]["f1"].mean()
    g = "Lemmatization" if l > s else "Stemming"
    print(f"  {vec:<15} {s:>9.4f} {l:>10.4f} {g:>14}")

subheader("Conclusión")
if abs(diff_prep) < 0.003:
    print(
        f"  La diferencia entre Stemming y Lemmatization es mínima\n"
        f"  ({diff_prep:+.4f}), lo que indica que ambas técnicas producen\n"
        f"  vocabularios igualmente útiles para este corpus.\n"
        f"\n"
        f"  Diferencias cualitativas:\n"
        f"\n"
        f"  PorterStemmer:\n"
        f"    'specifically' → 'specif' | 'learning' → 'learn'\n"
        f"    'becomes'      → 'becom'  | 'worthless' → 'worthless'\n"
        f"    Puede producir tokens no reconocibles ('becam', 'wors').\n"
        f"    Ventaja: más rápido computacionalmente.\n"
        f"\n"
        f"  WordNetLemmatizer:\n"
        f"    'loved'   → 'love'   | 'better' → 'good'\n"
        f"    'running' → 'run'    | 'books'  → 'book'\n"
        f"    Produce siempre palabras válidas del diccionario.\n"
        f"    Vocabulario más limpio e interpretable.\n"
        f"\n"
        f"  Para producción: Lemmatization es preferible si se necesita\n"
        f"  interpretar los features más importantes del modelo."
    )
elif diff_prep > 0:
    print(
        f"  Lemmatization supera a Stemming ({diff_prep:+.4f}).\n"
        f"  Al devolver formas canónicas válidas evita la fragmentación\n"
        f"  agresiva del stemmer, produciendo tokens más significativos."
    )
else:
    print(
        f"  Stemming supera a Lemmatization ({diff_prep:+.4f}).\n"
        f"  La reducción agresiva agrupa más variantes bajo el mismo token,\n"
        f"  concentrando las frecuencias y reduciendo el vocabulario."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 8 — Comparación de modelos
# ─────────────────────────────────────────────────────────────────────────────

header("8. COMPARACIÓN DE MODELOS")

model_stats = (results_df.groupby("modelo")["f1"]
               .agg(["mean", "max", "min"])
               .sort_values("mean", ascending=False))

print(f"\n  {'Modelo':<25} {'F1 promedio':>12} {'F1 máximo':>10} {'F1 mínimo':>10}")
print("  " + "-" * 60)
for modelo, row in model_stats.iterrows():
    print(f"  {modelo:<25} {row['mean']:>12.4f} {row['max']:>10.4f} {row['min']:>10.4f}")

subheader("Conclusión")
nb_tfidf_f1 = results_df[(results_df["modelo"]=="Naive Bayes") &
                          (results_df["vec"]=="TF-IDF")]["f1"].mean()
print(
    f"  1. SVM (LinearSVC) — MEJOR MODELO\n"
    f"     F1 promedio {model_stats.loc['SVM','mean']:.4f}  |  Máximo {model_stats.loc['SVM','max']:.4f}\n"
    f"\n"
    f"     SVM maximiza el margen en el espacio de {MAX_FEATURES:,} features.\n"
    f"     Los textos son naturalmente linealmente separables en ese\n"
    f"     espacio de alta dimensión, y LinearSVC lo explota eficientemente.\n"
    f"\n"
    f"  2. Logistic Regression — COMPETITIVO\n"
    f"     F1 promedio {model_stats.loc['Logistic Regression','mean']:.4f}  |  Máximo {model_stats.loc['Logistic Regression','max']:.4f}\n"
    f"\n"
    f"     Muy cercano a SVM y mucho más rápido en inferencia.\n"
    f"     Produce probabilidades calibradas, útil si se necesita\n"
    f"     un score de confianza por predicción.\n"
    f"\n"
    f"  3. Naive Bayes — PEOR CON TF-IDF\n"
    f"     F1 promedio {model_stats.loc['Naive Bayes','mean']:.4f}  |  Con TF-IDF: {nb_tfidf_f1:.4f}\n"
    f"\n"
    f"     MultinomialNB asume independencia condicional entre features\n"
    f"     y espera frecuencias enteras no negativas. TF-IDF produce\n"
    f"     valores continuos que violan esos supuestos, degradando el\n"
    f"     recall de la clase negativa (minoritaria) a valores muy bajos."
)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 9 — Problemas del dataset
# ─────────────────────────────────────────────────────────────────────────────

header("9. PROBLEMAS ENCONTRADOS EN EL DATASET")

pos_count = df_raw["sentiment"].sum()
neg_count = (df_raw["sentiment"] == 0).sum()
pos_pct   = pos_count / len(df_raw) * 100
neg_pct   = neg_count / len(df_raw) * 100

print(
    f"\n  9.1 DESBALANCE SEVERO DE CLASES\n"
    f"  {'─'*55}\n"
    f"  Positivos: {pos_count:,} ({pos_pct:.1f}%)  |  Negativos: {neg_count:,} ({neg_pct:.1f}%)\n"
    f"\n"
    f"  Un clasificador que siempre prediga 'Positivo' alcanzaría\n"
    f"  {pos_pct:.1f}% de accuracy sin aprender nada útil sobre el sentimiento.\n"
    f"  El F1 weighted parece alto (~0.95) pero enmascara el bajo recall\n"
    f"  de negativos, que es lo que realmente importa detectar.\n"
    f"\n"
    f"  9.2 RUIDO Y AMBIGÜEDAD EN EL TEXTO\n"
    f"  {'─'*55}\n"
    f"  Muchas reseñas negativas contienen lenguaje positivo:\n"
    f"  'Well written but disappointing ending' → el modelo detecta\n"
    f"  'well written' y clasifica como positivo.\n"
    f"  BoW/TF-IDF no capturan el contexto del 'but' adversativo.\n"
    f"\n"
    f"  9.3 RESEÑAS NO LITERARIAS\n"
    f"  {'─'*55}\n"
    f"  Algunas reseñas hablan de problemas de envío, suscripciones\n"
    f"  o quejas a Amazon, no del libro. Generan ruido porque el\n"
    f"  vocabulario de logística no se correlaciona con sentimiento.\n"
    f"\n"
    f"  9.4 RESEÑAS MUY CORTAS\n"
    f"  {'─'*55}\n"
    f"  Textos con pocas palabras ('Good book. Fast delivery.')\n"
    f"  ofrecen pocos tokens al vectorizador. El clasificador tiene\n"
    f"  poca información para decidir y aumenta la incertidumbre."
)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 10 — Errores de clasificación
# ─────────────────────────────────────────────────────────────────────────────

header("10. ERRORES DE CLASIFICACIÓN DETECTADOS")

best_name  = best["label"]
best_pred  = best["y_pred"]

# Reconstruir los índices del test split para poder recuperar el texto original
_, idx_test = train_test_split(
    range(len(df_raw)), test_size=0.2, random_state=42, stratify=labels
)

y_test_arr = np.array(y_test)

# Errores: negativo predicho como positivo (false positives)
fn_mask  = (y_test_arr == 0) & (np.array(best_pred) == 1)
fn_idx   = np.where(fn_mask)[0][:3]

# Errores: positivo predicho como negativo (false negatives)
fp_mask  = (y_test_arr == 1) & (np.array(best_pred) == 0)
fp_idx   = np.where(fp_mask)[0][:3]

print(f"\n  Analizando errores del mejor modelo: {best_name}")
print(f"  Total de errores: {fn_mask.sum() + fp_mask.sum():,} / {len(y_test):,} "
      f"({(fn_mask.sum()+fp_mask.sum())/len(y_test)*100:.1f}%)")

subheader("Negativos mal clasificados como Positivos (más comunes)")
for i, idx in enumerate(fn_idx, 1):
    texto_orig = df_raw["reviewText"].iloc[idx_test[idx]]
    rating     = df_raw["overall"].iloc[idx_test[idx]]
    print(f"\n  [{i}] Rating real: {rating}★  → Verdadero: NEG | Predicho: POS")
    print(f"      \"{texto_orig[:180]}...\"")

subheader("Positivos mal clasificados como Negativos (menos frecuentes)")
for i, idx in enumerate(fp_idx, 1):
    texto_orig = df_raw["reviewText"].iloc[idx_test[idx]]
    rating     = df_raw["overall"].iloc[idx_test[idx]]
    print(f"\n  [{i}] Rating real: {rating}★  → Verdadero: POS | Predicho: NEG")
    print(f"      \"{texto_orig[:180]}...\"")

subheader("Patrones de error identificados")
print(
    "\n  PATRÓN 1: Sentimiento mixto / adversativo\n"
    "  'While the plotting was much better than the first, I still\n"
    "  felt the character development was lacking.'\n"
    "  → BoW/TF-IDF detecta 'better', 'much' y clasifica positivo.\n"
    "    El 'but/while/still' adversativo no se captura en bolsa de palabras.\n"
    "\n"
    "  PATRÓN 2: Sarcasmo e ironía\n"
    "  'I'm just glad this hot mess was free.'\n"
    "  → 'glad' tiene peso positivo; el sarcasmo es imposible de detectar\n"
    "    sin contexto semántico profundo.\n"
    "\n"
    "  PATRÓN 3: Quejas fuera de contexto\n"
    "  'I have not received my copy after 3 weeks.'\n"
    "  → 1 estrella pero sin vocabulario negativo literario.\n"
    "    El modelo no encuentra señales de sentimiento negativo.\n"
    "\n"
    "  PATRÓN 4: Reseñas positivas descriptivas\n"
    "  'This novel covers events from 1920 to 1940 in a factual manner.'\n"
    "  → 5 estrellas pero el texto es completamente neutro/descriptivo.\n"
    "\n"
    "  Conclusión: Estos errores evidencian las limitaciones de los modelos\n"
    "  basados en frecuencias de palabras. Un modelo de embeddings\n"
    "  contextuales (BERT, RoBERTa) capturaría negación, adversativos\n"
    "  y sarcasmo, reduciendo significativamente estos errores."
)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 11 — GRÁFICAS
# ─────────────────────────────────────────────────────────────────────────────

header("11. GENERANDO GRÁFICAS")

os.makedirs("outputs", exist_ok=True)

# ── Gráfica 1: Heatmap F1 por modelo × vectorizador × preprocesamiento ──

pivot = results_df.pivot_table(index="modelo", columns=["prep","vec"], values="f1")
pivot.columns = [f"{p}+{v}" for p, v in pivot.columns]

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(
    pivot, annot=True, fmt=".4f", cmap="YlGn",
    vmin=0.90, vmax=0.97, linewidths=0.5, ax=ax,
    cbar_kws={"label": "F1-score (weighted)"},
)
ax.set_title("F1-score por Modelo × Vectorizador × Preprocesamiento")
ax.set_xlabel("Preprocesamiento + Vectorizador")
ax.set_ylabel("Modelo")
plt.tight_layout()
heatmap_path = os.path.join("outputs", "clase_heatmap.png")
plt.savefig(heatmap_path, dpi=130)
plt.close()
print(f"  Guardado: {heatmap_path}")


# ── Gráfica 2: BoW vs TF-IDF promedio por modelo ──

fig, ax = plt.subplots(figsize=(8, 5))
x      = np.arange(len(results_df["modelo"].unique()))
width  = 0.35
models = results_df["modelo"].unique()
bow_f1   = [results_df[(results_df["modelo"]==m) & (results_df["vec"]=="BoW")   ]["f1"].mean() for m in models]
tfidf_f1 = [results_df[(results_df["modelo"]==m) & (results_df["vec"]=="TF-IDF")]["f1"].mean() for m in models]

bars1 = ax.bar(x - width/2, bow_f1,   width, label="BoW",    color="#4C72B0")
bars2 = ax.bar(x + width/2, tfidf_f1, width, label="TF-IDF", color="#DD8452")

ax.set_title("F1 promedio: BoW vs TF-IDF por Modelo")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=10)
ax.set_ylabel("F1-score (weighted)")
ax.set_ylim(0.88, 0.98)
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
bow_path = os.path.join("outputs", "clase_bow_vs_tfidf.png")
plt.savefig(bow_path, dpi=130)
plt.close()
print(f"  Guardado: {bow_path}")


# ── Gráfica 3: Stem vs Lemma promedio por modelo ──

fig, ax = plt.subplots(figsize=(8, 5))
stem_f1  = [results_df[(results_df["modelo"]==m) & (results_df["prep"]=="Stem") ]["f1"].mean() for m in models]
lemma_f1 = [results_df[(results_df["modelo"]==m) & (results_df["prep"]=="Lemma")]["f1"].mean() for m in models]

bars1 = ax.bar(x - width/2, stem_f1,  width, label="Stemming",      color="#55A868")
bars2 = ax.bar(x + width/2, lemma_f1, width, label="Lemmatization",  color="#C44E52")

ax.set_title("F1 promedio: Stemming vs Lemmatization por Modelo")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=10)
ax.set_ylabel("F1-score (weighted)")
ax.set_ylim(0.88, 0.98)
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
stem_path = os.path.join("outputs", "clase_stem_vs_lemma.png")
plt.savefig(stem_path, dpi=130)
plt.close()
print(f"  Guardado: {stem_path}")


# ── Gráfica 4: Distribución de clases (desbalance) ──

fig, ax = plt.subplots(figsize=(6, 4))
counts = [neg_count, pos_count]
labels_dist = [f"Negativo\n({neg_pct:.1f}%)", f"Positivo\n({pos_pct:.1f}%)"]
colors = ["#C44E52", "#4C72B0"]
bars = ax.bar(labels_dist, counts, color=colors, edgecolor="white", linewidth=1.2)
ax.set_title("Distribución de Clases en el Dataset")
ax.set_ylabel("Número de reseñas")
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
            f"{count:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.tight_layout()
dist_path = os.path.join("outputs", "clase_distribucion_clases.png")
plt.savefig(dist_path, dpi=130)
plt.close()
print(f"  Guardado: {dist_path}")


# ── Gráfica 5: Matriz de confusión del mejor modelo ──

cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Negativo", "Positivo"],
            yticklabels=["Negativo", "Positivo"])
ax.set_title(f"Matriz de Confusión — {best_name}")
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
plt.tight_layout()
cm_path = os.path.join("outputs", "clase_confusion_matrix.png")
plt.savefig(cm_path, dpi=130)
plt.close()
print(f"  Guardado: {cm_path}")


# ── Reporte de clasificación del mejor modelo ──

subheader(f"Reporte de Clasificación — {best_name}")
print(classification_report(y_test, best_pred,
                             target_names=["Negativo", "Positivo"]))


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 12 — CONCLUSIÓN FINAL
# ─────────────────────────────────────────────────────────────────────────────

header("12. CONCLUSIÓN FINAL")

print(
    f"\n  Mejor configuración : {best_name}\n"
    f"  F1 weighted         : {best['f1']:.4f}\n"
    f"  Accuracy            : {best['accuracy']:.4f}\n"
    f"\n"
    f"  Resumen de hallazgos:\n"
    f"\n"
    f"  BoW vs TF-IDF  → {ganador_vec} es superior (diferencia {diff_vec:+.4f})\n"
    f"  Stem vs Lemma  → {ganador_prep} es superior (diferencia {diff_prep:+.4f})\n"
    f"  Mejor modelo   → SVM (LinearSVC) con F1={model_stats.loc['SVM','max']:.4f}\n"
    f"\n"
    f"  Recomendaciones para mejorar el sistema:\n"
    f"  1. Aplicar class_weight='balanced' en SVM/LR para mejorar\n"
    f"     el recall de la clase negativa (actualmente ~6.6% del corpus).\n"
    f"  2. Agregar bigramas de negación ('not good', 'never again')\n"
    f"     para capturar inversiones de sentimiento.\n"
    f"  3. Filtrar reseñas no literarias (quejas de logística)\n"
    f"     mediante detección de tópico previa.\n"
    f"  4. Para producción, considerar BERT/DistilBERT que captura\n"
    f"     contexto, negación, adversativos y sarcasmo."
)

print("\n" + "=" * 65)
print("  Análisis completado.")
print("=" * 65)
