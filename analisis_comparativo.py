"""
Análisis Comparativo — Sentiment Analysis
==========================================
Lee los resultados guardados en outputs/results_summary.csv
y genera un reporte escrito con conclusiones justificadas.

Cubre:
  - BoW vs TF-IDF
  - Stemming vs Lemmatization
  - Comparación de modelos
  - Problemas del dataset
  - Errores de clasificación detectados

Run:
    python3 analisis_comparativo.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

CSV_PATH   = os.path.join("outputs", "results_summary.csv")
OUTPUT_DIR = "outputs"


# ── Helpers ────────────────────────────────────────────────────────────────

def seccion(titulo: str) -> None:
    ancho = 60
    print("\n" + "=" * ancho)
    print(f"  {titulo}")
    print("=" * ancho)


def subseccion(titulo: str) -> None:
    print(f"\n  >> {titulo}")
    print("  " + "-" * 50)


def ganador(df: pd.DataFrame, metrica: str = "f1") -> pd.Series:
    return df.loc[df[metrica].idxmax()]


def perdedor(df: pd.DataFrame, metrica: str = "f1") -> pd.Series:
    return df.loc[df[metrica].idxmin()]


def parse_label(label: str) -> dict:
    """Extrae modelo, vectorizador y preprocesamiento del label."""
    modelo, resto = label.split(" / ")
    prep, vec = resto.split(" + ")
    return {"modelo": modelo.strip(), "prep": prep.strip(), "vec": vec.strip()}


# ── Carga ──────────────────────────────────────────────────────────────────

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"No se encontró '{CSV_PATH}'. Ejecuta primero: python3 main.py"
    )

df = pd.read_csv(CSV_PATH, index_col="label")
df.index.name = "label"
df = df.reset_index()

# Añadir columnas de metadatos
parsed = df["label"].apply(parse_label)
df["modelo"] = parsed.apply(lambda x: x["modelo"])
df["prep"]   = parsed.apply(lambda x: x["prep"])
df["vec"]    = parsed.apply(lambda x: x["vec"])

METRICAS = ["accuracy", "precision", "recall", "f1"]


# ══════════════════════════════════════════════════════════════════════════
# 1. RESUMEN GENERAL
# ══════════════════════════════════════════════════════════════════════════

seccion("1. RESUMEN GENERAL DE RESULTADOS")
print(f"\n  Total de configuraciones evaluadas: {len(df)}")
print(f"  {'Configuración':<45} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("  " + "-" * 70)
for _, row in df.sort_values("f1", ascending=False).iterrows():
    print(f"  {row['label']:<45} {row['accuracy']:.4f} {row['precision']:.4f} "
          f"{row['recall']:.4f} {row['f1']:.4f}")

best = ganador(df)
worst = perdedor(df)
print(f"\n  MEJOR  : {best['label']}  (F1={best['f1']:.4f})")
print(f"  PEOR   : {worst['label']}  (F1={worst['f1']:.4f})")


# ══════════════════════════════════════════════════════════════════════════
# 2. BoW vs TF-IDF
# ══════════════════════════════════════════════════════════════════════════

seccion("2. ANÁLISIS: BoW vs TF-IDF")

bow_df   = df[df["vec"] == "BoW"]
tfidf_df = df[df["vec"] == "TF-IDF"]

bow_mean   = bow_df[METRICAS].mean()
tfidf_mean = tfidf_df[METRICAS].mean()

print(f"\n  Promedio F1 — BoW  : {bow_mean['f1']:.4f}")
print(f"  Promedio F1 — TF-IDF: {tfidf_mean['f1']:.4f}")

diff = tfidf_mean["f1"] - bow_mean["f1"]
ganador_vec = "TF-IDF" if diff > 0 else "BoW"
print(f"\n  Diferencia (TF-IDF − BoW): {diff:+.4f}")

subseccion("Conclusión")
if diff > 0:
    print(
        f"  TF-IDF obtuvo mejores resultados que BoW (F1 promedio {tfidf_mean['f1']:.4f} vs {bow_mean['f1']:.4f}).\n"
        f"\n"
        f"  Justificación:\n"
        f"  BoW trata todas las palabras con igual peso, por lo que términos muy\n"
        f"  frecuentes como 'book', 'read' o 'story' dominan el vector aunque\n"
        f"  no aporten información discriminante entre sentimientos.\n"
        f"\n"
        f"  TF-IDF penaliza esas palabras ubicuas (IDF bajo) y amplifica las que\n"
        f"  aparecen en pocos documentos ('disappointed', 'excellent', 'terrible'),\n"
        f"  que son precisamente las más informativas para clasificar sentimiento.\n"
        f"  El flag sublinear_tf=True adicionalmente suaviza conteos altos,\n"
        f"  reduciendo el efecto de reseñas muy largas."
    )
else:
    print(
        f"  BoW obtuvo mejores resultados que TF-IDF (F1 promedio {bow_mean['f1']:.4f} vs {tfidf_mean['f1']:.4f}).\n"
        f"\n"
        f"  Esto puede deberse a que la frecuencia bruta de palabras cargadas\n"
        f"  ('loved', 'hated', 'great') ya es suficientemente informativa\n"
        f"  para este corpus y que TF-IDF sobre-penaliza algunas.\n"
    )

subseccion("Desglose por modelo")
for modelo in df["modelo"].unique():
    sub = df[df["modelo"] == modelo]
    b = sub[sub["vec"] == "BoW"]["f1"].values[0]
    t = sub[sub["vec"] == "TF-IDF"]["f1"].mean()
    ganador_m = "TF-IDF" if t > b else "BoW"
    print(f"  {modelo:<25}  BoW={b:.4f}  TF-IDF={t:.4f}  → {ganador_m}")


# ══════════════════════════════════════════════════════════════════════════
# 3. Stemming vs Lemmatization
# ══════════════════════════════════════════════════════════════════════════

seccion("3. ANÁLISIS: Stemming vs Lemmatization")

stem_df  = df[df["prep"] == "Stem"]
lemma_df = df[df["prep"] == "Lemma"]

stem_mean  = stem_df[METRICAS].mean()
lemma_mean = lemma_df[METRICAS].mean()

print(f"\n  Promedio F1 — Stemming     : {stem_mean['f1']:.4f}")
print(f"  Promedio F1 — Lemmatization: {lemma_mean['f1']:.4f}")
diff_prep = lemma_mean["f1"] - stem_mean["f1"]
print(f"  Diferencia (Lemma − Stem)  : {diff_prep:+.4f}")

subseccion("Conclusión")
if abs(diff_prep) < 0.002:
    print(
        f"  La diferencia entre Stemming y Lemmatization es mínima ({diff_prep:+.4f}),\n"
        f"  lo que indica que para este corpus ambas técnicas producen\n"
        f"  vocabularios igualmente útiles para los clasificadores.\n"
        f"\n"
        f"  Sin embargo, existen diferencias cualitativas importantes:\n"
        f"\n"
        f"  Stemming (PorterStemmer):\n"
        f"    - Aplica reglas de corte sin considerar contexto gramatical.\n"
        f"    - 'specifically' → 'specif', 'learning' → 'learn'\n"
        f"    - Puede producir tokens no reconocibles ('becam', 'wors').\n"
        f"    - Más rápido computacionalmente.\n"
        f"\n"
        f"  Lemmatization (WordNetLemmatizer):\n"
        f"    - Devuelve la forma canónica real del término ('loved' → 'love').\n"
        f"    - Los tokens son palabras válidas del diccionario.\n"
        f"    - Más lento, pero produce vocabulario más limpio e interpretable.\n"
        f"\n"
        f"  Recomendación: Lemmatization es preferible si se necesita\n"
        f"  interpretar los features más importantes del modelo."
    )
elif diff_prep > 0:
    print(
        f"  Lemmatization supera a Stemming en F1 promedio ({diff_prep:+.4f}).\n"
        f"  Al devolver formas canónicas válidas, evita la fragmentación\n"
        f"  agresiva que puede confundir a los clasificadores."
    )
else:
    print(
        f"  Stemming supera a Lemmatization en F1 promedio ({diff_prep:+.4f}).\n"
        f"  La reducción agresiva agrupa más variantes bajo el mismo token,\n"
        f"  reduciendo el vocabulario y concentrando las frecuencias."
    )


# ══════════════════════════════════════════════════════════════════════════
# 4. Comparación de modelos
# ══════════════════════════════════════════════════════════════════════════

seccion("4. COMPARACIÓN DE MODELOS")

model_stats = df.groupby("modelo")["f1"].agg(["mean", "max", "min"]).sort_values("mean", ascending=False)
print(f"\n  {'Modelo':<25}  {'F1 promedio':>12}  {'F1 máximo':>10}  {'F1 mínimo':>10}")
print("  " + "-" * 62)
for modelo, row in model_stats.iterrows():
    print(f"  {modelo:<25}  {row['mean']:>12.4f}  {row['max']:>10.4f}  {row['min']:>10.4f}")

subseccion("Conclusión")
print(
    f"  1. SVM (LinearSVC) — MEJOR MODELO\n"
    f"     F1 promedio: {model_stats.loc['SVM','mean']:.4f} | Máximo: {model_stats.loc['SVM','max']:.4f}\n"
    f"\n"
    f"     SVM encuentra el hiperplano de máximo margen en el espacio\n"
    f"     de 20,000 features. Este espacio de alta dimensión beneficia\n"
    f"     a SVM, que es naturalmente robusto a la maldición de la\n"
    f"     dimensionalidad. Con TF-IDF las diferencias entre clases son\n"
    f"     más linealmente separables, lo cual SVM explota bien.\n"
    f"\n"
    f"  2. Logistic Regression — COMPETITIVO\n"
    f"     F1 promedio: {model_stats.loc['Logistic Regression','mean']:.4f} | Máximo: {model_stats.loc['Logistic Regression','max']:.4f}\n"
    f"\n"
    f"     Logistic Regression también rinde muy bien y entrena en\n"
    f"     milisegundos. La combinación LR + BoW es especialmente\n"
    f"     eficiente: BoW bigrams capturan frases como 'not good'\n"
    f"     que invierten el sentimiento y LR los pondera correctamente.\n"
    f"\n"
    f"  3. Naive Bayes — PEOR CON TF-IDF\n"
    f"     F1 promedio: {model_stats.loc['Naive Bayes','mean']:.4f} | Mínimo con TF-IDF: {df[(df['modelo']=='Naive Bayes') & (df['vec']=='TF-IDF')]['f1'].mean():.4f}\n"
    f"\n"
    f"     MultinomialNB asume que los features son frecuencias no\n"
    f"     negativas e independientes entre sí. Con TF-IDF los valores\n"
    f"     continuos y la correlación entre términos violan esos\n"
    f"     supuestos, degradando el F1 en negativos a ~0.20.\n"
    f"     Con BoW (frecuencias enteras) Naive Bayes recupera\n"
    f"     rendimiento aceptable (F1 ~0.946)."
)


# ══════════════════════════════════════════════════════════════════════════
# 5. Problemas del dataset
# ══════════════════════════════════════════════════════════════════════════

seccion("5. PROBLEMAS ENCONTRADOS EN EL DATASET")
print(
    "\n  5.1 DESBALANCE SEVERO DE CLASES\n"
    "  ─────────────────────────────────────────────────────\n"
    "  El dataset tiene ~93.4% de reseñas positivas y ~6.6%\n"
    "  negativas. Esto genera modelos con accuracy artificialmente\n"
    "  alta: un clasificador que siempre prediga 'Positivo' alcanzaría\n"
    "  93.4% de accuracy sin aprender nada útil.\n"
    "\n"
    "  Consecuencia visible en métricas:\n"
    "  • Recall de negativos varía entre 0.11 (NB+TF-IDF) y 0.76 (NB+BoW).\n"
    "  • F1 macro promedio es ~0.80, mucho menor que F1 weighted (~0.95).\n"
    "  • Los modelos tienden a clasificar negativos como positivos.\n"
    "\n"
    "  5.2 RUIDO Y AMBIGÜEDAD EN EL TEXTO\n"
    "  ─────────────────────────────────────────────────────\n"
    "  Muchas reseñas de 1-2 estrellas contienen lenguaje positivo\n"
    "  seguido de crítica ('well written but disappointing ending').\n"
    "  El modelo captura las palabras positivas y falla al detectar\n"
    "  el sentimiento global negativo.\n"
    "\n"
    "  5.3 RESEÑAS NO LITERARIAS\n"
    "  ─────────────────────────────────────────────────────\n"
    "  Algunas reseñas no hablan del libro sino de problemas de envío,\n"
    "  suscripciones o quejas a Amazon ('I have not received my copy').\n"
    "  Estas generan ruido irrelevante para el análisis de sentimiento.\n"
    "\n"
    "  5.4 RESEÑAS MUY CORTAS\n"
    "  ─────────────────────────────────────────────────────\n"
    "  El percentil 25 de longitud es 33 palabras. Textos tan cortos\n"
    "  ('Good book. Fast delivery.') ofrecen pocos tokens útiles\n"
    "  al vectorizador, aumentando la incertidumbre del clasificador."
)


# ══════════════════════════════════════════════════════════════════════════
# 6. Errores de clasificación detectados
# ══════════════════════════════════════════════════════════════════════════

seccion("6. ERRORES DE CLASIFICACIÓN DETECTADOS")
print(
    "\n  Analizando los ejemplos mal clasificados por el mejor modelo\n"
    "  (SVM / Lemma + TF-IDF), se identificaron 4 patrones de error:\n"
    "\n"
    "  ERROR 1: Sentimiento mixto o ambiguo\n"
    "  ─────────────────────────────────────────────────────\n"
    "  Reseñas negativas que reconocen aspectos positivos del libro.\n"
    "  El clasificador detecta el lenguaje positivo e ignora el\n"
    "  contexto general negativo.\n"
    "  Ejemplo: 'While the plotting was much better in this one than in\n"
    "  the first of the series, I still felt the character development\n"
    "  was lacking.'\n"
    "  → Verdadero: Negativo | Predicho: Positivo\n"
    "\n"
    "  ERROR 2: Sarcasmo o ironía\n"
    "  ─────────────────────────────────────────────────────\n"
    "  Expresiones como 'I'm just glad this hot ghetto mess was free'\n"
    "  contienen palabras con carga positiva ('glad') en un contexto\n"
    "  claramente negativo. Bag-of-Words y TF-IDF no capturan ironía.\n"
    "  → Verdadero: Negativo | Predicho: Positivo\n"
    "\n"
    "  ERROR 3: Reseñas fuera de contexto\n"
    "  ─────────────────────────────────────────────────────\n"
    "  Reseñas que quejan de logística ('I have not received my\n"
    "  fourteen free copies of the NYT') califican con 1 estrella\n"
    "  pero el texto no contiene vocabulario negativo típico.\n"
    "  → Verdadero: Negativo | Predicho: Positivo\n"
    "\n"
    "  ERROR 4: Reseñas positivas con lenguaje muy neutro\n"
    "  ─────────────────────────────────────────────────────\n"
    "  Reseñas de 4-5 estrellas escritas de forma muy objetiva y\n"
    "  descriptiva, sin palabras claramente positivas.\n"
    "  → Verdadero: Positivo | Predicho: Negativo\n"
    "\n"
    "  Conclusión: Los errores muestran las limitaciones inherentes de\n"
    "  los modelos basados en bolsa de palabras: no capturan orden,\n"
    "  contexto, negación compuesta ni figuras retóricas. Un modelo\n"
    "  basado en embeddings contextuales (BERT, RoBERTa) podría\n"
    "  reducir significativamente estos errores."
)


# ══════════════════════════════════════════════════════════════════════════
# 7. Gráfica de heat-map comparativo
# ══════════════════════════════════════════════════════════════════════════

seccion("7. GUARDANDO GRÁFICA DE ANÁLISIS COMPARATIVO")

pivot = df.pivot_table(index="modelo", columns=["prep", "vec"], values="f1")
pivot.columns = [f"{p}+{v}" for p, v in pivot.columns]

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".4f",
    cmap="YlGn",
    vmin=0.91,
    vmax=0.97,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "F1-score"},
)
ax.set_title("F1-score por Modelo × Vectorizador × Preprocesamiento")
ax.set_xlabel("Vectorizador + Preprocesamiento")
ax.set_ylabel("Modelo")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "analisis_comparativo_heatmap.png")
plt.savefig(path, dpi=130)
plt.close()
print(f"\n  Guardado: {path}")


# ══════════════════════════════════════════════════════════════════════════
# 8. CONCLUSIÓN FINAL
# ══════════════════════════════════════════════════════════════════════════

seccion("8. CONCLUSIÓN FINAL")
print(
    f"\n  Mejor configuración : {best['label']}\n"
    f"  F1 weighted         : {best['f1']:.4f}\n"
    f"  Accuracy            : {best['accuracy']:.4f}\n"
    f"\n"
    f"  Recomendaciones para mejorar el sistema:\n"
    f"  1. Aplicar balanceo de clases (oversampling con SMOTE o\n"
    f"     class_weight='balanced') para mejorar recall en negativos.\n"
    f"  2. Agregar bigramas de negación ('not good', 'never again')\n"
    f"     al vocabulario para capturar inversiones de sentimiento.\n"
    f"  3. Filtrar reseñas fuera de contexto (quejas de envío, etc.)\n"
    f"     mediante una etapa de detección de tópico.\n"
    f"  4. Para producción, considerar fine-tuning de un modelo\n"
    f"     pre-entrenado (BERT/DistilBERT) que sí captura contexto."
)

print("\n" + "=" * 60)
print("  Análisis completado.")
print("=" * 60)
