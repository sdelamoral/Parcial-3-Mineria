"""
Sentiment Analysis Dashboard — Kindle Reviews
==============================================
Run:
    venv/bin/streamlit run app.py
"""

import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from src.preprocessor import NLPPreprocessor

# ── Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis — Kindle Reviews",
    page_icon="📚",
    layout="wide",
)

OUTPUT_DIR = "outputs"
MODEL_DIR  = "model"
CSV_PATH   = os.path.join(OUTPUT_DIR, "results_summary.csv")

PAGES = [
    "Inicio",
    "Dataset",
    "Preprocesamiento",
    "Resultados",
    "Analisis Comparativo",
    "Demo",
]

ICONS = {
    "Inicio":               "🏠",
    "Dataset":              "📊",
    "Preprocesamiento":     "🔤",
    "Resultados":           "📈",
    "Analisis Comparativo": "🔍",
    "Demo":                 "🎯",
}

DESCRIPCIONES = {
    "Inicio":               "Resumen del proyecto",
    "Dataset":              "Explora los datos",
    "Preprocesamiento":     "Prueba el pipeline NLP",
    "Resultados":           "Métricas de los 12 modelos",
    "Analisis Comparativo": "BoW vs TF-IDF · modelos · errores",
    "Demo":                 "Clasifica texto en vivo",
}

# ── Session state ──────────────────────────────────────────────────────────
if "pagina" not in st.session_state:
    st.session_state.pagina = "Inicio"
if "ejemplo_texto" not in st.session_state:
    st.session_state.ejemplo_texto = ""


def ir_a(nombre: str):
    st.session_state.pagina = nombre


# ── Loaders ────────────────────────────────────────────────────────────────
@st.cache_data
def load_results() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH, index_col="label").reset_index()


@st.cache_resource
def load_model():
    m   = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    vec = joblib.load(os.path.join(MODEL_DIR, "best_vectorizer.pkl"))
    cfg = {}
    with open(os.path.join(MODEL_DIR, "best_config.txt")) as f:
        for line in f:
            k, v = line.strip().split("=", 1)
            cfg[k] = v
    return m, vec, cfg


def parse_label(label: str) -> dict:
    modelo, resto = label.split(" / ")
    prep, vec = resto.split(" + ")
    return {"modelo": modelo.strip(), "prep": prep.strip(), "vec": vec.strip()}


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Sentiment Analysis")
    st.caption("Amazon Kindle Reviews · NLP Pipeline")
    st.divider()

    for nombre in PAGES:
        activo = st.session_state.pagina == nombre
        label  = f"{ICONS[nombre]}  {nombre}"
        if st.button(
            label,
            key=f"nav_{nombre}",
            use_container_width=True,
            type="primary" if activo else "secondary",
        ):
            ir_a(nombre)
            st.rerun()

    st.divider()
    st.caption("💡 Sigue el orden de las secciones de arriba hacia abajo para explorar el proyecto completo.")


pagina = st.session_state.pagina


# ── Navegación inferior ────────────────────────────────────────────────────
def nav_buttons(actual: str):
    idx = PAGES.index(actual)
    st.divider()
    col_prev, col_next = st.columns(2)
    if idx > 0:
        prev = PAGES[idx - 1]
        if col_prev.button(f"← {ICONS[prev]} {prev}", use_container_width=True):
            ir_a(prev)
            st.rerun()
    if idx < len(PAGES) - 1:
        nxt = PAGES[idx + 1]
        if col_next.button(
            f"{ICONS[nxt]} {nxt} →",
            use_container_width=True,
            type="primary",
        ):
            ir_a(nxt)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# 🏠 INICIO
# ══════════════════════════════════════════════════════════════════════════
if pagina == "Inicio":
    st.title("📚 Sentiment Analysis — Kindle Reviews")
    st.markdown("#### Clasifica automáticamente si una reseña de libro es **positiva** o **negativa**")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📝 Reseñas analizadas",       "50,000")
    c2.metric("⚙️ Configuraciones evaluadas", "12")
    c3.metric("🏆 Mejor F1-score",            "0.9629")
    c4.metric("🤖 Mejor modelo",              "SVM + TF-IDF")

    st.divider()
    st.markdown("### 🗺️ ¿Por dónde empiezo?")

    cards = [
        ("Dataset",              "📊", "Mira de dónde vienen los datos y entiende el problema de desbalance de clases antes de entrenar."),
        ("Preprocesamiento",     "🔤", "Escribe cualquier texto y observa cómo el pipeline lo limpia, tokeniza y reduce paso a paso."),
        ("Resultados",           "📈", "Compara las 12 configuraciones entrenadas con sus métricas completas y matrices de confusión."),
        ("Analisis Comparativo", "🔍", "Lee las conclusiones: qué representación ganó, qué modelo funcionó mejor y por qué."),
        ("Demo",                 "🎯", "Escribe cualquier reseña en inglés y el modelo la clasifica al instante."),
    ]

    col1, col2, col3 = st.columns(3)
    col4, col5       = st.columns(2)
    cols_cards = [col1, col2, col3, col4, col5]

    for col, (nombre, icono, desc) in zip(cols_cards, cards):
        with col:
            with st.container(border=True):
                st.markdown(f"### {icono} {nombre}")
                st.markdown(desc)
                if st.button(f"Ir →", key=f"home_{nombre}", use_container_width=True, type="primary"):
                    ir_a(nombre)
                    st.rerun()

    st.divider()
    st.markdown("### ⚙️ Pipeline implementado")
    pipeline_df = pd.DataFrame({
        "Paso": [
            "1️⃣ Carga", "2️⃣ Limpieza", "3️⃣ Tokenización",
            "4️⃣ Stopwords", "5️⃣ Reducción", "6️⃣ Vectorización",
            "7️⃣ Clasificación", "8️⃣ Evaluación",
        ],
        "Descripción": [
            "982 K reseñas de Amazon Kindle (Kaggle)",
            "Minúsculas · sin puntuación · sin caracteres especiales",
            "NLTK word_tokenize — divide el texto en palabras",
            "Elimina palabras vacías (the, a, is, was…)",
            "Stemming (PorterStemmer) o Lemmatization (WordNetLemmatizer)",
            "Bag-of-Words o TF-IDF — convierte texto a números",
            "Naive Bayes · Logistic Regression · SVM",
            "Accuracy · Precision · Recall · F1 · Confusion Matrix",
        ],
    })
    st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

    nav_buttons("Inicio")


# ══════════════════════════════════════════════════════════════════════════
# 📊 DATASET
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "Dataset":
    st.title("📊 Dataset — Amazon Kindle Reviews")
    st.caption("Fuente: Kaggle · bharadwaj6/kindle-reviews · 982,619 reseñas originales")
    st.divider()

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("### 🏷️ Mapeo de sentimiento")
        st.markdown("""
        Las calificaciones originales (1–5 ⭐) se convirtieron a sentimiento binario:

        | Calificación | Etiqueta | Descripción |
        |---|---|---|
        | ⭐ 1 estrella | 🔴 Negativo (0) | El lector odió el libro |
        | ⭐⭐ 2 estrellas | 🔴 Negativo (0) | El lector no quedó satisfecho |
        | ⭐⭐⭐ 3 estrellas | ⚪ Descartado | Neutro / ambiguo |
        | ⭐⭐⭐⭐ 4 estrellas | 🟢 Positivo (1) | El lector quedó satisfecho |
        | ⭐⭐⭐⭐⭐ 5 estrellas | 🟢 Positivo (1) | El lector amó el libro |
        """)
    with col_b:
        st.metric("📁 Reseñas totales",   "982,619")
        st.metric("🎯 Usadas (muestra)",  "50,000")
        st.metric("📋 Columnas usadas",   "reviewText + overall")

    st.divider()
    st.markdown("### 📉 Distribución de ratings y sentimiento")
    img1 = os.path.join(OUTPUT_DIR, "01_dataset_distribution.png")
    if os.path.exists(img1):
        st.image(img1, use_container_width=True)
    else:
        st.warning("⚠️ Ejecuta `python3 main.py` para generar las gráficas.")

    st.markdown("### 📏 Longitud de las reseñas")
    img2 = os.path.join(OUTPUT_DIR, "02_review_lengths.png")
    if os.path.exists(img2):
        st.image(img2, use_container_width=True)

    st.divider()
    st.markdown("### ⚠️ Problema principal: Desbalance de clases")

    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Positivas", "93.4%", "clase mayoritaria")
    col2.metric("🔴 Negativas",  "6.6%", "-86.8 pp", delta_color="inverse")
    col3.metric("⚖️ Ratio",      "14 : 1", "positivas vs negativas", delta_color="off")

    st.error(
        "**¿Por qué importa?** Un modelo que siempre prediga 'Positivo' obtendría 93.4% de accuracy "
        "sin aprender nada útil. Por eso usamos **F1-score** como métrica principal en vez de accuracy."
    )

    nav_buttons("Dataset")


# ══════════════════════════════════════════════════════════════════════════
# 🔤 PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "Preprocesamiento":
    st.title("🔤 Pipeline de Preprocesamiento NLP")
    st.caption("Observa cómo el texto se transforma paso a paso antes de entrar al modelo")
    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### ✏️ Paso 1 — Escribe o pega un texto en inglés")
        ejemplo_default = "I absolutely loved this book! The characters were beautifully written and the plot was amazing. Couldn't put it down."
        texto_input = st.text_area(
            "Texto:",
            value=ejemplo_default,
            height=100,
            label_visibility="collapsed",
        )

        st.markdown("#### ⚙️ Paso 2 — Elige el método de reducción")
        metodo = st.radio(
            "Método:",
            ["stem", "lemma"],
            format_func=lambda x: (
                "🌿 Stemming — recorta sufijos (más rápido)"
                if x == "stem"
                else "📖 Lemmatization — forma canónica del diccionario (más preciso)"
            ),
            horizontal=False,
        )

        procesar = st.button("▶️ Procesar texto", type="primary", use_container_width=True)

    with col_right:
        st.markdown("#### 🆚 Comparación Stemming vs Lemmatization")
        comp = pd.DataFrame({
            "Aspecto":       ["Velocidad", "Tokens válidos", "Agresividad", "'loved'", "'running'", "'specifically'"],
            "🌿 Stemming":   ["Más rápido", "No garantizado", "Alta",  "love",  "run",     "specif"],
            "📖 Lemmatization": ["Más lento", "Siempre válido", "Baja", "loved", "running", "specifically"],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.caption("En este proyecto la diferencia de F1 entre ambos métodos es de solo +0.0005.")

    if procesar:
        STOPWORDS_SET = set(stopwords.words("english"))
        st.divider()
        st.markdown("#### 🔄 Transformación paso a paso")

        limpio = texto_input.lower()
        limpio = re.sub(r"http\S+|www\S+", " ", limpio)
        limpio = re.sub(r"[^a-z\s]", " ", limpio)
        limpio = re.sub(r"\s+", " ", limpio).strip()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📝 Original**")
            st.code(texto_input, language=None)
        with c2:
            st.markdown("**1️⃣ Limpieza** *(lowercase + sin puntuación)*")
            st.code(limpio, language=None)

        tokens = word_tokenize(limpio)
        st.markdown("**2️⃣ Tokenización** *(separar en palabras individuales)*")
        st.code(str(tokens), language=None)

        tokens_ns  = [t for t in tokens if t not in STOPWORDS_SET and len(t) > 1]
        eliminadas = [t for t in tokens if t in STOPWORDS_SET]
        st.markdown(f"**3️⃣ Eliminación de stopwords** *(removidas: `{eliminadas}`)*")
        st.code(str(tokens_ns), language=None)

        if metodo == "stem":
            stemmer = PorterStemmer()
            final = [stemmer.stem(t) for t in tokens_ns]
            st.markdown("**4️⃣ 🌿 Stemming** *(recorta sufijos)*")
        else:
            lemmatizer = WordNetLemmatizer()
            final = [lemmatizer.lemmatize(t) for t in tokens_ns]
            st.markdown("**4️⃣ 📖 Lemmatization** *(forma canónica del diccionario)*")
        st.code(str(final), language=None)

        st.markdown("**✅ Texto final que entra al vectorizador**")
        st.success(" ".join(final))

    nav_buttons("Preprocesamiento")


# ══════════════════════════════════════════════════════════════════════════
# 📈 RESULTADOS
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "Resultados":
    st.title("📈 Resultados de los 12 Modelos")
    st.caption("3 modelos × 2 preprocesadores × 2 vectorizadores = 12 configuraciones evaluadas")

    if not os.path.exists(CSV_PATH):
        st.error("⚠️ Ejecuta `python3 main.py` primero para generar los resultados.")
        st.stop()

    df = load_results()
    parsed       = df["label"].apply(parse_label)
    df["modelo"] = parsed.apply(lambda x: x["modelo"])
    df["prep"]   = parsed.apply(lambda x: x["prep"])
    df["vec"]    = parsed.apply(lambda x: x["vec"])
    best_row     = df.loc[df["f1"].idxmax()]

    st.divider()
    st.markdown("### 🏆 Mejor configuración encontrada")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🤖 Modelo",           best_row["label"].split(" / ")[0])
    c2.metric("🧮 Vectorizador",     best_row["vec"])
    c3.metric("🔤 Preprocesamiento", best_row["prep"])
    c4.metric("🎯 F1-score",         f"{best_row['f1']:.4f}")
    c5.metric("✅ Accuracy",         f"{best_row['accuracy']:.4f}")

    st.divider()
    st.markdown("### 📋 Tabla completa — ordenada por F1")
    st.caption("🟢 = mejor valor en esa métrica  ·  🔴 = peor valor")

    display = df[["label","accuracy","precision","recall","f1"]].sort_values("f1", ascending=False).copy()
    display.columns = ["Configuración","Accuracy","Precision","Recall","F1"]
    st.dataframe(
        display.style
            .highlight_max(subset=["Accuracy","Precision","Recall","F1"], color="#d4edda")
            .highlight_min(subset=["Accuracy","Precision","Recall","F1"], color="#f8d7da")
            .format({"Accuracy":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}","F1":"{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("### 📊 Gráfica comparativa")
    img_comp = os.path.join(OUTPUT_DIR, "model_comparison.png")
    if os.path.exists(img_comp):
        st.image(img_comp, use_container_width=True)

    st.divider()
    st.markdown("### 🔲 Explorar matriz de confusión")
    st.caption("Elige una configuración para ver cómo clasifica positivos y negativos:")

    selected = st.selectbox("Configuración:", df["label"].tolist(), label_visibility="collapsed")
    safe     = selected.lower().replace(" ","_").replace("/","_")
    cm_path  = os.path.join(OUTPUT_DIR, f"cm_{safe}.png")

    if os.path.exists(cm_path):
        col_img, col_exp = st.columns([1, 1])
        col_img.image(cm_path)
        row = df[df["label"] == selected].iloc[0]
        with col_exp:
            st.markdown("**🗺️ Cómo leer la matriz**")
            st.markdown("""
            - **Fila** = sentimiento real de la reseña
            - **Columna** = lo que predijo el modelo
            - **Diagonal principal** = predicciones correctas ✅
            - **Fuera de diagonal** = errores ❌

            Un valor alto abajo-izquierda (Negativo real → predijo Positivo)
            indica que el modelo falla detectando reseñas negativas.
            """)
            st.markdown("**📊 Métricas de esta configuración**")
            st.markdown(f"""
            | Métrica | Valor | Qué mide |
            |---------|-------|----------|
            | Accuracy  | `{row['accuracy']:.4f}` | % total de aciertos |
            | Precision | `{row['precision']:.4f}` | De los que predijo positivo, cuántos lo eran realmente |
            | Recall    | `{row['recall']:.4f}` | De todos los positivos reales, cuántos encontró |
            | F1-score  | `{row['f1']:.4f}` | Promedio armónico de Precision y Recall |
            """)
    else:
        st.warning(f"Archivo no encontrado: {cm_path}")

    nav_buttons("Resultados")


# ══════════════════════════════════════════════════════════════════════════
# 🔍 ANALISIS COMPARATIVO
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "Analisis Comparativo":
    st.title("🔍 Análisis Comparativo")
    st.caption("Conclusiones del proyecto: qué combinación ganó, por qué, y qué errores comete el modelo")

    if not os.path.exists(CSV_PATH):
        st.error("⚠️ Ejecuta `python3 main.py` primero.")
        st.stop()

    df = load_results()
    parsed       = df["label"].apply(parse_label)
    df["modelo"] = parsed.apply(lambda x: x["modelo"])
    df["prep"]   = parsed.apply(lambda x: x["prep"])
    df["vec"]    = parsed.apply(lambda x: x["vec"])

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧮 BoW vs TF-IDF",
        "🔤 Stemming vs Lemmatization",
        "🤖 Comparación de Modelos",
        "❌ Errores de Clasificación",
    ])

    with tab1:
        st.subheader("🧮 ¿Qué representación vectorial es mejor?")
        bow_f1   = df[df["vec"] == "BoW"]["f1"].mean()
        tfidf_f1 = df[df["vec"] == "TF-IDF"]["f1"].mean()
        c1, c2 = st.columns(2)
        c1.metric("F1 promedio — BoW",    f"{bow_f1:.4f}")
        c2.metric("F1 promedio — TF-IDF", f"{tfidf_f1:.4f}", delta=f"{tfidf_f1-bow_f1:+.4f}")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        df.groupby(["modelo","vec"])["f1"].mean().unstack("vec").plot(
            kind="bar", ax=ax, color=["#4c72b0","#dd8452"], edgecolor="black"
        )
        ax.set_title("F1 promedio: BoW vs TF-IDF por modelo")
        ax.set_ylabel("F1-score"); ax.set_ylim(0.88, 0.98)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Vectorizador"); plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""
        **📌 Conclusión:**
        - **BoW gana en promedio general** — pero solo porque Naive Bayes colapsa con TF-IDF (F1=0.918).
        - **TF-IDF gana con SVM** (el mejor modelo), porque sus valores ponderados crean hiperplanos de separación más claros.
        - TF-IDF reduce el peso de palabras frecuentes como *"book"* o *"read"* que no aportan información sobre el sentimiento, y amplifica las que sí importan como *"disappointed"* o *"excellent"*.
        """)

    with tab2:
        st.subheader("🔤 ¿Stemming o Lemmatization?")
        sm = df[df["prep"]=="Stem"]["f1"].mean()
        lm = df[df["prep"]=="Lemma"]["f1"].mean()
        c1, c2 = st.columns(2)
        c1.metric("F1 promedio — 🌿 Stemming",      f"{sm:.4f}")
        c2.metric("F1 promedio — 📖 Lemmatization",  f"{lm:.4f}", delta=f"{lm-sm:+.4f}")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        df.groupby(["modelo","prep"])["f1"].mean().unstack("prep").plot(
            kind="bar", ax=ax, color=["#55a868","#c44e52"], edgecolor="black"
        )
        ax.set_title("F1 promedio: Stemming vs Lemmatization por modelo")
        ax.set_ylabel("F1-score"); ax.set_ylim(0.88, 0.98)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Preprocesamiento"); plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown(f"""
        **📌 Conclusión:**
        La diferencia es mínima (**{lm-sm:+.4f}** en F1). Los dos métodos son equivalentes en rendimiento.

        | Aspecto | 🌿 Stemming | 📖 Lemmatization |
        |---------|----------|---------------|
        | Velocidad | Más rápido | Más lento |
        | Output | Puede no ser palabra real (`specif`, `becam`) | Siempre palabra válida (`specific`, `became`) |
        | Uso recomendado | Cuando la velocidad importa | Cuando importa interpretar los tokens |

        **Recomendación:** Lemmatization, por la legibilidad de los features.
        """)

    with tab3:
        st.subheader("🤖 ¿Qué modelo funciona mejor?")
        stats = df.groupby("modelo")["f1"].agg(["mean","max","min"]).sort_values("mean", ascending=False)
        c1, c2, c3 = st.columns(3)
        emojis_mod = {"SVM": "🥇", "Logistic Regression": "🥈", "Naive Bayes": "🥉"}
        for col, (mod, row) in zip([c1,c2,c3], stats.iterrows()):
            col.metric(f"{emojis_mod.get(mod,'')} {mod}", f"F1 max: {row['max']:.4f}", f"Promedio: {row['mean']:.4f}")

        hm = os.path.join(OUTPUT_DIR, "analisis_comparativo_heatmap.png")
        if os.path.exists(hm):
            st.image(hm, use_container_width=True)
        else:
            pivot = df.pivot_table(index="modelo", columns=["prep","vec"], values="f1")
            pivot.columns = [f"{p}+{v}" for p,v in pivot.columns]
            fig, ax = plt.subplots(figsize=(10,4))
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGn",
                        vmin=0.91, vmax=0.97, linewidths=0.5, ax=ax)
            ax.set_title("F1-score: Modelo × Configuración")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown(f"""
        **📌 Conclusiones:**

        🥇 **SVM — Mejor modelo** (F1 max: {stats.loc['SVM','max']:.4f})
        Encuentra el hiperplano de máximo margen en 20,000 dimensiones. Con TF-IDF las clases son más separables linealmente.

        🥈 **Logistic Regression — Muy competitivo** (F1 max: {stats.loc['Logistic Regression','max']:.4f})
        Entrena en menos de un segundo y rinde casi igual que SVM. Con BoW bigrams captura frases como *"not good"* que invierten el sentimiento.

        🥉 **Naive Bayes — Peor con TF-IDF** (F1 mín: {stats.loc['Naive Bayes','min']:.4f})
        Asume independencia total entre palabras e incompatibilidad con valores continuos de TF-IDF. El recall de negativos cae a 0.11.
        """)

    with tab4:
        st.subheader("❌ ¿Qué errores comete el mejor modelo?")
        st.markdown("Predicciones incorrectas del modelo **SVM / Lemma + TF-IDF:**")

        errores = [
            ("🌀 Sentimiento mixto",
             "La reseña critica el libro pero reconoce algo positivo. El modelo detecta el lenguaje positivo e ignora el contexto negativo general.",
             '"While the plotting was much better in this one, I still felt the character development was lacking."',
             "Negativo", "Positivo"),
            ("😏 Sarcasmo o ironía",
             "Palabras positivas en tono irónico. TF-IDF no distingue si 'glad' es genuino o sarcástico.",
             "\"I'm just glad this hot ghetto mess was free.\"",
             "Negativo", "Positivo"),
            ("📦 Queja fuera de contexto",
             "La crítica es al servicio de Amazon, no al libro. No hay vocabulario negativo literario.",
             '"I have not received my fourteen free copies of the NYT so please end my subscription."',
             "Negativo", "Positivo"),
            ("💬 Positivo con lenguaje neutro",
             "Reseña de 5 estrellas muy objetiva. Sin palabras como 'amazing' o 'loved', el modelo no lo detecta como positivo.",
             '"The book covers the topic thoroughly. Each chapter builds on the previous one."',
             "Positivo", "Negativo"),
        ]

        for tipo, desc, ej, verd, pred in errores:
            with st.expander(tipo):
                st.markdown(desc)
                st.code(ej, language=None)
                col1, col2 = st.columns(2)
                col1.success(f"✅ Real: **{verd}**")
                col2.error(f"❌ Predijo: **{pred}**")

        st.info(
            "💡 **Limitación de BoW/TF-IDF:** no capturan orden de palabras, negaciones compuestas "
            "('not at all bad'), ni figuras retóricas. Un modelo BERT o RoBERTa reduciría estos errores significativamente."
        )

    nav_buttons("Analisis Comparativo")


# ══════════════════════════════════════════════════════════════════════════
# 🎯 DEMO
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "Demo":
    st.title("🎯 Demo — Clasifica una reseña en tiempo real")
    st.caption("Escribe o elige un ejemplo y el modelo te dice si la reseña es positiva o negativa")

    model_exists = (
        os.path.exists(os.path.join(MODEL_DIR, "best_model.pkl")) and
        os.path.exists(os.path.join(MODEL_DIR, "best_vectorizer.pkl"))
    )
    if not model_exists:
        st.error("⚠️ El modelo no está guardado. Ejecuta `python3 main.py` primero.")
        st.stop()

    model, vectorizer, config = load_model()
    prep_method = config.get("prep", "lemma")

    with st.container(border=True):
        st.caption(f"🤖 Modelo activo: **{config.get('label','?')}**  ·  F1 = {config.get('f1','?')}")

    st.divider()
    st.markdown("### 💡 Elige un ejemplo rápido")

    ejemplos = {
        "⭐ Positivo claro":      "This book was absolutely fantastic! I couldn't put it down. The characters were so well developed and the story was gripping from start to finish. One of the best books I've read this year.",
        "💔 Negativo claro":      "Terrible book. Poorly written, boring plot, and completely unoriginal. I wasted my money and my time. I couldn't even finish it. Avoid at all costs.",
        "📚 No ficción positiva": "A brilliantly researched and well-argued book. The author brings together decades of scientific evidence in a way that is both accessible and compelling. I learned something new on every page.",
        "😴 Aburrido / negativo": "I struggled to get through this book. The pacing was painfully slow and the author repeated the same ideas over and over. By chapter five I had completely lost interest.",
        "🌀 Sentimiento mixto":   "The writing style is beautiful and the author clearly knows the topic, but the story felt slow and I lost interest halfway through. Good ideas, poor execution.",
        "😏 Sarcasmo":            "Oh great, another predictable romance novel with a billionaire CEO. Just what the world needed. Absolutely riveting — if you enjoy reading the same book for the hundredth time.",
        "📦 Queja de envío":      "I never received my order. Three weeks have passed and the book still hasn't arrived. This is a logistics problem, not a book review. One star for the service.",
        "💬 Reseña muy corta":    "Good book. Enjoyed it.",
        "🔥 Fan entusiasta":      "I devoured this book in one sitting! The author has an incredible gift for storytelling. Every page kept me on the edge of my seat. I immediately bought the sequel.",
        "🤔 Difícil de clasificar": "The book was exactly what I expected. It covered the main topics and had some interesting sections. Nothing groundbreaking but it gets the job done.",
    }

    # Fila 1: 5 ejemplos
    row1 = st.columns(5)
    for col, (nombre, txt) in zip(row1, list(ejemplos.items())[:5]):
        if col.button(nombre, use_container_width=True, key=f"ej_{nombre}"):
            st.session_state.ejemplo_texto = txt
            st.rerun()

    # Fila 2: 5 ejemplos
    row2 = st.columns(5)
    for col, (nombre, txt) in zip(row2, list(ejemplos.items())[5:]):
        if col.button(nombre, use_container_width=True, key=f"ej_{nombre}"):
            st.session_state.ejemplo_texto = txt
            st.rerun()

    st.divider()
    st.markdown("### ✏️ Escribe o edita tu reseña en inglés")
    texto = st.text_area(
        "Reseña:",
        value=st.session_state.ejemplo_texto,
        height=130,
        placeholder="Write a book review in English...",
        label_visibility="collapsed",
    )

    st.markdown("### ▶️ Clasifica")
    if st.button("Clasificar sentimiento", type="primary", use_container_width=True):
        if not texto.strip():
            st.warning("⚠️ Escribe algo primero.")
        else:
            prep   = NLPPreprocessor(method=prep_method)
            proc   = prep.process_single(texto)
            vector = vectorizer.transform([proc])
            pred   = model.predict(vector)[0]
            proba  = model.predict_proba(vector)[0]

            st.divider()

            if pred == 1:
                st.success("## ✅ Sentimiento: POSITIVO")
            else:
                st.error("## ❌ Sentimiento: NEGATIVO")

            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 Confianza positivo", f"{proba[1]*100:.1f}%")
            c2.metric("🔴 Confianza negativo", f"{proba[0]*100:.1f}%")
            c3.metric("🎯 Resultado", "Positivo ✅" if pred == 1 else "Negativo ❌")

            fig, ax = plt.subplots(figsize=(6, 1))
            ax.barh([""], [proba[1]], color="#28a745", label="Positivo")
            ax.barh([""], [proba[0]], left=[proba[1]], color="#dc3545", label="Negativo")
            ax.set_xlim(0, 1)
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.set_xticklabels(["0%","25%","50%","75%","100%"])
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title("Distribución de probabilidad")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            with st.expander("🔍 Ver texto tras el preprocesamiento"):
                st.code(proc, language=None)

    nav_buttons("Demo")
