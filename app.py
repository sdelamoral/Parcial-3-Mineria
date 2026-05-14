"""
Sentiment Analysis Dashboard — Kindle Reviews
Run: venv/bin/streamlit run app.py
"""

import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import nltk
for _r in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(_r, quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from src.preprocessor import NLPPreprocessor

st.set_page_config(
    page_title="Sentiment Analysis — Kindle Reviews",
    page_icon="📚",
    layout="wide",
)


OUTPUT_DIR = "outputs"
MODEL_DIR  = "model"
CSV_PATH   = os.path.join(OUTPUT_DIR, "results_summary.csv")

PAGES = ["Inicio", "Dataset", "Preprocesamiento", "Resultados", "Analisis Comparativo", "Demo"]

ICONS = {
    "Inicio":               ":material/home:",
    "Dataset":              ":material/bar_chart:",
    "Preprocesamiento":     ":material/text_fields:",
    "Resultados":           ":material/trending_up:",
    "Analisis Comparativo": ":material/manage_search:",
    "Demo":                 ":material/play_circle:",
}

if "pagina" not in st.session_state:
    st.session_state.pagina = "Inicio"
if "ejemplo_texto" not in st.session_state:
    st.session_state.ejemplo_texto = ""

def ir_a(p): st.session_state.pagina = p

@st.cache_data
def load_results():
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

@st.cache_resource
def get_nlp_resources():
    return {
        "stopwords": set(stopwords.words("english")),
        "stemmer":   PorterStemmer(),
        "lemma":     WordNetLemmatizer(),
    }

def parse_label(label):
    modelo, resto = label.split(" / ")
    prep, vec = resto.split(" + ")
    return {"modelo": modelo.strip(), "prep": prep.strip(), "vec": vec.strip()}

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Sentiment Analysis")
    st.caption("Kindle Reviews · NLP Pipeline")
    st.write("")
    for nombre in PAGES:
        activo = st.session_state.pagina == nombre
        if st.button(nombre, key=f"nav_{nombre}", icon=ICONS[nombre],
                     use_container_width=True,
                     type="primary" if activo else "secondary"):
            ir_a(nombre)
            st.rerun()

pagina = st.session_state.pagina

def sig_btn(actual):
    idx = PAGES.index(actual)
    if idx < len(PAGES) - 1:
        nxt = PAGES[idx + 1]
        st.write("")
        if st.button(f"Siguiente: {nxt}", icon=":material/arrow_forward:",
                     type="primary"):
            ir_a(nxt)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# INICIO
# ══════════════════════════════════════════════════════════════════════════
if pagina == "Inicio":
    st.title("Sentiment Analysis sobre resenas de Kindle")

    st.markdown("""
    Este proyecto entrena un sistema capaz de leer resenas de libros en ingles
    y determinar automaticamente si el sentimiento expresado es **positivo** o **negativo**.

    Se evaluaron **12 configuraciones** combinando tres modelos (Naive Bayes, Logistic Regression, SVM),
    dos tecnicas de preprocesamiento (Stemming y Lemmatization) y dos representaciones vectoriales
    (Bag-of-Words y TF-IDF). La mejor combinacion fue **SVM + Lemmatization + TF-IDF**
    con un F1-score de **0.9629** sobre 50,000 resenas del dataset de Amazon Kindle (Kaggle).
    """)

    st.write("")
    st.markdown("**Que hay en cada seccion:**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Dataset** — De donde vienen los datos, como se etiquetaron
        los sentimientos y cual es el problema principal del corpus.

        **Preprocesamiento** — Escribe cualquier texto y observa como
        el pipeline lo transforma paso a paso antes de llegar al modelo.

        **Resultados** — Tabla con las metricas de las 12 configuraciones
        y matrices de confusion para explorar cada una.
        """)
    with col2:
        st.markdown("""
        **Analisis Comparativo** — Conclusions sobre que representacion
        gano, que modelo funciono mejor, que problemas tiene el dataset
        y que tipos de errores comete el clasificador.

        **Demo** — Clasifica cualquier resena en ingles en tiempo real
        usando el mejor modelo entrenado.
        """)

    sig_btn("Inicio")


# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "Dataset":
    st.title("Dataset")

    st.markdown("""
    El dataset proviene de [Amazon Kindle Reviews (Kaggle)](https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews)
    y contiene 982,619 resenas originales. Para este proyecto se trabajo con una muestra
    de **50,000 resenas** usando solo las columnas `reviewText` y `overall`.

    Las calificaciones se convirtieron a sentimiento binario: ratings 1-2 se mapearon
    a **Negativo (0)**, ratings 4-5 a **Positivo (1)**, y los ratings 3 se descartaron
    por ser ambiguos.
    """)

    img1 = os.path.join(OUTPUT_DIR, "01_dataset_distribution.png")
    if os.path.exists(img1):
        st.image(img1, use_container_width=True)

    img2 = os.path.join(OUTPUT_DIR, "02_review_lengths.png")
    if os.path.exists(img2):
        st.image(img2, use_container_width=True)

    st.markdown("#### Desbalance de clases")
    st.markdown("""
    El problema mas importante del dataset es el desbalance severo: el **93.4%** de las
    resenas son positivas y solo el **6.6%** son negativas (ratio 14:1). Esto significa
    que un modelo que siempre prediga "Positivo" alcanzaria 93.4% de accuracy sin
    aprender nada util sobre el sentimiento. Por eso se uso **F1-score** como metrica
    principal en lugar de accuracy.
    """)

    sig_btn("Dataset")


# ══════════════════════════════════════════════════════════════════════════
# PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "Preprocesamiento":
    st.title("Preprocesamiento NLP")

    st.markdown("""
    Antes de entrenar cualquier modelo, el texto pasa por un pipeline de limpieza:
    se convierte a minusculas, se eliminan puntuacion y caracteres especiales,
    se tokeniza, se quitan las stopwords y finalmente se aplica Stemming o Lemmatization.
    """)

    st.write("")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        ejemplo_default = "I absolutely loved this book! The characters were beautifully written and the plot was amazing. Couldn't put it down."
        texto_input = st.text_area("Texto de prueba:", value=ejemplo_default, height=110)

        metodo = st.radio(
            "Metodo de reduccion:",
            ["stem", "lemma"],
            format_func=lambda x: "Stemming (PorterStemmer)" if x == "stem" else "Lemmatization (WordNetLemmatizer)",
            horizontal=True,
        )
        procesar = st.button("Procesar", icon=":material/play_arrow:", type="primary")

    with col_right:
        st.markdown("**Stemming vs Lemmatization**")
        st.dataframe(pd.DataFrame({
            "":              ["Velocidad", "Output", "'loved'", "'running'", "'specifically'"],
            "Stemming":      ["Rapido",    "Puede no ser palabra real", "love", "run", "specif"],
            "Lemmatization": ["Lento",     "Siempre palabra valida",    "loved","running","specifically"],
        }), hide_index=True, use_container_width=True)
        st.caption("La diferencia en F1 entre ambos metodos fue de solo 0.0005 en este proyecto.")

    if procesar:
        nlp = get_nlp_resources()
        st.write("")

        limpio = re.sub(r"http\S+|www\S+", " ", texto_input.lower())
        limpio = re.sub(r"[^a-z\s]", " ", limpio)
        limpio = re.sub(r"\s+", " ", limpio).strip()

        c1, c2 = st.columns(2)
        c1.markdown("**Original**"); c1.code(texto_input, language=None)
        c2.markdown("**Limpieza**"); c2.code(limpio, language=None)

        tokens = word_tokenize(limpio)
        st.markdown("**Tokenizacion**"); st.code(str(tokens), language=None)

        tokens_ns = [t for t in tokens if t not in nlp["stopwords"] and len(t) > 1]
        eliminadas = [t for t in tokens if t in nlp["stopwords"]]
        st.markdown(f"**Sin stopwords** (eliminadas: `{eliminadas}`)"); st.code(str(tokens_ns), language=None)

        if metodo == "stem":
            final = [nlp["stemmer"].stem(t) for t in tokens_ns]
            st.markdown("**Stemming**")
        else:
            final = [nlp["lemma"].lemmatize(t) for t in tokens_ns]
            st.markdown("**Lemmatization**")
        st.code(str(final), language=None)

        st.markdown("**Texto final al vectorizador:**")
        st.success(" ".join(final), icon=":material/check_circle:")

    sig_btn("Preprocesamiento")


# ══════════════════════════════════════════════════════════════════════════
# RESULTADOS
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "Resultados":
    st.title("Resultados")

    if not os.path.exists(CSV_PATH):
        st.error("Ejecuta python3 main.py primero.", icon=":material/error:")
        st.stop()

    df = load_results()
    parsed       = df["label"].apply(parse_label)
    df["modelo"] = parsed.apply(lambda x: x["modelo"])
    df["prep"]   = parsed.apply(lambda x: x["prep"])
    df["vec"]    = parsed.apply(lambda x: x["vec"])
    best_row     = df.loc[df["f1"].idxmax()]

    st.markdown(f"""
    Se evaluaron 12 configuraciones: 3 modelos × 2 preprocesadores × 2 vectorizadores.
    La mejor fue **{best_row['label']}** con F1 de `{best_row['f1']:.4f}`
    y accuracy de `{best_row['accuracy']:.4f}`.
    """)

    display = df[["label","accuracy","precision","recall","f1"]].sort_values("f1", ascending=False).copy()
    display.columns = ["Configuracion","Accuracy","Precision","Recall","F1"]
    st.dataframe(
        display.style
            .highlight_max(subset=["Accuracy","Precision","Recall","F1"], color="#d4edda")
            .highlight_min(subset=["Accuracy","Precision","Recall","F1"], color="#f8d7da")
            .format({"Accuracy":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}","F1":"{:.4f}"}),
        use_container_width=True, hide_index=True,
    )

    img_comp = os.path.join(OUTPUT_DIR, "model_comparison.png")
    if os.path.exists(img_comp):
        st.image(img_comp, use_container_width=True)

    st.write("")
    st.markdown("#### Matriz de confusion")
    st.caption("Selecciona una configuracion para ver como clasifica cada clase.")
    selected = st.selectbox("", df["label"].tolist(), label_visibility="collapsed")
    safe     = selected.lower().replace(" ","_").replace("/","_")
    cm_path  = os.path.join(OUTPUT_DIR, f"cm_{safe}.png")

    if os.path.exists(cm_path):
        col_img, col_txt = st.columns([1, 1])
        col_img.image(cm_path)
        row = df[df["label"] == selected].iloc[0]
        with col_txt:
            st.markdown("La diagonal principal son las predicciones correctas. "
                        "Un valor alto en la esquina inferior izquierda indica que el modelo "
                        "falla detectando resenas negativas, lo que es esperable dado el desbalance del dataset.")
            st.write("")
            st.dataframe(pd.DataFrame({
                "Metrica":  ["Accuracy","Precision","Recall","F1-score"],
                "Valor":    [f"{row['accuracy']:.4f}", f"{row['precision']:.4f}",
                             f"{row['recall']:.4f}",   f"{row['f1']:.4f}"],
            }), hide_index=True, use_container_width=True)

    sig_btn("Resultados")


# ══════════════════════════════════════════════════════════════════════════
# ANALISIS COMPARATIVO
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "Analisis Comparativo":
    st.title("Analisis Comparativo")

    if not os.path.exists(CSV_PATH):
        st.error("Ejecuta python3 main.py primero.", icon=":material/error:")
        st.stop()

    df = load_results()
    parsed       = df["label"].apply(parse_label)
    df["modelo"] = parsed.apply(lambda x: x["modelo"])
    df["prep"]   = parsed.apply(lambda x: x["prep"])
    df["vec"]    = parsed.apply(lambda x: x["vec"])

    tab1, tab2, tab3, tab4 = st.tabs([
        "BoW vs TF-IDF",
        "Stemming vs Lemmatization",
        "Comparacion de modelos",
        "Errores de clasificacion",
    ])

    with tab1:
        bow_f1   = df[df["vec"] == "BoW"]["f1"].mean()
        tfidf_f1 = df[df["vec"] == "TF-IDF"]["f1"].mean()

        fig, ax = plt.subplots(figsize=(7, 3.5))
        df.groupby(["modelo","vec"])["f1"].mean().unstack("vec").plot(
            kind="bar", ax=ax, color=["#4c72b0","#dd8452"], edgecolor="white", width=0.6
        )
        ax.set_title("F1 promedio: BoW vs TF-IDF por modelo", pad=12)
        ax.set_ylabel("F1-score"); ax.set_ylim(0.88, 0.98)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(title="Vectorizador", frameon=False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown(f"""
        En promedio, **BoW obtuvo F1 = {bow_f1:.4f}** y **TF-IDF obtuvo F1 = {tfidf_f1:.4f}**.
        BoW gana en el promedio general, pero eso se debe a que Naive Bayes colapsa con TF-IDF
        (F1 = 0.918) porque MultinomialNB espera frecuencias enteras y TF-IDF produce valores
        continuos que violan ese supuesto.

        Si se comparan solo SVM y Logistic Regression, TF-IDF es superior: penaliza palabras
        muy comunes como *"book"* o *"read"* que no aportan informacion sobre el sentimiento,
        y le da mas peso a palabras discriminantes como *"disappointed"* o *"excellent"*.
        """)

    with tab2:
        sm = df[df["prep"]=="Stem"]["f1"].mean()
        lm = df[df["prep"]=="Lemma"]["f1"].mean()

        fig, ax = plt.subplots(figsize=(7, 3.5))
        df.groupby(["modelo","prep"])["f1"].mean().unstack("prep").plot(
            kind="bar", ax=ax, color=["#55a868","#c44e52"], edgecolor="white", width=0.6
        )
        ax.set_title("F1 promedio: Stemming vs Lemmatization por modelo", pad=12)
        ax.set_ylabel("F1-score"); ax.set_ylim(0.88, 0.98)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(title="Preprocesamiento", frameon=False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown(f"""
        La diferencia entre Stemming (F1 = {sm:.4f}) y Lemmatization (F1 = {lm:.4f})
        es practicamente insignificante ({lm-sm:+.4f}). Para este corpus ambos metodos
        producen vocabularios igualmente utiles para los clasificadores.

        La diferencia cualitativa es que Stemming recorta sufijos agresivamente y puede
        producir tokens que no son palabras reales (*"specif"*, *"becam"*), mientras que
        Lemmatization siempre devuelve la forma canonica del diccionario (*"specific"*, *"became"*).
        Lemmatization es preferible si se necesita interpretar cuales palabras pesan mas en el modelo.
        """)

    with tab3:
        stats = df.groupby("modelo")["f1"].agg(["mean","max","min"]).sort_values("mean", ascending=False)

        hm = os.path.join(OUTPUT_DIR, "analisis_comparativo_heatmap.png")
        if os.path.exists(hm):
            st.image(hm, use_container_width=True)
        else:
            pivot = df.pivot_table(index="modelo", columns=["prep","vec"], values="f1")
            pivot.columns = [f"{p}+{v}" for p,v in pivot.columns]
            fig, ax = plt.subplots(figsize=(10,4))
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGn",
                        vmin=0.91, vmax=0.97, linewidths=0.5, ax=ax)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown(f"""
        **SVM** es el mejor modelo con F1 maximo de {stats.loc['SVM','max']:.4f}.
        Funciona bien en espacios de alta dimension como el que generan BoW y TF-IDF
        (20,000 features), donde los textos son linealmente separables.

        **Logistic Regression** queda muy cerca con F1 de {stats.loc['Logistic Regression','max']:.4f}
        y es mucho mas rapido en inferencia. Es una alternativa viable si el tiempo de respuesta importa.

        **Naive Bayes** rinde bien con BoW pero es el peor con TF-IDF porque asume que los
        features son frecuencias enteras no negativas e independientes entre si, supuestos que
        TF-IDF viola. Su F1 minimo es {stats.loc['Naive Bayes','min']:.4f}.
        """)

    with tab4:
        st.markdown("""
        Analizando las predicciones incorrectas del mejor modelo (SVM / Lemma + TF-IDF)
        se identificaron cuatro patrones principales:
        """)

        errores = [
            ("Sentimiento mixto",
             "La resena critica el libro pero reconoce aspectos positivos. El modelo detecta el lenguaje positivo e ignora el contexto negativo general.",
             '"While the plotting was much better in this one, I still felt the character development was lacking."',
             "Negativo", "Positivo"),
            ("Sarcasmo o ironia",
             "Palabras con carga positiva usadas en tono ironico. TF-IDF no distingue si 'glad' es genuino o sarcastico.",
             "\"I'm just glad this hot ghetto mess was free.\"",
             "Negativo", "Positivo"),
            ("Queja fuera de contexto",
             "La critica es al servicio de Amazon, no al libro. No hay vocabulario negativo literario que el modelo pueda detectar.",
             '"I have not received my fourteen free copies of the NYT so please end my subscription."',
             "Negativo", "Positivo"),
            ("Positivo con lenguaje neutro",
             "Resena de 5 estrellas escrita de forma objetiva y descriptiva, sin palabras claramente positivas.",
             '"The book covers the topic thoroughly. Each chapter builds on the previous one."',
             "Positivo", "Negativo"),
        ]

        for tipo, desc, ej, verd, pred in errores:
            with st.expander(tipo):
                st.markdown(desc)
                st.code(ej, language=None)
                c1, c2 = st.columns(2)
                c1.success(f"Real: {verd}", icon=":material/check:")
                c2.error(f"Predijo: {pred}", icon=":material/close:")

        st.info(
            "BoW y TF-IDF no capturan orden de palabras, negaciones compuestas ni figuras "
            "retoricas. Un modelo basado en embeddings contextuales como BERT reduciria "
            "significativamente estos errores.",
            icon=":material/info:",
        )

    sig_btn("Analisis Comparativo")


# ══════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "Demo":
    st.title("Demo")

    model_exists = (
        os.path.exists(os.path.join(MODEL_DIR, "best_model.pkl")) and
        os.path.exists(os.path.join(MODEL_DIR, "best_vectorizer.pkl"))
    )
    if not model_exists:
        st.error("Ejecuta python3 main.py primero.", icon=":material/error:")
        st.stop()

    model, vectorizer, config = load_model()
    prep_method = config.get("prep", "lemma")
    st.caption(f"Modelo: {config.get('label','?')}  ·  F1 = {config.get('f1','?')}")
    st.write("")

    ejemplos = {
        "Positivo claro":        "This book was absolutely fantastic! I couldn't put it down. The characters were so well developed and the story was gripping from start to finish.",
        "Negativo claro":        "Terrible book. Poorly written, boring plot, and completely unoriginal. I wasted my money and my time. I couldn't even finish it.",
        "No ficcion positiva":   "A brilliantly researched book. The author brings together decades of scientific evidence in a way that is both accessible and compelling.",
        "Aburrido / negativo":   "I struggled to get through this book. The pacing was painfully slow and the author repeated the same ideas over and over.",
        "Sentimiento mixto":     "The writing style is beautiful and the author clearly knows the topic, but the story felt slow and I lost interest halfway through.",
        "Sarcasmo":              "Oh great, another predictable romance novel with a billionaire CEO. Just what the world needed. Absolutely riveting.",
        "Queja de envio":        "I never received my order. Three weeks have passed and the book still hasn't arrived. One star for the service.",
        "Resena muy corta":      "Good book. Enjoyed it.",
        "Fan entusiasta":        "I devoured this book in one sitting! Every page kept me on the edge of my seat. I immediately bought the sequel.",
        "Dificil de clasificar": "The book covered the main topics and had some interesting sections. Nothing groundbreaking but it gets the job done.",
    }

    st.markdown("**Ejemplos rapidos:**")
    row1 = st.columns(5)
    for col, (nombre, txt) in zip(row1, list(ejemplos.items())[:5]):
        if col.button(nombre, use_container_width=True, key=f"ej1_{nombre}"):
            st.session_state.ejemplo_texto = txt
            st.rerun()

    row2 = st.columns(5)
    for col, (nombre, txt) in zip(row2, list(ejemplos.items())[5:]):
        if col.button(nombre, use_container_width=True, key=f"ej2_{nombre}"):
            st.session_state.ejemplo_texto = txt
            st.rerun()

    st.write("")
    texto = st.text_area(
        "Resena en ingles:",
        value=st.session_state.ejemplo_texto,
        height=120,
        placeholder="Write a book review in English...",
    )

    if st.button("Clasificar", icon=":material/search:", type="primary"):
        if not texto.strip():
            st.warning("Escribe algo primero.", icon=":material/warning:")
        else:
            prep   = NLPPreprocessor(method=prep_method)
            proc   = prep.process_single(texto)
            vector = vectorizer.transform([proc])
            pred   = model.predict(vector)[0]
            proba  = model.predict_proba(vector)[0]

            st.write("")
            if pred == 1:
                st.success(f"POSITIVO  —  confianza {proba[1]*100:.1f}%", icon=":material/thumb_up:")
            else:
                st.error(f"NEGATIVO  —  confianza {proba[0]*100:.1f}%", icon=":material/thumb_down:")

            fig, ax = plt.subplots(figsize=(6, 0.8))
            ax.barh([""], [proba[1]], color="#28a745", label="Positivo")
            ax.barh([""], [proba[0]], left=[proba[1]], color="#dc3545", label="Negativo")
            ax.set_xlim(0, 1)
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.set_xticklabels(["0%","25%","50%","75%","100%"])
            ax.spines[["top","right","left"]].set_visible(False)
            ax.legend(loc="upper right", fontsize=8, frameon=False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            with st.expander("Ver texto preprocesado", icon=":material/text_snippet:"):
                st.code(proc, language=None)
