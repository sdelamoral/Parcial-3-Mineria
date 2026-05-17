"""
Sentiment Analysis Dashboard — Kindle Reviews
Run: venv/bin/python3 flask_app.py
Then open: http://localhost:5000
"""

import os, re, base64
from io import BytesIO

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.data.path.append("/Users/tabatha/nltk_data")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_sw
from nltk.stem import PorterStemmer, WordNetLemmatizer
from src.preprocessor import NLPPreprocessor
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

OUTPUT_DIR = "outputs"
MODEL_DIR  = "model"
CSV_PATH   = os.path.join(OUTPUT_DIR, "results_summary.csv")

_model = _vec = _cfg = _nlp = None

def get_model():
    global _model, _vec, _cfg
    if _model is None:
        _model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
        _vec   = joblib.load(os.path.join(MODEL_DIR, "best_vectorizer.pkl"))
        _cfg   = {}
        with open(os.path.join(MODEL_DIR, "best_config.txt")) as f:
            for line in f:
                k, v = line.strip().split("=", 1)
                _cfg[k] = v
    return _model, _vec, _cfg

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = {
            "stopwords": set(nltk_sw.words("english")),
            "stemmer":   PorterStemmer(),
            "lemma":     WordNetLemmatizer(),
        }
    return _nlp

def load_df():
    df = pd.read_csv(CSV_PATH, index_col="label").reset_index()
    parsed       = df["label"].apply(lambda l: dict(zip(["modelo","prep","vec"], [x.strip() for x in (l.split(" / ")[0], *l.split(" / ")[1].split(" + "))]) ))
    df["modelo"] = parsed.apply(lambda x: x["modelo"])
    df["prep"]   = parsed.apply(lambda x: x["prep"])
    df["vec"]    = parsed.apply(lambda x: x["vec"])
    return df

def b64png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data

def file_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def inicio():
    return render_template("inicio.html")


@app.route("/dataset")
def dataset():
    imgs = {}
    for key, fname in [("dist","01_dataset_distribution.png"), ("len","02_review_lengths.png")]:
        p = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(p):
            imgs[key] = file_b64(p)
    return render_template("dataset.html", imgs=imgs)


@app.route("/preprocesamiento", methods=["GET","POST"])
def preprocesamiento():
    default = "I absolutely loved this book! The characters were beautifully written and the plot was amazing. Couldn't put it down."
    texto  = default
    metodo = "lemma"
    res    = None

    if request.method == "POST":
        texto  = request.form.get("texto", default)
        metodo = request.form.get("metodo", "lemma")
        nlp    = get_nlp()

        limpio = re.sub(r"http\S+|www\S+", " ", texto.lower())
        limpio = re.sub(r"[^a-z\s]", " ", limpio)
        limpio = re.sub(r"\s+", " ", limpio).strip()

        tokens    = word_tokenize(limpio)
        eliminadas = [t for t in tokens if t in nlp["stopwords"]]
        tokens_ns  = [t for t in tokens if t not in nlp["stopwords"] and len(t) > 1]
        final = ([nlp["stemmer"].stem(t) for t in tokens_ns] if metodo == "stem"
                 else [nlp["lemma"].lemmatize(t) for t in tokens_ns])

        res = dict(limpio=limpio, tokens=tokens, eliminadas=eliminadas,
                   tokens_ns=tokens_ns, final=final, texto_final=" ".join(final))

    return render_template("preprocesamiento.html", texto=texto, metodo=metodo, res=res)


@app.route("/resultados")
def resultados():
    if not os.path.exists(CSV_PATH):
        return render_template("resultados.html", error=True)

    df       = load_df()
    best     = df.loc[df["f1"].idxmax()]
    display  = df[["label","accuracy","precision","recall","f1"]].sort_values("f1", ascending=False).copy()
    display.columns = ["Configuracion","Accuracy","Precision","Recall","F1"]

    chart = None
    p = os.path.join(OUTPUT_DIR, "model_comparison.png")
    if os.path.exists(p):
        chart = file_b64(p)

    selected = request.args.get("config", df["label"].iloc[0])
    safe     = selected.lower().replace(" ","_").replace("/","_")
    cm_path  = os.path.join(OUTPUT_DIR, f"cm_{safe}.png")
    cm_b64   = file_b64(cm_path) if os.path.exists(cm_path) else None
    row      = df[df["label"] == selected].iloc[0] if selected in df["label"].values else None

    return render_template("resultados.html", error=False, best=best,
        tabla=display.to_dict("records"), configs=df["label"].tolist(),
        selected=selected, chart=chart, cm_b64=cm_b64, row=row)


@app.route("/analisis")
def analisis():
    if not os.path.exists(CSV_PATH):
        return render_template("analisis.html", error=True)

    df      = load_df()
    bow_f1  = df[df["vec"]=="BoW"]["f1"].mean()
    tfidf_f1= df[df["vec"]=="TF-IDF"]["f1"].mean()
    sm      = df[df["prep"]=="Stem"]["f1"].mean()
    lm      = df[df["prep"]=="Lemma"]["f1"].mean()
    stats   = df.groupby("modelo")["f1"].agg(["mean","max","min"]).sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(7,3.5))
    df.groupby(["modelo","vec"])["f1"].mean().unstack("vec").plot(
        kind="bar", ax=ax, color=["#0a2342","#4a9fd4"], edgecolor="white", width=0.6)
    ax.set_title("F1 promedio: BoW vs TF-IDF por modelo", pad=12)
    ax.set_ylabel("F1-score"); ax.set_ylim(0.88, 0.98)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(title="Vectorizador", frameon=False)
    plt.tight_layout()
    chart_bow = b64png(fig)

    fig, ax = plt.subplots(figsize=(7,3.5))
    df.groupby(["modelo","prep"])["f1"].mean().unstack("prep").plot(
        kind="bar", ax=ax, color=["#1e3a5f","#2e6da4"], edgecolor="white", width=0.6)
    ax.set_title("F1 promedio: Stemming vs Lemmatization por modelo", pad=12)
    ax.set_ylabel("F1-score"); ax.set_ylim(0.88, 0.98)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(title="Preprocesamiento", frameon=False)
    plt.tight_layout()
    chart_stem = b64png(fig)

    hm = os.path.join(OUTPUT_DIR, "analisis_comparativo_heatmap.png")
    if os.path.exists(hm):
        chart_hm = file_b64(hm)
    else:
        pivot = df.pivot_table(index="modelo", columns=["prep","vec"], values="f1")
        pivot.columns = [f"{p}+{v}" for p,v in pivot.columns]
        fig, ax = plt.subplots(figsize=(10,4))
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="Blues", vmin=0.91, vmax=0.97, linewidths=0.5, ax=ax)
        plt.tight_layout()
        chart_hm = b64png(fig)

    errores = [
        ("Sentimiento mixto",
         "La reseña critica el libro pero reconoce aspectos positivos. El modelo detecta el lenguaje positivo e ignora el contexto negativo general.",
         '"While the plotting was much better in this one, I still felt the character development was lacking."',
         "Negativo", "Positivo"),
        ("Sarcasmo o ironía",
         "Palabras con carga positiva usadas en tono irónico. TF-IDF no distingue si 'glad' es genuino o sarcástico.",
         "\"I'm just glad this hot ghetto mess was free.\"",
         "Negativo", "Positivo"),
        ("Queja fuera de contexto",
         "La crítica es al servicio de Amazon, no al libro. No hay vocabulario negativo literario que el modelo pueda detectar.",
         '"I have not received my fourteen free copies of the NYT so please end my subscription."',
         "Negativo", "Positivo"),
        ("Positivo con lenguaje neutro",
         "Reseña de 5 estrellas escrita de forma objetiva, sin palabras claramente positivas.",
         '"The book covers the topic thoroughly. Each chapter builds on the previous one."',
         "Positivo", "Negativo"),
    ]

    return render_template("analisis.html", error=False,
        bow_f1=bow_f1, tfidf_f1=tfidf_f1, sm=sm, lm=lm, diff=lm-sm,
        stats=stats, chart_bow=chart_bow, chart_stem=chart_stem, chart_hm=chart_hm,
        errores=errores)


EJEMPLOS = {
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

@app.route("/demo", methods=["GET","POST"])
def demo():
    model_exists = (os.path.exists(os.path.join(MODEL_DIR,"best_model.pkl")) and
                    os.path.exists(os.path.join(MODEL_DIR,"best_vectorizer.pkl")))
    if not model_exists:
        return render_template("demo.html", error=True)

    model, vectorizer, config = get_model()
    prep_method = config.get("prep","lemma")
    texto = ""
    res   = None

    if request.method == "POST":
        ej_key = request.form.get("ejemplo","")
        texto  = EJEMPLOS.get(ej_key, request.form.get("texto","")).strip()

        if texto:
            prep   = NLPPreprocessor(method=prep_method)
            proc   = prep.process_single(texto)
            vector = vectorizer.transform([proc])
            pred   = int(model.predict(vector)[0])
            proba  = model.predict_proba(vector)[0]

            fig, ax = plt.subplots(figsize=(6, 0.8))
            ax.barh([""], [proba[1]], color="#2e6da4", label="Positivo")
            ax.barh([""], [proba[0]], left=[proba[1]], color="#0a2342", label="Negativo")
            ax.set_xlim(0,1)
            ax.set_xticks([0,.25,.5,.75,1])
            ax.set_xticklabels(["0%","25%","50%","75%","100%"])
            ax.spines[["top","right","left"]].set_visible(False)
            ax.legend(loc="upper right", fontsize=8, frameon=False)
            plt.tight_layout()
            chart = b64png(fig)

            res = dict(pred=pred, proba_pos=proba[1]*100,
                       proba_neg=proba[0]*100, proc=proc, chart=chart)

    return render_template("demo.html", error=False,
        model_label=config.get("label","?"), model_f1=config.get("f1","?"),
        ejemplos=EJEMPLOS, texto=texto, res=res)


if __name__ == "__main__":
    print("\n  Abre en tu navegador: http://localhost:8080\n")
    app.run(debug=False, port=8080)
