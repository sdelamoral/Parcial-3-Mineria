# Sentiment Analysis — Amazon Kindle Reviews
**Mini Proyecto 3 · Minería de Datos**  
Universidad Autónoma de Guadalajara

---

## Descripción

Sistema de clasificación de sentimientos entrenado sobre reseñas de libros de Amazon Kindle. El modelo determina automáticamente si una reseña expresa un sentimiento **positivo** o **negativo** usando técnicas de Procesamiento de Lenguaje Natural (NLP).

Se evaluaron **12 configuraciones** combinando:
- 3 modelos: Naive Bayes, Logistic Regression, SVM (LinearSVC)
- 2 técnicas de preprocesamiento: Stemming (PorterStemmer) y Lemmatization (WordNetLemmatizer)
- 2 representaciones vectoriales: Bag-of-Words y TF-IDF

**Mejor resultado:** SVM + Lemmatization + TF-IDF → F1-score = **0.9629**

---

## Dataset

**Fuente:** [Amazon Kindle Reviews — Kaggle](https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews)  
**Autor del dataset:** bharadwaj6  
**Tamaño original:** 982,619 reseñas  
**Muestra utilizada:** 50,000 reseñas (selección aleatoria con `random_state=42`)

Mapeo de sentimiento:

| Rating | Etiqueta |
|--------|----------|
| 1 – 2 estrellas | Negativo (0) |
| 3 estrellas | Descartado (neutral ambiguo) |
| 4 – 5 estrellas | Positivo (1) |

---

## Estructura del proyecto

```
P3 Mini Proyecto/
│
├── data/
│   └── kindle_reviews.csv          # Dataset descargado de Kaggle
│
├── src/
│   ├── data_loader.py              # Carga y exploración del dataset
│   ├── preprocessor.py             # Pipeline NLP (NLPPreprocessor)
│   ├── vectorizer.py               # CountVectorizer y TfidfVectorizer
│   ├── models.py                   # Naive Bayes, Logistic Regression, SVM
│   └── evaluator.py                # Métricas, confusion matrix, comparación
│
├── outputs/                        # Gráficas y resultados generados
│   ├── results_summary.csv         # Tabla de las 12 configuraciones
│   ├── model_comparison.png        # Gráfica comparativa de modelos
│   ├── cm_*.png                    # Matrices de confusión por configuración
│   └── ...
│
├── model/
│   ├── best_model.pkl              # Mejor modelo entrenado (SVM)
│   ├── best_vectorizer.pkl         # Vectorizador correspondiente (TF-IDF)
│   └── best_config.txt             # Metadatos del mejor modelo
│
├── main.py                         # Pipeline completo de entrenamiento
├── analisis_comparativo.py         # Análisis comparativo desde CSV
├── analisis_clase.py               # Pipeline en estilo notebook de clase
├── download_data.py                # Descarga del dataset desde Kaggle
├── app.py                          # Dashboard interactivo (Streamlit)
└── requirements.txt
```

---

## Instalación y ejecución

### Requisitos
- Python 3.9+
- Cuenta de Kaggle (para descargar el dataset)

### Pasos

```bash
# 1. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar el dataset (solo la primera vez)
python download_data.py

# 4. Entrenar todos los modelos y generar resultados
python main.py

# 5. Lanzar el dashboard
venv/bin/streamlit run app.py
```

El dashboard queda disponible en `http://localhost:8501`.

---

## Pipeline NLP

```
Texto original
    │
    ▼
Limpieza          re.sub('[^a-zA-Z]', ' ', text) → lowercase
    │
    ▼
Tokenización      word_tokenize (NLTK)
    │
    ▼
Stopwords         eliminación con nltk.corpus.stopwords ('english')
    │
    ├──── Stemming ────► PorterStemmer  → tokens reducidos por sufijo
    └──── Lemmatization → WordNetLemmatizer → forma canónica del diccionario
    │
    ▼
Vectorización
    ├──── Bag-of-Words  → CountVectorizer  (max_features=20,000, ngram_range=(1,2))
    └──── TF-IDF        → TfidfVectorizer  (max_features=20,000, sublinear_tf=True)
    │
    ▼
Clasificador      Naive Bayes / Logistic Regression / SVM
```

---

## Resultados

| Configuración | Accuracy | F1-score |
|---|---|---|
| SVM / Lemma + TF-IDF | 0.9659 | 0.9629 |
| SVM / Stem + TF-IDF | 0.9659 | 0.9627 |
| Logistic Regression / Lemma + BoW | 0.9641 | 0.9617 |
| Logistic Regression / Stem + BoW | 0.9624 | 0.9599 |
| Naive Bayes / Lemma + TF-IDF | 0.9408 | 0.9184 |

**Problema detectado:** el dataset tiene un desbalance severo (93.4% positivo / 6.6% negativo). El recall de la clase negativa varía entre 0.11 (Naive Bayes + TF-IDF) y 0.76, lo que indica que los modelos tienden a clasificar reseñas negativas como positivas.

---

## Código generado por IA

Este proyecto fue desarrollado con asistencia de **Claude Code (Anthropic)** como herramienta de apoyo en el proceso de implementación. A continuación se detalla qué partes fueron generadas con IA y cuáles fueron de autoría propia:

### Generado con asistencia de IA

| Archivo | Descripción |
|---|---|
| `src/preprocessor.py` | Clase `NLPPreprocessor` con los dos métodos de reducción |
| `src/vectorizer.py` | Funciones `build_bow` y `build_tfidf` |
| `src/models.py` | Funciones de entrenamiento para los tres clasificadores |
| `src/evaluator.py` | Cálculo de métricas, gráficas de confusion matrix y comparación |
| `src/data_loader.py` | Carga del CSV, mapeo binario de sentimiento, exploración |
| `main.py` | Orquestador del pipeline completo |
| `analisis_comparativo.py` | Script de análisis comparativo sobre resultados pre-calculados |
| `analisis_clase.py` | Pipeline completo en estilo notebook (para referencia académica) |
| `app.py` | Dashboard Streamlit con las 6 secciones interactivas |

### Definido por el estudiante

- Selección del dataset (Amazon Kindle Reviews, Kaggle)
- Decisión de evaluar 12 configuraciones (3 × 2 × 2)
- Mapeo de sentimiento: descartar rating 3, binarizar 1-2 y 4-5
- Definición de las secciones del dashboard y su contenido
- Iteraciones de diseño y ajustes visuales de la interfaz
- Interpretación de los resultados y conclusiones del análisis comparativo

---

## Por qué se utilizó IA

1. **Velocidad de implementación:** el proyecto requería un pipeline completo (carga, preprocesamiento, vectorización, entrenamiento, evaluación y visualización) en un tiempo limitado. La IA permitió construir la estructura base de forma rápida.

2. **Reducción de errores de sintaxis:** trabajar con múltiples librerías (NLTK, scikit-learn, Streamlit, matplotlib, seaborn, joblib) al mismo tiempo aumenta la probabilidad de errores. La IA ayudó a integrarlas correctamente.

3. **Referencia a patrones del curso:** se usó el notebook de la clase como referencia (`PorterStemmer`, `CountVectorizer`, `TfidfVectorizer`) para que la IA implementara el mismo patrón pero aplicado al dataset de Kindle Reviews.

4. **Aprendizaje:** al revisar el código generado se pudo entender cómo cada componente del pipeline interactúa con los demás, lo que complementó lo visto en clase.

---

## Pruebas realizadas

### 1. Verificación del pipeline de entrenamiento (`main.py`)
- Se ejecutó el script completo y se verificó que las 12 configuraciones entrenaran sin errores.
- Se revisó que `results_summary.csv` contenía exactamente 12 filas con valores en rango esperado (accuracy > 0.90).
- Se verificó que los archivos `.pkl` del modelo se guardaran correctamente.

### 2. Prueba del análisis comparativo (`analisis_clase.py`)
- Se ejecutó el script completo y se verificó la salida impresa para cada una de las 12 secciones.
- Se comprobó que las 5 gráficas se generaran y guardaran en `outputs/`.
- Se verificó que los ejemplos de errores de clasificación mostraran textos reales del dataset.

### 3. Prueba del dashboard (`app.py`)
- Se navegó por las 6 páginas del dashboard verificando que cargaran sin errores.
- **Dataset:** se comprobó que las gráficas de distribución y longitud de reseñas aparecieran.
- **Preprocesamiento:** se probó el pipeline paso a paso con textos en inglés, verificando que cada etapa (limpieza, tokenización, stopwords, stemming/lemmatization) mostrara la transformación correcta.
- **Resultados:** se verificó que la tabla de 12 configuraciones mostrara los valores correctos con resaltado de máximos y mínimos, y que las matrices de confusión se cargaran para cada configuración seleccionada.
- **Análisis Comparativo:** se verificó que las gráficas de barras se generaran dinámicamente y que las conclusiones reflejaran los datos reales del CSV.
- **Demo:** se probaron los 10 ejemplos predefinidos y se introdujeron textos propios para verificar que el modelo clasificara y mostrara el porcentaje de confianza.

### 4. Prueba de integración con GitHub
- Se verificó que el repositorio se subiera correctamente con `git push`.
- Se comprobó que los archivos excluidos en `.gitignore` (dataset, modelos `.pkl`, entorno virtual) no aparecieran en el repositorio.

---

## Errores encontrados y correcciones

### Error 1 — `TypeError` en `analisis_clase.py`

**Descripción:** Al generar la matriz de confusión del mejor modelo, el código intentaba acceder a `results["y_pred"][best_idx]` usando una clave de string sobre una lista.

```python
# Código con error:
best_pred_arr = np.array(results["y_pred"][best_idx] if isinstance(results, list)
                          else results[best_idx]["y_pred"])
```

**Error:** `TypeError: list indices must be integers or slices, not str`

**Corrección:** Se eliminó la línea ya que `best_pred` ya estaba correctamente definido más arriba desde el DataFrame de resultados. La matriz de confusión usaba `best_pred` directamente.

```python
# Corrección: eliminar la línea innecesaria y usar best_pred directamente
cm = confusion_matrix(y_test, best_pred)
```

---

### Error 2 — `ModuleNotFoundError` al ejecutar los scripts

**Descripción:** Al intentar ejecutar `analisis_clase.py` con el Python del sistema (`/opt/homebrew/opt/python@3.14`), los módulos `pandas`, `seaborn`, `nltk` y `scikit-learn` no estaban instalados.

**Error:** `ModuleNotFoundError: No module named 'pandas'`

**Causa:** La Mac tiene múltiples instalaciones de Python (sistema, Anaconda, Homebrew) y los paquetes estaban instalados en el entorno de Anaconda (`curso_bigdata`), no en el Python del sistema.

**Corrección:** Se creó un entorno virtual propio dentro del proyecto:

```bash
python3 -m venv venv
venv/bin/pip install pandas numpy scikit-learn nltk matplotlib seaborn streamlit joblib
```

A partir de ese punto todos los scripts se ejecutan con `venv/bin/python3` y el dashboard con `venv/bin/streamlit run app.py`.

---

### Error 3 — Claude aparecía como contribuidor en GitHub

**Descripción:** El primer commit incluía la línea `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>` en el mensaje, lo que hizo que GitHub registrara a "claude" como contribuidor del repositorio.

**Corrección (intento 1):** Se hizo `git commit --amend` para eliminar la línea y `git push --force`. GitHub no actualizó la lista de contribuidores porque ya había procesado el primer push.

**Corrección (definitiva):** Se eliminó el repositorio en GitHub y se creó uno nuevo. Se usó `git checkout --orphan` para crear un historial limpio sin ningún commit anterior, asegurando que el autor y committer fueran exclusivamente `samanthaMora`:

```bash
git checkout --orphan limpio
GIT_AUTHOR_NAME="samanthaMora" GIT_AUTHOR_EMAIL="..." \
GIT_COMMITTER_NAME="samanthaMora" GIT_COMMITTER_EMAIL="..." \
git commit -m "Sentiment Analysis — NLP pipeline completo"
git push --force origin limpio:main
```

---

### Error 4 — Imposibilidad de eliminar el repositorio via API

**Descripción:** Al intentar eliminar el repositorio de GitHub con la API REST usando el token generado, la solicitud devolvió un error 403.

**Error:** `{"message": "Must have admin rights to Repository.", "status": "403"}`

**Causa:** El token de acceso personal fue creado con el scope `repo` pero sin el scope `delete_repo`, que es necesario para eliminar repositorios via API.

**Corrección:** Se eliminó el repositorio manualmente desde la interfaz de GitHub (Settings → Danger Zone → Delete this repository) y se recreó con el nombre correcto.

---

## Dependencias

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk>=3.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.31.0
joblib
kagglehub>=0.2.0
```

---

## Autora

**Samantha de la Mora**  
Minería de Datos — Universidad Autónoma de Guadalajara  
2026
