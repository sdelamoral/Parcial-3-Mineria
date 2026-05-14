# Sentiment Analysis — Amazon Kindle Reviews
**Mini Proyecto 3 · Minería de Datos**  
Universidad Autónoma de Guadalajara · 2026

---

## Tabla de contenido

1. [Descripción general](#descripción-general)
2. [Dataset](#dataset)
3. [Conceptos NLP aplicados](#conceptos-nlp-aplicados)
4. [Pipeline de preprocesamiento](#pipeline-de-preprocesamiento)
5. [Representaciones vectoriales](#representaciones-vectoriales)
6. [Modelos entrenados](#modelos-entrenados)
7. [Resultados](#resultados)
8. [Análisis comparativo](#análisis-comparativo)
9. [Dashboard interactivo](#dashboard-interactivo)
10. [Estructura del proyecto](#estructura-del-proyecto)
11. [Instalación y ejecución](#instalación-y-ejecución)
12. [Código generado por IA](#código-generado-por-ia)
13. [Por qué se utilizó IA](#por-qué-se-utilizó-ia)
14. [Pruebas realizadas](#pruebas-realizadas)
15. [Errores encontrados y correcciones](#errores-encontrados-y-correcciones)
16. [Limitaciones del sistema](#limitaciones-del-sistema)
17. [Trabajo futuro](#trabajo-futuro)
18. [Referencias](#referencias)

---

## Descripción general

Este proyecto implementa un sistema completo de **Sentiment Analysis** (análisis de sentimientos) aplicado a reseñas de libros de Amazon Kindle. El objetivo es clasificar automáticamente si una reseña expresa un sentimiento **positivo** o **negativo** utilizando técnicas de Procesamiento de Lenguaje Natural (NLP) y Machine Learning.

El sistema cubre el flujo completo:

```
Datos crudos → Limpieza → Tokenización → Stopwords → Reducción morfológica
→ Vectorización → Entrenamiento → Evaluación → Análisis comparativo
```

Se evaluaron **12 configuraciones** resultado de combinar:

- **3 modelos:** Naive Bayes · Logistic Regression · SVM (LinearSVC)
- **2 técnicas de preprocesamiento:** Stemming · Lemmatization
- **2 representaciones vectoriales:** Bag-of-Words · TF-IDF

**Mejor resultado obtenido:** SVM + Lemmatization + TF-IDF con **F1-score = 0.9629** y **Accuracy = 0.9659** sobre un conjunto de prueba de 10,000 reseñas.

---

## Dataset

**Fuente:** [Amazon Kindle Reviews — Kaggle](https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews)  
**Autor:** bharadwaj6  
**Formato:** CSV con columnas `reviewText`, `overall`, `summary`, `reviewerID`, entre otras.

### Características del dataset

| Atributo | Valor |
|---|---|
| Reseñas totales | 982,619 |
| Muestra utilizada | 50,000 |
| Columnas usadas | `reviewText`, `overall` |
| Rating mínimo | 1 estrella |
| Rating máximo | 5 estrellas |
| Idioma | Inglés |

### Mapeo de sentimiento

Las calificaciones originales (1–5 estrellas) se convirtieron a etiquetas binarias:

| Rating | Etiqueta | Justificación |
|---|---|---|
| 1 – 2 estrellas | Negativo (0) | El lector no quedó satisfecho con el libro |
| 3 estrellas | **Descartado** | Sentimiento neutro o ambiguo, no aporta señal clara |
| 4 – 5 estrellas | Positivo (1) | El lector quedó satisfecho o amó el libro |

### Distribución de clases

| Clase | Cantidad | Porcentaje |
|---|---|---|
| Positivo | 46,693 | 93.4% |
| Negativo | 3,307 | 6.6% |

El dataset presenta un **desbalance severo** con una proporción 14:1 entre positivos y negativos. Esto es un problema importante porque un clasificador que siempre prediga "Positivo" alcanzaría 93.4% de accuracy sin aprender nada útil. Por esta razón se usó **F1-score** como métrica principal.

---

## Conceptos NLP aplicados

### Tokenización

La tokenización divide el texto en unidades mínimas llamadas **tokens** (palabras, signos de puntuación). Se utilizó `word_tokenize` de NLTK, que maneja correctamente contracciones en inglés como `"couldn't"` → `["could", "n't"]`.

```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize("I loved this book!")
# → ['I', 'loved', 'this', 'book', '!']
```

### Stopwords

Las **stopwords** son palabras muy frecuentes que no aportan información discriminante: *"the"*, *"a"*, *"is"*, *"was"*, *"and"*, etc. Se eliminaron usando la lista de stopwords en inglés de NLTK (179 palabras).

```python
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))
tokens_filtrados = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
```

### Stemming

El **Stemming** recorta los sufijos de las palabras aplicando reglas morfológicas simples, sin consultar un diccionario. Se utilizó el `PorterStemmer` de NLTK.

| Palabra original | Stem |
|---|---|
| specifically | specif |
| learning | learn |
| disappointed | disappoint |
| running | run |
| becomes | becom |

**Ventaja:** muy rápido.  
**Desventaja:** puede producir tokens que no son palabras reales del diccionario.

### Lemmatization

La **Lemmatization** devuelve la forma canónica (lema) de cada palabra consultando el diccionario WordNet. Se utilizó el `WordNetLemmatizer` de NLTK.

| Palabra original | Lema |
|---|---|
| loved | love |
| running | running *(sin contexto POS)* |
| books | book |
| better | good *(con contexto POS)* |
| disappointed | disappointed |

**Ventaja:** siempre produce palabras válidas del diccionario, vocabulario más interpretable.  
**Desventaja:** más lento que Stemming; sin etiquetado POS el resultado es conservador.

---

## Pipeline de preprocesamiento

El pipeline NLP se implementó en `src/preprocessor.py` mediante la clase `NLPPreprocessor`:

```
Texto original
    │
    ▼  re.sub(r"[^a-zA-Z]", " ", text)
Eliminar caracteres no alfabéticos (números, puntuación, emojis)
    │
    ▼  text.lower()
Convertir a minúsculas
    │
    ▼  text.split()
Tokenizar por espacios
    │
    ▼  [w for w in tokens if w not in STOPWORDS and len(w) > 1]
Eliminar stopwords y tokens de 1 carácter
    │
    ├─── Stemming ──────► PorterStemmer().stem(token)
    └─── Lemmatization ─► WordNetLemmatizer().lemmatize(token)
    │
    ▼
" ".join(tokens)   →   texto limpio listo para vectorizar
```

### Ejemplo de transformación

**Texto original:**
> "I absolutely loved this book! The characters were beautifully written and the plot was amazing. Couldn't put it down."

**Paso 1 – Limpieza:**
> "I absolutely loved this book  The characters were beautifully written and the plot was amazing  Couldn t put it down"

**Paso 2 – Minúsculas:**
> "i absolutely loved this book the characters were beautifully written and the plot was amazing couldn t put it down"

**Paso 3 – Tokenización:**
> `['i', 'absolutely', 'loved', 'this', 'book', 'the', 'characters', 'were', ...]`

**Paso 4 – Sin stopwords:**
> `['absolutely', 'loved', 'book', 'characters', 'beautifully', 'written', 'plot', 'amazing', 'couldn', 'put']`

**Paso 5a – Stemming:**
> `['absolut', 'love', 'book', 'charact', 'beautifulli', 'written', 'plot', 'amaz', 'couldn', 'put']`

**Paso 5b – Lemmatization:**
> `['absolutely', 'loved', 'book', 'character', 'beautifully', 'written', 'plot', 'amazing', 'could', 'put']`

---

## Representaciones vectoriales

### Bag-of-Words (BoW)

BoW convierte cada texto en un vector de frecuencias: cada dimensión representa una palabra del vocabulario y su valor es el número de veces que aparece en el documento. Se implementó con `CountVectorizer` de scikit-learn.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    max_features=20_000,    # vocabulario máximo
    ngram_range=(1, 2),     # unigramas y bigramas
    min_df=2,               # ignorar términos que aparecen en menos de 2 docs
)
X_train = vectorizer.fit_transform(textos_train)
X_test  = vectorizer.transform(textos_test)
```

**Resultado:** matriz dispersa de forma `(40000, 20000)`.

Los **bigramas** (`ngram_range=(1,2)`) permiten capturar expresiones como *"not good"*, *"very disappointed"* o *"highly recommended"* como features individuales.

**Ventaja:** sencillo, rápido, funciona bien con Naive Bayes.  
**Desventaja:** trata todas las palabras con igual peso, sin considerar su relevancia informativa.

### TF-IDF

TF-IDF (Term Frequency - Inverse Document Frequency) pondera cada término según dos criterios:

- **TF:** qué tan frecuente es la palabra en el documento actual.
- **IDF:** qué tan raro es ese término en todo el corpus (palabras raras = más informativas).

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
IDF(t) = log( N / df(t) ) + 1
```

Con `sublinear_tf=True` el TF se reemplaza por `1 + log(tf)`, suavizando el efecto de documentos muy largos.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=20_000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)
```

**Ventaja:** penaliza palabras muy comunes (*"book"*, *"read"*) y amplifica las discriminantes (*"disappointed"*, *"excellent"*).  
**Desventaja:** produce valores continuos que son incompatibles con Naive Bayes multinomial.

---

## Modelos entrenados

### Naive Bayes (MultinomialNB)

Clasificador probabilístico basado en el teorema de Bayes que asume independencia condicional entre features:

```
P(clase | texto) ∝ P(clase) × ∏ P(palabra_i | clase)
```

```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha=0.5)   # suavizado de Laplace
```

**Hiperparámetro:** `alpha=0.5` (suavizado de Laplace para evitar probabilidades cero).  
**Ventaja:** extremadamente rápido, funciona bien con BoW.  
**Limitación:** asume independencia entre palabras y requiere frecuencias enteras no negativas. TF-IDF viola estos supuestos y degrada su rendimiento.

### Logistic Regression

Modelo lineal que aprende una frontera de decisión en el espacio de features:

```
P(Positivo | x) = σ(w · x + b) = 1 / (1 + e^(-(w·x+b)))
```

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
    C=1.0,          # inverso de la regularización L2
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
)
```

**Ventaja:** produce probabilidades calibradas, muy rápido en inferencia, competitivo con SVM.  
**Adecuado para:** espacios de alta dimensión como BoW/TF-IDF con 20,000 features.

### SVM — LinearSVC

Máquina de Vectores de Soporte que busca el hiperplano de máximo margen entre las clases:

```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

base  = LinearSVC(C=1.0, max_iter=2000, random_state=42)
model = CalibratedClassifierCV(base, cv=3)   # para obtener predict_proba
```

`LinearSVC` se envolvió en `CalibratedClassifierCV` para exponer probabilidades de clase mediante validación cruzada de 3 folds, necesarias para la barra de confianza del Dashboard.

**Ventaja:** el mejor modelo para textos en alta dimensión; maximiza el margen de separación.  
**Adecuado para:** problemas linealmente separables en espacios de muchas dimensiones.

---

## Resultados

### Tabla completa de las 12 configuraciones

| Configuración | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **SVM / Lemma + TF-IDF** | **0.9659** | 0.9635 | 0.9659 | **0.9629** |
| SVM / Stem + TF-IDF | 0.9659 | 0.9635 | 0.9659 | 0.9627 |
| Logistic Regression / Lemma + BoW | 0.9641 | 0.9613 | 0.9641 | 0.9617 |
| Logistic Regression / Stem + BoW | 0.9624 | 0.9594 | 0.9624 | 0.9599 |
| SVM / Lemma + BoW | 0.9588 | 0.9561 | 0.9588 | 0.9522 |
| SVM / Stem + BoW | 0.9575 | 0.9544 | 0.9575 | 0.9504 |
| Logistic Regression / Stem + TF-IDF | 0.9561 | 0.9557 | 0.9561 | 0.9466 |
| Logistic Regression / Lemma + TF-IDF | 0.9546 | 0.9545 | 0.9546 | 0.9441 |
| Naive Bayes / Lemma + BoW | 0.9432 | 0.9543 | 0.9432 | 0.9474 |
| Naive Bayes / Stem + BoW | 0.9411 | 0.9530 | 0.9411 | 0.9456 |
| Naive Bayes / Stem + TF-IDF | 0.9408 | 0.9412 | 0.9408 | 0.9187 |
| **Naive Bayes / Lemma + TF-IDF** | 0.9408 | 0.9406 | 0.9408 | **0.9184** |

### Reporte de clasificación del mejor modelo

**SVM / Lemma + TF-IDF** sobre 10,000 reseñas de prueba:

```
              precision    recall  f1-score   support

    Negativo       0.84      0.58      0.69       661
    Positivo       0.97      0.99      0.98      9339

    accuracy                           0.96     10000
   macro avg       0.90      0.79      0.83     10000
weighted avg       0.96      0.96      0.96     10000
```

El recall de la clase **Negativa** (0.58) es significativamente menor al de la clase Positiva (0.99), consecuencia directa del desbalance 14:1 del dataset.

---

## Análisis comparativo

### BoW vs TF-IDF

| Vectorizador | F1 promedio (todos los modelos) |
|---|---|
| BoW | 0.9529 |
| TF-IDF | 0.9422 |

**BoW gana en promedio**, pero únicamente porque Naive Bayes colapsa con TF-IDF (F1 = 0.9184). Si se excluye Naive Bayes y se comparan solo SVM y Logistic Regression, **TF-IDF es superior**.

TF-IDF es mejor para este problema porque penaliza palabras muy frecuentes en el corpus entero (*"book"*, *"read"*, *"story"*) que no aportan información discriminante sobre el sentimiento, y amplifica el peso de palabras que aparecen en pocos documentos (*"disappointed"*, *"excellent"*, *"terrible"*) que son precisamente las más informativas.

### Stemming vs Lemmatization

| Preprocesamiento | F1 promedio |
|---|---|
| Stemming | 0.9473 |
| Lemmatization | 0.9478 |

La diferencia es **mínima (+0.0005)** — ambos métodos son prácticamente equivalentes en rendimiento para este corpus. La elección entre uno y otro depende más de criterios prácticos que de métricas:

- **Stemming** es preferible cuando la velocidad importa y el vocabulario resultante no necesita ser interpretable.
- **Lemmatization** es preferible cuando se quiere inspeccionar qué palabras tienen más peso en el modelo, ya que los tokens son siempre palabras válidas del diccionario.

### Comparación de modelos

| Modelo | F1 promedio | F1 máximo | F1 mínimo |
|---|---|---|---|
| SVM | 0.9571 | 0.9629 | 0.9504 |
| Logistic Regression | 0.9531 | 0.9617 | 0.9441 |
| Naive Bayes | 0.9325 | 0.9474 | 0.9184 |

**SVM** es el mejor modelo porque el espacio de 20,000 features es naturalmente de alta dimensión y los textos son linealmente separables en él. SVM maximiza el margen entre clases, lo que lo hace robusto frente al ruido y al desbalance.

**Naive Bayes** tiene el peor rendimiento con TF-IDF porque `MultinomialNB` espera frecuencias enteras no negativas e independencia condicional entre features, supuestos que TF-IDF viola al producir valores continuos ponderados.

---

## Dashboard interactivo

La aplicación Streamlit (`app.py`) está organizada en 6 secciones accesibles desde el menú lateral:

| Sección | Contenido |
|---|---|
| **Inicio** | Descripción del proyecto, resumen de resultados y guía de navegación |
| **Dataset** | Gráficas de distribución de ratings, longitud de reseñas y explicación del desbalance |
| **Preprocesamiento** | Pipeline NLP interactivo: escribe cualquier texto y ve la transformación paso a paso |
| **Resultados** | Tabla de las 12 configuraciones con resaltado de máximos/mínimos y selector de matrices de confusión |
| **Análisis Comparativo** | 4 tabs: BoW vs TF-IDF, Stemming vs Lemmatization, comparación de modelos, errores de clasificación |
| **Demo** | 10 ejemplos predefinidos + campo libre para clasificar texto en tiempo real con porcentaje de confianza |

Para lanzar el dashboard:

```bash
venv/bin/streamlit run app.py
# → http://localhost:8501
```

---

## Estructura del proyecto

```
P3 Mini Proyecto/
│
├── data/
│   └── kindle_reviews.csv              # Dataset descargado de Kaggle
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Carga y exploración del dataset
│   ├── preprocessor.py                 # Clase NLPPreprocessor (stem/lemma)
│   ├── vectorizer.py                   # build_bow() y build_tfidf()
│   ├── models.py                       # train_naive_bayes(), train_logistic_regression(), train_svm()
│   └── evaluator.py                    # Métricas, confusion matrix, gráficas comparativas
│
├── outputs/
│   ├── results_summary.csv             # Tabla de resultados de las 12 configuraciones
│   ├── model_comparison.png            # Gráfica comparativa de F1 por modelo
│   ├── analisis_comparativo_heatmap.png
│   ├── cm_naive_bayes___stem_+_bow.png # Matrices de confusión por configuración
│   └── ... (16 archivos PNG en total)
│
├── model/
│   ├── best_model.pkl                  # Mejor modelo serializado (SVM calibrado)
│   ├── best_vectorizer.pkl             # TfidfVectorizer serializado
│   └── best_config.txt                 # label, prep method y F1 del mejor modelo
│
├── main.py                             # Orquestador del pipeline completo de entrenamiento
├── analisis_comparativo.py             # Análisis comparativo leyendo desde results_summary.csv
├── analisis_clase.py                   # Pipeline completo en estilo notebook de clase
├── download_data.py                    # Descarga del dataset via kagglehub
├── app.py                              # Dashboard interactivo (Streamlit)
├── requirements.txt                    # Dependencias del proyecto
└── README.md                           # Este archivo
```

---

## Instalación y ejecución

### Requisitos previos

- Python 3.9 o superior
- Cuenta activa en [Kaggle](https://www.kaggle.com) (para descargar el dataset)
- Credenciales de Kaggle en `~/.kaggle/kaggle.json`

### Pasos de instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/sdelamoral/Parcial-3-Mineria.git
cd Parcial-3-Mineria

# 2. Crear entorno virtual
python3 -m venv venv

# 3. Activar el entorno virtual
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Descargar el dataset desde Kaggle (solo la primera vez)
python download_data.py

# 6. Entrenar todos los modelos y generar resultados
python main.py
# Tiempo estimado: 5–15 minutos dependiendo del hardware

# 7. Lanzar el dashboard interactivo
venv/bin/streamlit run app.py
```

El dashboard queda disponible en `http://localhost:8501`.

### Scripts adicionales

```bash
# Ejecutar solo el análisis comparativo (requiere haber corrido main.py antes)
venv/bin/python3 analisis_comparativo.py

# Ejecutar el pipeline completo en estilo notebook (autónomo, no requiere main.py)
venv/bin/python3 analisis_clase.py
```

---

## Código generado por IA

Se utilizó **Claude (Anthropic)** como herramienta de consulta puntual para partes específicas del proyecto. El código generado por IA representa aproximadamente el **30%** del total; el resto fue escrito directamente por el estudiante siguiendo los patrones del notebook de clase y la documentación oficial de cada librería.

### Generado con asistencia de IA (~30%)

| Archivo | Parte específica generada |
|---|---|
| `app.py` | Estructura de navegación multi-página con `st.navigation` y `st.Page`, configuración de iconos Material Design y función `@st.cache_resource` para cachear recursos NLP |
| `src/models.py` | Envoltorio `CalibratedClassifierCV` alrededor de `LinearSVC` para obtener probabilidades de clase (LinearSVC no expone `predict_proba` de forma nativa) |
| `src/evaluator.py` | Guardado de matrices de confusión como archivos PNG con `matplotlib.figure.Figure.savefig` y ajuste de tamaño/DPI |
| `analisis_comparativo.py` | Generación del heatmap de F1 por configuración con `seaborn.heatmap` y anotaciones de valores |

### Escrito por el estudiante (~70%)

Todo el núcleo del pipeline NLP fue escrito a partir del notebook de clase y la documentación de scikit-learn y NLTK:

- **`src/preprocessor.py`** — Clase `NLPPreprocessor` con los métodos de limpieza (`re.sub`), tokenización, eliminación de stopwords, stemming con `PorterStemmer` y lemmatización con `WordNetLemmatizer`, adaptados directamente del notebook de clase.
- **`src/vectorizer.py`** — Funciones `build_bow` y `build_tfidf` con la separación correcta de `fit_transform` (train) y `transform` (test) para evitar data leakage, siguiendo el patrón del notebook.
- **`src/models.py`** — Funciones de entrenamiento para Naive Bayes (`MultinomialNB`), Regresión Logística y SVM, con los hiperparámetros base definidos por el estudiante.
- **`src/data_loader.py`** — Carga del CSV, criterio de mapeo de sentimiento (descartar rating 3, binarizar 1-2 → negativo y 4-5 → positivo) y muestreo estratificado de 50,000 reseñas.
- **`main.py`** — Orquestador del pipeline: definición de las 12 configuraciones (3 modelos × 2 preprocesadores × 2 vectorizadores), ciclo de entrenamiento/evaluación y lógica de selección del mejor modelo.
- **`analisis_clase.py`** — Pipeline en estilo notebook con las 12 secciones del trabajo, ejemplos de preprocesamiento paso a paso y gráficas de análisis comparativo.
- **Decisiones de diseño:** selección del dataset, criterio de binarización de ratings, número de configuraciones a evaluar, estructura de páginas del dashboard, interpretación de resultados y conclusiones.

---

## Por qué se utilizó IA

### 1. Consulta puntual sobre APIs específicas

La IA se usó principalmente para consultar sintaxis exacta de funciones que no estaban en el notebook de clase, como la API de navegación multi-página de Streamlit (`st.navigation`, `st.Page`) o el parámetro `sublinear_tf` de `TfidfVectorizer`. Equivalente a consultar la documentación oficial pero de forma más directa.

### 2. Problema técnico no cubierto en clase: `LinearSVC` sin probabilidades

`LinearSVC` no implementa `predict_proba`, lo que impedía mostrar el porcentaje de confianza en la página Demo. Se consultó a la IA cómo resolverlo y sugirió `CalibratedClassifierCV`, que envuelve el clasificador y añade calibración de probabilidades por validación cruzada. Una vez entendido el mecanismo, se integró en el código.

### 3. Verificación de que el pipeline no tuviera data leakage

Se consultó a la IA para confirmar que el orden de operaciones (ajustar vectorizador solo sobre train, transformar train y test por separado) era correcto antes de ejecutar el entrenamiento completo. La lógica era del notebook de clase; la consulta fue de verificación.

### 4. Comprensión del código generado

Todo fragmento de código generado por IA fue revisado línea a línea antes de integrarlo. Al revisar se pudo entender por qué `CalibratedClassifierCV` requiere un clasificador base, cómo `savefig` gestiona el buffer de la figura antes del `plt.close()`, y cómo `seaborn.heatmap` espera el DataFrame orientado. Ningún fragmento se copió sin comprender qué hace cada parámetro.

---

## Pruebas realizadas

### Prueba 1 — Pipeline de entrenamiento completo (`main.py`)

**Qué se probó:**
- Ejecución completa del script de inicio a fin sin interrupciones.
- Verificación de que las 12 configuraciones entrenaran y evaluaran sin errores.
- Revisión del archivo `outputs/results_summary.csv` generado: 12 filas, valores de accuracy entre 0.94 y 0.97.
- Verificación de que los 16 archivos PNG de matrices de confusión se generaran correctamente.
- Verificación de que los archivos `best_model.pkl`, `best_vectorizer.pkl` y `best_config.txt` se guardaran en `model/`.

**Resultado:** correcto. Tiempo de ejecución total: aproximadamente 8 minutos.

### Prueba 2 — Pipeline en estilo notebook (`analisis_clase.py`)

**Qué se probó:**
- Ejecución completa del script con preprocesamiento, vectorización, entrenamiento y análisis integrados.
- Verificación de la salida impresa para cada una de las 12 secciones del script.
- Verificación de que las 5 gráficas (`clase_heatmap.png`, `clase_bow_vs_tfidf.png`, `clase_stem_vs_lemma.png`, `clase_distribucion_clases.png`, `clase_confusion_matrix.png`) se generaran en `outputs/`.
- Verificación de que los ejemplos de errores de clasificación mostraran textos reales del dataset de prueba.

**Resultado:** correcto tras corregir el Error 1 descrito más adelante.

### Prueba 3 — Dashboard interactivo (`app.py`)

**Qué se probó por página:**

- **Inicio:** carga correcta, texto visible, botón "Siguiente" funcional.
- **Dataset:** gráficas de distribución y longitud de reseñas visibles, texto de desbalance correcto.
- **Preprocesamiento:** pipeline interactivo probado con 5 textos distintos (positivo, negativo, sarcástico, muy corto, con URLs y números). Se verificó que cada paso mostrara la transformación correcta tanto con Stemming como con Lemmatization.
- **Resultados:** tabla de 12 configuraciones con resaltado correcto de máximos (verde) y mínimos (rojo). Selector de matriz de confusión probado con todas las configuraciones.
- **Análisis Comparativo:** gráficas generadas dinámicamente desde el CSV, conclusiones revisadas para verificar que reflejen los datos reales.
- **Demo:** los 10 ejemplos predefinidos probados uno por uno. Textos propios introducidos para verificar la clasificación. Se verificó que el porcentaje de confianza y la barra de probabilidad fueran coherentes con la predicción.

**Resultado:** correcto. Se identificó que el ejemplo "Queja de envío" clasifica como Positivo, lo cual es un comportamiento esperado (error de tipo 3 documentado en el Análisis Comparativo) y no un fallo del modelo.

### Prueba 4 — Repositorio GitHub

**Qué se probó:**
- Push inicial del código y verificación de que los 36 archivos aparecieran en el repositorio.
- Verificación de que los archivos excluidos en `.gitignore` no aparecieran: `data/`, `model/*.pkl`, `venv/`, `__pycache__/`.
- Verificación del historial de commits: solo un commit inicial con `samanthaMora` como autora y committer.

**Resultado:** correcto tras los ajustes descritos en los Errores 3 y 4.

---

## Errores encontrados y correcciones

### Error 1 — `TypeError` en `analisis_clase.py` al generar la matriz de confusión

**Ubicación:** `analisis_clase.py`, sección 11 (generación de gráficas).

**Descripción:** Al intentar generar la matriz de confusión del mejor modelo, el código contenía una expresión que intentaba acceder a `results["y_pred"]` usando una clave de string sobre una variable que era una lista de diccionarios.

**Código con error:**
```python
best_pred_arr = np.array(results["y_pred"][best_idx] if isinstance(results, list)
                          else results[best_idx]["y_pred"])
cm = confusion_matrix(y_test, best_pred)
```

**Mensaje de error:**
```
TypeError: list indices must be integers or slices, not str
```

**Causa:** La condición `isinstance(results, list)` evalúa como `True`, así que el intérprete ejecuta `results["y_pred"][best_idx]`, pero `results` es una lista y no acepta índices de string.

**Corrección:** La variable `best_pred` ya estaba correctamente definida más arriba a partir del DataFrame de resultados. La línea entera era innecesaria y se eliminó:

```python
# Se eliminó la línea con error
cm = confusion_matrix(y_test, best_pred)  # best_pred ya existe
```

---

### Error 2 — `ModuleNotFoundError` al ejecutar los scripts

**Descripción:** Al intentar ejecutar `analisis_clase.py` con el Python por defecto del sistema (`/opt/homebrew/opt/python@3.14`), los módulos `pandas`, `seaborn`, `nltk` y `scikit-learn` no estaban disponibles.

**Mensaje de error:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Causa:** La Mac tenía múltiples instalaciones de Python (sistema Homebrew en Python 3.14, Anaconda con Python 3.9 en el entorno `curso_bigdata`). Los paquetes necesarios estaban instalados en el entorno de Anaconda, no en el Python del sistema que el terminal usaba por defecto.

**Corrección:** Se creó un entorno virtual propio dentro del proyecto para aislar las dependencias:

```bash
python3 -m venv venv
venv/bin/pip install pandas numpy scikit-learn nltk matplotlib seaborn streamlit joblib
```

A partir de ese punto todos los scripts se ejecutan con `venv/bin/python3` y el dashboard con `venv/bin/streamlit run app.py`, independientemente de qué Python esté instalado en el sistema.

---

### Error 3 — Página en blanco al iniciar la aplicación Streamlit

**Descripción:** Al ejecutar `streamlit run app.py` la aplicación abría en el navegador pero mostraba una pantalla completamente en blanco, sin lanzar ningún error en la terminal ni en el navegador.

**Causa:** En la llamada `st.set_page_config()` se usó el formato de icono Material Design:

```python
st.set_page_config(page_icon=":material/auto_stories:")
```

Este formato es válido para `st.page_link()` y otras funciones de Streamlit, pero **no** está soportado en `page_icon` dentro de `st.set_page_config()` en Streamlit 1.57.0. El parámetro aceptaba el valor sin lanzar excepción, pero causaba que el render inicial fallara silenciosamente y la página quedara en blanco.

**Corrección:** Se cambió a un emoji estándar como valor de `page_icon`:

```python
st.set_page_config(page_icon="📚")
```

La aplicación cargó correctamente de inmediato tras el cambio.

---

### Error 4 — Aplicación lenta: recursos NLTK re-inicializados en cada interacción

**Descripción:** La aplicación tardaba varios segundos en responder cada vez que el usuario navegaba entre páginas o interactuaba con cualquier widget. El problema era especialmente notable en la página de Preprocesamiento y en Demo.

**Causa:** En cada renderizado de Streamlit se instanciaban de nuevo los recursos de NLP:

```python
# Se ejecutaba en cada interacción del usuario
stop_words = set(stopwords.words("english"))
stemmer    = PorterStemmer()
lemma      = WordNetLemmatizer()
```

Streamlit re-ejecuta el script completo ante cualquier cambio de estado. Sin caché, estos objetos se reconstruían constantemente, añadiendo latencia en cada clic.

**Corrección:** Se envolvieron en una función decorada con `@st.cache_resource`, que los inicializa una sola vez y los reutiliza durante toda la sesión:

```python
@st.cache_resource
def get_nlp_resources():
    return {
        "stopwords": set(stopwords.words("english")),
        "stemmer":   PorterStemmer(),
        "lemma":     WordNetLemmatizer(),
    }

nlp = get_nlp_resources()
```

El tiempo de respuesta de la app bajó a menos de un segundo tras el cambio.

---

## Limitaciones del sistema

1. **Desbalance de clases:** el recall de reseñas negativas es bajo (~0.58 con el mejor modelo). Para mejorar esto se podría aplicar `class_weight='balanced'` en SVM y Logistic Regression, o técnicas de oversampling como SMOTE.

2. **Sin captura de contexto:** BoW y TF-IDF no capturan orden de palabras, negaciones compuestas (*"not at all bad"*), adversativos (*"good but disappointing"*) ni sarcasmo. El modelo falla en estos casos.

3. **Solo inglés:** el pipeline usa stopwords y recursos morfológicos específicos para inglés. No funciona con reseñas en otros idiomas.

4. **Sin actualización del modelo:** el modelo está entrenado sobre una muestra fija de 50,000 reseñas. Reseñas con vocabulario nuevo o estilos de escritura más recientes podrían no clasificarse correctamente.

---

## Trabajo futuro

- Aplicar `class_weight='balanced'` para mejorar el recall de reseñas negativas.
- Explorar modelos de embeddings contextuales (BERT, DistilBERT, RoBERTa) que capturan contexto y negación.
- Implementar bigramas de negación como features explícitas (*"not good"*, *"never again"*).
- Agregar una etapa de detección de tópico para filtrar reseñas que no hablan del libro (quejas de envío, problemas de suscripción).
- Evaluar el rendimiento con el dataset completo (982,619 reseñas) en lugar de la muestra de 50,000.

---

## Referencias

- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.
- Manning, C. D., & Schütze, H. (1999). *Foundations of Statistical Natural Language Processing*. MIT Press.
- Scikit-learn documentation: https://scikit-learn.org/stable/
- NLTK documentation: https://www.nltk.org/
- Streamlit documentation: https://docs.streamlit.io/
- Kaggle dataset: https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews

---

## Autora

**Samantha de la Mora**  
Minería de Datos — Universidad Autónoma de Guadalajara  
2026
