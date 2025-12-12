# Proyecto con análisis y modelos sobre un dataset para exploración/predicción.
Una pequeña suite para explorar un dataset tipo Kaggle (IMDB top movies), gestionar reseñas de usuarios (SQLite + JSON) y generar un informe PDF con análisis y visualizaciones.
-  Es una pequeña aplicación para explorar un dataset de películas (estilo Kaggle IMDB) y gestionar/visualizar reseñas de usuarios.
-  Soporta tanto UI estática (HTML/CSS/JS) como backend en Flask que persiste reseñas en SQLite, actualiza un JSON de películas con reseñas y regenera un PDF con análisis/visualizaciones.
-  También incluye utilidades para preparar el dataset desde Kaggle, combinar reseñas desde la BD/JSON y una app Streamlit alternativa para ver películas que tienen reseñas en la BD.

## Requisitos e instalación
Se recomienda usar entorno virtual.

Con virtualenv / pip:
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Ejecución
Este proyecto se ejecuta simplemente con:
```bash
python server.py
```
- Asegúrate de haber instalado las dependencias y de colocar los datos necesarios en `data/` si aplican.
- Si el servidor expone un puerto, se indicará en la salida al ejecutarlo (por ejemplo, "Listening on http://localhost:5000").

## Archivos principales
- Generar movies.json:
python dataset_prepare.py
- Generar movies_with_reviews.json desde DB:
python merge_reviews.py
- Generar PDF:
python grafos.py
- Levantar servidor (sirve UI + endpoints):
python server.py

## Cómo escribe el PDF:
- Escribe primero a un temporal (`model_results.pdf.tmp`) y luego reemplaza el archivo final de forma atómica para evitar inconsistencias.

Qué gráficos / páginas incluye (qué significa cada uno)
- Portada / resumen:
  - Conteos de reseñas (DB/JSON), lista de modelos entrenados, breve resumen numérico.

- Top movies (barra horizontal):
  - Top N películas por rating promedio (usando reseñas agregadas).

- TF‑IDF top terms:
  - Palabras con mayor TF‑IDF en reseñas JSON. Indica temas y vocabulario representativo.

- Distribución de sentimiento (barra y pie):
  - Polaridad global (VADER): positivos / neutrales / negativos.

- Polaridad por película (stacked bars):
  - Para películas con más reseñas, se muestra la distribución por polaridad.

- Boxplots de ratings:
  - Dos paneles: boxplot global y boxplots por polaridad (positive/neutral/negative).

- Boxplot por película (top N):
  - Distribución de ratings por las películas con más reseñas.

- KMeans clusters (scatter 2D) + centroides:
  - Reduce TF‑IDF a 2D (TruncatedSVD), clusteriza con KMeans, dibuja puntos por cluster y marca centroides proyectados.
  - Pagina(s) adicionales: por cluster muestra top TF‑IDF terms del centroide.

- Matrices de confusión, ROC y PR:
  - Si se entrenaron modelos (Naive Bayes, Linear SVM, Decision Tree) y hay etiquetas binarias en los reviews (rating >= 7), se generarán matrices y curvas.

- Métricas comparadas:
  - Barras comparando accuracy/precision/recall/f1 entre modelos.

- Feature importances (Decision Tree):
  - Palabras con mayor importancia para el árbol de decisión (si se entrenó).

- LDA topics:
  - Tópicos no supervisados y sus palabras top.

- Classification reports (texto):
  - Informes detallados (precision/recall/f1) por modelo.

### Cambiar películas
Si deseas puedes cambiar las películas con las que trabaja la página con:
1) Crear `site_data/movies.json` (dataset de películas)
- Opción automática (Kaggle via kagglehub):
  - Ejecuta:
    ```bash
    python dataset_prepare.py
    ```
  - Resultado: `site_data/movies.json` (lista JSON) con columnas normalizadas: `Poster_Link`, `Series_Title`, `Released_Year`, `Runtime`, `Genre`, `IMDB_Rating`, `Overview`, `Director`, `Star1..Star4`, `Gross`, etc.
- Opción manual:
  - Proveer un CSV (descargado de Kaggle u otra fuente) y adaptar/renombrar columnas. `dataset_prepare.py` intenta renombrar columnas comunes automáticamente.

Estructura típica (ejemplo simplificado)
```json
{
  "Poster_Link": "https://...",
  "Series_Title": "Movie Title",
  "Released_Year": 2010,
  "Runtime": 120,
  "Genre": "Drama, Thriller",
  "IMDB_Rating": 7.8,
  "Overview": "Breve sinopsis...",
  "Director": "Nombre",
  "Star1": "Actor 1"
}
```

---

2) Crear `site_data/movies_with_reviews.json` (películas + reseñas)
Este JSON es la versión enriquecida de `movies.json` que incluye reseñas (`user_reviews`, `reviews`, `positive_review`, `negative_review`, etc.). Hay tres caminos para generarlo:

- A) Con `merge_reviews.py` (recomendado cuando ya tienes una BD o `reviews.json`):
  - Lee `site_data/movies.json`.
  - Carga reseñas (preferencia: `IMDB_Movies_2021.db` SQLite; fallback: `site_data/reviews.json`).
  - Normaliza títulos (elimina acentos/paréntesis, lowercase) y hace emparejamiento por similitud (difflib, cutoff ~0.80).
  - Escribe `site_data/movies_with_reviews.json`.
  - Ejecutar:
    ```bash
    python merge_reviews.py
    ```

- B) Usando el servidor (`server.py`) al añadir reseñas:
  - Si arrancas `server.py`, al llamar a `/api/add_review` con `movie_index` el servidor actualizará o creará `site_data/movies_with_reviews.json` de forma atómica. Esto añade `db_rowid`, `posted` y guarda también en SQLite.
  - Ejemplo curl:
    ```bash
    curl -X POST -H "Content-Type: application/json" \
      -d '{"title":"Movie Title","author":"Ana","review":"Excelente","rating":9,"movie_index":12}' \
      http://localhost:5000/api/add_review
    ```

- C) Edición manual:
  - Edita `site_data/movies_with_reviews.json`. Es válido para pequeñas correcciones, pero preferible usar script/servidor.

Consejos sobre emparejamiento:
- `merge_reviews.py` usa difflib.get_close_matches con cutoff 0.8. Si muchos títulos no casan baja el cutoff (ej. 0.6) o usa RapidFuzz/Ratcliff-Obershelp para fuzzy matching más robusto.

---

3) Generar `site_data/model_results.pdf` (análisis y gráficos)
Script: `grafos.py`. Lee reseñas desde:
- Preferido: `site_data/movies_with_reviews.json`
- Fallback: `site_data/movies.json`
- También puede leer la BD `IMDB_Movies_2021.db` para entrenar modelos.

Ejecución:
```bash
python grafos.py
# o dejar que server.py lo invoque automáticamente tras cambios
```

Créditos
- Autora del proyecto: Cinthia Camila Bravo Marmolejo
- Este README fue adaptado para explicar claramente cómo recrear los JSON y el PDF generado.