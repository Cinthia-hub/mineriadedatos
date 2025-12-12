# Proyecto con análisis y modelos sobre un dataset para exploración/predicción.
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
- Si el servidor expone un puerto, se indicará en la salida al ejecutarlo (por ejemplo, "Listening on http://localhost:8000").
