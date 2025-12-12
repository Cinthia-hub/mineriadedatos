# Proyecto con análisis y modelos sobre un dataset para exploración/predicción.
Incluye notebooks, scripts y un servidor simple para interactuar con resultados.

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
