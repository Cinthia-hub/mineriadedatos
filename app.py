"""
Aplicación Streamlit (app.py)

Muestra SOLO las películas que tienen reseña en una base de datos SQLite de reseñas,
mostrando el poster y los datos (tomados del dataset tipo Kaggle mencionado).

Cómo usar:
1) Instala dependencias:
   pip install streamlit pandas numpy pillow sqlalchemy
   # Opcional (si quieres usar kagglehub en lugar de CSV local):
   pip install kagglehub

2) Coloca:
   - la base de datos SQLite de reseñas (ej.: IMDB_Movies_2021.db)
     - la tabla de reseñas debe contener al menos: AUTHOR, TITLE, REVIEW, RATING
     - si existe columna de fecha, el script también la mostrará si su nombre es POSTED, POSTED_DATE, DATE, CREATED_AT, TIMESTAMP
   - el dataset de películas (CSV descargado del dataset de Kaggle "harshitshankhdhar/...")
     - columnas esperadas (si no existen, intenta usar nombres alternativos): 
       Poster_Link, Series_Title, Released_Year, Certificate, Runtime, Genre, IMDB_Rating,
       Overview, Meta_score, Director, Star1, Star2, Star3, Star4, No_of_votes, Gross

3) Ejecuta:
   streamlit run app.py

Interfaz:
- En la barra lateral puedes:
  * Indicar ruta al archivo .db (o dejar la por defecto './IMDB_Movies_2021.db')
  * Cargar el CSV del dataset de películas (o intentar cargar desde KaggleHub si lo tienes instalado)
  * Filtrar por género, año, texto
- En la página principal se lista únicamente las películas que tienen al menos 1 reseña en la BD.
- Al seleccionar una película verás: poster, datos, y la(s) reseña(s) (author, rating, texto, posted si existe).

Notas:
- El emparejamiento entre las reseñas y el dataset se hace por título (Series_Title vs TITLE),
  con comparación insensible a mayúsculas y espacios finales/iniciales. Si los títulos no coinciden exactamente,
  puedes usar la búsqueda en la barra lateral para encontrar manualmente.
- Si deseas mejorar el emparejamiento (fuzzy matching), puedo añadirlo si me lo pides.
"""

import os
import sqlite3
import io
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import requests

st.set_page_config(page_title="Películas con Reseñas - IMDB", layout="wide")

# --------------------------
# Helpers
# --------------------------
@st.cache_data
def load_reviews_db(sqlite_db_path: str, reviews_table: str = None):
    """
    Intenta cargar la tabla REVIEWS (u otra tabla) de la BD SQLite.
    Devuelve DataFrame de reseñas.
    """
    if not os.path.exists(sqlite_db_path):
        raise FileNotFoundError(f"No existe el archivo de BD en: {sqlite_db_path}")
    conn = sqlite3.connect(sqlite_db_path)
    # intentar varias tablas/nombres comunes
    try:
        if reviews_table:
            df = pd.read_sql_query(f"SELECT * FROM {reviews_table}", conn)
        else:
            # preferencia por tabla REVIEWS
            try:
                df = pd.read_sql_query("SELECT * FROM REVIEWS", conn)
            except Exception:
                # leer las tablas disponibles y elegir la primera que contenga columnas relevantes
                tbls = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
                chosen = None
                for t in tbls['name'].tolist():
                    tmp = pd.read_sql_query(f"PRAGMA table_info('{t}')", conn)
                    cols = tmp['name'].str.upper().tolist()
                    if 'REVIEW' in cols or 'TITLE' in cols:
                        chosen = t
                        break
                if chosen is None:
                    # tomar la primera tabla
                    chosen = tbls['name'].iloc[0]
                df = pd.read_sql_query(f"SELECT * FROM '{chosen}'", conn)
    finally:
        conn.close()
    return df

@st.cache_data
def load_movies_from_csv(csv_bytes):
    """
    Carga DataFrame desde bytes de CSV (Streamlit file_uploader) o ruta local.
    """
    if isinstance(csv_bytes, (str, os.PathLike)):
        return pd.read_csv(csv_bytes)
    else:
        # csv_bytes es BytesIO
        return pd.read_csv(io.BytesIO(csv_bytes.read()))

@st.cache_data
def load_movies_from_kagglehub(adapter_name: str, dataset_ref: str, file_path: str = ""):
    """
    Intento de carga vía kagglehub si está instalado. adapter_name no se usa realmente aquí,
    se mantiene por compatibilidad con el snippet del usuario.
    """
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except Exception as e:
        raise RuntimeError("kagglehub no está instalado. Usa el uploader de CSV en su lugar.") from e

    # Usar la API tal como indicó el usuario
    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, dataset_ref, file_path)
    return df

def normalize_title(s: str):
    if pd.isna(s):
        return ""
    return "".join(s).strip().lower()

def get_possible_date_column(df_reviews: pd.DataFrame):
    # busca columnas comunes que indiquen fecha/publicación
    candidates = ['posted', 'posted_date', 'date', 'created_at', 'timestamp']
    cols = [c.lower() for c in df_reviews.columns]
    for cand in candidates:
        if cand in cols:
            # devolver el nombre exacto original
            return df_reviews.columns[cols.index(cand)]
    return None

def show_movie_card(movie_row, reviews_for_movie):
    """
    Muestra poster, datos principales y la lista de reseñas para esa película.
    movie_row: pandas Series de dataset de películas
    reviews_for_movie: DataFrame de reseñas relacionadas
    """
    col1, col2 = st.columns([1,2])
    with col1:
        poster = movie_row.get('Poster_Link') or movie_row.get('poster_link') or movie_row.get('Poster link')
        if pd.isna(poster) or poster == "":
            st.image(Image.new('RGB', (300, 450), color=(200,200,200)), caption="Poster no disponible", use_column_width=True)
        else:
            try:
                st.image(poster, use_column_width=True)
            except Exception:
                # intentar descargar y mostrar
                try:
                    resp = requests.get(poster, timeout=6)
                    img = Image.open(io.BytesIO(resp.content))
                    st.image(img, use_column_width=True)
                except Exception:
                    st.image(Image.new('RGB', (300, 450), color=(200,200,200)), caption="Poster no disponible", use_column_width=True)

    with col2:
        st.header(str(movie_row.get('Series_Title') or movie_row.get('series_title') or movie_row.get('Series Title') or "Título desconocido"))
        # mostrar campos relevantes si existen
        fields = [
            ('Released_Year', 'Año'),
            ('Released Year', 'Año'),
            ('Released_Year (or Year)', 'Año'),
            ('Certificate', 'Certificado'),
            ('Runtime', 'Runtime'),
            ('Genre', 'Género'),
            ('IMDB_Rating', 'Rating IMDB'),
            ('Overview', 'Resumen'),
            ('Meta_score', 'Meta Score'),
            ('Director', 'Director'),
            ('Star1','Star 1'),
            ('Star2','Star 2'),
            ('Star3','Star 3'),
            ('Star4','Star 4'),
            ('No_of_votes','No. de votos'),
            ('Gross','Gross')
        ]
        displayed = False
        for colname, label in fields:
            if colname in movie_row.index and not pd.isna(movie_row[colname]) and str(movie_row[colname]).strip() != "":
                st.markdown(f"**{label}:** {movie_row[colname]}")
                displayed = True
        if not displayed:
            st.write("No hay información adicional disponible en el dataset para esta película.")

        st.markdown("---")
        st.subheader("Reseñas en la BD")
        if reviews_for_movie.empty:
            st.write("No se encontraron reseñas asociadas (tras normalizar títulos).")
        else:
            date_col = get_possible_date_column(reviews_for_movie)
            for idx, r in reviews_for_movie.iterrows():
                author = r.get('AUTHOR') or r.get('Author') or r.get('author') or r.get('NAME') or "Desconocido"
                rating = r.get('RATING') if 'RATING' in r.index else r.get('rating') if 'rating' in r.index else None
                st.markdown(f"**{author}**" + (f" — rating: {rating}" if rating is not None else ""))
                if date_col:
                    st.caption(f"Publicado: {r.get(date_col)}")
                # mostrar review con bloque
                review_text = r.get('REVIEW') or r.get('review') or ""
                st.write(review_text)
                st.markdown("---")

# --------------------------
# UI - Barra lateral
# --------------------------
st.sidebar.title("Configuración / Carga de datos")

sqlite_db_path = st.sidebar.text_input("Ruta a la BD de reseñas (.db)", value="./IMDB_Movies_2021.db")
use_kagglehub = st.sidebar.checkbox("Cargar dataset desde KaggleHub (si tienes kagglehub instalado)", value=False)

movies_df = None
reviews_df = None

if use_kagglehub:
    st.sidebar.markdown("Cargar dataset desde KaggleHub")
    dataset_ref = st.sidebar.text_input("Dataset Kaggle (owner/repo)", value="harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
    file_path = st.sidebar.text_input("file_path dentro del dataset (dejar vacío para el CSV por defecto)", value="")
    if st.sidebar.button("Cargar desde KaggleHub"):
        try:
            with st.spinner("Cargando desde KaggleHub ..."):
                movies_df = load_movies_from_kagglehub("PANDAS", dataset_ref, file_path)
            st.success("Dataset de películas cargado desde KaggleHub.")
        except Exception as e:
            st.error(f"No se pudo cargar desde KaggleHub: {e}")
            movies_df = None
else:
    st.sidebar.markdown("Cargar dataset local CSV")
    uploaded_file = st.sidebar.file_uploader("Sube el CSV del dataset de películas (si no, indica ruta local)", type=['csv'])
    csv_path = st.sidebar.text_input("Ruta local al CSV (opcional)", value="")
    if uploaded_file is not None:
        try:
            movies_df = load_movies_from_csv(uploaded_file)
            st.sidebar.success("CSV cargado desde uploader.")
        except Exception as e:
            st.sidebar.error(f"Error al leer CSV subido: {e}")
    elif csv_path.strip() != "":
        if os.path.exists(csv_path):
            try:
                movies_df = load_movies_from_csv(csv_path)
                st.sidebar.success("CSV cargado desde ruta local.")
            except Exception as e:
                st.sidebar.error(f"Error al leer CSV en ruta: {e}")
        else:
            st.sidebar.warning("La ruta indicada no existe.")

# Carga BD de reseñas (si se presiona)
if st.sidebar.button("Cargar reseñas desde BD"):
    try:
        with st.spinner("Cargando reseñas desde la base de datos..."):
            reviews_df = load_reviews_db(sqlite_db_path)
        st.sidebar.success("Reseñas cargadas desde BD.")
    except Exception as e:
        st.sidebar.error(f"No se pudo cargar la BD de reseñas: {e}")
        reviews_df = None

# Si no se cargó reviews_df aún, intentar cargar automáticamente (mejor UX)
if reviews_df is None:
    try:
        reviews_df = load_reviews_db(sqlite_db_path)
    except Exception:
        reviews_df = None

# --------------------------
# Validaciones y procesamiento
# --------------------------
if movies_df is None:
    st.info("Carga el dataset de películas (CSV local o desde KaggleHub) en la barra lateral para empezar.")
    st.stop()

if reviews_df is None:
    st.info("No se pudo cargar la BD de reseñas. Verifica la ruta en la barra lateral.")
    st.stop()

# Normalizar títulos de dataset de películas y de reseñas
# Nombres de columnas en el dataset de películas:
title_cols_movies = [c for c in movies_df.columns if c.lower() in ('series_title','series title','title','name')]
if title_cols_movies:
    movies_title_col = title_cols_movies[0]
else:
    # si no encuentra columna, intentar Series_Title
    movies_title_col = 'Series_Title' if 'Series_Title' in movies_df.columns else movies_df.columns[0]

# Nombres columnas en reseñas
title_cols_reviews = [c for c in reviews_df.columns if c.lower() in ('title','serie_title','series_title','movie','name')]
if title_cols_reviews:
    reviews_title_col = title_cols_reviews[0]
else:
    # intentar 'TITLE'
    reviews_title_col = 'TITLE' if 'TITLE' in reviews_df.columns else reviews_df.columns[0]

# Crear columna normalizada en ambos
movies_df['_title_norm'] = movies_df[movies_title_col].astype(str).str.strip().str.lower()
reviews_df['_title_norm'] = reviews_df[reviews_title_col].astype(str).str.strip().str.lower()

# Obtener títulos únicos que tienen reseñas
titles_with_reviews = set(reviews_df['_title_norm'].unique())

# Filtrar películas que estén en titles_with_reviews (match exacto normalizado)
matched_mask = movies_df['_title_norm'].isin(titles_with_reviews)
movies_with_reviews = movies_df[matched_mask].copy()

# Mostrar conteos
st.title("Películas que TIENEN reseña en la BD")
st.markdown(f"Total películas en dataset: **{len(movies_df)}**  — Películas con reseña encontradas (coincidencia exacta por título): **{len(movies_with_reviews)}**")

# Sidebar filters
st.sidebar.markdown("Filtros rápidos")
genre_col = None
for c in movies_df.columns:
    if c.lower() == 'genre':
        genre_col = c
        break
if genre_col:
    all_genres = sorted(set(",".join(movies_with_reviews[genre_col].dropna().astype(str).tolist()).split(",")))
    selected_genre = st.sidebar.selectbox("Filtrar por género (parcial)", options=["(todos)"] + all_genres, index=0)
else:
    selected_genre = "(todos)"

year_col = None
for c in movies_df.columns:
    if c.lower() in ('released_year','year'):
        year_col = c
        break
if year_col:
    years = sorted(movies_with_reviews[year_col].dropna().unique().tolist())
    years = [int(y) for y in years if pd.notna(y) and str(y).strip()!='']
    selected_year = st.sidebar.selectbox("Filtrar por año", options=["(todos)"] + sorted(set(years)), index=0)
else:
    selected_year = "(todos)"

search_text = st.sidebar.text_input("Buscar por texto en título o resumen (sensible a palabras)", value="")

# Aplicar filtros
filtered = movies_with_reviews.copy()
if selected_genre != "(todos)":
    # Filtrar si el género contiene la opción
    filtered = filtered[filtered[genre_col].astype(str).str.contains(selected_genre, na=False)]
if selected_year != "(todos)":
    filtered = filtered[filtered[year_col].astype(str) == str(selected_year)]
if search_text.strip() != "":
    mask_search = filtered[movies_title_col].astype(str).str.contains(search_text, case=False, na=False) | \
                  filtered.apply(lambda r: (str(r.get('Overview',''))).lower().find(search_text.lower())>=0, axis=1)
    filtered = filtered[mask_search]

st.markdown(f"Películas disponibles tras filtros: **{len(filtered)}**")

# Mostrar lista (mini tarjetas)
if filtered.empty:
    st.warning("No hay películas que cumplan los filtros y que además tengan reseñas en la BD.")
    st.stop()

# Selector o lista
with st.expander("Mostrar lista compacta de títulos (clic para navegar)"):
    # mostrar como tabla con poster pequeño
    sel = st.selectbox("Selecciona una película para ver detalles:", options=filtered.index.tolist(),
                       format_func=lambda i: f"{filtered.loc[i, movies_title_col]} ({filtered.loc[i, year_col] if year_col in filtered.columns else ''})")

# Mostrar tarjeta para la película seleccionada
selected_idx = sel
movie_row = filtered.loc[selected_idx]

# Obtener reseñas asociadas (por título normalizado)
movie_title_norm = movie_row['_title_norm']
reviews_for_movie = reviews_df[reviews_df['_title_norm'] == movie_title_norm].copy()

show_movie_card(movie_row, reviews_for_movie)

# También ofrecer descarga CSV de las reseñas y datos combinados
if not reviews_for_movie.empty:
    combined = reviews_for_movie.copy()
    # añadir columnas de película importantes
    for col in ['Poster_Link','Series_Title','Released_Year','IMDB_Rating','Director']:
        if col in movie_row.index:
            combined[col] = movie_row[col]
    csv = combined.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar reseñas + datos de la película (CSV)", data=csv, file_name=f"{movie_row[movies_title_col]}_reviews.csv", mime="text/csv")

st.markdown("---")
st.caption("Si los títulos no coinciden exactamente, puedo añadir emparejamiento difuso (fuzzy matching) para encontrar correspondencias aproximadas. Pídelo si lo necesitas.")