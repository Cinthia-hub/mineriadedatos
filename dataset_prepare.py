#!/usr/bin/env python3
"""
dataset_prepare.py

Descarga y prepara el dataset de Kaggle:
harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows

Salida: site_data/movies.json (lista de objetos con los campos solicitados)

Uso:
  pip install pandas kagglehub
  python dataset_prepare.py
"""
import os
import zipfile
import json
from pathlib import Path

import pandas as pd
import kagglehub

OUT_DIR = Path("site_data")
OUT_DIR.mkdir(exist_ok=True)
OUT_JSON = OUT_DIR / "movies.json"

print("Descargando dataset desde Kaggle...")
download_path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
print("Ruta:", download_path)

def find_csv_in_path(p: Path):
    # Busca recursivamente CSVs o zips que contengan CSVs
    csvs = list(p.rglob("*.csv"))
    if csvs:
        # Priorizar nombres esperados
        preferred = [c for c in csvs if any(x in c.name.lower() for x in ("imdb", "top", "movies", "tv", "dataset"))]
        if preferred:
            return sorted(preferred, key=lambda x: x.stat().st_size, reverse=True)[0]
        # si no hay preferidos, devolver el csv mas grande
        return sorted(csvs, key=lambda x: x.stat().st_size, reverse=True)[0]
    # si no hay csvs, buscar zips que posiblemente contengan csv
    zips = list(p.rglob("*.zip"))
    if zips:
        # extraer el primero en OUT_DIR/extracted_zip
        z = zips[0]
        extract_dir = OUT_DIR / "extracted_from_zip"
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Encontré zip dentro del directorio: {z}. Extrayendo a {extract_dir}...")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(extract_dir)
        return find_csv_in_path(extract_dir)
    return None

csv_path = None
p = Path(download_path)

if p.exists():
    if p.is_file():
        # archivo individual (puede ser zip o csv)
        if zipfile.is_zipfile(p):
            extract_dir = OUT_DIR / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            print("Es un zip: extrayendo...", p)
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(extract_dir)
            csv_path = find_csv_in_path(extract_dir)
        elif p.suffix.lower() == ".csv":
            csv_path = p
        else:
            # intentar buscar CSVs en la misma carpeta del archivo
            csv_path = find_csv_in_path(p.parent)
    elif p.is_dir():
        # si es un directorio, buscar recursivamente
        print("La ruta es un directorio. Buscando CSVs recursivamente dentro de:", p)
        # listar contenidos para depuración
        try:
            contents = list(p.iterdir())
            print("Contenido directo del directorio:", [c.name for c in contents])
        except Exception:
            pass
        csv_path = find_csv_in_path(p)

if not csv_path:
    # También intentar buscar CSV en el parent del path (por si el downloader creó estructura diferente)
    parent = p.parent
    print("Intentando buscar CSV en el directorio padre:", parent)
    csv_path = find_csv_in_path(parent) if parent.exists() else None

if not csv_path:
    raise FileNotFoundError("No pude encontrar un archivo CSV en el contenido descargado. Revisa la ruta de descarga e inspecciona el directorio impreso arriba.")

print("Usando CSV:", csv_path)

df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

# Normalizar nombres de columnas (algunos datasets usan nombres distintos)
col_map = {
    "Poster_Link": "Poster_Link",
    "Poster Link": "Poster_Link",
    "Poster": "Poster_Link",
    "Series_Title": "Series_Title",
    "Title": "Series_Title",
    "Released_Year": "Released_Year",
    "Year": "Released_Year",
    "Certificate": "Certificate",
    "Runtime": "Runtime",
    "Genre": "Genre",
    "IMDB_Rating": "IMDB_Rating",
    "IMDB Rating": "IMDB_Rating",
    "Overview": "Overview",
    "Summary": "Overview",
    "Meta_score": "Meta_score",
    "Meta Score": "Meta_score",
    "Director": "Director",
    "Star1": "Star1",
    "Star 1": "Star1",
    "Star2": "Star2",
    "Star 2": "Star2",
    "Star3": "Star3",
    "Star 3": "Star3",
    "Star4": "Star4",
    "Star 4": "Star4",
    "No_of_votes": "No_of_votes",
    "Votes": "No_of_votes",
    "Gross": "Gross",
    "Gross ($)": "Gross",
}

rename = {}
for src, dst in col_map.items():
    if src in df.columns and dst not in df.columns:
        rename[src] = dst

if rename:
    df = df.rename(columns=rename)

fields = [
    "Poster_Link",
    "Series_Title",
    "Released_Year",
    "Certificate",
    "Runtime",
    "Genre",
    "IMDB_Rating",
    "Overview",
    "Meta_score",
    "Director",
    "Star1",
    "Star2",
    "Star3",
    "Star4",
    "No_of_votes",
    "Gross",
]

for f in fields:
    if f not in df.columns:
        df[f] = None

def parse_runtime(x):
    try:
        if pd.isna(x):
            return None
        s = str(x)
        digits = ''.join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else None
    except:
        return None

df["Runtime"] = df["Runtime"].apply(parse_runtime)

for col in ("IMDB_Rating", "Meta_score"):
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["No_of_votes"] = pd.to_numeric(df["No_of_votes"], errors="coerce")

def parse_money(x):
    try:
        if pd.isna(x):
            return None
        s = str(x).replace("$", "").replace(",", "").strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except:
        return None

df["Gross"] = df["Gross"].apply(parse_money)
df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce").astype("Int64")

records = []
for _, row in df.iterrows():
    rec = {k: (None if pd.isna(row[k]) else (int(row[k]) if k=="Released_Year" and not pd.isna(row[k]) else row[k])) for k in fields}
    records.append(rec)

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Generado {OUT_JSON} con {len(records)} registros.")
print("Coloca index.html, styles.css, app.js y la carpeta site_data en el servidor estático y abre http://localhost:8000")