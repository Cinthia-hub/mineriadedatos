#!/usr/bin/env python3
"""
merge_reviews.py

Reconstruye site_data/movies_with_reviews.json uniendo site_data/movies.json
con las reseñas existentes (prefiere BD SQLite si existe, sino site_data/reviews.json).

Uso:
  python merge_reviews.py

Salida:
  site_data/movies_with_reviews.json
"""
import os, json, sqlite3, difflib, unicodedata, math
from pathlib import Path

SITE = Path("site_data")
SITE.mkdir(exist_ok=True)
MOVIES_JSON = SITE / "movies.json"
OUT_JSON = SITE / "movies_with_reviews.json"
DB_FILE = Path("IMDB_Movies_2021.db")

def _normalize_title(s):
    if s is None: return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.split('(')[0].split('-')[0].strip()
    s = s.strip(" .,:;-—")
    return s

def load_movies():
    if not MOVIES_JSON.exists():
        raise FileNotFoundError(f"{MOVIES_JSON} not found")
    with open(MOVIES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def load_reviews_from_db():
    if not DB_FILE.exists():
        return []
    conn = sqlite3.connect(str(DB_FILE))
    try:
        cur = conn.cursor()
        cur.execute("SELECT ID, AUTHOR, TITLE, REVIEW, RATING, POSTED FROM REVIEWS")
        rows = cur.fetchall()
        cols = ["id","author","title","review","rating","posted"]
        out = []
        for r in rows:
            obj = dict(zip(cols, r))
            out.append(obj)
        return out
    finally:
        conn.close()

def load_reviews_from_json():
    p = SITE / "reviews.json"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def find_best_match(title_norm, bd_keys, cutoff=0.80):
    if not bd_keys:
        return None
    if title_norm in bd_keys:
        return title_norm
    matches = difflib.get_close_matches(title_norm, bd_keys, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def main():
    movies = load_movies()
    # prepare movie norms
    for m in movies:
        m["_title_norm"] = _normalize_title(m.get("Series_Title") or "")

    # load reviews: prefer DB
    reviews = load_reviews_from_db()
    if not reviews:
        reviews = load_reviews_from_json()

    # build review map by normalized title
    rev_map = {}
    for r in reviews:
        title = r.get("title") or r.get("TITLE") or ""
        key = _normalize_title(title)
        rev_map.setdefault(key, []).append({
            "id": int(r.get("id") or 0),
            "author": r.get("author") or r.get("AUTHOR") or "Anon",
            "rating": (float(r.get("rating")) if r.get("rating") is not None else None),
            "review": r.get("review") or r.get("REVIEW") or "",
            "posted": r.get("posted") or r.get("POSTED") or ""
        })

    bd_keys = list(rev_map.keys())
    matched = 0
    for m in movies:
        tnorm = m.get("_title_norm","")
        match = find_best_match(tnorm, bd_keys, cutoff=0.80)
        if match:
            lst = rev_map.get(match, [])
            m["reviews"] = lst
            ratings = [r["rating"] for r in lst if r.get("rating") is not None]
            m["review_count"] = len(lst)
            m["avg_rating"] = float(sum(ratings)/len(ratings)) if ratings else None
            if ratings:
                try:
                    prod = math.prod([int(x) if float(x).is_integer() else float(x) for x in ratings])
                except Exception:
                    prod = 1
                    for rv in ratings:
                        try:
                            prod *= (int(rv) if float(rv).is_integer() else float(rv))
                        except Exception:
                            pass
                m["review_product"] = prod
                m["No_of_votes"] = prod
            else:
                m["review_product"] = None
            m["_matched_rev_key"] = match
            matched += 1
        else:
            m["reviews"] = []
            m["review_count"] = 0
            m["avg_rating"] = None
            m["review_product"] = None
            m["_matched_rev_key"] = None

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT_JSON} — matched {matched} movies with reviews (total reviews source: {len(reviews)})")

if __name__ == "__main__":
    main()