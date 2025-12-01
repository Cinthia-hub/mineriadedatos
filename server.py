#!/usr/bin/env python3
"""
server.py - Flask server that persists reviews and regenerates the PDF by invoking
grafos.py via subprocess (always uses subprocess to avoid matplotlib/thread issues).

Endpoints:
- POST /api/add_review
- POST /api/delete_review
- POST /api/edit_review  <-- nuevo: editar reseña (JSON + DB opcional) y regenerar
- GET  /api/pdf_status
- GET/POST /api/regen
- GET  /api/reviews
- Static file serving for UI

Run:
  python server.py
"""
import os
import sqlite3
import threading
import subprocess
import traceback
import logging
import json
from datetime import datetime
from pathlib import Path
import sys

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

APP = Flask(__name__, static_folder=None)
CORS(APP)

# Config
DB_PATH = Path("IMDB_Movies_2021.db")
REVIEWS_TABLE = "REVIEWS"
PDF_PATH = Path("site_data/model_results.pdf")
GRAFOS_SCRIPT = Path("grafos.py")
LOG_FILE = Path("server.log")
SITE_DIR = Path("site_data")
MOVIES_WITH_REVIEWS = SITE_DIR / "movies_with_reviews.json"
MOVIES_JSON = SITE_DIR / "movies.json"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("server")

# Regeneration state (thread-safe)
_regen_lock = threading.Lock()
_regen_flag = {"running": False, "started_at": None, "last_error": None}

# File lock for JSON edits
_file_lock = threading.Lock()


# ----------------------------
# DB helpers (ensure table & insert/delete/update)
# ----------------------------
def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols_info = cur.fetchall()
    cols = [row[1].upper() for row in cols_info]
    return column.upper() in cols

def ensure_db_and_table():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {REVIEWS_TABLE} (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                AUTHOR TEXT,
                TITLE TEXT,
                REVIEW TEXT,
                RATING REAL,
                POSTED TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        try:
            if not _table_has_column(conn, REVIEWS_TABLE, "POSTED"):
                logger.info("Adding POSTED column to REVIEWS (no DEFAULT to keep compatibility).")
                cur.execute(f"ALTER TABLE {REVIEWS_TABLE} ADD COLUMN POSTED TIMESTAMP")
                conn.commit()
                try:
                    cur.execute(f"UPDATE {REVIEWS_TABLE} SET POSTED = CURRENT_TIMESTAMP WHERE POSTED IS NULL")
                    conn.commit()
                except Exception:
                    logger.warning("Could not populate POSTED column for existing rows.")
        except Exception as e:
            logger.exception(f"Could not check/add POSTED column: {e}")
    finally:
        conn.close()

def insert_review(author: str, title: str, review: str, rating):
    ensure_db_and_table()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        has_posted = _table_has_column(conn, REVIEWS_TABLE, "POSTED")
        if has_posted:
            cur.execute(f"""
                INSERT INTO {REVIEWS_TABLE} (AUTHOR, TITLE, REVIEW, RATING, POSTED)
                VALUES (?, ?, ?, ?, ?)
            """, (author, title, review, rating, datetime.utcnow().isoformat()))
        else:
            cur.execute(f"""
                INSERT INTO {REVIEWS_TABLE} (AUTHOR, TITLE, REVIEW, RATING)
                VALUES (?, ?, ?, ?)
            """, (author, title, review, rating))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()

def delete_review_from_db(rowid):
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(f"DELETE FROM {REVIEWS_TABLE} WHERE ID = ?", (rowid,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()

def update_review_in_db(rowid, author=None, review=None, rating=None):
    """
    Actualiza fila en la tabla REVIEWS por ID. Devuelve True si afectó fila.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        fields = []
        params = []
        if author is not None:
            fields.append("AUTHOR = ?")
            params.append(author)
        if review is not None:
            fields.append("REVIEW = ?")
            params.append(review)
        if rating is not None:
            fields.append("RATING = ?")
            params.append(rating)
        # actualizar POSTED a ahora para reflejar edición
        fields.append("POSTED = ?")
        params.append(datetime.utcnow().isoformat())
        if not fields:
            return False
        params.append(int(rowid))
        sql = f"UPDATE {REVIEWS_TABLE} SET {', '.join(fields)} WHERE ID = ?"
        cur.execute(sql, tuple(params))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ----------------------------
# JSON movies helpers
# ----------------------------
def _load_movies_json():
    path = MOVIES_WITH_REVIEWS if MOVIES_WITH_REVIEWS.exists() else MOVIES_JSON if MOVIES_JSON.exists() else None
    if path is None:
        return None, None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return path, data
    except Exception as e:
        logger.exception(f"Could not load {path}: {e}")
        return None, None

def _atomic_write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def add_review_to_movies_json(movie_index: int, review_obj: dict):
    with _file_lock:
        try:
            if not MOVIES_WITH_REVIEWS.exists() and MOVIES_JSON.exists():
                logger.info("Creating movies_with_reviews.json from movies.json")
                try:
                    with MOVIES_JSON.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    with MOVIES_WITH_REVIEWS.open("w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.exception("Failed to create movies_with_reviews.json:")
                    return False, str(e)
        except Exception as e:
            logger.exception("Error while ensuring movies_with_reviews.json exists:")
            return False, str(e)

        path = MOVIES_WITH_REVIEWS if MOVIES_WITH_REVIEWS.exists() else (MOVIES_JSON if MOVIES_JSON.exists() else None)
        if path is None:
            return False, "no_movies_json"
        try:
            with path.open("r", encoding="utf-8") as f:
                movies = json.load(f)
        except Exception as e:
            logger.exception(f"Could not load {path}: {e}")
            return False, str(e)

        if not (0 <= movie_index < len(movies)):
            logger.warning(f"movie_index out of range: {movie_index}")
            return False, "index_oob"

        movie = movies[movie_index]
        if not isinstance(movie.get("user_reviews"), list):
            movie["user_reviews"] = []
        try:
            existing_ids = {int(x.get("id")) for x in movie["user_reviews"] if x.get("id") is not None}
        except Exception:
            existing_ids = set()
        if review_obj.get("id") is None or int(review_obj.get("id")) in existing_ids:
            review_obj["id"] = int(datetime.utcnow().timestamp() * 1000)
        movie["user_reviews"].append(review_obj)

        try:
            _atomic_write_json(path, movies)
            logger.info(f"Added review to JSON movies index={movie_index}, path={path}")
            return True, None
        except Exception as e:
            logger.exception("Error writing movies JSON:")
            return False, str(e)

def remove_review_from_movies_json(movie_index: int, review_id):
    with _file_lock:
        path, movies = _load_movies_json()
        if path is None or movies is None:
            return False, "no_movies_json"
        if not (0 <= movie_index < len(movies)):
            return False, "index_oob"
        movie = movies[movie_index]
        if not isinstance(movie.get("user_reviews"), list):
            return False, "no_user_reviews"
        old_len = len(movie["user_reviews"])
        movie["user_reviews"] = [r for r in movie["user_reviews"] if str(r.get("id")) != str(review_id)]
        if len(movie["user_reviews"]) == old_len:
            return False, "not_found"
        try:
            _atomic_write_json(path, movies)
            logger.info(f"Removed review id={review_id} from JSON movies index={movie_index}, path={path}")
            return True, None
        except Exception as e:
            logger.exception("Error writing movies JSON:")
            return False, str(e)

def update_review_in_movies_json(movie_index: int, review_id, updates: dict):
    """
    Busca la reseña por movie_index y review_id dentro de user_reviews y aplica los campos de 'updates'.
    Devuelve (True, None) si actualizó, (False, error) si no.
    """
    with _file_lock:
        path, movies = _load_movies_json()
        if path is None or movies is None:
            return False, "no_movies_json"
        if not (0 <= movie_index < len(movies)):
            return False, "index_oob"
        movie = movies[movie_index]
        if not isinstance(movie.get("user_reviews"), list):
            return False, "no_user_reviews"
        found = False
        for r in movie["user_reviews"]:
            if str(r.get("id")) == str(review_id):
                # aplicar actualizaciones (solo campos permitidos)
                if "author" in updates:
                    r["author"] = updates["author"]
                if "review" in updates:
                    r["review"] = updates["review"]
                if "rating" in updates:
                    try:
                        r["rating"] = float(updates["rating"]) if updates["rating"] is not None else None
                    except Exception:
                        r["rating"] = updates["rating"]
                # actualizar posted para indicar edición
                r["posted"] = datetime.utcnow().isoformat()
                found = True
                break
        if not found:
            return False, "not_found"
        try:
            _atomic_write_json(path, movies)
            logger.info(f"Updated review id={review_id} in JSON movies index={movie_index}, path={path}")
            return True, None
        except Exception as e:
            logger.exception("Error writing movies JSON (update):")
            return False, str(e)


# ----------------------------
# Regeneration: always use subprocess (recommended)
# ----------------------------
def _run_regeneration_in_thread():
    def _job():
        global _regen_flag
        with _regen_lock:
            if _regen_flag["running"]:
                logger.info("Regeneration already running; skipping new start.")
                return
            _regen_flag["running"] = True
            _regen_flag["started_at"] = datetime.utcnow().isoformat()
            _regen_flag["last_error"] = None

        logger.info("Starting PDF regeneration (subprocess thread).")
        try:
            python_exec = os.environ.get("PYTHON_EXECUTABLE") or sys.executable
            cmd = [python_exec, str(GRAFOS_SCRIPT)]
            logger.info("Launching subprocess: " + " ".join(cmd))
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(Path.cwd()))
                logger.info(f"Subprocess returncode: {proc.returncode}")
                if proc.stdout:
                    logger.info("grafos.py stdout:\n" + proc.stdout)
                if proc.stderr:
                    logger.error("grafos.py stderr:\n" + proc.stderr)
                if proc.returncode != 0:
                    _regen_flag["last_error"] = proc.stderr or proc.stdout or f"returncode={proc.returncode}"
            except Exception as e:
                logger.exception("Error running grafos.py subprocess:")
                _regen_flag["last_error"] = traceback.format_exc()

        except Exception:
            logger.exception("Unexpected error during regeneration:")
            _regen_flag["last_error"] = traceback.format_exc()
        finally:
            with _regen_lock:
                _regen_flag["running"] = False
            logger.info("PDF regeneration finished (flag updated).")

    thread = threading.Thread(target=_job, daemon=True)
    thread.start()


# ----------------------------
# API endpoints
# ----------------------------
@APP.route("/api/add_review", methods=["POST"])
def api_add_review():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    title = data.get("title") or data.get("movie_title") or data.get("movie") or ""
    author = data.get("author") or "Anon"
    review = data.get("review") or ""
    rating = data.get("rating", None)
    movie_index = data.get("movie_index", None)

    if not title or not review:
        return jsonify({"ok": False, "error": "Missing 'title' or 'review'"}), 400

    logger.info(f"POST /api/add_review - title='{title[:80]}' author='{author}' rating={rating} movie_index={movie_index}")

    try:
        rowid = insert_review(author, title, review, rating)
    except Exception as e:
        logger.exception("Failed to insert into DB:")
        return jsonify({"ok": False, "error": f"DB insert failed: {e}"}), 500

    json_added = False
    json_error = None
    if movie_index is not None:
        try:
            rev_obj = {
                "id": rowid,
                "author": author,
                "rating": float(rating) if rating is not None else None,
                "review": review,
                "posted": datetime.utcnow().isoformat(),
                "db_rowid": rowid
            }
            ok, err = add_review_to_movies_json(int(movie_index), rev_obj)
            json_added = ok
            json_error = err
            if not ok:
                logger.warning(f"Could not update movies JSON: {err}")
        except Exception as e:
            logger.exception("Error updating movies JSON:")
            json_error = str(e)

    with _regen_lock:
        already_running = _regen_flag["running"]

    if not already_running:
        logger.info("Triggering regeneration in background (subprocess).")
        _run_regeneration_in_thread()
        started = True
    else:
        logger.info("Regeneration already running; not starting another.")
        started = False

    resp = {"ok": True, "rowid": rowid, "regeneration_started": started, "json_added": json_added}
    if json_error:
        resp["json_error"] = json_error
    return jsonify(resp)


@APP.route("/api/delete_review", methods=["POST"])
def api_delete_review():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    movie_index = data.get("movie_index")
    review_id = data.get("review_id")
    db_rowid = data.get("db_rowid", None)

    if movie_index is None or review_id is None:
        return jsonify({"ok": False, "error": "Missing 'movie_index' or 'review_id'"}), 400

    logger.info(f"POST /api/delete_review - movie_index={movie_index} review_id={review_id} db_rowid={db_rowid}")

    json_ok, json_err = remove_review_from_movies_json(int(movie_index), review_id)

    db_deleted = False
    db_err = None
    if db_rowid:
        try:
            deleted = delete_review_from_db(int(db_rowid))
            db_deleted = deleted > 0
            if not db_deleted:
                db_err = "no_row_deleted"
        except Exception as e:
            logger.exception("Error deleting review from DB:")
            db_err = str(e)

    if not json_ok and not db_deleted:
        return jsonify({"ok": False, "json_ok": json_ok, "json_err": json_err, "db_deleted": db_deleted, "db_err": db_err}), 500

    with _regen_lock:
        already_running = _regen_flag["running"]

    if not already_running:
        logger.info("No hay regeneración en curso: se iniciará una nueva tras eliminación.")
        _run_regeneration_in_thread()
        regen_started = True
    else:
        logger.info("Regeneración ya en curso: no se inicia otra tras eliminación.")
        regen_started = False

    resp = {"ok": True, "json_ok": json_ok, "db_deleted": db_deleted, "regeneration_started": regen_started}
    if json_err:
        resp["json_err"] = json_err
    if db_err:
        resp["db_err"] = db_err
    return jsonify(resp)


@APP.route("/api/edit_review", methods=["POST"])
def api_edit_review():
    """
    Edita una reseña existente.
    Request JSON expected:
    {
      "movie_index": int,
      "review_id": <id in JSON user_reviews>,
      "db_rowid": <optional DB row id>,
      "author": <optional>,
      "rating": <optional>,
      "review": <optional>
    }
    Returns json with flags about JSON/DB update and whether regeneration started.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    movie_index = data.get("movie_index")
    review_id = data.get("review_id")
    db_rowid = data.get("db_rowid", None)
    author = data.get("author", None)
    rating = data.get("rating", None)
    review_text = data.get("review", None)

    if movie_index is None or review_id is None:
        return jsonify({"ok": False, "error": "Missing 'movie_index' or 'review_id'"}), 400

    logger.info(f"POST /api/edit_review - movie_index={movie_index} review_id={review_id} db_rowid={db_rowid}")

    json_ok = False
    json_err = None
    try:
        updates = {}
        if author is not None: updates["author"] = author
        if review_text is not None: updates["review"] = review_text
        if rating is not None: updates["rating"] = rating
        json_ok, json_err = update_review_in_movies_json(int(movie_index), review_id, updates)
        if not json_ok:
            logger.warning(f"Could not update movies JSON: {json_err}")
    except Exception as e:
        logger.exception("Error updating movies JSON:")
        json_ok = False
        json_err = str(e)

    db_updated = False
    db_err = None
    if db_rowid:
        try:
            db_updated = update_review_in_db(int(db_rowid), author=author, review=review_text, rating=rating)
            if not db_updated:
                db_err = "no_row_updated"
        except Exception as e:
            logger.exception("Error updating review in DB:")
            db_err = str(e)

    if not json_ok and not db_updated:
        return jsonify({"ok": False, "json_ok": json_ok, "json_err": json_err, "db_updated": db_updated, "db_err": db_err}), 500

    # Lanzar regeneración en background si hubo cambio
    with _regen_lock:
        already_running = _regen_flag["running"]

    if not already_running:
        logger.info("No hay regeneración en curso: se iniciará una nueva tras edición.")
        _run_regeneration_in_thread()
        regen_started = True
    else:
        logger.info("Regeneración ya en curso: no se inicia otra tras edición.")
        regen_started = False

    resp = {"ok": True, "json_ok": json_ok, "db_updated": db_updated, "regeneration_started": regen_started}
    if json_err:
        resp["json_err"] = json_err
    if db_err:
        resp["db_err"] = db_err
    return jsonify(resp)


@APP.route("/api/pdf_status", methods=["GET"])
def api_pdf_status():
    exists = PDF_PATH.exists()
    mtime = None
    try:
        if exists:
            mtime = datetime.utcfromtimestamp(PDF_PATH.stat().st_mtime).isoformat()
    except Exception:
        mtime = None
    with _regen_lock:
        regenerating = _regen_flag["running"]
        last_error = _regen_flag.get("last_error")
    return jsonify({
        "exists": exists,
        "mtime": mtime,
        "regenerating": regenerating,
        "last_error": last_error
    })


@APP.route("/api/reviews", methods=["GET"])
def api_get_reviews():
    title = request.args.get("title", "").strip()
    ensure_db_and_table()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        if _table_has_column(conn, REVIEWS_TABLE, "POSTED"):
            posted_expr = "POSTED"
        else:
            posted_expr = "NULL as POSTED"
        if title:
            cur.execute(f"SELECT ID, AUTHOR, TITLE, REVIEW, RATING, {posted_expr} FROM {REVIEWS_TABLE} WHERE lower(TITLE)=? ORDER BY ID DESC", (title.lower(),))
        else:
            cur.execute(f"SELECT ID, AUTHOR, TITLE, REVIEW, RATING, {posted_expr} FROM {REVIEWS_TABLE} ORDER BY ID DESC LIMIT 200")
        rows = cur.fetchall()
        cols = ["id", "author", "title", "review", "rating", "posted"]
        data = [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()
    return jsonify({"ok": True, "reviews": data})


@APP.route("/api/regen", methods=["GET", "POST"])
def api_regen():
    with _regen_lock:
        if _regen_flag["running"]:
            return jsonify({"ok": False, "message": "Regeneration already running"}), 409
    logger.info("/api/regen called - starting regeneration in background (subprocess).")
    _run_regeneration_in_thread()
    return jsonify({"ok": True, "message": "Regeneration started"})


# Static file serving for UI and assets
@APP.route('/', defaults={'req_path': ''})
@APP.route('/<path:req_path>')
def serve_ui(req_path):
    if req_path.startswith('api'):
        return jsonify({"error": "API endpoint not found"}), 404

    if req_path == "" or req_path is None:
        target = "index.html"
    else:
        target = req_path

    target_path = Path(target)
    if target_path.exists() and target_path.is_file():
        return send_from_directory('.', target)

    if target.startswith('site_data/') or target.lower().endswith('.pdf') or target.startswith('site_data\\'):
        return jsonify({"error": f"Static asset not found: {target}"}), 404

    index = Path("index.html")
    if index.exists():
        return send_from_directory('.', "index.html")
    return jsonify({"error": "index.html not found in project directory"}), 404


if __name__ == "__main__":
    ensure_db_and_table()
    logger.info("Server starting. Static files served from: %s", os.getcwd())

    force = os.environ.get("FORCE_REGEN_ON_START", "").lower() in ("1", "true", "yes")
    pdf_exists = PDF_PATH.exists()
    if force or not pdf_exists:
        logger.info(f"Triggering regeneration on startup (force={force}, pdf_exists={pdf_exists})")
        _run_regeneration_in_thread()
    else:
        logger.info("PDF exists and FORCE_REGEN_ON_START not set: not regenerating on startup.")

    APP.run(host="0.0.0.0", port=5000, debug=False)