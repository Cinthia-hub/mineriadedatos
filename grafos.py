#!/usr/bin/env python3
"""
grafos.py - Generate PDF reports using ONLY the reviews present in
site_data/movies_with_reviews.json (preferred) or site_data/movies.json.

This version writes the report to a temporary file and atomically replaces
site_data/model_results.pdf when the PDF is fully generated. This avoids
leaving no file available during regeneration (prevents 404s between delete & write).
"""
import os
import sqlite3
import re
import warnings
import unicodedata
from collections import OrderedDict, Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve,
                             auc, precision_recall_curve, classification_report)

from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set(style="whitegrid", rc={"figure.figsize": (8, 6)})

# ----------------------------
# NLTK resources & text utils
# ----------------------------
def ensure_nltk_resources():
    resources = ['punkt', 'stopwords', 'vader_lexicon']
    for res in resources:
        try:
            if res == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif res == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif res == 'vader_lexicon':
                nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            try:
                nltk.download(res)
            except Exception:
                pass

stemmer = SnowballStemmer("english")

def tokenization_and_stemming(text, stopwords_set):
    if not isinstance(text, str):
        text = str(text or "")
    tokens = []
    try:
        for word in nltk.word_tokenize(text):
            w = word.lower()
            if w not in stopwords_set:
                tokens.append(w)
    except Exception:
        tokens = re.findall(r"[A-Za-z]+", text.lower())
        tokens = [t for t in tokens if t not in stopwords_set]
    filtered = [t for t in tokens if t.isalpha()]
    stems = [stemmer.stem(t) for t in filtered if t not in stopwords_set]
    return stems

def _normalize_title(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.split('(')[0].split('-')[0].strip()
    s = s.strip(" .,:;-—")
    return s

# ----------------------------
# Plot helpers
# ----------------------------
def save_figure_to_pdf(pp, fig, tight=True):
    """
    Save fig into PdfPages. Use bbox_inches='tight' by default to avoid cropping
    and ensure the figure is as large as it needs to be on the page.
    """
    try:
        if tight:
            pp.savefig(fig, bbox_inches='tight')  # \x1b[33mMODIFIED\x1b[0m >>> línea modificada: usé bbox_inches='tight'
        else:
            pp.savefig(fig)
    finally:
        plt.close(fig)

def plot_top_movies_diagram(movies_df=None, top_n=10):
    try:
        if movies_df is None or movies_df.empty:
            return None

        rows = []
        for idx, row in movies_df.iterrows():
            title = row.get('Series_Title') or row.get('series_title') or row.get('Series Title') or f"Untitled {idx}"
            ratings = []
            ur = row.get('user_reviews') or []
            if isinstance(ur, list):
                for u in ur:
                    try:
                        r = u.get('rating') if isinstance(u, dict) else None
                        if r is None:
                            continue
                        ratings.append(float(r))
                    except Exception:
                        continue
            rr = row.get('reviews') or []
            if isinstance(rr, list):
                for u in rr:
                    try:
                        r = u.get('rating') if isinstance(u, dict) else None
                        if r is None:
                            continue
                        ratings.append(float(r))
                    except Exception:
                        continue
            for key in ('positive_review','negative_review'):
                pr = row.get(key)
                if isinstance(pr, dict):
                    try:
                        r = pr.get('rating')
                        if r is not None:
                            ratings.append(float(r))
                    except Exception:
                        pass
            if ratings:
                avg = float(sum(ratings)) / len(ratings)
                rows.append((title, avg))
        if not rows:
            return None

        df = pd.DataFrame(rows, columns=['title','avg_rating'])
        df = df.sort_values('avg_rating', ascending=False).reset_index(drop=True)
        df_top = df.head(top_n).copy()
        n = len(df_top)
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        y_pos = np.arange(n)
        ax.barh(y_pos, df_top['avg_rating'], align='center', color='tab:blue', alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_top['title'].tolist(), fontsize=9)
        ax.set_xlabel("Rating (promedio de reseñas)")
        ax.set_xlim(0, 10)
        ax.set_title(f"Top {n} películas (rating por promedio de reseñas)")
        ax.invert_yaxis()
        plt.tight_layout()
        return fig
    except Exception as e:
        print("Error generating top movies diagram:", e)
        return None

def plot_confusion_matrix(y_true, y_pred, labels, title=None):
    fig, ax = plt.subplots(figsize=(6,5))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    ax.set_title(title or "Confusion Matrix")
    return fig

def plot_roc_curves(models_scores, y_true, title="ROC Curves"):
    fig, ax = plt.subplots(figsize=(7,6))
    for name, score_info in models_scores.items():
        y_score = score_info.get('score')
        if y_score is None or len(y_score) != len(y_true):
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],'k--', lw=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc='lower right')
    return fig

def plot_precision_recall(models_scores, y_true, title="Precision-Recall Curves"):
    fig, ax = plt.subplots(figsize=(7,6))
    for name, score_info in models_scores.items():
        y_score = score_info.get('score')
        if y_score is None or len(y_score) != len(y_true):
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc='best')
    return fig

def plot_metric_bars(metrics_dict, title="Model comparison"):
    """
    Create a bar chart of metrics. The figure size is chosen dynamically
    depending on the number of models, so the plot uses the space required.
    """
    if not metrics_dict:
        return None
    df = pd.DataFrame(metrics_dict).T
    keep = [c for c in ['accuracy','precision','recall','f1'] if c in df.columns]
    if not keep:
        return None
    # dynamic sizing: width scales with number of models
    n_models = max(1, df.shape[0])
    width = max(8, n_models * 1.5)  # \x1b[33mMODIFIED\x1b[0m >>> línea modificada: tamaño dinámico según # de modelos
    fig, ax = plt.subplots(figsize=(width, 6))
    df[keep].plot(kind='bar', ax=ax)
    ax.set_title(title)
    ax.set_ylim(0,1)
    ax.legend(loc='lower right')
    plt.tight_layout()
    return fig

# ----------------------------
# Sentiment plots (existing)
# ----------------------------
def compute_sentiment_labels(df_reviews):
    if df_reviews is None or df_reviews.empty:
        return df_reviews.copy(), None
    sia = SentimentIntensityAnalyzer()
    df = df_reviews.copy()
    compounds = []
    labels = []
    for txt in df['review'].astype(str).tolist():
        try:
            sc = sia.polarity_scores(txt).get('compound', 0.0)
        except Exception:
            sc = 0.0
        compounds.append(float(sc))
        if sc >= 0.05:
            labels.append('positive')
        elif sc <= -0.05:
            labels.append('negative')
        else:
            labels.append('neutral')
    df['sentiment_compound'] = compounds
    df['sentiment'] = labels
    return df, Counter(labels)

def plot_sentiment_distribution_overall(df_reviews_with_sentiment):
    if df_reviews_with_sentiment is None or df_reviews_with_sentiment.empty:
        return None
    counts = df_reviews_with_sentiment['sentiment'].value_counts().reindex(['positive','neutral','negative']).fillna(0)
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=counts.index, y=counts.values, palette=['#2ca02c','#7f7f7f','#d62728'], ax=ax)
    ax.set_title("Distribución de polaridad (reseñas JSON)")
    ax.set_ylabel("Número de reseñas")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(1, total*0.01), f"{int(v)}", ha='center')
    plt.tight_layout()
    return fig

def plot_sentiment_pie(df_reviews_with_sentiment):
    if df_reviews_with_sentiment is None or df_reviews_with_sentiment.empty:
        return None
    counts = df_reviews_with_sentiment['sentiment'].value_counts().reindex(['positive','neutral','negative']).fillna(0)
    labels = [f"{lab} ({int(cnt)})" for lab,cnt in zip(counts.index, counts.values)]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(counts.values, labels=labels, colors=['#2ca02c','#7f7f7f','#d62728'], autopct='%1.1f%%', startangle=140)
    ax.set_title("Porcentaje de polaridad (reseñas JSON)")
    plt.tight_layout()
    return fig

def plot_sentiment_by_top_movies(movies_df, df_reviews_with_sentiment, top_n=10):
    if movies_df is None or movies_df.empty or df_reviews_with_sentiment is None or df_reviews_with_sentiment.empty:
        return None
    grp = df_reviews_with_sentiment.groupby('movie_idx')['sentiment'].value_counts().unstack(fill_value=0)
    grp['total'] = grp.sum(axis=1)
    top_idx = grp.sort_values('total', ascending=False).head(top_n).index.tolist()
    if not top_idx:
        return None
    sub = grp.loc[top_idx, ['positive','neutral','negative']].fillna(0)
    titles = []
    for mi in sub.index:
        try:
            t = movies_df.iloc[int(mi)].get('Series_Title') or movies_df.iloc[int(mi)].get('series_title') or ""
        except Exception:
            t = str(mi)
        titles.append(t)
    sub.index = titles
    fig, ax = plt.subplots(figsize=(9, max(4, 0.6*len(titles))))
    sub[['positive','neutral','negative']].plot(kind='barh', stacked=True, color=['#2ca02c','#7f7f7f','#d62728'], ax=ax)
    ax.set_xlabel("Número de reseñas")
    ax.set_ylabel("Película")
    ax.set_title(f"Polaridad por película (top {len(titles)} por #reseñas en JSON)")
    plt.tight_layout()
    return fig

# ----------------------------
# NEW: Boxplot diagrams (diagramas de bigotes)
# ----------------------------
def plot_rating_boxplots_overall_and_by_sentiment(df_reviews_with_sentiment):
    """
    Two-panel figure:
     - Left: boxplot of numeric ratings overall
     - Right: boxplot of numeric ratings grouped by sentiment (positive/neutral/negative)
    Requires df_reviews_with_sentiment to contain numeric 'rating' and 'sentiment'.
    """
    if df_reviews_with_sentiment is None or df_reviews_with_sentiment.empty:
        return None
    # Filter numeric ratings
    df = df_reviews_with_sentiment.copy()
    df['rating_num'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[~df['rating_num'].isna()].copy()
    if df.empty:
        return None

    # order sentiments for consistent coloring
    order = ['positive', 'neutral', 'negative']
    palette = {'positive': '#2ca02c', 'neutral': '#7f7f7f', 'negative': '#d62728'}

    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    # Overall boxplot
    sns.boxplot(x=df['rating_num'], ax=axes[0], color='tab:blue')
    axes[0].set_xlabel("Rating")
    axes[0].set_title("Distribución de calificaciones (boxplot) - global")
    axes[0].set_xlim(0, 10)

    # By sentiment
    sns.boxplot(x='sentiment', y='rating_num', data=df, order=order, palette=[palette.get(o) for o in order], ax=axes[1])
    axes[1].set_xlabel("Polaridad")
    axes[1].set_ylabel("Rating")
    axes[1].set_title("Distribución de calificaciones por polaridad (boxplot)")
    axes[1].set_ylim(0, 10)

    plt.tight_layout()
    return fig

def plot_rating_boxplot_by_top_movies(movies_df, df_reviews_with_sentiment, top_n=8):
    """
    Boxplot of ratings for the top-N movies by number of JSON reviews.
    Each box shows the distribution of numeric ratings for that movie.
    """
    if movies_df is None or movies_df.empty or df_reviews_with_sentiment is None or df_reviews_with_sentiment.empty:
        return None
    df = df_reviews_with_sentiment.copy()
    df['rating_num'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[~df['rating_num'].isna()].copy()
    if df.empty:
        return None

    counts = df.groupby('movie_idx').size().sort_values(ascending=False)
    top_idx = counts.head(top_n).index.tolist()
    if not top_idx:
        return None

    # build plot DataFrame with movie titles
    rows = []
    titles = []
    for mi in top_idx:
        try:
            title = movies_df.iloc[int(mi)].get('Series_Title') or movies_df.iloc[int(mi)].get('series_title') or ""
        except Exception:
            title = str(mi)
        sub = df[df['movie_idx'] == mi]
        if sub.empty:
            continue
        for v in sub['rating_num'].tolist():
            rows.append({'movie': title, 'rating': v})
        titles.append(title)

    if not rows:
        return None
    df_plot = pd.DataFrame(rows)
    # order movies by median descending so highest median on top
    medians = df_plot.groupby('movie')['rating'].median().sort_values(ascending=False)
    ordered = medians.index.tolist()

    fig, ax = plt.subplots(figsize=(9, max(4, 0.6*len(ordered))))
    sns.boxplot(x='rating', y='movie', data=df_plot, order=ordered, ax=ax, palette='Blues')
    ax.set_xlabel("Rating")
    ax.set_ylabel("Película")
    ax.set_title(f"Distribución de calificaciones por película (top {len(ordered)} por #reseñas)")
    ax.set_xlim(0, 10)
    plt.tight_layout()
    return fig

# ----------------------------
# Main pipeline
# ----------------------------
def main(sqlite_db_path='./IMDB_Movies_2021.db', pdf_out='./site_data/model_results.pdf'):
    ensure_nltk_resources()
    pdf_out = Path(pdf_out)
    pdf_out.parent.mkdir(parents=True, exist_ok=True)

    # We will write to a temporary file and then atomically replace the final PDF
    pdf_tmp = pdf_out.with_name(pdf_out.name + '.tmp')

    # 1) TRAINING: read DB and prepare df_model (first 4000) - unchanged
    conn = None
    try:
        conn = sqlite3.connect(sqlite_db_path)
        try:
            df_all = pd.read_sql_query("SELECT ID, AUTHOR, TITLE, REVIEW, RATING, POSTED FROM REVIEWS", conn)
        except Exception:
            df_all = pd.read_sql_query("SELECT * FROM REVIEWS", conn)
    except Exception:
        df_all = pd.DataFrame(columns=['ID','AUTHOR','TITLE','REVIEW','RATING','POSTED'])
    finally:
        if conn:
            conn.close()

    if not df_all.empty:
        df_all = df_all.replace('\\n', ' ', regex=True)
        df_model = df_all.loc[:3999].copy()
        df_model.reset_index(drop=True, inplace=True)
        if 'REVIEW' in df_model.columns:
            df_model['REVIEW'] = df_model['REVIEW'].astype(str)
        else:
            df_model['REVIEW'] = ""
        df_model.drop_duplicates(subset="REVIEW", inplace=True)
        df_model.reset_index(drop=True, inplace=True)
    else:
        df_model = pd.DataFrame(columns=['ID','AUTHOR','TITLE','REVIEW','RATING','POSTED'])

    # 2) Load movies_with_reviews.json (preferred) and build df_json_reviews
    movies_df = None
    try:
        pref = Path("site_data/movies_with_reviews.json")
        default = Path("site_data/movies.json")
        if pref.exists():
            movies_df = pd.read_json(pref, orient='records')
        elif default.exists():
            movies_df = pd.read_json(default, orient='records')
        else:
            movies_df = pd.DataFrame()
    except Exception:
        movies_df = pd.DataFrame()

    df_json_reviews = pd.DataFrame(columns=['title','review','rating','movie_idx'])
    try:
        rows = []
        if movies_df is not None and not movies_df.empty:
            for idx, r in movies_df.iterrows():
                title = r.get('Series_Title') or r.get('series_title') or r.get('Series Title') or ""
                # user_reviews
                ur = r.get('user_reviews') or []
                if isinstance(ur, list):
                    for u in ur:
                        try:
                            txt = u.get('review') if isinstance(u, dict) else None
                            rt = u.get('rating') if isinstance(u, dict) else None
                            if txt:
                                rows.append({'title': title, 'review': str(txt), 'rating': (float(rt) if rt is not None else None), 'movie_idx': idx})
                        except Exception:
                            continue
                # reviews
                rr = r.get('reviews') or []
                if isinstance(rr, list):
                    for u in rr:
                        try:
                            txt = u.get('review') if isinstance(u, dict) else None
                            rt = u.get('rating') if isinstance(u, dict) else None
                            if txt:
                                rows.append({'title': title, 'review': str(txt), 'rating': (float(rt) if rt is not None else None), 'movie_idx': idx})
                        except Exception:
                            continue
                # positive/negative
                for key in ('positive_review','negative_review'):
                    pr = r.get(key)
                    if isinstance(pr, dict):
                        try:
                            txt = pr.get('review') or pr.get('REVIEW')
                            rt = pr.get('rating')
                            if txt:
                                rows.append({'title': title, 'review': str(txt), 'rating': (float(rt) if rt is not None else None), 'movie_idx': idx})
                        except Exception:
                            pass
        if rows:
            df_json_reviews = pd.DataFrame(rows)
    except Exception:
        df_json_reviews = pd.DataFrame(columns=['title','review','rating','movie_idx'])

    # 3) TF-IDF and representations (training vs JSON visuals - unchanged)
    stopwords = []
    try:
        stopwords = nltk.corpus.stopwords.words('english')
    except Exception:
        stopwords = []
    stopwords += ["'s","'m","n't","br","movie","film"]
    stopwords_set = set(stopwords)

    preprocessed_model = []
    for doc in (df_model['REVIEW'].tolist() if not df_model.empty else []):
        preprocessed_model.append(" ".join(tokenization_and_stemming(doc, stopwords_set)))

    tfidf_model = None
    X_model = None
    feature_names_model = []
    if preprocessed_model:
        tfidf_model = TfidfVectorizer(max_df=0.99, max_features=2000, min_df=0.01, use_idf=True, ngram_range=(1,1))
        X_model = tfidf_model.fit_transform(preprocessed_model)
        try:
            feature_names_model = tfidf_model.get_feature_names_out()
        except Exception:
            feature_names_model = tfidf_model.get_feature_names()

    preprocessed_json = []
    if not df_json_reviews.empty:
        for doc in df_json_reviews['review'].tolist():
            preprocessed_json.append(" ".join(tokenization_and_stemming(doc, stopwords_set)))

    X_json = None
    feature_names_plot = []
    if preprocessed_json:
        if tfidf_model is not None:
            try:
                X_json = tfidf_model.transform(preprocessed_json)
                feature_names_plot = feature_names_model
            except Exception:
                tfidf_plot = TfidfVectorizer(max_df=0.99, max_features=2000, min_df=0.01, use_idf=True, ngram_range=(1,1))
                X_json = tfidf_plot.fit_transform(preprocessed_json)
                try:
                    feature_names_plot = tfidf_plot.get_feature_names_out()
                except Exception:
                    feature_names_plot = tfidf_plot.get_feature_names()
        else:
            tfidf_plot = TfidfVectorizer(max_df=0.99, max_features=2000, min_df=0.01, use_idf=True, ngram_range=(1,1))
            X_json = tfidf_plot.fit_transform(preprocessed_json)
            try:
                feature_names_plot = tfidf_plot.get_feature_names_out()
            except Exception:
                feature_names_plot = tfidf_plot.get_feature_names()

    # 4) TRAINING (unchanged)
    fitted = {}
    kmeans_model = None
    X_2d_model = None
    if X_model is not None and X_model.shape[0] > 0:
        try:
            kmeans_model = KMeans(n_clusters=7, random_state=42)
            kmeans_preds_model = kmeans_model.fit_predict(X_model)
            svd_model = TruncatedSVD(n_components=2, random_state=42)
            X_2d_model = svd_model.fit_transform(X_model)
        except Exception:
            kmeans_preds_model = None
            X_2d_model = None

    metrics_summary_train = {}
    if X_model is not None and X_model.shape[0] > 0 and 'RATING' in df_model.columns:
        ratings = pd.to_numeric(df_model['RATING'], errors='coerce')
        mask = (ratings <= 4) | (ratings >= 7)
        df_bin = df_model[mask].copy()
        df_bin['label'] = (pd.to_numeric(df_bin['RATING'], errors='coerce') >= 7).astype(int)
        X_bin = X_model[mask.values] if X_model is not None else None
        if X_bin is not None and X_bin.shape[0] == mask.sum():
            y_bin = df_bin['label'].values
            if X_bin.shape[0] >= 10 and len(np.unique(y_bin)) > 1:
                X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.25, random_state=42, stratify=y_bin)
                models = OrderedDict()
                models['Naive Bayes'] = MultinomialNB()
                models['Linear SVM'] = LinearSVC(random_state=42)
                models['Decision Tree'] = DecisionTreeClassifier(max_depth=10, random_state=42)
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        fitted[name] = model
                        y_pred = model.predict(X_test)
                        y_score = None
                        if hasattr(model, "predict_proba"):
                            try:
                                y_score = model.predict_proba(X_test)[:,1]
                            except Exception:
                                y_score = None
                        elif hasattr(model, "decision_function"):
                            try:
                                y_score = model.decision_function(X_test)
                            except Exception:
                                y_score = None
                        acc = metrics.accuracy_score(y_test, y_pred)
                        prec = metrics.precision_score(y_test, y_pred, zero_division=0)
                        rec = metrics.recall_score(y_test, y_pred, zero_division=0)
                        f1 = metrics.f1_score(y_test, y_pred, zero_division=0)
                        metrics_summary_train[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'y_pred': y_pred}
                    except Exception:
                        continue

    # 5) EVAL ON JSON
    metrics_summary_json = {}
    models_scores_json = {}
    y_true_json = None

    if (not df_json_reviews.empty) and (X_json is not None) and fitted:
        df_eval = df_json_reviews.copy()
        df_eval['rating_num'] = pd.to_numeric(df_eval['rating'], errors='coerce')
        mask_valid = df_eval['rating_num'].notna().values
        if mask_valid.any():
            y_true_json = (df_eval.loc[mask_valid, 'rating_num'] >= 7).astype(int).values
            try:
                X_json_valid = X_json[mask_valid, :]
            except Exception:
                X_json_valid = None
            if X_json_valid is not None and X_json_valid.shape[0] == len(y_true_json):
                for name, model in fitted.items():
                    try:
                        y_pred = model.predict(X_json_valid)
                    except Exception:
                        y_pred = None
                    y_score = None
                    if hasattr(model, "predict_proba"):
                        try:
                            y_score = model.predict_proba(X_json_valid)[:,1]
                        except Exception:
                            y_score = None
                    elif hasattr(model, "decision_function"):
                        try:
                            y_score = model.decision_function(X_json_valid)
                        except Exception:
                            y_score = None
                    if y_pred is not None:
                        acc = metrics.accuracy_score(y_true_json, y_pred)
                        prec = metrics.precision_score(y_true_json, y_pred, zero_division=0)
                        rec = metrics.recall_score(y_true_json, y_pred, zero_division=0)
                        f1 = metrics.f1_score(y_true_json, y_pred, zero_division=0)
                        metrics_summary_json[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'y_pred': y_pred}
                    else:
                        metrics_summary_json[name] = {'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'y_pred': None}
                    models_scores_json[name] = {'score': y_score}

    metrics_display = metrics_summary_json if metrics_summary_json else metrics_summary_train
    models_scores_display = models_scores_json if models_scores_json else {}

    # 6) SENTIMENT
    df_json_with_sentiment = None
    sentiment_counts = None
    try:
        if not df_json_reviews.empty:
            df_json_with_sentiment, sentiment_counts = compute_sentiment_labels(df_json_reviews)
    except Exception:
        df_json_with_sentiment = None
        sentiment_counts = None

    # 7) Generate PDF - include sentiment and boxplots
    try:
        with PdfPages(pdf_tmp) as pp:
            # Top movies
            try:
                fig_top = plot_top_movies_diagram(movies_df=movies_df, top_n=10)
                if fig_top is not None:
                    save_figure_to_pdf(pp, fig_top)  # \x1b[33mMODIFIED\x1b[0m >>> uso de la nueva función de guardado
            except Exception as e:
                print("Top movies error:", e)

            # Summary text
            fig = plt.figure(figsize=(8.27,11.69))
            plt.axis('off')
            text = []
            text.append("IMDB - Model Results")
            text.append("")
            text.append(f"Total reviews in DB: {len(df_all) if 'df_all' in locals() else 0}")
            text.append(f"Reviews used for modeling (first 4000): {len(df_model)}")
            if not df_json_reviews.empty:
                text.append(f"Reviews used for PDF visuals/evaluation (JSON): {len(df_json_reviews)}")
            text.append("")
            text.append("Models trained: " + (", ".join(list(metrics_summary_train.keys())) or "none"))
            plt.text(0.01, 0.99, "\n".join(text), va='top', fontsize=11, family='monospace')
            save_figure_to_pdf(pp, fig)

            # TF-IDF top terms (JSON)
            try:
                if X_json is not None and len(feature_names_plot) > 0:
                    tfidf_sum = np.asarray(X_json.sum(axis=0)).ravel()
                    top_n = min(30, tfidf_sum.shape[0])
                    top_idx = tfidf_sum.argsort()[::-1][:top_n]
                    top_terms = [feature_names_plot[i] for i in top_idx]
                    top_vals = tfidf_sum[top_idx]
                    fig, ax = plt.subplots(figsize=(9,7))
                    sns.barplot(x=top_vals, y=top_terms, palette='viridis', ax=ax)
                    ax.set_title("Top TF-IDF terms (reviews from JSON)")
                    ax.set_xlabel("Sum TF-IDF")
                    save_figure_to_pdf(pp, fig)
            except Exception as e:
                print("TF-IDF (JSON) error:", e)

            # Sentiment distribution and pie
            try:
                if df_json_with_sentiment is not None and not df_json_with_sentiment.empty:
                    fig = plot_sentiment_distribution_overall(df_json_with_sentiment)
                    if fig is not None:
                        save_figure_to_pdf(pp, fig)
                    fig2 = plot_sentiment_pie(df_json_with_sentiment)
                    if fig2 is not None:
                        save_figure_to_pdf(pp, fig2)
                    fig3 = plot_sentiment_by_top_movies(movies_df, df_json_with_sentiment, top_n=10)
                    if fig3 is not None:
                        save_figure_to_pdf(pp, fig3)
            except Exception as e:
                print("Sentiment plotting error:", e)

            # Boxplots: overall & by sentiment
            try:
                fig_box = plot_rating_boxplots_overall_and_by_sentiment(df_json_with_sentiment)
                if fig_box is not None:
                    save_figure_to_pdf(pp, fig_box)
            except Exception as e:
                print("Boxplot overall/by sentiment error:", e)

            # Boxplot by top movies
            try:
                fig_box2 = plot_rating_boxplot_by_top_movies(movies_df, df_json_with_sentiment, top_n=8)
                if fig_box2 is not None:
                    save_figure_to_pdf(pp, fig_box2)
            except Exception as e:
                print("Boxplot by movie error:", e)

            # KMeans + scatter on JSON (bigger, full figure / dynamic sizing) + CENTROIDS + TOP TERMS POR CLUSTER
            try:
                if X_json is not None:
                    kmeans_plot = KMeans(n_clusters=7, random_state=42)
                    preds_plot = kmeans_plot.fit_predict(X_json)
                    svd_plot = TruncatedSVD(n_components=2, random_state=42)
                    X_2d_plot = svd_plot.fit_transform(X_json)

                    # dynamic sizing: increase width/height when many clusters or many points
                    n_clusters_plot = len(np.unique(preds_plot))
                    width = max(11, n_clusters_plot * 1.2)  # \x1b[33mMODIFIED\x1b[0m >>> línea modificada: ancho dinámico para KMeans
                    height = 10  # \x1b[33mMODIFIED\x1b[0m >>> altura mayor para mostrar todo el grafo
                    fig, ax = plt.subplots(figsize=(width, height))
                    palette = sns.color_palette("tab10", n_colors=max(10, n_clusters_plot))
                    for k in range(n_clusters_plot):
                        idx = (preds_plot == k)
                        ax.scatter(X_2d_plot[idx,0], X_2d_plot[idx,1], s=12, color=palette[k % len(palette)], label=f"Cluster {k}", alpha=0.7)  # \x1b[33mMODIFIED\x1b[0m >>> marcador y tamaño ajustados
                    # compute and plot centroids projected to 2D so user sees centers
                    try:
                        centroids = kmeans_plot.cluster_centers_  # \x1b[33mMODIFIED\x1b[0m >>> añadida: obtener centroides en espacio TF-IDF
                        # Project centroids to SVD 2D space: centroid_2d = centroid dot components_.T
                        centroid_2d = centroids.dot(svd_plot.components_.T)  # \x1b[33mMODIFIED\x1b[0m >>> añadida: proyección de centroides a 2D
                        for k in range(n_clusters_plot):
                            ax.scatter(centroid_2d[k,0], centroid_2d[k,1], marker='X', s=220, edgecolor='k', linewidth=0.8, color=palette[k % len(palette)], label=f"Centroid {k}")  # \x1b[33mMODIFIED\x1b[0m >>> centroides marcados
                    except Exception:
                        # safe fallback: no centroid plotting if shape mismatch
                        pass
                    ax.set_title("KMeans clusters (TruncatedSVD 2D) - JSON reviews")
                    # place legend outside to avoid covering points and allow full use of plot area
                    ax.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc='upper left')  # \x1b[33mMODIFIED\x1b[0m >>> leyenda fuera del área del gráfico
                    # annotate cluster sizes near legend area
                    try:
                        counts = np.bincount(preds_plot, minlength=n_clusters_plot)
                        # build text with counts
                        count_text = "\n".join([f"Cluster {i}: {counts[i]} pts" for i in range(n_clusters_plot)])
                        ax.text(1.02, 0.5, count_text, transform=ax.transAxes, va='center', fontsize=9, family='monospace')  # \x1b[33mMODIFIED\x1b[0m >>> tamaños de clusters añadidos como texto
                    except Exception:
                        pass
                    plt.tight_layout()
                    save_figure_to_pdf(pp, fig)

                    # Additionally: create one page per cluster with top TF-IDF terms (centroid weights)
                    if len(feature_names_plot) > 0:
                        try:
                            # centroids variable already computed above (in TF-IDF space)
                            if 'centroids' not in locals():
                                centroids = kmeans_plot.cluster_centers_  # \x1b[33mMODIFIED\x1b[0m >>> asegurando centroides disponibles
                            for k in range(n_clusters_plot):
                                try:
                                    cw = centroids[k]
                                    top_n_terms = 12
                                    top_idx = np.argsort(cw)[::-1][:top_n_terms]
                                    kws = [feature_names_plot[i] for i in top_idx]
                                    vals = cw[top_idx]
                                    fig, ax = plt.subplots(figsize=(11, max(3, 0.4 * len(kws) + 1.5)))  # \x1b[33mMODIFIED\x1b[0m >>> tamaño dinámico por número de palabras
                                    sns.barplot(x=vals, y=kws, palette='crest', ax=ax)
                                    ax.set_title(f"Cluster {k} - top TF-IDF words (centroid)")  # \x1b[33mMODIFIED\x1b[0m >>> título para top términos por cluster
                                    ax.set_xlabel("Centroid weight (TF-IDF feature space)")
                                    plt.tight_layout()
                                    save_figure_to_pdf(pp, fig)
                                except Exception:
                                    continue
                        except Exception:
                            pass
            except Exception as e:
                print("KMeans (JSON) error:", e)

            # Confusion matrices / ROC / PR computed on JSON if available
            try:
                if metrics_display and 'y_pred' in list(metrics_display.values())[0].keys():
                    y_plot = None
                    if 'y_true_json' in locals() and y_true_json is not None:
                        y_plot = y_true_json
                    elif 'y_test' in locals():
                        y_plot = y_test
                    if y_plot is not None:
                        for name, info in metrics_display.items():
                            y_pred = info.get('y_pred')
                            if y_pred is None:
                                continue
                            if len(y_pred) == len(y_plot):
                                fig = plot_confusion_matrix(y_plot, y_pred, labels=[0,1], title=f"{name} - Confusion Matrix")
                                save_figure_to_pdf(pp, fig)
                        if models_scores_display and y_plot is not None:
                            fig = plot_roc_curves(models_scores_display, y_plot, title="ROC Curves (JSON eval)")
                            save_figure_to_pdf(pp, fig)
                            fig = plot_precision_recall(models_scores_display, y_plot, title="Precision-Recall Curves (JSON eval)")
                            save_figure_to_pdf(pp, fig)
                if metrics_display:
                    # metrics figure now uses dynamic sizing inside the function
                    fig = plot_metric_bars(metrics_display, title="Model metrics (evaluated on JSON reviews if available)")
                    if fig is not None:
                        save_figure_to_pdf(pp, fig)
            except Exception as e:
                print("Metrics (JSON) error:", e)

            # Feature importance / model insights (training)
            try:
                if fitted:
                    dt = fitted.get('Decision Tree', None)
                    if dt is not None and len(feature_names_model) > 0:
                        importances = dt.feature_importances_
                        top_n = min(20, len(importances))
                        top_idx = np.argsort(importances)[::-1][:top_n]
                        top_feats = [feature_names_model[i] for i in top_idx]
                        top_vals = importances[top_idx]
                        fig, ax = plt.subplots(figsize=(9,6))
                        sns.barplot(x=top_vals, y=top_feats, palette='magma', ax=ax)
                        ax.set_title("Decision Tree - top feature importances (training)")
                        save_figure_to_pdf(pp, fig)
            except Exception as e:
                print("Feature importance error:", e)

            # LDA topics on JSON reviews (if X_json present)
            # -> produce one (readable) barplot per topic with top words and their weights.
            try:
                if X_json is not None and len(feature_names_plot) > 0:
                    n_topics = 5
                    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
                    lda_out = lda.fit_transform(X_json)
                    for i, topic_weights in enumerate(lda.components_):
                        top_idx = topic_weights.argsort()[::-1][:10]
                        kws = [feature_names_plot[j] for j in top_idx]
                        vals = topic_weights[top_idx]
                        # dynamic height depending on number of kws
                        height = max(3, 0.4 * len(kws) + 1.5)  # \x1b[33mMODIFIED\x1b[0m >>> altura dinámica por tópico
                        fig, ax = plt.subplots(figsize=(11, height))
                        sns.barplot(x=vals, y=kws, palette='viridis', ax=ax)
                        ax.set_title(f"LDA Topic {i} - top words (LDA)")
                        ax.set_xlabel("Weight")
                        plt.tight_layout()
                        save_figure_to_pdf(pp, fig)
            except Exception as e:
                print("LDA (JSON) error:", e)

            # Classification reports - computed on JSON if available, else training
            try:
                if metrics_display:
                    fig = plt.figure(figsize=(8.27,11.69))
                    plt.axis('off')
                    bigtext = "Classification reports (evaluated on JSON reviews if available)\n\n"
                    for name, info in metrics_display.items():
                        if (not df_json_reviews.empty) and ('y_true_json' in locals()) and name in metrics_summary_json:
                            y_pred_local = info.get('y_pred')
                            if y_pred_local is not None and len(y_pred_local) == len(y_true_json):
                                cr = classification_report(y_true_json, y_pred_local, target_names=['NEG','POS'], zero_division=0)
                            else:
                                cr = "No aligned predictions for JSON evaluation."
                        else:
                            if name in metrics_summary_train and 'y_test' in locals():
                                cr = classification_report(y_test, metrics_summary_train[name].get('y_pred'), target_names=['NEG','POS'], zero_division=0)
                            else:
                                cr = "No evaluation data available."
                        bigtext += f"Model: {name}\n{cr}\n\n"
                    plt.text(0.01, 0.99, bigtext, va='top', fontsize=8, family='monospace')
                    save_figure_to_pdf(pp, fig)
            except Exception as e:
                print("Classification reports error:", e)
    except Exception as e:
        # If any unexpected error occurs while generating the PDF, ensure tmp file is removed
        try:
            if pdf_tmp.exists():
                pdf_tmp.unlink()
        except Exception:
            pass
        raise

    # Atomically replace the final file with the tmp file
    try:
        if pdf_tmp.exists():
            pdf_tmp.replace(pdf_out)
    except Exception as e:
        print(f"Could not replace final PDF atomically: {e}")
        raise

    print(f"PDF with results written to: {os.path.abspath(pdf_out)}")


if __name__ == '__main__':
    main()