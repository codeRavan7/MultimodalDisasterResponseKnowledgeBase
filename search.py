"""Query layer: SQL filtering, keyword search (inverted index) and
similarity search (exact k-NN and LSH) over the knowledge base."""

import re
import sqlite3
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DB_PATH = ROOT / "disaster_kb.db"
DATA_DIR = ROOT / "data" / "CrisisMMD_v2.0"
MODEL_NAME = "clip-ViT-B-32"

# Columns shown for every result row
RESULT_COLUMNS = """
    i.image_id, i.image_path, t.tweet_id, t.tweet_text, t.tweet_date,
    e.name AS event, i.informative, i.humanitarian, i.damage
"""
RESULT_JOIN = """
    FROM image i
    JOIN tweet t ON t.tweet_id = i.tweet_id
    JOIN event e ON e.event_id = t.event_id
"""


def connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def distinct_values(con, table, column):
    rows = con.execute(
        f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY 1"
    )
    return [r[0] for r in rows]


# ---------- 1. Structured search (plain SQL) ----------

def filter_reports(con, events=(), humanitarian=(), damage=(), informative=None,
                   date_from=None, date_to=None, limit=50):
    """Builds a parameterised SELECT from the chosen filters.
    Returns (sql, params, rows) so the UI can show the query that ran."""
    where, params = [], []

    def add_in(column, values):
        if values:
            where.append(f"{column} IN ({','.join('?' * len(values))})")
            params.extend(values)

    add_in("t.event_id", events)
    add_in("i.humanitarian", humanitarian)
    add_in("i.damage", damage)
    if informative:
        where.append("i.informative = ?")
        params.append(informative)
    if date_from:
        where.append("t.tweet_date >= ?")
        params.append(str(date_from))
    if date_to:
        where.append("t.tweet_date <= ?")
        params.append(str(date_to))

    sql = f"SELECT {RESULT_COLUMNS} {RESULT_JOIN}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY t.tweet_date LIMIT ?"
    params.append(limit)
    return sql, params, con.execute(sql, params).fetchall()


# ---------- 2. Keyword search (FTS5 inverted index + BM25) ----------

def keyword_search(con, query, limit=50):
    """Each word must appear in the tweet text (implicit AND).
    bm25() is negative in FTS5, so ascending order = best match first."""
    words = re.sub(r"[^\w\s]", " ", query).split()  # punctuation breaks FTS5 syntax
    if not words:
        return "", []
    match = " ".join(f'"{w}"' for w in words)
    sql = f"""
        SELECT {RESULT_COLUMNS}, f.score
        FROM (SELECT rowid, bm25(tweet_fts) AS score
              FROM tweet_fts WHERE tweet_fts MATCH ?
              ORDER BY score LIMIT ?) f
        JOIN tweet t ON t.tweet_id = f.rowid
        JOIN image i ON i.image_id = (SELECT MIN(image_id) FROM image WHERE tweet_id = t.tweet_id)
        JOIN event e ON e.event_id = t.event_id
        ORDER BY f.score                 -- one row per tweet (its first image)
    """
    return match, con.execute(sql, (match, limit)).fetchall()


# ---------- 3. Similarity search (vectors) ----------

def load_vectors(con):
    """Reads every image embedding into one (N, 512) matrix."""
    rows = con.execute(
        "SELECT image_id, embedding FROM image WHERE embedding IS NOT NULL ORDER BY image_id"
    ).fetchall()
    ids = [r[0] for r in rows]
    matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return ids, matrix


def knn_exact(matrix, q, k):
    """Brute-force scan: cosine similarity = dot product of unit vectors."""
    scores = matrix @ q
    top = np.argpartition(-scores, k)[:k]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]


class LSHIndex:
    """Random-hyperplane LSH for cosine similarity (Charikar, 2002).

    Each table hashes a vector to `bits` sign bits (which side of each random
    hyperplane it falls on). Similar vectors tend to share a bucket, so a query
    only compares itself with the vectors in its buckets, not the whole table.
    """

    def __init__(self, matrix, n_tables=10, bits=6, seed=42):
        rng = np.random.default_rng(seed)
        self.matrix = matrix
        self.planes = rng.standard_normal((n_tables, matrix.shape[1], bits)).astype(np.float32)
        self.powers = 1 << np.arange(bits)
        self.tables = []
        for planes in self.planes:
            keys = ((matrix @ planes) > 0) @ self.powers
            buckets = {}
            for idx, key in enumerate(keys):
                buckets.setdefault(int(key), []).append(idx)
            self.tables.append(buckets)

    def query(self, q, k):
        candidates = set()
        for planes, buckets in zip(self.planes, self.tables):
            key = int(((q @ planes) > 0) @ self.powers)
            candidates.update(buckets.get(key, []))
        if not candidates:
            return np.array([], dtype=int), np.array([]), 0
        cand = np.fromiter(candidates, dtype=int)
        scores = self.matrix[cand] @ q
        order = np.argsort(-scores)[:k]
        return cand[order], scores[order], len(cand)


def compare_search(matrix, lsh, q, k):
    """Runs exact and LSH search on the same query; returns timings and recall@k."""
    t0 = time.perf_counter()
    exact_idx, exact_scores = knn_exact(matrix, q, k)
    t1 = time.perf_counter()
    lsh_idx, lsh_scores, n_candidates = lsh.query(q, k)
    t2 = time.perf_counter()
    recall = len(set(exact_idx) & set(lsh_idx)) / k
    stats = {
        "exact_ms": (t1 - t0) * 1000,
        "lsh_ms": (t2 - t1) * 1000,
        "candidates": n_candidates,
        "total": len(matrix),
        "recall": recall,
    }
    return (exact_idx, exact_scores), (lsh_idx, lsh_scores), stats


def fetch_by_ids(con, image_ids):
    """Returns result rows in the same order as image_ids."""
    if not image_ids:
        return []
    marks = ",".join("?" * len(image_ids))
    rows = con.execute(
        f"SELECT {RESULT_COLUMNS} {RESULT_JOIN} WHERE i.image_id IN ({marks})", image_ids
    ).fetchall()
    by_id = {r["image_id"]: r for r in rows}
    return [by_id[i] for i in image_ids]
