"""Step 3: the web interface. Run with:  streamlit run app.py"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import search
from search import DATA_DIR, MODEL_NAME

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
st.set_page_config(page_title="Disaster Response KB", layout="wide")


@st.cache_resource
def get_connection():
    return search.connect()


@st.cache_resource
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device="cpu")


@st.cache_resource
def get_vector_index():
    ids, matrix = search.load_vectors(get_connection())
    return ids, matrix, search.LSHIndex(matrix)


def label(value):
    return value.replace("_", " ").capitalize() if value else "Not assessed"


def show_results(rows, scores=None, columns=4):
    if not rows:
        st.info("No matching reports.")
        return
    st.caption(f"{len(rows)} result(s)")
    for start in range(0, len(rows), columns):
        cols = st.columns(columns)
        for j, row in enumerate(rows[start:start + columns]):
            with cols[j]:
                path = DATA_DIR / row["image_path"]
                if path.exists():
                    st.image(str(path), width="stretch")
                score = f" · score {scores[start + j]:.3f}" if scores is not None else ""
                st.markdown(f"**{row['event']}** · {row['tweet_date']}{score}")
                st.caption(f"{label(row['humanitarian'])} · damage: {label(row['damage']).lower()}")
                with st.expander("Tweet text"):
                    st.write(row["tweet_text"])


con = get_connection()
st.title("Multimodal Disaster Response Knowledge Base")
st.caption("Tweets and photos from real 2017 disasters (CrisisMMD), searchable by "
           "structured filters, keywords and visual similarity.")

tab_filter, tab_keyword, tab_similar, tab_overview = st.tabs(
    ["Filter (SQL)", "Keyword search", "Similarity search", "Database overview"]
)

# ---------- 1. Structured search ----------
with tab_filter:
    events = con.execute("SELECT event_id, name FROM event ORDER BY name").fetchall()
    event_names = {r["event_id"]: r["name"] for r in events}
    c1, c2, c3 = st.columns(3)
    chosen_events = c1.multiselect("Event", list(event_names), format_func=event_names.get)
    chosen_human = c2.multiselect("Humanitarian category",
                                  search.distinct_values(con, "image", "humanitarian"),
                                  format_func=label)
    chosen_damage = c3.multiselect("Damage severity",
                                   search.distinct_values(con, "image", "damage"),
                                   format_func=label)
    c4, c5, c6, c7 = st.columns(4)
    informative = c4.selectbox("Informative?", ["", "informative", "not_informative"],
                               format_func=lambda v: label(v) if v else "Any")
    min_date, max_date = con.execute("SELECT MIN(tweet_date), MAX(tweet_date) FROM tweet").fetchone()
    date_from = c5.date_input("From", pd.to_datetime(min_date))
    date_to = c6.date_input("To", pd.to_datetime(max_date))
    limit = c7.number_input("Max results", 4, 200, 24, step=4)

    sql, params, rows = search.filter_reports(
        con, chosen_events, chosen_human, chosen_damage, informative or None,
        date_from, date_to, limit,
    )
    with st.expander("SQL query that ran"):
        st.code(sql.strip(), language="sql")
        st.write("Parameters:", params)
    show_results(rows)

# ---------- 2. Keyword search ----------
with tab_keyword:
    query = st.text_input("Keywords (all words must appear)", placeholder="e.g. bridge collapsed")
    if query:
        match, rows = search.keyword_search(con, query)
        st.caption(f"FTS5 MATCH expression: `{match}` · ranked by BM25 (lower = better)")
        show_results(rows, [r["score"] for r in rows])

# ---------- 3. Similarity search ----------
with tab_similar:
    mode = st.radio("Search by", ["Text description", "Example photo"], horizontal=True)
    k = st.slider("Number of results (k)", 4, 40, 12, step=4)
    model = get_model()
    q = None
    if mode == "Text description":
        text = st.text_input("Describe the scene", placeholder="e.g. flooded street with people wading")
        if text:
            q = model.encode(text, normalize_embeddings=True)
    else:
        upload = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        if upload:
            img = Image.open(upload).convert("RGB")
            st.image(img, width=200)
            q = model.encode(img, normalize_embeddings=True)

    if q is not None:
        ids, matrix, lsh = get_vector_index()
        q = np.asarray(q, dtype=np.float32)
        (exact_idx, exact_scores), (lsh_idx, lsh_scores), stats = search.compare_search(matrix, lsh, q, k)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Exact scan time", f"{stats['exact_ms']:.2f} ms", help=f"Compared with all {stats['total']} vectors")
        m2.metric("LSH time", f"{stats['lsh_ms']:.2f} ms")
        m3.metric("LSH vectors compared", f"{stats['candidates']} / {stats['total']}")
        m4.metric(f"LSH recall@{k}", f"{stats['recall']:.0%}", help="Share of the exact top-k that LSH also found")

        st.caption("LSH recall is high for photo queries but low for text queries: a text vector is "
                   "far from every image vector (cosine ≈ 0.3), so it rarely shares a hash bucket with them.")
        method = st.radio("Show results from", ["Exact k-NN", "LSH"], horizontal=True)
        idx, scores = (exact_idx, exact_scores) if method == "Exact k-NN" else (lsh_idx, lsh_scores)
        st.caption("Score = cosine similarity (higher = more similar)")
        show_results(search.fetch_by_ids(con, [ids[i] for i in idx]), list(scores))

# ---------- 4. Overview ----------
with tab_overview:
    st.subheader("Contents")
    st.dataframe(pd.read_sql_query("""
        SELECT e.name AS event, e.disaster_type, COUNT(DISTINCT t.tweet_id) AS tweets,
               COUNT(i.image_id) AS images, MIN(t.tweet_date) AS first_day, MAX(t.tweet_date) AS last_day
        FROM event e JOIN tweet t ON t.event_id = e.event_id JOIN image i ON i.tweet_id = t.tweet_id
        GROUP BY e.event_id ORDER BY images DESC""", con), hide_index=True)
    c1, c2 = st.columns(2)
    c1.dataframe(pd.read_sql_query("""
        SELECT humanitarian AS category, COUNT(*) AS images FROM image
        GROUP BY humanitarian ORDER BY images DESC""", con), hide_index=True)
    c2.dataframe(pd.read_sql_query("""
        SELECT COALESCE(damage, 'not assessed') AS damage, COUNT(*) AS images FROM image
        GROUP BY damage ORDER BY images DESC""", con), hide_index=True)
    st.subheader("Schema")
    st.code((search.ROOT / "schema.sql").read_text(encoding="utf-8"), language="sql")
