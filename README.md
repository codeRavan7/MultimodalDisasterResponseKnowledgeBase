# Multimodal Disaster Response Knowledge Base

A small database of real disaster reports (tweet text + photo + human labels)
from three 2017 disasters, searchable in three ways:

| Search | How it works | Course topic |
|---|---|---|
| **Filter** by event, date, category, damage | Plain SQL `SELECT … JOIN … WHERE` on B+ tree indexes | Module 1: ER model, relational model, SQL, normalization, B+ tree |
| **Keyword** search in tweet text | SQLite FTS5 inverted index, ranked by BM25 | Module 3: inverted file / inverted list |
| **Similarity** search: text → photos, photo → photos | CLIP turns every photo into a 512-d vector; k-nearest neighbours by cosine similarity, exact scan vs. LSH | Module 2: nearest-neighbour search in high dimensions; Module 3: locality-sensitive hashing; Module 5: multimedia similarity search |

The human labels (informative, humanitarian category, damage severity) act as
semantic tags on the images (Module 5).

## Data

CrisisMMD v2.0 (Alam, Ofli & Imran, ICWSM 2018), the official QCRI copy on
Hugging Face. Subset used: **California Wildfires, Mexico Earthquake, Sri Lanka
Floods** (3,991 images from 3,554 tweets, three different hazard types).
License: CC BY-NC-SA 4.0. The `data/` folder and the `.db` file must not be
shared publicly (they are in `.gitignore`).

## Design

```
event 1 ──< tweet 1 ──< image
```

- `event(event_id, name, disaster_type, start_date, end_date)`
- `tweet(tweet_id, event_id → event, tweet_date, tweet_text)`
- `image(image_id, tweet_id → tweet, image_path, informative, humanitarian, damage, embedding)`
- `tweet_fts`: FTS5 inverted index over `tweet.tweet_text`

**Why three tables (3NF):** the source files have one row per *image*, so a tweet
with 3 photos repeats its text 3 times, and event details would repeat on every
row. Splitting into event / tweet / image stores each fact once.

**Dates:** the source files have no date column; the date is taken from the image
folder name (`data_image/<event>/<D_M_YYYY>/…`).

**Embeddings:** stored in the `image` table as a 2,048-byte BLOB (512 × float32),
normalised to length 1 so cosine similarity is just a dot product.

**Exact k-NN vs. LSH:** exact search compares the query with every vector.
LSH (random hyperplanes, 10 tables × 6 bits) only compares it with vectors that
land in the same hash buckets. The app shows the time of both, how many vectors
LSH compared, and its recall (how many of the true top-k it found). With only
~4,000 vectors the exact scan is already fast; LSH is there to show the idea
and its accuracy trade-off.

**Why not a K-D tree or R-tree:** in 512 dimensions, tree indexes end up
visiting almost every node (curse of dimensionality, Weber et al., VLDB 1998),
so they are no faster than a plain scan.

## Files

| File | Purpose |
|---|---|
| `schema.sql` | Table, index and FTS5 definitions |
| `download_data.py` | Step 1: downloads annotations + images of the chosen events |
| `build_db.py` | Step 2: ETL into SQLite + CLIP image embeddings |
| `search.py` | SQL filter, keyword search, exact k-NN, LSH |
| `app.py` | Step 3: Streamlit web interface |

## Run (Windows, CPU only)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers streamlit pandas pillow numpy
python download_data.py
python build_db.py
streamlit run app.py
```

## References

1. Alam, F., Ofli, F., Imran, M. (2018). CrisisMMD: Multimodal Twitter Datasets from Natural Disasters. ICWSM.
2. Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML.
3. Charikar, M. (2002). Similarity Estimation Techniques from Rounding Algorithms. STOC.
4. Weber, R., Schek, H.-J., Blott, S. (1998). A Quantitative Analysis and Performance Study for Similarity-Search Methods in High-Dimensional Spaces. VLDB.
5. Silberschatz, Korth, Sudarshan. Database System Concepts, 7th ed.
6. Manning, Raghavan, Schütze (2008). Introduction to Information Retrieval.
