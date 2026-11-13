"""Step 2: build the SQLite knowledge base from the downloaded CrisisMMD files.

ETL: read the three annotation tasks, join them on image_id, split the flat
rows into event / tweet / image tables, build the full-text index, then
encode every image with CLIP and store its vector as a BLOB.
"""

import json
import os
import sqlite3
import time
from datetime import datetime

import numpy as np
from PIL import Image

from download_data import DATA_DIR, EVENTS, SPLITS
from search import DB_PATH, MODEL_NAME, ROOT

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# From the CrisisMMD dataset page (collection windows of each event)
EVENT_INFO = {
    "california_wildfires": ("California Wildfires", "wildfire", "2017-10-10", "2017-10-27"),
    "hurricane_harvey": ("Hurricane Harvey", "hurricane", "2017-08-25", "2017-09-20"),
    "hurricane_irma": ("Hurricane Irma", "hurricane", "2017-09-06", "2017-09-21"),
    "hurricane_maria": ("Hurricane Maria", "hurricane", "2017-09-20", "2017-11-13"),
    "iraq_iran_earthquake": ("Iraq-Iran Earthquake", "earthquake", "2017-11-13", "2017-11-19"),
    "mexico_earthquake": ("Mexico Earthquake", "earthquake", "2017-09-20", "2017-10-06"),
    "srilanka_floods": ("Sri Lanka Floods", "flood", "2017-05-31", "2017-07-03"),
}


def read_task(task):
    """All rows of one annotation task (train + dev + test), keyed by image_id."""
    rows = {}
    for split in SPLITS:
        for r in json.loads((DATA_DIR / task / f"{split}.json").read_text(encoding="utf-8")):
            if r["event_name"] in EVENTS:
                rows[r["image_id"]] = r
    return rows


def date_from_path(image_path):
    # data_image/<event>/<D_M_YYYY>/<image_id>.jpg -> YYYY-MM-DD
    folder = image_path.split("/")[2]
    return datetime.strptime(folder, "%d_%m_%Y").strftime("%Y-%m-%d")


def load_tables(con):
    informative = read_task("informative")
    humanitarian = read_task("humanitarian")
    damage = read_task("damage")

    for event_id in EVENTS:
        con.execute("INSERT INTO event VALUES (?, ?, ?, ?, ?)", (event_id, *EVENT_INFO[event_id]))

    skipped = 0
    for image_id, r in informative.items():
        if not (DATA_DIR / r["image_path"]).exists():
            skipped += 1
            continue
        con.execute(
            "INSERT OR IGNORE INTO tweet VALUES (?, ?, ?, ?)",
            (int(r["tweet_id"]), r["event_name"], date_from_path(r["image_path"]), r["tweet_text"]),
        )
        con.execute(
            "INSERT INTO image (image_id, tweet_id, image_path, informative, humanitarian, damage) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                image_id,
                int(r["tweet_id"]),
                r["image_path"],
                r["label"],
                humanitarian.get(image_id, {}).get("label"),
                damage.get(image_id, {}).get("label"),
            ),
        )
    con.execute("INSERT INTO tweet_fts(tweet_fts) VALUES ('rebuild')")  # fill the inverted index
    con.commit()
    if skipped:
        print(f"  skipped {skipped} rows whose image file is missing")


def open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def embed_images(con, batch_size=32):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    rows = con.execute("SELECT image_id, image_path FROM image ORDER BY image_id").fetchall()
    start = time.time()
    for i in range(0, len(rows), batch_size):
        batch = [(image_id, open_image(DATA_DIR / path)) for image_id, path in rows[i:i + batch_size]]
        batch = [(image_id, img) for image_id, img in batch if img is not None]
        vectors = model.encode([img for _, img in batch], normalize_embeddings=True)
        con.executemany(
            "UPDATE image SET embedding = ? WHERE image_id = ?",
            [(v.astype(np.float32).tobytes(), image_id) for (image_id, _), v in zip(batch, vectors)],
        )
        con.commit()
        done = min(i + batch_size, len(rows))
        print(f"\r  encoded {done}/{len(rows)} images ({time.time() - start:.0f}s)", end="")
    print()


def main():
    DB_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))

    print("Loading annotations into event / tweet / image tables...")
    load_tables(con)
    for table in ("event", "tweet", "image"):
        print(f"  {table}: {con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]} rows")

    print(f"Encoding images with CLIP ({MODEL_NAME}); the first run downloads the model...")
    embed_images(con)
    con.close()
    print(f"Done: {DB_PATH}")


if __name__ == "__main__":
    main()
