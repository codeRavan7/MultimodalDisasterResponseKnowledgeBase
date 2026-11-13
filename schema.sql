-- Multimodal Disaster Response Knowledge Base: relational schema (3NF)
--
--   event 1 ──< tweet 1 ──< image
--
-- A tweet belongs to one disaster event; a tweet can carry up to 4 images.
-- The tweet text is stored once (in tweet), not repeated for every image.

CREATE TABLE event (
    event_id      TEXT PRIMARY KEY,          -- e.g. 'mexico_earthquake'
    name          TEXT NOT NULL,             -- e.g. 'Mexico Earthquake'
    disaster_type TEXT NOT NULL,             -- wildfire / earthquake / flood / hurricane
    start_date    TEXT NOT NULL,             -- ISO date, YYYY-MM-DD
    end_date      TEXT NOT NULL
);

CREATE TABLE tweet (
    tweet_id   INTEGER PRIMARY KEY,          -- Twitter status id (also the SQLite rowid)
    event_id   TEXT NOT NULL REFERENCES event(event_id),
    tweet_date TEXT NOT NULL,                -- ISO date, YYYY-MM-DD
    tweet_text TEXT NOT NULL
);

CREATE TABLE image (
    image_id     TEXT PRIMARY KEY,           -- '<tweet_id>_<n>'
    tweet_id     INTEGER NOT NULL REFERENCES tweet(tweet_id),
    image_path   TEXT NOT NULL,              -- relative to data/CrisisMMD_v2.0
    informative  TEXT,                       -- semantic tags from the human annotators
    humanitarian TEXT,
    damage       TEXT,                       -- NULL when the image was not assessed
    embedding    BLOB                        -- 512 float32 values from CLIP (2048 bytes)
);

-- B+ tree indexes on the columns used in WHERE / JOIN clauses
CREATE INDEX idx_tweet_event  ON tweet(event_id);
CREATE INDEX idx_tweet_date   ON tweet(tweet_date);
CREATE INDEX idx_image_tweet  ON image(tweet_id);
CREATE INDEX idx_image_human  ON image(humanitarian);
CREATE INDEX idx_image_damage ON image(damage);

-- Inverted index over tweet text (SQLite FTS5, ranked with BM25).
-- External-content table: the text itself stays in tweet.
CREATE VIRTUAL TABLE tweet_fts USING fts5(
    tweet_text,
    content='tweet',
    content_rowid='tweet_id'
);
