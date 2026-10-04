from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from medrag.embeddings import blob_to_vector, vector_to_blob
from medrag.models import Chunk, Record

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
  pmid TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  journal TEXT,
  year INTEGER,
  pub_types TEXT,
  mesh_terms TEXT,
  doi TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pmid TEXT NOT NULL REFERENCES records(pmid),
  section TEXT,
  header TEXT NOT NULL,
  text TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id');

CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id),
  model TEXT NOT NULL,
  vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT
);
"""


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- records -----------------------------------------------------
    def upsert_record(self, record: Record) -> None:
        self.conn.execute(
            """
            INSERT INTO records (pmid, title, journal, year, pub_types, mesh_terms, doi)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pmid) DO UPDATE SET
              title=excluded.title, journal=excluded.journal, year=excluded.year,
              pub_types=excluded.pub_types, mesh_terms=excluded.mesh_terms, doi=excluded.doi
            """,
            (
                record.pmid,
                record.title,
                record.journal,
                record.year,
                json.dumps(record.pub_types),
                json.dumps(record.mesh_terms),
                record.doi,
            ),
        )
        self.conn.commit()

    def get_record(self, pmid: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM records WHERE pmid = ?", (pmid,)).fetchone()

    def record_has_chunks(self, pmid: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM chunks WHERE pmid = ? LIMIT 1", (pmid,)).fetchone()
        return row is not None

    # --- chunks --------------------------------------------------------
    def insert_chunk(self, chunk: Chunk) -> int:
        cur = self.conn.execute(
            "INSERT INTO chunks (pmid, section, header, text) VALUES (?, ?, ?, ?)",
            (chunk.pmid, chunk.section, chunk.header, chunk.text),
        )
        chunk_id = cur.lastrowid
        self.conn.execute(
            "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
            (chunk_id, chunk.text),
        )
        self.conn.commit()
        return chunk_id

    def get_chunk(self, chunk_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT chunks.*, records.title AS rec_title, records.year AS rec_year,
                   records.journal AS rec_journal, records.pub_types AS rec_pub_types
            FROM chunks JOIN records ON chunks.pmid = records.pmid
            WHERE chunks.id = ?
            """,
            (chunk_id,),
        ).fetchone()

    def all_chunk_ids(self) -> list[int]:
        return [r[0] for r in self.conn.execute("SELECT id FROM chunks").fetchall()]

    # --- embeddings ------------------------------------------------------
    def chunk_ids_missing_embedding(self, model: str) -> list[int]:
        rows = self.conn.execute(
            """
            SELECT chunks.id FROM chunks
            LEFT JOIN embeddings ON chunks.id = embeddings.chunk_id AND embeddings.model = ?
            WHERE embeddings.chunk_id IS NULL
            """,
            (model,),
        ).fetchall()
        return [r[0] for r in rows]

    def insert_embedding(self, chunk_id: int, model: str, vector: list[float]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (chunk_id, model, vector) VALUES (?, ?, ?)",
            (chunk_id, model, vector_to_blob(vector)),
        )
        self.conn.commit()

    def clear_embeddings(self) -> None:
        self.conn.execute("DELETE FROM embeddings")
        self.conn.commit()

    def get_embedding_matrix(self, model: str) -> tuple[list[int], np.ndarray]:
        rows = self.conn.execute(
            "SELECT chunk_id, vector FROM embeddings WHERE model = ?", (model,)
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        chunk_ids = [r["chunk_id"] for r in rows]
        vectors = np.stack([blob_to_vector(r["vector"]) for r in rows])
        return chunk_ids, vectors

    def get_embeddings_for_chunk_ids(self, chunk_ids: list[int], model: str) -> dict[int, np.ndarray]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.conn.execute(
            f"SELECT chunk_id, vector FROM embeddings WHERE model = ? AND chunk_id IN ({placeholders})",
            (model, *chunk_ids),
        ).fetchall()
        return {r["chunk_id"]: blob_to_vector(r["vector"]) for r in rows}

    # --- meta ------------------------------------------------------------
    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    # --- search ------------------------------------------------------------
    def fts_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        escaped = _escape_fts_query(query)
        if not escaped:
            return []
        try:
            rows = self.conn.execute(
                """
                SELECT chunks_fts.rowid AS chunk_id, bm25(chunks_fts) AS score
                FROM chunks_fts WHERE chunks_fts MATCH ?
                ORDER BY score LIMIT ?
                """,
                (escaped, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r["chunk_id"], r["score"]) for r in rows]


def _escape_fts_query(query: str) -> str:
    terms = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if t]
    return " OR ".join(f'"{t}"' for t in terms)
