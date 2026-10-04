# Indexing workflow — a literal step-by-step trace

This is a companion to [03-indexing.md](03-indexing.md). That module explains the *mental model*
(why chunking branches the way it does, why indexing is idempotent, what the scale trade-offs
are). This doc is the literal call-by-call trace of what happens when you run `medrag index` —
useful when you're debugging a specific run rather than building first-time intuition.

## Precondition

`data/records.jsonl` must already exist (written by `medrag fetch`). Each line is one PubMed
`Record` — title, abstract sections, journal, year, pub types, MeSH terms, DOI. If this file is
missing or empty, `index_cmd()` prints an error and exits before touching anything else
(`cli.py:50-52`).

## Step 1 — Load records back into memory

```python
records = load_records_jsonl(config.RECORDS_PATH)
```
`ncbi.py:95-117` reads the JSONL line by line and reconstructs `Record` dataclass instances.
Nothing from the database is consulted yet — this is pure deserialization from the fetch step's
output.

## Step 2 — Open the store, create schema if needed

```python
store = Store(config.INDEX_DB_PATH)
```
`Store.__init__` (`store.py:48-53`) opens (or creates) `data/index.db`, sets
`row_factory = sqlite3.Row` (so you get dict-like rows back), and runs the full `SCHEMA` script
with `executescript`. Since every `CREATE TABLE` uses `IF NOT EXISTS`, this is safe to run every
single time `index` is invoked — on a fresh DB it creates five objects (`records`, `chunks`,
`chunks_fts`, `embeddings`, `meta`); on an existing DB it's a no-op.

## Step 3 — Construct the embedder

```python
embedder = OllamaEmbedder(model=config.embed_model(), base_url=config.OLLAMA_BASE_URL)
```
No network call happens here — this just sets up an `httpx.Client` and remembers the model name
(default `nomic-embed-text`). The actual embedding calls happen later, batched.

## Step 4 — `index_records()`: the embedding-model bookkeeping

This is the first thing `index_records` does (`indexer.py:15-18`), *before* touching any record:

```python
stored_model = store.get_meta("embedding_model")
if stored_model is not None and stored_model != model:
    store.clear_embeddings()
store.set_meta("embedding_model", model)
```

It reads the `meta` table for the last embedding model used. If you've switched models (e.g.
`MEDRAG_EMBED_MODEL` changed), it wipes **every row in `embeddings`**, regardless of which model
they belonged to — `clear_embeddings()` is `DELETE FROM embeddings` with no `WHERE model = ...`
clause. This is a real gotcha: switching from model A to model B and back to A later means
re-embedding from scratch both times — nothing is cached per-model for a future switch back.

## Step 5 — Per-record loop: upsert + idempotency check

For each `Record` in the loaded list:

```python
store.upsert_record(record)
if store.record_has_chunks(record.pmid):
    continue
```

`upsert_record` (`store.py:59-78`) is an `INSERT ... ON CONFLICT(pmid) DO UPDATE` — so re-running
`fetch` + `index` on overlapping queries always refreshes the record's metadata (title, year,
journal, etc.) even if nothing else happens.

`record_has_chunks` (`store.py:83-85`) is the idempotency gate:
`SELECT 1 FROM chunks WHERE pmid = ? LIMIT 1`. If this PMID has been chunked before, the record is
skipped entirely — **no re-chunking happens even if the record's metadata just changed in the
upsert above.** This is why a repeat `medrag index` reports `0 new chunks`.

## Step 6 — Chunking (only for records reaching this point)

```python
for chunk in chunk_record(record):
    store.insert_chunk(chunk)
    chunks_created += 1
records_indexed += 1
```

`chunk_record` (`chunking.py:16-34`) branches on whether the abstract has labeled sections:

- **Labeled** (e.g. `BACKGROUND`/`METHODS`/`RESULTS`/`CONCLUSIONS`): one `Chunk` per section,
  however long that section is.
- **Unlabeled**: the abstract is treated as one blob, split every `WORDS_PER_CHUNK = 250` words.

Either way, every chunk's stored `text` is built as `header + "\n" + body`, where `header` is
`PMID:<id> | <title> | <year> | <pub_type> [| <section>]`. That header string is baked into the
text that later gets embedded *and* full-text indexed — it's not kept separately.

## Step 7 — Writing a chunk: two tables, one call

```python
def insert_chunk(self, chunk: Chunk) -> int:
    cur = self.conn.execute("INSERT INTO chunks (...) VALUES (...)", ...)
    chunk_id = cur.lastrowid
    self.conn.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (chunk_id, chunk.text))
    self.conn.commit()
    return chunk_id
```
(`store.py:88-99`) Every chunk insert is immediately mirrored into the `chunks_fts` virtual table
(SQLite FTS5), using the same rowid. There's no trigger doing this automatically — it's two
explicit inserts in the same function. If any future code path ever inserts into `chunks` without
going through `insert_chunk`, keyword search silently stops seeing those rows.

At this point, if you're chunking a PMID into 4 labeled sections, you'd see 4 new rows in `chunks`
and 4 new rows in `chunks_fts`, all sharing rowids.

## Step 8 — After all records: find what still needs an embedding

```python
missing_ids = store.chunk_ids_missing_embedding(model)
```
`store.py:116-125` runs a `LEFT JOIN` between `chunks` and `embeddings` filtered to the current
model, returning chunk ids with no matching embedding row. This is what makes embedding
generation incremental and separate from chunk creation — a chunk can exist for a while with no
embedding (e.g. if Ollama was down during a previous `index` run) and this query will pick it up
on the next run.

## Step 9 — Batch-fetch the text, batch-call Ollama

```python
texts = [store.get_chunk(cid)["text"] for cid in missing_ids]
vectors = embedder.embed_documents_batched(texts)
```
`embed_documents_batched` (`embeddings.py:39-41`) chunks the text list into groups of 16
(`EMBED_BATCH_SIZE`) and calls `embed()` per batch. Each call:
```python
prefixed = [f"{prefix}{t}" for t in texts]   # prefix = "search_document: "
resp = self.client.post(f"{base_url}/api/embed", json={"model": model, "input": prefixed})
```
(`embeddings.py:29-37`) — one HTTP POST per batch of 16 to Ollama's `/api/embed` endpoint. The
`"search_document: "` prefix is `nomic-embed-text`'s asymmetric convention: documents get one
prefix, queries get a different one (`"search_query: "`) at retrieval time — that asymmetry is
part of why the same model can be used for both sides of retrieval without the index and the
query collapsing into a simple string-match.

## Step 10 — Store each vector as a raw blob

```python
for chunk_id, vector in zip(missing_ids, vectors):
    store.insert_embedding(chunk_id, model, vector)
    embeddings_created += 1
```
`insert_embedding` (`store.py:127-132`) does `INSERT OR REPLACE`, converting the Python float list
to bytes via `vector_to_blob` (`embeddings.py:53-54`:
`np.asarray(vector, dtype=np.float32).tobytes()`). No vector index is built — it's just raw bytes
in a BLOB column, later loaded wholesale into a NumPy matrix at retrieval time.

## Step 11 — Return stats, close, print

```python
return {"records_indexed": ..., "chunks_created": ..., "embeddings_created": ...}
```
Back in `cli.py:60-67`, `index_cmd()` closes the store and prints the three counts plus elapsed
wall-clock time (measured only around the `index_records()` call, not the file load).

## The shape of a typical run

| Scenario | records_indexed | chunks_created | embeddings_created |
|---|---|---|---|
| First `index` on fresh data | N (all) | sum of chunks across all records | same as chunks_created |
| Re-run, nothing changed | 0 | 0 | 0 |
| Re-run after `fetch`ing *more* records | only the new ones | only for new ones | only for new ones |
| Re-run after switching `MEDRAG_EMBED_MODEL` | 0 (chunks already exist) | 0 | **all** chunks (full re-embed) |
| Re-run after Ollama was down mid-run last time | 0 | 0 | just the ones that failed last time |

## Verify it yourself

```bash
sqlite3 data/index.db "select pmid, section, length(text) from chunks limit 10;"
medrag index   # re-run; should print 0, 0, 0 if nothing changed
```
This proves the idempotency chain (upsert → has-chunks check → missing-embedding check) held.
